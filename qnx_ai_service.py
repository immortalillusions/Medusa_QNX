"""QNX-resident Medusa service using checked-in TFLite models."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from medusa.arduino_serial import ArduinoController
from medusa.control import DEFAULT_ACTUAL_PORT, ControlClient
from medusa.handoff import DEFAULT_PATH as DEFAULT_HANDOFF_MAP, HandoffLearner
from medusa.decision import SCENE_CLEAR_FRAMES, SnooperSelector
from medusa.models import SnooperDecision
from medusa.nose_offset import measure_nose_offset
from medusa.preview import PreviewServer
from medusa.snooper import select_snoopers
from medusa.servo import ServoMapper
from medusa.tflite_detector import (
    DEFAULT_NUM_THREADS, DEFAULT_SCORE_THRESHOLD, TFLiteFaceGazeDetector, draw_face_boxes,
)
from medusa.trigger import (
    FALLBACK_REASON, FaceConfirm, FrameRate, LaptopTrigger, PanTracker, decide_aim, draw_aim_boxes,
)


def send(sock, target, active, target_x, reason, *, scene_clear: bool = False):
    event = {
        "type": "medusa-event",
        "active": active,
        "target_x": target_x,
        "reason": reason,
        "scene_clear": scene_clear,
    }
    sock.sendto(json.dumps(event).encode(), target)


# Flags of the old proportional pan step. Still accepted so existing launch lines run.
_DEPRECATED_PAN_FLAGS = ("--pan-gain", "--pan-deadband", "--pan-max-step")
# BlazeFace variants: short-range (128x128, faces within ~2 m) and back (256x256, longer range).
FACE_MODELS = {
    "short": Path("qnx/blaze_face_short_range.tflite"),
    "back": Path("qnx/face_detector.tflite"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    cameras = parser.add_mutually_exclusive_group()
    cameras.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    cameras.add_argument("--qnx-camera-unit", type=int, help="QNX Sensor Framework camera unit")
    parser.add_argument("--face-model", choices=sorted(FACE_MODELS),
                        help="BlazeFace variant: back (256x256, finds smaller/farther faces; default in "
                             "laptop-trigger mode) or short (128x128; default otherwise)")
    parser.add_argument("--face-detector", type=Path,
                        help="face model file; default: the --face-model file (the variant is read "
                             "from the model's input size)")
    parser.add_argument("--face-score", type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help="minimum BlazeFace score (sigmoid) for a face box")
    parser.add_argument("--tflite-threads", type=int, default=DEFAULT_NUM_THREADS,
                        help="TFLite interpreter threads for the face and landmark models")
    parser.add_argument("--face-landmarker", type=Path, default=Path("qnx/face_landmark_detector.tflite"))
    parser.add_argument("--event-host", help="laptop IP for medusa-event UDP; required unless --trigger-udp-port")
    parser.add_argument("--event-port", type=int, default=8766)
    parser.add_argument("--preview-port", type=int, default=8080, help="0 disables LAN preview")
    parser.add_argument("--min-faces", type=int, default=2,
                        help="screen-facing faces needed to trigger; 1 aims at any face (bench calibration only)")
    parser.add_argument("--hold-frames", type=int, default=8,
                        help="frames a snooper must persist before triggering (~0.5 s at 15 fps)")
    parser.add_argument("--arduino-port", help="QNX serial device connected to the Arduino, e.g. /dev/ser1")
    parser.add_argument("--control-port", type=int, metavar="PORT",
                        help="send targets to medusa_control.py on 127.0.0.1:PORT (e.g. 8768) instead of "
                             "opening the serial port; the control process owns it and the heartbeat")
    parser.add_argument("--actual-port", type=int, default=DEFAULT_ACTUAL_PORT,
                        help="localhost UDP port for the control process's medusa-actual packets "
                             "(real turret angle); 0 disables")
    parser.add_argument("--servo-min", type=int, default=65)
    parser.add_argument("--servo-max", type=int, default=115)
    parser.add_argument("--servo-home", type=int, default=90)
    parser.add_argument("--servo-reversed", action="store_true", help="reverse left/right aiming")
    parser.add_argument("--tilt-min", type=int, default=65)
    parser.add_argument("--tilt-max", type=int, default=115)
    parser.add_argument("--tilt-home", type=int, default=90)
    parser.add_argument(
        "--tilt-reversed", action="store_true",
        help="reverse up/down aiming; by default a face at the top of the frame maps to --tilt-min",
    )
    trigger = parser.add_argument_group(
        "laptop-trigger mode",
        "the laptop decides WHETHER (medusa-peeper UDP), the turret camera decides WHERE; "
        "no medusa-event packets are sent and --min-faces/--hold-frames are unused",
    )
    trigger.add_argument("--trigger-udp-port", type=int, metavar="PORT",
                         help="enable laptop-trigger mode, listening on 0.0.0.0:PORT (off by default)")
    trigger.add_argument("--trigger-from", metavar="IP", help="accept trigger packets only from this IP")
    trigger.add_argument("--trigger-timeout", type=float, default=0.5,
                         help="seconds a laptop 'active' message stays fresh")
    trigger.add_argument("--face-confirm-frames", type=int, default=2,
                         help="consecutive frames a turret face must be seen near the same spot "
                              "before it is aimed at")
    trigger.add_argument("--face-dropout-frames", type=int, default=2,
                         help="frames the last aimed face is kept through a detection dropout")
    turret = parser.add_argument_group(
        "camera on turret (laptop-trigger mode only)",
        "the camera pans with the turret, so pan tracks the face's world bearing: the Pi models the "
        "firmware's eased motion to know where the camera pointed, filters bearing and velocity, "
        "and leads the face; tilt still maps face y through the tilt range",
    )
    turret.add_argument("--camera-on-turret", action="store_true",
                        help="closed-loop pan for a camera clipped on the pan stage (off by default)")
    turret.add_argument("--camera-hfov", type=float, default=50.0,
                        help="camera horizontal field of view in degrees")
    turret.add_argument("--pan-mode", choices=("proportional", "predictive"), default="proportional",
                        help="proportional (default): nudge toward centring the face each frame, robust to "
                             "camera delay; predictive: model + filter + lead, needs an accurate --camera-latency")
    turret.add_argument("--pan-step-gain", type=float, default=0.12,
                        help="proportional mode: fraction of the remaining error corrected per 1/15 s "
                             "(frame-rate independent); higher is faster but overshoots with camera delay")
    turret.add_argument("--camera-latency", type=float, default=0.07,
                        help="seconds from frame capture to the pan update (about one frame)")
    turret.add_argument("--pan-lead", type=float, default=0.15,
                        help="seconds of face velocity to lead by, counted from frame capture")
    turret.add_argument("--pan-coast", type=float, default=0.3,
                        help="seconds to coast on the prediction after losing the target")
    turret.add_argument("--pan-hold", type=float, default=None,
                        help="seconds after losing the target before gliding home; "
                             "default and minimum: --pan-coast")
    turret.add_argument("--pan-return-speed", type=float, default=60.0,
                        help="degrees per second for the glide home")
    turret.add_argument("--handoff-map", default=DEFAULT_HANDOFF_MAP,
                        help="laptop_x -> pan map (python3 -m medusa.handoff), learned while tracking and "
                             "saved here every 5 s; with 2+ points a lost face is aimed at from the laptop's "
                             "x with the light on; a missing file means no map yet")
    turret.add_argument("--handoff-timeout", type=float, default=0.6,
                        help="seconds at the handoff angle without a face before the search sweep")
    turret.add_argument("--search-speed", type=float, default=60.0,
                        help="degrees per second for the search sweep when the laptop sees a peeper "
                             "but the turret camera does not")
    turret.add_argument("--no-search", action="store_true",
                        help="no handoff, search, map learning or laptop-guided fallback: coast, then glide "
                             "home, as before")
    for flag in _DEPRECATED_PAN_FLAGS:
        turret.add_argument(flag, type=float, default=None,
                            help="deprecated and ignored: the bearing tracker replaced the proportional step")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.trigger_udp_port is None and args.event_host is None:
        parser.error("the following arguments are required: --event-host")
    if args.trigger_timeout <= 0:
        parser.error("--trigger-timeout must be positive")
    if not 0 < args.face_score < 1:
        parser.error("--face-score must be between 0 and 1")
    if args.tflite_threads < 1:
        parser.error("--tflite-threads must be at least 1")
    if args.face_confirm_frames < 1 or args.face_dropout_frames < 0:
        parser.error("--face-confirm-frames must be at least 1 and --face-dropout-frames not negative")
    if args.face_model is None:
        args.face_model = "back" if args.trigger_udp_port is not None else "short"
    if args.face_detector is None:
        args.face_detector = FACE_MODELS[args.face_model]
    if args.camera_on_turret and args.trigger_udp_port is None:
        parser.error("--camera-on-turret requires --trigger-udp-port")
    if args.camera_hfov <= 0 or args.pan_return_speed <= 0:
        parser.error("--camera-hfov and --pan-return-speed must be positive")
    if min(args.camera_latency, args.pan_lead, args.pan_coast, args.pan_hold or 0.0) < 0:
        parser.error("--camera-latency, --pan-lead, --pan-coast and --pan-hold must not be negative")
    if args.search_speed < 0 or args.handoff_timeout < 0:
        parser.error("--search-speed and --handoff-timeout must not be negative")
    if args.control_port is not None and args.arduino_port:
        print("warning: --control-port given: --arduino-port is ignored (the control process owns it)",
              file=sys.stderr)
    for flag in _DEPRECATED_PAN_FLAGS:
        if getattr(args, flag[2:].replace("-", "_")) is not None:
            print(f"warning: {flag} is deprecated and ignored", file=sys.stderr)
    return args


def _laptop_x(trigger: LaptopTrigger, laptop_active: bool) -> float | None:
    return trigger.latest.target_x if laptop_active and trigger.latest is not None else None


def _trigger_status(trigger: LaptopTrigger, laptop_active: bool, aim, pan=None, pan_error=None,
                    fps: float | None = None, learned_points: int | None = None) -> dict:
    age = trigger.age()
    status = {
        "mode": "laptop-trigger",
        "fps": fps,
        "laptop_active": laptop_active,
        "laptop_age_s": round(age, 3) if age is not None else None,
        "laptop_target_x": _laptop_x(trigger, laptop_active),
        "aim_target": [aim.target_x, aim.target_y] if aim is not None and aim.active else None,
    }
    if pan is not None:
        status["pan_angle"] = round(pan.angle, 1)
        status["pan_error"] = round(pan_error, 4) if pan_error is not None else None
        status["pan_bearing"] = round(pan.bearing, 1) if pan.bearing is not None else None
        status["pan_velocity"] = round(pan.velocity, 1)
        status["pan_state"] = pan.state
        status["pan_actual"] = round(pan.actual[1], 1) if pan.actual is not None else None
        status["fallback_aim"] = pan.state == "fallback"
        status["learned_points"] = learned_points
    return status


def _pan_tracker(args) -> PanTracker | None:
    if not args.camera_on_turret:
        return None
    return PanTracker(
        args.servo_min, args.servo_max, args.servo_home, reversed=args.servo_reversed,
        hfov_deg=args.camera_hfov, latency_s=args.camera_latency, lead_s=args.pan_lead,
        coast_s=args.pan_coast, hold_s=args.pan_hold, return_speed=args.pan_return_speed,
        search_speed=0.0 if args.no_search else args.search_speed, handoff_timeout=args.handoff_timeout,
        mode=args.pan_mode, gain=args.pan_step_gain,
    )


def run_laptop_trigger(args, camera, detector, preview, arduino) -> int:
    """Aim at the turret camera's smallest face while the laptop reports a fresh peeper."""
    import cv2

    trigger = None
    pan = _pan_tracker(args)
    confirm = FaceConfirm(args.face_confirm_frames, args.face_dropout_frames)
    rate = FrameRate()
    handoff = HandoffLearner(args.handoff_map, minimum=args.servo_min, maximum=args.servo_max) if (
        pan and pan.search_speed > 0) else None
    try:
        try:
            trigger = LaptopTrigger.bind(
                args.trigger_udp_port, timeout=args.trigger_timeout, allowed_sender=args.trigger_from,
            )
        except OSError as error:
            print(f"Medusa cannot start safely: laptop trigger port {args.trigger_udp_port} "
                  f"unavailable: {error}", file=sys.stderr)
            return 2
        while True:
            ok, frame = camera.read()
            trigger.poll()
            laptop_active, laptop_reason = trigger.status()
            if not ok:
                confirm.reset()
                if pan:
                    pan.reset()  # A camera fault is not a brief detection drop: park, matching safe_stop.
                if arduino:
                    arduino.safe_stop()
                if preview:
                    preview.state.update_status(False, "camera frame unavailable",
                                                extra=_trigger_status(trigger, laptop_active, None, pan,
                                                                      fps=rate.fps(), learned_points=(
                                                                          handoff.learned_points
                                                                          if handoff else None)))
                continue
            rate.tick()
            height, width = frame.shape[:2]
            try:
                boxes = detector.face_detector.detect(frame)
            except (RuntimeError, ValueError, cv2.error):
                boxes = None
            aim = decide_aim(laptop_active, laptop_reason, boxes, width, height, confirm=confirm)
            pan_error = learned = laptop_x = None
            if pan:
                actual = arduino.poll_actual() if arduino else None
                if actual is not None:
                    pan.observe_actual(actual.pan, actual.t, actual.armed)
                else:
                    pan.observe_actual(None, None)
                laptop_x = _laptop_x(trigger, laptop_active)
                learned = handoff.current() if handoff else None
                # A held target is a stale image position on a camera that has since moved: the
                # tracker coasts on its own prediction instead.
                pan_error = pan.update(
                    aim.target_x if aim.active and not aim.held else None,
                    laptop_active=laptop_active, laptop_x=laptop_x,
                    handoff_angle=learned.angle(laptop_x) if learned else None,
                    fallback_map=learned.angle if learned else None,
                )
                if handoff:
                    handoff.observe(pan.last_update, laptop_x, pan_error, pan.real_pan())
                    handoff.maybe_save(pan.last_update)
            # The light is on for a turret-camera face and in laptop-guided fallback (user decision),
            # and through the short coast in between when fallback is ready, so it does not blink.
            fallback = pan is not None and pan.state == "fallback"
            light = aim.active or fallback or (
                pan is not None and pan.state == "coasting" and learned is not None
                and learned.angle(laptop_x) is not None)
            if arduino and pan:
                pitch, _ = arduino.tilt_mapper.command(aim.active, aim.target_y)
                arduino.send_angles(light, pan.command, pitch)
            elif arduino:
                arduino.send(aim.active, aim.target_x, aim.target_y)
            if preview:
                nose_offset = None
                if aim.target_box is not None and aim.target_box.nose is not None:
                    nose_offset = measure_nose_offset(aim.target_box.nose, width, height)
                if boxes:
                    draw_aim_boxes(frame, boxes, aim.target_box)
                preview.state.update_frame(frame)
                preview.state.update_status(
                    light, FALLBACK_REASON if fallback else aim.reason,
                    face_count=len(boxes) if boxes is not None else None,
                    snooper_count=1 if aim.active else 0,
                    nose_offset=nose_offset,
                    extra=_trigger_status(trigger, laptop_active, aim, pan, pan_error, fps=rate.fps(),
                                          learned_points=handoff.learned_points if handoff else None),
                )
    except KeyboardInterrupt:
        pass
    finally:
        if trigger:
            trigger.close()
        if handoff and pan and pan.last_update is not None:
            handoff.maybe_save(pan.last_update, force=True)  # Keep what was learned since the last save.
        if arduino:
            arduino.safe_stop()
            arduino.close()
        camera.release()
        if preview:
            preview.stop()
    return 0


def main() -> int:
    args = parse_args()
    if not args.face_detector.is_file() or not args.face_landmarker.is_file():
        print("Medusa cannot start safely: required TFLite model is missing.", file=sys.stderr)
        return 2
    try:
        import cv2
        detector = TFLiteFaceGazeDetector(str(args.face_detector), str(args.face_landmarker),
                                          score_threshold=args.face_score, num_threads=args.tflite_threads)
        if args.qnx_camera_unit is None:
            camera = cv2.VideoCapture(args.camera)
            if not camera.isOpened():
                camera.release()
                raise RuntimeError("OpenCV camera is unavailable")
        else:
            from medusa.qnx_camera import QnxCamera

            camera = QnxCamera(args.qnx_camera_unit)
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        print(f"Medusa cannot start safely: {error}", file=sys.stderr)
        return 2

    preview = PreviewServer(args.preview_port) if args.preview_port else None
    if preview:
        preview.start()
    arduino = None
    if args.arduino_port or args.control_port is not None:
        try:
            mapper = ServoMapper(
                minimum=args.servo_min,
                maximum=args.servo_max,
                home=args.servo_home,
                reversed=args.servo_reversed,
            )
            if not mapper.minimum <= mapper.home <= mapper.maximum:
                raise ValueError("servo home must be inside the safe servo range")
            tilt_mapper = ServoMapper(
                minimum=args.tilt_min,
                maximum=args.tilt_max,
                home=args.tilt_home,
                reversed=args.tilt_reversed,
            )
            if args.control_port is not None:
                arduino = ControlClient(args.control_port, mapper, tilt_mapper,
                                        actual_port=args.actual_port or None)
            else:
                arduino = ArduinoController(args.arduino_port, mapper, tilt_mapper=tilt_mapper)
            arduino.safe_stop()
        except (OSError, ValueError, AttributeError) as error:
            camera.release()
            if preview:
                preview.stop()
            what = "control link" if args.control_port is not None else "Arduino serial"
            print(f"Medusa cannot start safely: {what} setup failed: {error}", file=sys.stderr)
            return 2
    if args.trigger_udp_port is not None:
        return run_laptop_trigger(args, camera, detector, preview, arduino)
    selector, target = SnooperSelector(args.hold_frames, min_viewers=args.min_faces), (args.event_host, args.event_port)
    clear_frames = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as events:
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    clear_frames = 0
                    send(events, target, False, None, "camera frame unavailable")
                    if arduino:
                        arduino.safe_stop()
                    if preview:
                        preview.state.update_status(False, "camera frame unavailable")
                    continue
                boxes = None
                try:
                    boxes = detector.face_detector.detect(frame)
                    decision = selector.update(detector.detect(frame, boxes=boxes))
                except (RuntimeError, ValueError, cv2.error):
                    decision = SnooperDecision(active=False, reason="AI inference unavailable")
                # Count clear frames at the camera, where missed UDP packets cannot affect the streak.
                if boxes is not None and len(boxes) < 2:
                    clear_frames = min(clear_frames + 1, SCENE_CLEAR_FRAMES)
                else:
                    clear_frames = 0
                send(events, target, decision.active, decision.target_x, decision.reason,
                     scene_clear=clear_frames == SCENE_CLEAR_FRAMES)
                if arduino:
                    arduino.send(decision.active, decision.target_x, decision.target_y)
                if preview:
                    snoopers = select_snoopers(boxes) if boxes is not None else []
                    closest_snooper = snoopers[0] if snoopers else None
                    nose_offset = None
                    if closest_snooper is not None and closest_snooper.nose is not None:
                        height, width = frame.shape[:2]
                        nose_offset = measure_nose_offset(closest_snooper.nose, width, height)
                    if boxes:
                        draw_face_boxes(frame, boxes, snoopers=snoopers)
                    preview.state.update_frame(frame)
                    preview.state.update_status(
                        decision.active, decision.reason,
                        face_count=len(boxes) if boxes is not None else None,
                        snooper_count=len(snoopers),
                        nose_offset=nose_offset,
                    )
        except KeyboardInterrupt:
            pass
        finally:
            send(events, target, False, None, "QNX service stopped")
            if arduino:
                arduino.safe_stop()
                arduino.close()
            camera.release()
            if preview:
                preview.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

