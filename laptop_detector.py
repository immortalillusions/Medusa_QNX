"""Decide on the MacBook whether a peeper is looking, and tell the Pi every frame.

The built-in camera above the screen sees who is looking at the laptop. The default
FaceLandmarker path judges gaze (head pose + iris), so a side-eye with the head turned away
still counts; --detector blazeface keeps the old head-direction-only rule. The Pi aims the
turret with its own camera; target_x/target_y here are informational only.

The decision comes from medusa.peeper_tracker: faces keep an id across frames, gaze is
smoothed per face, and ACTIVE needs --enter-s of looking and ends after --exit-s away.

The Google Docs switch is gated on the turret being ON (armed, gate closed), read from the
Pi's /control.json; --docs-always skips the gate. Detection, heartbeats and the preview run
either way.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import socket
import sys
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from medusa.actions import PrivacyAction
from medusa.decision import screen_facing_from_landmarks
from medusa.gaze import NOSE_TIP, GazeConfig, GazeEstimate, estimate_gaze
from medusa.models import FaceObservation, SnooperDecision
from medusa.peeper_tracker import Detection, PeeperTracker, Track, TrackerConfig
from medusa.preview import PreviewServer, RuntimeSettings
from qnx.face_webcam import detect_faces  # lazy cv2/mediapipe imports inside


_LOG = logging.getLogger("laptop_detector")
SOURCE = "mac-camera"
MODELS = {
    "landmarker": Path(__file__).resolve().parent / "qnx" / "face_landmarker.task",
    "blazeface": Path(__file__).resolve().parent / "qnx" / "blaze_face_short_range.tflite",
}
MAX_FACES = 4
PEEPER_COLOR = (0, 0, 255)  # BGR red
USER_COLOR = (0, 255, 0)
AWAY_COLOR = (160, 160, 160)
DEBUG_GAZE_INTERVAL_S = 1.0
CONTROL_PORT = 8082
CONTROL_POLL_S = 0.5
CONTROL_TIMEOUT_S = 1.0
CONTROL_STALE_S = 2.0


@dataclass(frozen=True)
class TrackedFace:
    """One face this frame: the selector's observation plus what the preview draws."""

    raw: object
    observation: FaceObservation
    box: tuple[int, int, int, int]  # left, top, width, height in pixels
    nose: tuple[int, int] | None = None
    gaze: GazeEstimate | None = None


def face_observations(detections, width: int, height: int) -> list[TrackedFace]:
    """BlazeFace detections -> faces, head direction only (same rules as face_webcam)."""
    faces = []
    for detection in detections:
        box = detection.bounding_box
        points = detection.keypoints or ()
        if len(points) < 3 or box.width <= 0 or box.height <= 0 or not detection.categories:
            continue
        face_scale = width / box.width
        eye_a, eye_b, nose = points[:3]
        facing = screen_facing_from_landmarks(
            nose.x * face_scale, eye_a.x * face_scale, eye_b.x * face_scale,
        )
        if facing is None:
            continue
        faces.append(TrackedFace(detection, FaceObservation(
            source=SOURCE,
            center_x=(box.origin_x + box.width / 2) / width,
            center_y=(box.origin_y + box.height / 2) / height,
            relative_size=box.width / width,
            screen_facing=facing,
            confidence=detection.categories[0].score,
        ), (box.origin_x, box.origin_y, box.width, box.height)))
    return faces


def landmarker_faces(result, width: int, height: int,
                     config: GazeConfig = GazeConfig()) -> list[TrackedFace]:
    """FaceLandmarker result -> faces judged by gaze. Faces without a pose matrix are skipped."""
    matrices = result.facial_transformation_matrixes or []
    faces = []
    for landmarks, matrix in zip(result.face_landmarks or [], matrices):
        if not landmarks:
            continue
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
        if right <= left or bottom <= top:
            continue
        gaze = estimate_gaze(landmarks, matrix, config, width, height)
        nose = landmarks[NOSE_TIP] if len(landmarks) > NOSE_TIP else None
        faces.append(TrackedFace(landmarks, FaceObservation(
            source=SOURCE,
            center_x=(left + right) / 2,
            center_y=(top + bottom) / 2,
            relative_size=right - left,
            screen_facing=gaze.looking,
            confidence=1.0,  # the landmarker reports no per-face score
        ), (round(left * width), round(top * height),
            round((right - left) * width), round((bottom - top) * height)),
            nose=None if nose is None else (round(nose.x * width), round(nose.y * height)),
            gaze=gaze))
    return faces


def detection(face: TrackedFace, width: int, height: int) -> Detection:
    _, _, box_w, box_h = face.box
    obs = face.observation
    return Detection(obs.center_x, obs.center_y, box_w / width, box_h / height,
                     looking=obs.screen_facing, gaze=face.gaze, source=obs.source)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def face_status(face: TrackedFace, track: Track, now: float) -> dict:
    """Position is this frame's; looking and gaze are the track's smoothed values."""
    return {
        "id": track.id,
        "x": round(face.observation.center_x, 3),
        "y": round(face.observation.center_y, 3),
        "looking": track.looking,
        "looking_s": round(track.looking_s(now), 2),
        "gaze_yaw": _round(track.gaze_yaw),
        "gaze_pitch": _round(track.gaze_pitch),
        "method": track.method,
    }


def heartbeat(decision: SnooperDecision, seq: int, user_x: float | None = None,
              peeper_id: int | None = None) -> bytes:
    message = {
        "type": "medusa-peeper",
        "active": decision.active,
        "target_x": decision.target_x if decision.active else None,
        "target_y": decision.target_y if decision.active else None,
        "seq": seq,
    }
    if user_x is not None:
        message["user_x"] = user_x  # optional: the user's (largest track's) x, for the Pi's matcher
    if decision.active and peeper_id is not None:
        message["peeper_id"] = peeper_id  # optional: a change tells the Pi to drop its face lock
    return json.dumps(message).encode()


def turret_on_from_control(payload) -> tuple[bool, str]:
    """/control.json -> (on, why). ON = Arduino link up, armed, and the gate not open.

    A missing or null ``gate_closed`` (older firmware) counts as closed, but only when armed.
    Raises ValueError when there is no "arduino" object.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("arduino"), dict):
        raise ValueError("no arduino object")
    arduino = payload["arduino"]
    if arduino.get("link_ok") is not True:
        return False, "Arduino link down"
    if arduino.get("armed") is not True:
        return False, "disarmed"
    if arduino.get("gate_closed") is False:
        return False, "gate open"
    return True, "armed, gate closed"


class TurretPoller:
    """Polls the Pi's /control.json on a daemon thread. Fails safe: any doubt reads OFF.

    A failed or malformed read is OFF at once; a good read older than ``stale_s`` is OFF too
    (covers a wedged poll thread).
    """

    def __init__(self, url: str, *, interval_s: float = CONTROL_POLL_S,
                 timeout_s: float = CONTROL_TIMEOUT_S, stale_s: float = CONTROL_STALE_S,
                 opener=urllib.request.urlopen, clock=time.monotonic) -> None:
        self.url = url
        self.interval_s, self.timeout_s, self.stale_s = interval_s, timeout_s, stale_s
        self.opener, self.clock = opener, clock
        self._lock = threading.Lock()
        self._on, self._reason, self._good_at = False, "no control.json read yet", None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> bool:
        try:
            with self.opener(self.url, timeout=self.timeout_s) as response:
                payload = json.loads(response.read(65536))
            on, reason = turret_on_from_control(payload)
        except Exception as error:  # unreachable, timeout, HTTP error, bad JSON: all OFF
            with self._lock:
                self._on, self._reason = False, f"control.json unreadable: {error}"
            return False
        with self._lock:
            self._on, self._reason, self._good_at = on, reason, self.clock()
        return on

    def status(self) -> tuple[bool, str]:
        with self._lock:
            on, reason, good_at = self._on, self._reason, self._good_at
        if on and (good_at is None or self.clock() - good_at > self.stale_s):
            return False, "control.json stale"
        return on, reason

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_once()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="medusa-turret-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


class PiLink:
    """Fire-and-forget UDP heartbeats. A send error is logged once, never raised."""

    def __init__(self, host: str, port: int, sock: socket.socket | None = None) -> None:
        self.address = (host, port)
        self.sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self._failing = False

    def send(self, decision: SnooperDecision, user_x: float | None = None,
             peeper_id: int | None = None) -> None:
        self.seq += 1
        try:
            self.sock.sendto(heartbeat(decision, self.seq, user_x, peeper_id), self.address)
        except OSError as error:
            if not self._failing:
                _LOG.warning("UDP send to %s:%d failed: %s", *self.address, error)
            self._failing = True
            return
        if self._failing:
            _LOG.info("UDP send to %s:%d recovered.", *self.address)
            self._failing = False

    def close(self) -> None:
        self.sock.close()


class LaptopDetector:
    """One frame in, one decision out: drives the docs action, the Pi link and the preview.

    ``extract`` turns the raw detector output into TrackedFaces (BlazeFace by default).
    ``clock`` must advance between frames: every dwell and timeout is in seconds.
    ``turret`` (a TurretPoller) gates the docs switch: it fires once per encounter, when the
    peeper is active and the turret is ON, whichever came second. No turret = OFF, unless
    ``docs_always``. ``settings`` (RuntimeSettings, from the dashboard) are applied at the start
    of each step: peeper mode one/two (tracker min_faces 1/2) and screen switching on/off.
    """

    def __init__(self, action: PrivacyAction, link: PiLink | None, preview_state=None,
                 tracker: PeeperTracker | None = None, extract=face_observations,
                 debug_gaze: bool = False, clock=time.monotonic,
                 turret: TurretPoller | None = None, docs_always: bool = False,
                 settings: RuntimeSettings | None = None) -> None:
        self.tracker = tracker or PeeperTracker()
        self.action = action
        self.link = link
        self.preview_state = preview_state
        self.extract = extract
        self.debug_gaze = debug_gaze
        self.clock = clock
        self.turret = turret
        self.docs_always = docs_always
        self.settings = settings
        self.screen_switch = True  # dashboard switch; OFF never switches to Docs
        self.user_id: int | None = None
        self.active = False
        self.turret_on = False
        self._docs_latched = False  # this encounter has met the gate; cleared when it ends
        self.faces: list[TrackedFace] = []
        self.face_tracks: list[Track] = []  # aligned with self.faces
        self._frame_times: deque[float] = deque(maxlen=30)
        self._last_debug = float("-inf")

    def fps(self) -> float | None:
        times = self._frame_times
        if len(times) < 2 or times[-1] <= times[0]:
            return None
        return round((len(times) - 1) / (times[-1] - times[0]), 1)

    @property
    def peeper_mode(self) -> str:
        return "one" if self.tracker.config.min_faces <= 1 else "two"

    def _apply_settings(self) -> None:
        update = self.settings.take() if self.settings is not None else {}
        mode = update.get("peeper_mode")
        if mode is not None and mode != self.peeper_mode:
            self.tracker.config = dataclasses.replace(self.tracker.config,
                                                      min_faces=1 if mode == "one" else 2)
            # Who may be the peeper changed: drop the lock so the next frame decides afresh.
            self.tracker.peeper_id = None
            _LOG.info("peeper mode: %s-person", mode)
        if "screen_switch" in update and update["screen_switch"] != self.screen_switch:
            self.screen_switch = update["screen_switch"]
            _LOG.info("screen switching %s", "ON" if self.screen_switch else "OFF")

    def step(self, raw, width: int, height: int) -> tuple[SnooperDecision, TrackedFace | None]:
        """Returns the decision and the peeper's face (None unless active)."""
        self._apply_settings()
        now = self.clock()
        self._frame_times.append(now)
        faces = self.faces = self.extract(raw, width, height)
        result = self.tracker.update([detection(face, width, height) for face in faces], now,
                                     aspect=height / width if width else 1.0)
        decision, tracks = result.decision, result.tracks
        self.face_tracks = tracks
        self.user_id = result.user_id if self.peeper_mode == "two" else None  # one-person: no USER
        peeper = None
        if decision.active:
            peeper = next((face for face, track in zip(faces, tracks)
                           if track.id == result.peeper_id), None)
        if decision.active != self.active:
            if decision.active:
                _LOG.info("ACTIVE: peeper %s at x=%.2f y=%.2f (%s)", result.peeper_id,
                          decision.target_x, decision.target_y, decision.reason)
            else:
                _LOG.info("inactive: %s", decision.reason)
            self.active = decision.active
        if self.debug_gaze and now - self._last_debug >= DEBUG_GAZE_INTERVAL_S:
            self._last_debug = now
            self._log_gaze(faces, tracks, now, result.user_id)
        docs_enabled = self._update_gate()
        # Latched for the whole encounter, so a turret flicker cannot re-fire the switch and
        # PrivacyAction's own rising edge stays once per encounter.
        self._docs_latched = decision.active and (self._docs_latched or docs_enabled)
        self.action.update(self._docs_latched)
        if self.link is not None:
            user = next((face for face, track in zip(faces, tracks)
                         if track.id == result.user_id), None)
            self.link.send(decision, None if user is None else user.observation.center_x,
                           result.peeper_id)
        if self.preview_state is not None:
            self.preview_state.update_status(
                decision.active, decision.reason, face_count=len(faces),
                snooper_count=1 if decision.active else 0,
                extra={"faces": [face_status(face, track, now) for face, track in zip(faces, tracks)],
                       "fps": self.fps(), "peeper_id": result.peeper_id,
                       "turret_on": self.turret_on, "docs_enabled": docs_enabled,
                       "peeper_mode": self.peeper_mode, "screen_switch": self.screen_switch,
                       "docs_available": bool(self.action.enabled), "docs_fired": self._docs_latched},
            )
        return decision, peeper

    def _update_gate(self) -> bool:
        """Refresh turret_on (logging changes); True when a docs switch is allowed now."""
        on, reason = self.turret.status() if self.turret is not None else (False, "no turret poll")
        if on != self.turret_on:
            _LOG.info("turret %s (%s): docs switching %s", "ON" if on else "OFF", reason,
                      "enabled" if on or self.docs_always else "paused")
            self.turret_on = on
        return bool(self.action.enabled) and self.screen_switch and (on or self.docs_always)

    def _log_gaze(self, faces: list[TrackedFace], tracks: list[Track], now: float,
                  user_id: int | None) -> None:
        if not faces:
            _LOG.info("gaze: no faces")
        for face, track in zip(faces, tracks):
            gaze = face.gaze
            role = "user" if track.id == user_id else "other"
            state = "LOOKING %.1fs" % track.looking_s(now) if track.looking else "away"
            if gaze is None:
                _LOG.info("gaze[id %d %s] x=%.2f size=%.2f head-only facing=%s %s", track.id,
                          role, face.observation.center_x, face.observation.relative_size,
                          face.observation.screen_facing, state)
                continue
            _LOG.info("gaze[id %d %s] x=%.2f size=%.2f head %+.0f/%+.0f eye %+.0f/%+.0f "
                      "raw %+.0f/%+.0f smooth %+.0f/%+.0f (yaw/pitch) %s %s", track.id, role,
                      face.observation.center_x, face.observation.relative_size,
                      gaze.head_yaw, gaze.head_pitch, gaze.eye_yaw, gaze.eye_pitch,
                      gaze.gaze_yaw, gaze.gaze_pitch, track.gaze_yaw, track.gaze_pitch,
                      track.method, state)

    def shutdown(self) -> None:
        if self.link is not None:
            self.link.send(SnooperDecision(False, reason="detector stopped"))


def draw(frame, faces: list[TrackedFace], peeper: TrackedFace | None,
         tracks: list[Track]) -> None:
    """Box per face (green looking / red PEEPER / grey not looking), smoothed gaze arrow, tag."""
    import cv2

    for face, track in zip(faces, tracks):
        left, top, box_w, box_h = face.box
        is_peeper = face is peeper
        looking = track.looking
        color = PEEPER_COLOR if is_peeper else USER_COLOR if looking else AWAY_COLOR
        cv2.rectangle(frame, (left, top), (left + box_w, top + box_h), color, 3 if is_peeper else 2)
        tag = f"#{track.id} " + ("looking" if looking else "not looking")
        if track.gaze_yaw is not None:
            tag += f" {track.gaze_yaw:+.0f}/{track.gaze_pitch:+.0f}"
        tag += f" | {track.method}"
        if is_peeper:
            tag = "PEEPER " + tag
        cv2.putText(frame, tag, (max(0, left), max(18, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if track.gaze_yaw is not None and face.nose is not None:
            # Length grows with the angle: a face width at 60 degrees.
            scale = box_w / 60.0
            tip = (round(face.nose[0] + track.gaze_yaw * scale),
                   round(face.nose[1] - track.gaze_pitch * scale))
            cv2.arrowedLine(frame, face.nose, tip, color, 2, tipLength=0.25)


def detect_landmarks(landmarker, frame, timestamp_ms: int):
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(image, timestamp_ms)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="AVFoundation camera index (default: 0)")
    parser.add_argument("--pi-host", help="Pi address for the UDP heartbeat (required unless --no-pi)")
    parser.add_argument("--pi-port", type=int, default=8767)
    parser.add_argument("--no-pi", action="store_true", help="do not send heartbeats")
    parser.add_argument("--no-browser", action="store_true", help="do not switch to Google Docs")
    parser.add_argument("--docs-always", action="store_true",
                        help="switch to Google Docs even when the turret is off (bench use)")
    parser.add_argument("--control-url",
                        help="the Pi's control status, polled to gate the docs switch "
                             f"(default: http://<pi-host>:{CONTROL_PORT}/control.json)")
    parser.add_argument("--preview-port", type=int, default=8081)
    parser.add_argument("--detector", choices=sorted(MODELS), default="landmarker",
                        help="landmarker = gaze (head + iris, default); blazeface = head direction only")
    parser.add_argument("--model", type=Path, help="model file (default: the one for --detector)")
    parser.add_argument("--min-faces", type=int, default=2,
                        help="faces needed to trigger (the largest is the user, never the peeper); "
                             "1 = any face may be the peeper (bench calibration only)")
    tracking = TrackerConfig()
    parser.add_argument("--enter-s", type=float, default=tracking.enter_s,
                        help=f"seconds one face must keep looking before triggering (default: {tracking.enter_s:g})")
    parser.add_argument("--exit-s", type=float, default=tracking.exit_s,
                        help=f"seconds the peeper must be away before releasing (default: {tracking.exit_s:g})")
    parser.add_argument("--gaze-yaw-margin", type=float, default=tracking.yaw_margin,
                        help="extra |yaw| degrees before a looking face counts as away "
                             f"(default: {tracking.yaw_margin:g})")
    parser.add_argument("--gaze-pitch-margin", type=float, default=tracking.pitch_margin,
                        help="extra pitch degrees outside the range before a looking face counts "
                             f"as away (default: {tracking.pitch_margin:g})")
    defaults = GazeConfig()
    parser.add_argument("--eye-gain", type=float, default=defaults.eye_gain_deg,
                        help=f"eye-in-head degrees at full iris deflection (default: {defaults.eye_gain_deg:g})")
    parser.add_argument("--gaze-yaw-max", type=float, default=defaults.yaw_max_deg,
                        help=f"|gaze yaw| in degrees that counts as on-screen (default: {defaults.yaw_max_deg:g})")
    parser.add_argument("--gaze-pitch-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=(defaults.pitch_min_deg, defaults.pitch_max_deg),
                        help="gaze pitch in degrees that counts as on-screen, down is negative "
                             f"(default: {defaults.pitch_min_deg:g} {defaults.pitch_max_deg:g})")
    parser.add_argument("--debug-gaze", action="store_true",
                        help="log every face's id, raw and smoothed gaze angles once per second")
    args = parser.parse_args(argv)
    if not args.no_pi and not args.pi_host:
        parser.error("--pi-host is required unless --no-pi is given")
    model = args.model or MODELS[args.detector]
    if not model.is_file():
        parser.error(f"Face model not found: {model}")
    if args.enter_s < 0 or args.exit_s < 0 or args.gaze_yaw_margin < 0 or args.gaze_pitch_margin < 0:
        parser.error("--enter-s, --exit-s and the gaze margins must not be negative")
    if args.min_faces < 1:
        parser.error("--min-faces must be at least 1")
    pitch_min, pitch_max = args.gaze_pitch_range
    if pitch_min > pitch_max:
        parser.error("--gaze-pitch-range MIN must not exceed MAX")
    gaze_config = GazeConfig(eye_gain_deg=args.eye_gain, yaw_max_deg=args.gaze_yaw_max,
                             pitch_min_deg=pitch_min, pitch_max_deg=pitch_max)
    tracker_config = TrackerConfig(
        yaw_max=args.gaze_yaw_max, pitch_min=pitch_min, pitch_max=pitch_max,
        yaw_margin=args.gaze_yaw_margin, pitch_margin=args.gaze_pitch_margin,
        enter_s=args.enter_s, exit_s=args.exit_s, min_faces=args.min_faces,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        print(f"Could not load OpenCV/MediaPipe with {sys.executable}: {error}\n"
              "Run it with the camvenv Python (see run_laptop_detector.command).", file=sys.stderr)
        return 1

    camera = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not camera.isOpened():
        camera.release()
        print(f"Could not open Mac camera {args.camera}.\n"
              "macOS needs camera permission for the app running this script (usually Terminal):\n"
              "System Settings > Privacy & Security > Camera > turn on Terminal, then quit and\n"
              "reopen Terminal and run it again. Also check that no other app holds the camera.",
              file=sys.stderr)
        return 2

    link = None if args.no_pi else PiLink(args.pi_host, args.pi_port)
    control_url = args.control_url or (
        f"http://{args.pi_host}:{CONTROL_PORT}/control.json" if args.pi_host else None)
    turret = None if control_url is None else TurretPoller(control_url)
    if turret is None and not args.docs_always and not args.no_browser:
        _LOG.warning("No --pi-host or --control-url: the turret reads OFF, so Docs never "
                     "switches (pass --docs-always on the bench).")
    preview = None
    detector = None
    vision = mp.tasks.vision
    landmarker = args.detector == "landmarker"
    try:
        settings = RuntimeSettings()
        preview = PreviewServer(args.preview_port, settings)
        preview.start()
        tracker = PeeperTracker(tracker_config)
        extract = partial(landmarker_faces, config=gaze_config) if landmarker else face_observations
        if turret is not None:
            turret.start()
        detector = LaptopDetector(PrivacyAction(enabled=not args.no_browser), link, preview.state,
                                  tracker=tracker, extract=extract, debug_gaze=args.debug_gaze,
                                  turret=turret, docs_always=args.docs_always, settings=settings)
        _LOG.info("Camera %d open, %s. Preview http://localhost:%d/stream.mjpg . Heartbeats to %s. "
                  "Docs switch: %s.", args.camera, args.detector, args.preview_port,
                  "nobody (--no-pi)" if link is None else f"{args.pi_host}:{args.pi_port}",
                  "off (--no-browser)" if args.no_browser else
                  "always (--docs-always)" if args.docs_always else
                  "never (no control URL)" if turret is None else f"when the turret is on ({control_url})")
        base = mp.tasks.BaseOptions(model_asset_path=str(model))
        if landmarker:
            options = vision.FaceLandmarkerOptions(
                base_options=base, running_mode=vision.RunningMode.VIDEO, num_faces=MAX_FACES,
                output_facial_transformation_matrixes=True,
            )
            model_context = vision.FaceLandmarker.create_from_options(options)
        else:
            options = vision.FaceDetectorOptions(base_options=base, min_detection_confidence=0.5)
            model_context = vision.FaceDetector.create_from_options(options)
        last_ms = -1
        with model_context as face_model:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print(f"Camera {args.camera} stopped delivering frames.", file=sys.stderr)
                    return 1
                height, width = frame.shape[:2]
                if landmarker:
                    # VIDEO mode needs strictly increasing timestamps.
                    last_ms = max(last_ms + 1, int(time.monotonic() * 1000))
                    raw = detect_landmarks(face_model, frame, last_ms)
                else:
                    raw = detect_faces(face_model, frame)
                _, peeper = detector.step(raw, width, height)
                draw(frame, detector.faces, peeper, detector.face_tracks)
                preview.state.update_frame(frame)
    except KeyboardInterrupt:
        _LOG.info("Stopping.")
        return 0
    except OSError as error:
        print(f"Laptop detector failed: {error}", file=sys.stderr)
        return 1
    finally:
        if detector is not None:
            detector.shutdown()
        elif link is not None:
            link.send(SnooperDecision(False, reason="detector stopped"))
        if link is not None:
            link.close()
        camera.release()
        if turret is not None:
            turret.stop()
        if preview is not None:
            preview.stop()


if __name__ == "__main__":
    raise SystemExit(main())
