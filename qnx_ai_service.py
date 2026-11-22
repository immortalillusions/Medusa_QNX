"""QNX-resident Medusa service using checked-in TFLite models."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from medusa.arduino_serial import ArduinoController
from medusa.decision import SnooperSelector
from medusa.models import SnooperDecision
from medusa.nose_offset import measure_nose_offset
from medusa.preview import PreviewServer
from medusa.snooper import select_snooper
from medusa.servo import ServoMapper
from medusa.tflite_detector import TFLiteFaceGazeDetector, draw_face_boxes


def send(sock, target, active, target_x, reason):
    sock.sendto(json.dumps({"type":"medusa-event","active":active,
                "target_x":target_x,"reason":reason}).encode(), target)


def main() -> int:
    parser = argparse.ArgumentParser()
    cameras = parser.add_mutually_exclusive_group()
    cameras.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    cameras.add_argument("--qnx-camera-unit", type=int, help="QNX Sensor Framework camera unit")
    parser.add_argument("--face-detector", type=Path, default=Path("qnx/blaze_face_short_range.tflite"))
    parser.add_argument("--face-landmarker", type=Path, default=Path("qnx/face_landmark_detector.tflite"))
    parser.add_argument("--event-host", required=True)
    parser.add_argument("--event-port", type=int, default=8766)
    parser.add_argument("--preview-port", type=int, default=8080, help="0 disables LAN preview")
    parser.add_argument("--hold-frames", type=int, default=15)
    parser.add_argument("--arduino-port", help="QNX serial device connected to the Arduino, e.g. /dev/ser1")
    parser.add_argument("--servo-min", type=int, default=35)
    parser.add_argument("--servo-max", type=int, default=145)
    parser.add_argument("--servo-home", type=int, default=90)
    parser.add_argument("--servo-reversed", action="store_true", help="reverse left/right aiming")
    args = parser.parse_args()
    if not args.face_detector.is_file() or not args.face_landmarker.is_file():
        print("Medusa cannot start safely: required TFLite model is missing.", file=sys.stderr)
        return 2
    try:
        import cv2
        detector = TFLiteFaceGazeDetector(str(args.face_detector), str(args.face_landmarker))
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
    if args.arduino_port:
        try:
            mapper = ServoMapper(
                minimum=args.servo_min,
                maximum=args.servo_max,
                home=args.servo_home,
                reversed=args.servo_reversed,
            )
            if not mapper.minimum <= mapper.home <= mapper.maximum:
                raise ValueError("servo home must be inside the safe servo range")
            arduino = ArduinoController(args.arduino_port, mapper)
            arduino.safe_stop()
        except (OSError, ValueError, AttributeError) as error:
            camera.release()
            if preview:
                preview.stop()
            print(f"Medusa cannot start safely: Arduino serial setup failed: {error}", file=sys.stderr)
            return 2
    selector, target = SnooperSelector(args.hold_frames), (args.event_host, args.event_port)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as events:
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
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
                send(events, target, decision.active, decision.target_x, decision.reason)
                if arduino:
                    arduino.send(decision.active, decision.target_x)
                if preview:
                    snooper = select_snooper(boxes) if boxes is not None else None
                    nose_offset = None
                    if snooper is not None and snooper.nose is not None:
                        height, width = frame.shape[:2]
                        nose_offset = measure_nose_offset(snooper.nose, width, height)
                    if boxes:
                        draw_face_boxes(frame, boxes, snooper=snooper)
                    preview.state.update_frame(frame)
                    preview.state.update_status(
                        decision.active, decision.reason,
                        face_count=len(boxes) if boxes is not None else None,
                        snooper_detected=snooper is not None,
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

