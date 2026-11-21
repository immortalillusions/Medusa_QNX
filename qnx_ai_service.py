"""QNX-resident Medusa service using checked-in TFLite models."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from medusa.decision import SnooperSelector
from medusa.models import SnooperDecision
from medusa.nose_offset import measure_nose_offset
from medusa.preview import PreviewServer
from medusa.snooper import select_snooper
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
    selector, target = SnooperSelector(args.hold_frames), (args.event_host, args.event_port)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as events:
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    send(events, target, False, None, "camera frame unavailable")
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
            camera.release()
            if preview:
                preview.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

