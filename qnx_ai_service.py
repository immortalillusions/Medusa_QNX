"""QNX-resident Medusa service using checked-in TFLite models."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from medusa.decision import SnooperSelector
from medusa.preview import PreviewServer
from medusa.tflite_detector import TFLiteFaceGazeDetector


def send(sock, target, active, target_x, reason):
    sock.sendto(json.dumps({"type":"medusa-event","active":active,
                "target_x":target_x,"reason":reason}).encode(), target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
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
    except (RuntimeError, ValueError) as error:
        print(f"Medusa cannot start safely: {error}", file=sys.stderr)
        return 2
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print("Medusa cannot start safely: QNX camera is unavailable", file=sys.stderr)
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
                if preview:
                    preview.state.update_frame(frame)
                try:
                    decision = selector.update(detector.detect(frame))
                except (RuntimeError, ValueError):
                    send(events, target, False, None, "AI inference unavailable")
                    if preview:
                        preview.state.update_status(False, "AI inference unavailable")
                    continue
                send(events, target, decision.active, decision.target_x, decision.reason)
                if preview:
                    preview.state.update_status(decision.active, decision.reason)
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

