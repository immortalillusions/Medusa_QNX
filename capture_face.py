"""Take one photo on a QNX Raspberry Pi and mark detected faces."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from medusa.tflite_detector import TFLiteFaceDetector, draw_face_boxes


MODEL = Path(__file__).resolve().parent / "qnx" / "blaze_face_short_range.tflite"


def take_photo(unit: int):
    from medusa.qnx_camera import QnxCamera

    camera = QnxCamera(unit)
    try:
        time.sleep(0.5)  # Allow the camera exposure to settle.
        for _ in range(5):
            ok, frame = camera.read()
            if ok:
                return frame
        raise RuntimeError(f"Timed out waiting for a frame from QNX camera unit {unit}.")
    finally:
        camera.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-unit", type=int, default=5,
                        help="QNX camera unit (default: 5, the Pi 5 USB camera)")
    parser.add_argument("--output", type=Path, default=Path("captures/face.jpg"),
                        help="Output photo path (default: captures/face.jpg)")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError as error:
        print(f"Snapshot failed to load OpenCV: {error}", file=sys.stderr)
        return 1

    try:
        detector = TFLiteFaceDetector(str(MODEL))
        photo = take_photo(args.camera_unit)
        faces = detector.detect(photo)
        draw_face_boxes(photo, faces)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=args.output.suffix,
                                         delete=False) as file:
            temporary_photo = Path(file.name)
        try:
            if not cv2.imwrite(str(temporary_photo), photo):
                raise RuntimeError(f"Could not write {args.output}.")
            temporary_photo.replace(args.output)
        finally:
            temporary_photo.unlink(missing_ok=True)
    except (ImportError, OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Snapshot failed: {error}", file=sys.stderr)
        return 1

    print(f"Faces detected: {len(faces)}")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
