"""Run the Medusa BlazeFace model on a Windows camera."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Direct execution adds qnx/ to the import path. The shared code lives one level up.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medusa.decision import SnooperSelector, screen_facing_from_landmarks
from medusa.models import FaceObservation
from medusa.windows_desktop import TASKBAR_SLOTS, DesktopSwitchTrigger, SnooperSignal


_LOG = logging.getLogger(__name__)


def detect_faces(detector, frame):
    """Return detections in the original frame's pixel coordinates."""
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return detector.detect(image).detections


def draw_faces(frame, detections):
    import cv2

    height, width = frame.shape[:2]
    for detection in detections:
        box = detection.bounding_box
        left, top = box.origin_x, box.origin_y
        right, bottom = left + box.width, top + box.height
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        if detection.categories:
            score = detection.categories[0].score
            cv2.putText(frame, f"{score:.2f}", (max(0, left), max(16, top - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for point in detection.keypoints or ():
            cv2.circle(frame, (round(point.x * width), round(point.y * height)),
                       2, (0, 255, 255), -1)


def confirmed_snooper_signal(
    selector: SnooperSelector, detections, width: int, height: int,
) -> tuple[SnooperSignal, str]:
    observations = []
    for detection in detections:
        box = detection.bounding_box
        points = detection.keypoints or ()
        if len(points) < 3 or box.width <= 0 or box.height <= 0 or not detection.categories:
            continue
        # BlazeFace gives the two eyes and nose tip first, normalized to the full image.
        face_scale = width / box.width
        eye_a, eye_b, nose = points[:3]
        facing = screen_facing_from_landmarks(
            nose.x * face_scale, eye_a.x * face_scale, eye_b.x * face_scale,
        )
        if facing is not None:
            observations.append(FaceObservation(
                source="windows-webcam",
                center_x=(box.origin_x + box.width / 2) / width,
                center_y=(box.origin_y + box.height / 2) / height,
                relative_size=box.width / width,
                screen_facing=facing,
                confidence=detection.categories[0].score,
            ))

    decision = selector.update(observations)
    if decision.active:
        return SnooperSignal.CONFIRMED, decision.reason
    if len(detections) < 2:
        return SnooperSignal.CLEAR, "fewer than two faces in the webcam frame"
    return SnooperSignal.UNCONFIRMED, decision.reason


def simulate_desktop_switch(slot: int, countdown: int) -> int:
    _LOG.info("Simulating a confirmed snooper in %d seconds. Put your game in front now.", countdown)
    time.sleep(countdown)
    switcher = DesktopSwitchTrigger(slot)
    try:
        attempt = switcher.update(SnooperSignal.CONFIRMED, "a camera-free simulated confirmation")
        return 0 if attempt is not None and attempt.result() else 1
    finally:
        switcher.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--camera", type=int, help="Windows camera index, usually 0 or 1")
    mode.add_argument("--test-switch", action="store_true",
                      help="send one shortcut from a simulated confirmation without opening a camera")
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).with_name("blaze_face_short_range.tflite"))
    parser.add_argument("--switch-desktop", action="store_true",
                        help="enable taskbar switching on a confirmed snooper")
    parser.add_argument("--taskbar-slot", type=int, choices=TASKBAR_SLOTS, default=1,
                        help="pinned app position, 1 through 10 (default: 1)")
    parser.add_argument("--countdown", type=int, default=15,
                        help="seconds before the camera-free test sends the shortcut (default: 15)")
    parser.add_argument("--verbose", action="store_true", help="log why individual frames were skipped")
    args = parser.parse_args(argv)
    if args.countdown < 0:
        parser.error("--countdown must be nonnegative")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    if args.test_switch:
        return simulate_desktop_switch(args.taskbar_slot, args.countdown)

    if not args.model.is_file():
        parser.error(f"Face detector model not found: {args.model}")

    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        parser.exit(1, f"Could not load webcam dependencies with {sys.executable}.\n"
                       f"{error}\n"
                       "Use the Python environment where MediaPipe and OpenCV are installed.\n"
                       "Setup instructions: docs/windows-desktop-switching.md\n")

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        min_detection_confidence=0.5,
    )
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    switcher = None
    try:
        if not camera.isOpened():
            parser.exit(1, f"Could not open camera {args.camera}. Check the USB "
                           "connection and Windows camera permissions, or try another index.\n")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if args.switch_desktop:
            selector = SnooperSelector()
            switcher = DesktopSwitchTrigger(args.taskbar_slot)
            _LOG.info("Desktop switching enabled for taskbar slot %d.", args.taskbar_slot)
        else:
            _LOG.info("Desktop switching skipped because --switch-desktop was omitted.")
        with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    parser.exit(1, f"Camera {args.camera} stopped delivering frames.\n")
                detections = detect_faces(detector, frame)
                if switcher is not None:
                    height, width = frame.shape[:2]
                    signal, reason = confirmed_snooper_signal(selector, detections, width, height)
                    switcher.update(signal, reason)
                draw_faces(frame, detections)
                cv2.imshow("Medusa BlazeFace | Q to quit", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if switcher is not None:
            switcher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
