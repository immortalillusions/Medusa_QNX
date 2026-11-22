"""The Pi's TFLite-first detector with a rate-limited YuNet fallback, and its flag."""

import unittest
from pathlib import Path

import numpy as np

import qnx_ai_service
from medusa.tflite_detector import FaceBox
from medusa.yunet_detector import FallbackFaceDetector, YuNetFaceDetector

USER = FaceBox(100, 100, 200, 220, 0.9)
MASKED = FaceBox(400, 120, 460, 190, 0.5)


class Fake:
    def __init__(self, boxes):
        self.boxes, self.calls, self.input_size = boxes, 0, 256

    def detect(self, frame):
        self.calls += 1
        return list(self.boxes)


class FallbackTests(unittest.TestCase):
    frame = np.zeros((480, 640, 3), np.uint8)

    def test_primary_faces_mean_no_fallback_run(self):
        primary, fallback = Fake([USER]), Fake([MASKED])
        detector = FallbackFaceDetector(primary, fallback, want=1)
        for _ in range(5):
            self.assertEqual(detector.detect(self.frame), [USER])
        self.assertEqual((primary.calls, fallback.calls), (5, 0))

    def test_no_primary_face_runs_the_fallback_every_third_frame_and_holds_it(self):
        primary, fallback = Fake([]), Fake([MASKED])
        detector = FallbackFaceDetector(primary, fallback, want=1, every=3)
        seen = [detector.detect(self.frame) for _ in range(7)]
        self.assertEqual(fallback.calls, 3)                 # frames 1, 4, 7
        self.assertTrue(all(boxes == [MASKED] for boxes in seen))
        self.assertEqual(primary.calls, 7)                  # TFLite still runs every frame

    def test_fallback_boxes_on_a_primary_face_are_dropped_and_others_added(self):
        same = FaceBox(110, 110, 190, 210, 0.7)
        detector = FallbackFaceDetector(Fake([USER]), Fake([same, MASKED]), want=2)
        self.assertEqual(detector.detect(self.frame), [USER, MASKED])

    def test_a_primary_face_clears_the_held_fallback_boxes(self):
        primary, fallback = Fake([]), Fake([MASKED])
        detector = FallbackFaceDetector(primary, fallback, want=1)
        detector.detect(self.frame)
        primary.boxes = [USER]
        self.assertEqual(detector.detect(self.frame), [USER])
        primary.boxes = []
        self.assertEqual(detector.detect(self.frame), [])   # not stale MASKED; next run is due later

    def test_attributes_pass_through_to_the_primary(self):
        self.assertEqual(FallbackFaceDetector(Fake([]), Fake([])).input_size, 256)

    def test_the_real_yunet_runs_on_a_pi_sized_frame(self):
        yunet = YuNetFaceDetector(str(qnx_ai_service.YUNET_MODEL), scale=1.0)
        self.assertEqual(yunet.detect(self.frame), [])


class FlagTests(unittest.TestCase):
    def test_back_plus_yunet_is_a_face_model_and_the_default_is_unchanged(self):
        base = ["--trigger-udp-port", "8767"]
        args = qnx_ai_service.parse_args(base)
        self.assertEqual(args.face_model, "back")
        args = qnx_ai_service.parse_args(base + ["--face-model", "back+yunet"])
        self.assertEqual(args.face_detector, Path("qnx/face_detector.tflite"))  # TFLite primary
        self.assertTrue(qnx_ai_service.YUNET_MODEL.is_file())


if __name__ == "__main__":
    unittest.main()
