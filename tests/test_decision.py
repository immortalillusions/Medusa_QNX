import unittest

from medusa.decision import SnooperSelector
from medusa.models import FaceObservation
from medusa.servo import ServoMapper
from medusa.tflite_detector import TFLiteFaceDetector


def face(x: float, size: float, looking: bool = True) -> FaceObservation:
    return FaceObservation("laptop", x, 0.5, size, looking, 0.9)


class FaceDetectorTests(unittest.TestCase):
    def test_blazeface_short_range_has_896_anchors(self) -> None:
        self.assertEqual(len(TFLiteFaceDetector._anchors(128, 128)), 896)

    def test_decode_removes_letterboxing_and_applies_sigmoid(self) -> None:
        detector = TFLiteFaceDetector.__new__(TFLiteFaceDetector)
        detector.anchors = [(0.5, 0.5), (0.5, 0.5)]
        raw = (0, 0, 64, 32, 0, 0, 0, 0, 4, -2, 0, 0, 0, 0, 0, 0)
        boxes = detector._decode([raw] * 2, [0.64, 0.63],
                                 640, 480, 0, 16, 128, 96)

        self.assertEqual(len(boxes), 1)
        box = boxes[0]
        self.assertEqual((box.left, box.top, box.right, box.bottom), (160, 160, 480, 320))
        self.assertAlmostEqual(box.score, 0.65475346)
        self.assertEqual(box.nose, (340, 230))


class SnooperSelectorTests(unittest.TestCase):
    def test_servo_mapper_bounds_and_parks(self) -> None:
        mapper = ServoMapper(minimum=35, maximum=145, home=90, smoothing=1.0)
        self.assertEqual(mapper.command(True, 0.0), (35, True))
        self.assertEqual(mapper.command(True, 1.0), (145, True))
        self.assertEqual(mapper.command(False, None), (90, False))

    def test_only_one_face_never_triggers(self) -> None:
        selector = SnooperSelector(hold_frames=2)
        self.assertFalse(selector.update([face(0.5, 0.3)]).active)
        self.assertFalse(selector.update([face(0.5, 0.3)]).active)


    def test_farthest_face_triggers_after_hold_period(self) -> None:
        selector = SnooperSelector(hold_frames=2)
        faces = [face(0.5, 0.35), face(0.8, 0.15)]
        self.assertFalse(selector.update(faces).active)
        decision = selector.update(faces)
        self.assertTrue(decision.active)
        self.assertEqual(decision.target_x, 0.8)


    def test_non_screen_facing_face_is_ignored(self) -> None:
        selector = SnooperSelector(hold_frames=1)
        self.assertFalse(
            selector.update([face(0.5, 0.3), face(0.8, 0.1, looking=False)]).active
        )

