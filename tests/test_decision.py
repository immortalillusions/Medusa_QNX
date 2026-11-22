import unittest

from medusa.decision import SnooperSelector
from medusa.models import FaceObservation
from medusa.servo import ServoMapper
from medusa.tflite_detector import TFLiteFaceDetector, ssd_anchors


def face(x: float, size: float, looking: bool = True, y: float = 0.5) -> FaceObservation:
    return FaceObservation("laptop", x, y, size, looking, 0.9)


class FaceDetectorTests(unittest.TestCase):
    def test_blazeface_short_range_has_896_anchors(self) -> None:
        self.assertEqual(len(ssd_anchors(128, (8, 16, 16, 16))), 896)

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

    def test_tilt_mapper_top_of_frame_is_minimum_unless_reversed(self) -> None:
        # Image y grows downward: y=0 is the top of the frame.
        tilt = ServoMapper(minimum=65, maximum=115, home=90, smoothing=1.0)
        self.assertEqual(tilt.command(True, 0.0), (65, True))
        self.assertEqual(tilt.command(True, 1.0), (115, True))
        self.assertEqual(tilt.command(True, None), (90, True))
        flipped = ServoMapper(minimum=65, maximum=115, home=90, reversed=True, smoothing=1.0)
        self.assertEqual(flipped.command(True, 0.0), (115, True))
        self.assertEqual(flipped.command(True, 1.0), (65, True))

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

    def test_active_decision_carries_snooper_vertical_position(self) -> None:
        selector = SnooperSelector(hold_frames=1)
        decision = selector.update([face(0.5, 0.35, y=0.6), face(0.8, 0.15, y=0.25)])
        self.assertTrue(decision.active)
        self.assertEqual((decision.target_x, decision.target_y), (0.8, 0.25))

    def test_inactive_decision_has_no_vertical_target(self) -> None:
        selector = SnooperSelector(hold_frames=2)
        decision = selector.update([face(0.5, 0.35), face(0.8, 0.15, y=0.25)])
        self.assertFalse(decision.active)
        self.assertIsNone(decision.target_y)


    def test_non_screen_facing_face_is_ignored(self) -> None:
        selector = SnooperSelector(hold_frames=1)
        self.assertFalse(
            selector.update([face(0.5, 0.3), face(0.8, 0.1, looking=False)]).active
        )

