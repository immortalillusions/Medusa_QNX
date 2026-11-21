import unittest

from medusa.decision import SnooperSelector
from medusa.models import FaceObservation
from medusa.tflite_detector import TFLiteFaceGazeDetector


def face(x: float, size: float, looking: bool = True) -> FaceObservation:
    return FaceObservation("laptop", x, 0.5, size, looking, 0.9)


class SnooperSelectorTests(unittest.TestCase):
    def test_blazeface_short_range_has_896_anchors(self) -> None:
        self.assertEqual(len(TFLiteFaceGazeDetector._anchors(128, 128)), 896)

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

