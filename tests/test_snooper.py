import unittest

from medusa.nose_offset import NoseOffset, measure_nose_offset
from medusa.snooper import select_snooper
from medusa.tflite_detector import FaceBox


class SnooperTests(unittest.TestCase):
    def test_selection_uses_box_area_and_requires_two_faces(self) -> None:
        near = FaceBox(0, 10, 120, 150, 0.95)
        narrow = FaceBox(150, 10, 175, 110, 0.92)
        smallest = FaceBox(200, 60, 240, 100, 0.88)

        self.assertIsNone(select_snooper([]))
        self.assertIsNone(select_snooper([near]))
        self.assertIs(select_snooper([near, narrow, smallest]), smallest)

    def test_pixel_distance_and_direction(self) -> None:
        self.assertEqual(measure_nose_offset((83, 41), 160, 90), NoseOffset(3, -4, 5))
        self.assertEqual(measure_nose_offset((80, 45), 160, 90), NoseOffset(0, 0, 0))
