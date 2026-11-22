"""Synthetic face masks (tools/mask_clips.py) and the eval's head-yaw error."""

import unittest

import numpy as np

import laptop_detector
from medusa.upper_pose import canonical_vertices
from tools.eval_distance import yaw_error
from tools.mask_clips import CLOTH_BGR, mask_polygon, paint_mask

SCALE, CX, CY = 20.0, 960, 540


def frontal_face():
    """The canonical mesh seen face-on, 20 px per cm, as pixel (x, y) with y down."""
    return [(CX + x * SCALE, CY - y * SCALE)
            for x, y, _ in canonical_vertices(str(laptop_detector.MODELS["landmarker"]))]


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.points = frontal_face()
        self.frame = np.full((1080, 1920, 3), 200, np.uint8)
        self.region = paint_mask(self.frame, self.points, np.random.default_rng(0))

    def covered(self, index):
        x, y = self.points[index]
        return bool(self.region[round(y), round(x)])

    def test_the_mouth_chin_and_nose_tip_are_covered(self):
        for index in (13, 14, 61, 291, 152, 1, 2):  # lips, mouth corners, chin, nose tip, below it
            self.assertTrue(self.covered(index), index)

    def test_the_eyes_brows_forehead_and_bridge_stay_visible(self):
        for index in (33, 133, 362, 263, 159, 386, 105, 334, 10, 151, 168, 6):
            self.assertFalse(self.covered(index), index)

    def test_cloth_colour_with_slight_noise_and_nothing_painted_outside(self):
        cloth = self.frame[self.region].astype(float)
        self.assertLess(abs(cloth.mean(axis=0) - np.array(CLOTH_BGR)).max(), 2.0)
        self.assertGreater(cloth.std(), 2.0)
        self.assertTrue((self.frame[~self.region] == 200).all())

    def test_the_polygon_spans_the_face_width_below_the_eyes(self):
        poly = mask_polygon(self.points)
        self.assertLess(poly[:, 0].min(), self.points[234][0] + 5)
        self.assertGreater(poly[:, 0].max(), self.points[454][0] - 5)
        self.assertGreater(poly[:, 1].min(), self.points[145][1])  # top edge under the eyes


class YawErrorTests(unittest.TestCase):
    def test_median_over_frames_both_found(self):
        self.assertEqual(yaw_error([1.0, None, 5.0, 2.0], [0.0, 3.0, None, 6.0]), 2.5)
        self.assertIsNone(yaw_error([None], [1.0]))


if __name__ == "__main__":
    unittest.main()
