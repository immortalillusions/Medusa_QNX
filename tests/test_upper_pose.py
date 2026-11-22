"""Upper-face head pose (solvePnP on eyes/brows/forehead/bridge) and when gaze uses it."""

import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

import laptop_detector
from medusa.gaze import GazeConfig, estimate_gaze
from medusa.upper_pose import FOCAL_PER_WIDTH, UPPER_FACE, canonical_vertices, upper_face_pose
from tests.gaze_fixtures import pose

W, H = 1920, 1080
VERTICES = canonical_vertices(str(laptop_detector.MODELS["landmarker"]))


def project(yaw=0.0, pitch=0.0, at=(0.0, 0.0, -100.0)):
    """Canonical face turned to yaw/pitch, centred at ``at`` cm (x right, y up, z to viewer),
    projected to pixel (x, y), y down, with the module's focal length."""
    y, p = math.radians(yaw), math.radians(pitch)
    # Ry(yaw) @ Rx(-pitch), orthonormal: forward (0, 0, 1) -> (sin y cos p, sin p, cos y cos p).
    rotation = [[math.cos(y), -math.sin(y) * math.sin(p), math.sin(y) * math.cos(p)],
                [0.0, math.cos(p), math.sin(p)],
                [-math.sin(y), -math.cos(y) * math.sin(p), math.cos(y) * math.cos(p)]]
    f = FOCAL_PER_WIDTH * W
    points = []
    for v in VERTICES:
        x, y, z = (sum(rotation[r][c] * v[c] for c in range(3)) + at[r] for r in range(3))
        points.append((W / 2 + f * x / -z, H / 2 - f * y / -z))
    return points + [points[1]] * 10  # 478 like the landmarker (iris points unused here)


class UpperPoseTests(unittest.TestCase):
    def test_the_constants_are_the_bundled_canonical_mesh(self):
        self.assertEqual(len(VERTICES), 468)
        for index, xyz in UPPER_FACE.items():
            for a, b in zip(xyz, VERTICES[index]):
                self.assertAlmostEqual(a, b, places=3)

    def test_known_poses_are_recovered_at_the_frame_centre(self):
        for yaw, pitch in ((0, 0), (20, 0), (-30, 5), (10, -20), (0, 15)):
            got = upper_face_pose(project(yaw, pitch), W, H)
            self.assertAlmostEqual(got[0], yaw, delta=0.5, msg=(yaw, pitch))
            self.assertAlmostEqual(got[1], pitch, delta=0.5, msg=(yaw, pitch))

    def test_an_off_centre_face_looking_at_the_lens_reads_zero(self):
        at = (60.0, -25.0, -150.0)
        dist = math.sqrt(sum(c * c for c in at))
        yaw = math.degrees(math.atan2(-at[0], -at[2]))
        pitch = math.degrees(math.asin(-at[1] / dist))
        got = upper_face_pose(project(yaw, pitch, at), W, H)
        self.assertAlmostEqual(got[0], 0.0, delta=0.5)
        self.assertAlmostEqual(got[1], 0.0, delta=0.5)

    def test_the_lower_face_does_not_matter(self):
        points = project(15, -10)
        for index in (1, 2, 4, 13, 14, 61, 152, 291, 199):  # nose tip, lips, chin: under a mask
            points[index] = (points[index][0] + 40, points[index][1] - 60)
        got = upper_face_pose(points, W, H)
        self.assertAlmostEqual(got[0], 15, delta=0.5)
        self.assertAlmostEqual(got[1], -10, delta=0.5)

    def test_too_few_points_is_none(self):
        self.assertIsNone(upper_face_pose([(0.0, 0.0)] * 100, W, H))


def normalized(points):
    return [SimpleNamespace(x=x / W, y=y / H) for x, y in points]


class GazePoseSourceTests(unittest.TestCase):
    def test_auto_keeps_the_matrix_on_a_plausible_face(self):
        gaze = estimate_gaze(normalized(project(20, 0)), pose(5, 0), GazeConfig(), W, H)
        self.assertEqual(gaze.pose, "matrix")
        self.assertAlmostEqual(gaze.head_yaw, 5, places=6)

    def test_auto_switches_to_the_upper_face_when_the_jaw_is_implausible(self):
        points = project(20, 0)
        points[152] = points[1]  # chin pulled up to the nose tip: ratio ~0.9
        gaze = estimate_gaze(normalized(points), pose(5, 0), GazeConfig(), W, H)
        self.assertEqual(gaze.pose, "upper")
        self.assertAlmostEqual(gaze.head_yaw, 20, delta=0.5)

    def test_no_matrix_uses_the_upper_face_and_matrix_mode_never_does(self):
        gaze = estimate_gaze(normalized(project(-15, 5)), None, GazeConfig(), W, H)
        self.assertEqual(gaze.pose, "upper")
        self.assertAlmostEqual(gaze.head_yaw, -15, delta=0.5)
        points = project(20, 0)
        points[152] = points[1]
        gaze = estimate_gaze(normalized(points), pose(5, 0),
                             replace(GazeConfig(), head_pose="matrix"), W, H)
        self.assertEqual(gaze.pose, "matrix")

    def test_upper_mode_always_uses_it(self):
        gaze = estimate_gaze(normalized(project(-12, 0)), pose(5, 0),
                             replace(GazeConfig(), head_pose="upper"), W, H)
        self.assertEqual(gaze.pose, "upper")
        self.assertAlmostEqual(gaze.head_yaw, -12, delta=0.5)


if __name__ == "__main__":
    unittest.main()
