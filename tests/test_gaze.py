import unittest

from medusa.gaze import GazeConfig, estimate_gaze, head_pose_from_matrix, looking_at_screen
from tests.gaze_fixtures import landmarks, pose


class HeadPoseTests(unittest.TestCase):
    def test_yaw_and_pitch_round_trip_through_the_matrix(self):
        for yaw, pitch in ((0, 0), (40, 0), (-30, 0), (0, -20), (25, 15), (-60, -30)):
            got_yaw, got_pitch = head_pose_from_matrix(pose(yaw, pitch))
            self.assertAlmostEqual(got_yaw, yaw, places=6)
            self.assertAlmostEqual(got_pitch, pitch, places=6)


class GazeTests(unittest.TestCase):
    def gaze(self, head=(0, 0), eyes=(0, 0), config=GazeConfig(), **face):
        return estimate_gaze(landmarks(*eyes, **face), pose(*head), config)

    def test_head_straight_eyes_centred_is_looking(self):
        gaze = self.gaze()
        self.assertTrue(gaze.looking)
        self.assertEqual(gaze.method, "eyes")
        for value in (gaze.head_yaw, gaze.eye_yaw, gaze.gaze_yaw, gaze.gaze_pitch):
            self.assertAlmostEqual(value, 0.0, places=6)

    def test_side_eye_head_turned_away_eyes_back_on_the_screen_is_looking(self):
        for sign in (1, -1):
            gaze = self.gaze(head=(40 * sign, 0), eyes=(-30 * sign, 0))
            self.assertAlmostEqual(gaze.head_yaw, 40 * sign, places=6)
            self.assertAlmostEqual(gaze.eye_yaw, -30 * sign, places=6)
            self.assertAlmostEqual(gaze.gaze_yaw, 10 * sign, places=6)
            self.assertTrue(gaze.looking)
            self.assertEqual(gaze.method, "eyes")

    def test_head_straight_eyes_far_to_the_side_is_not_looking(self):
        for sign in (1, -1):
            gaze = self.gaze(eyes=(35 * sign, 0))
            self.assertAlmostEqual(gaze.gaze_yaw, 35 * sign, places=6)
            self.assertFalse(gaze.looking)

    def test_head_turned_eyes_straight_is_not_looking(self):
        for sign in (1, -1):
            gaze = self.gaze(head=(40 * sign, 0))
            self.assertAlmostEqual(gaze.gaze_yaw, 40 * sign, places=6)
            self.assertFalse(gaze.looking)

    def test_iris_past_full_deflection_is_clamped_to_the_gain(self):
        gaze = self.gaze(eyes=(80, 0))
        self.assertAlmostEqual(gaze.eye_yaw, GazeConfig().eye_gain_deg, places=6)

    def test_closed_eyes_fall_back_to_the_head(self):
        turned = self.gaze(head=(40, 0), eyes=(-30, 0), closed=("right", "left"))
        self.assertEqual(turned.method, "head")
        self.assertEqual((turned.eye_yaw, turned.eye_pitch), (0.0, 0.0))
        self.assertAlmostEqual(turned.gaze_yaw, 40, places=6)
        self.assertFalse(turned.looking)
        straight = self.gaze(eyes=(30, 0), closed=("right", "left"))
        self.assertEqual(straight.method, "head")
        self.assertTrue(straight.looking)

    def test_one_closed_eye_uses_the_open_one(self):
        gaze = self.gaze(head=(40, 0), eyes=(-30, 0), closed=("left",))
        self.assertEqual(gaze.method, "eyes")
        self.assertAlmostEqual(gaze.eye_yaw, -30, places=6)
        self.assertTrue(gaze.looking)

    def test_no_iris_landmarks_is_head_only(self):
        gaze = estimate_gaze(landmarks(count=468), pose(10, 0))
        self.assertEqual(gaze.method, "head")
        self.assertAlmostEqual(gaze.gaze_yaw, 10, places=6)

    def test_pitch_limits(self):
        cases = [
            ((0, -20), (0, 0), True),     # head tipped down at the keyboard end of the screen
            ((0, -45), (0, 0), False),    # looking at the desk
            ((0, 15), (0, 0), False),     # looking over the laptop
            ((0, 0), (0, -28), True),     # eyes alone dropped to the screen
            ((0, -20), (0, -28), False),  # head and eyes down together: below the screen
            ((0, 15), (0, -10), True),    # head up, eyes back down onto the camera
        ]
        for head, eyes, looking in cases:
            with self.subTest(head=head, eyes=eyes):
                self.assertEqual(self.gaze(head=head, eyes=eyes).looking, looking)

    def test_eye_pitch_sign_is_up_positive(self):
        self.assertAlmostEqual(self.gaze(eyes=(0, 20)).eye_pitch, 20, places=6)
        self.assertAlmostEqual(self.gaze(eyes=(0, -20)).eye_pitch, -20, places=6)

    def test_thresholds_are_tunable(self):
        wide = GazeConfig(yaw_max_deg=45)
        self.assertTrue(self.gaze(head=(40, 0), config=wide).looking)
        self.assertTrue(looking_at_screen(0, -40, GazeConfig(pitch_min_deg=-50)))
        self.assertFalse(looking_at_screen(0, 5, GazeConfig(pitch_max_deg=0)))
        stronger_eyes = GazeConfig(eye_gain_deg=70)
        gaze = estimate_gaze(landmarks(-35, 0), pose(40, 0), stronger_eyes)  # same iris, doubled gain
        self.assertAlmostEqual(gaze.eye_yaw, -70, places=6)

    def test_non_square_frames_use_pixel_proportions(self):
        # A centred iris stays centred whatever the frame aspect.
        gaze = estimate_gaze(landmarks(), pose(), GazeConfig(), width=1280, height=720)
        self.assertAlmostEqual(gaze.eye_yaw, 0.0, places=6)
        self.assertAlmostEqual(gaze.eye_pitch, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
