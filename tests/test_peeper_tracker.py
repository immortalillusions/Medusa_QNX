import random
import unittest

from medusa.decision import SnooperSelector
from medusa.peeper_tracker import PeeperTracker, TrackerConfig
from tests.peeper_stream import ASPECT, det, edges, gaze, noisy_viewer, observation

USER_FACE = dict(size=0.3, cy=0.55)


def user(x=0.3):
    return det(x, gaze(0, 0), **USER_FACE)


def run(tracker, frames, fps, start=0.0):
    """frames: list of detection lists, one per frame. Returns (t, result) per frame."""
    out = []
    for n, faces in enumerate(frames):
        t = start + n / fps
        out.append((t, tracker.update(faces, t, aspect=ASPECT)))
    return out


def looking_peeper(x=0.7, yaw_eye=0.0):
    return det(x, gaze(0, yaw_eye))


def away_peeper(x=0.7):
    return det(x, gaze(0, 60))


class StabilityTests(unittest.TestCase):
    def test_still_side_eyeing_peeper_with_noise_and_blinks_stays_active(self):
        # Head turned 30 deg, eyes back -16: every blink drops to head-only (30 = not looking).
        stream = list(noisy_viewer(30, 30, seed=3, with_user=True, head_yaw=30, eye_yaw=-16,
                                   x0=0.7, noise_deg=3, spike_p=0.0))
        tracker = PeeperTracker(TrackerConfig(min_faces=2))
        actives = [tracker.update(faces, t, ASPECT).decision.active for t, faces in stream]
        self.assertLessEqual(edges(actives)[0], 1)
        self.assertTrue(all(actives[30:]))
        # The same stream flickers under the old 15-frame selector.
        old = SnooperSelector(hold_frames=15, min_viewers=2)
        old_actives = [old.update([observation(d) for d in faces]).active for _, faces in stream]
        self.assertGreater(edges(old_actives)[0], 3)

    def test_calibration_viewer_with_spikes_stays_active(self):
        stream = list(noisy_viewer(30, 30, seed=5))
        tracker = PeeperTracker(TrackerConfig(min_faces=1))
        actives = [tracker.update(faces, t, ASPECT).decision.active for t, faces in stream]
        self.assertEqual(edges(actives)[0], 1)
        self.assertTrue(all(actives[30:]))

    def test_peeper_drifting_across_bucket_edges_keeps_one_id_and_stays_active(self):
        fps, seconds = 30, 10
        frames = [[looking_peeper(0.40 + 0.2 * n / (fps * seconds))] for n in range(fps * seconds)]
        results = run(PeeperTracker(TrackerConfig(min_faces=1)), frames, fps)
        actives = [r.decision.active for _, r in results]
        self.assertEqual(edges(actives), (1, 1))
        self.assertEqual({r.tracks[0].id for _, r in results}, {1})
        self.assertAlmostEqual(results[-1][1].decision.target_x, 0.6, delta=0.01)

    def test_glances_shorter_than_enter_never_activate(self):
        # Away at 60 deg, just past the exit band (36 deg), or a face that appears looking.
        for away, absent in ((60, False), (36, False), (60, True)):
            frames = []
            for _ in range(10):
                frames += [[user(), looking_peeper()]] * 14  # 0.47 s
                frames += [[user()] if absent else [user(), det(0.7, gaze(0, away))]] * 30
            actives = [r.decision.active for _, r in run(PeeperTracker(), frames, 30)]
            self.assertFalse(any(actives), (away, absent))

    def test_a_steady_look_activates_after_enter_s(self):
        results = run(PeeperTracker(), [[user(), looking_peeper()]] * 30, 30)
        first = next(t for t, r in results if r.decision.active)
        self.assertAlmostEqual(first, 0.5, delta=1e-6)

    def test_looking_away_under_exit_s_holds_and_over_exit_s_releases(self):
        fps = 30
        frames = ([[user(), looking_peeper()]] * 30 + [[user(), away_peeper()]] * 18   # 0.6 s
                  + [[user(), looking_peeper()]] * 30 + [[user(), away_peeper()]] * 36)  # 1.2 s
        actives = [r.decision.active for _, r in run(PeeperTracker(), frames, fps)]
        self.assertTrue(all(actives[15:78]))
        self.assertFalse(actives[-1])
        off = actives.index(False, 78)
        self.assertGreater((off - 78) / fps, 0.7)
        self.assertLess((off - 78) / fps, 1.0)  # 0.7 s plus the smoothing lag

    def test_peeper_missing_under_exit_s_holds(self):
        frames = ([[user(), looking_peeper()]] * 30 + [[user()]] * 18   # gone 0.6 s > track ttl
                  + [[user(), looking_peeper()]] * 30)
        actives = [r.decision.active for _, r in run(PeeperTracker(), frames, 30)]
        self.assertEqual(edges(actives), (1, 1))

    def test_peeper_missing_over_exit_s_releases(self):
        frames = [[user(), looking_peeper()]] * 30 + [[user()]] * 30
        results = run(PeeperTracker(), frames, 30)
        self.assertFalse(results[-1][1].decision.active)
        self.assertEqual(results[-1][1].decision.reason, "fewer than 2 faces")


class IdentityTests(unittest.TestCase):
    def test_user_and_peeper_do_not_swap_on_a_brief_size_change(self):
        frames = ([[user(), looking_peeper()]] * 30
                  + [[user(), det(0.7, gaze(0, 0), size=0.4)]] * 9      # leans in for 0.3 s
                  + [[user(), looking_peeper()]] * 30)
        results = run(PeeperTracker(), frames, 30)
        self.assertEqual({r.user_id for _, r in results}, {1})
        self.assertTrue(all(r.decision.active for _, r in results[15:]))
        self.assertEqual({r.peeper_id for _, r in results[15:]}, {2})

    def test_a_sustained_bigger_face_becomes_the_user(self):
        frames = [[user(), looking_peeper()]] * 30 + [[user(), det(0.7, gaze(0, 0), size=0.5)]] * 45
        results = run(PeeperTracker(), frames, 30)
        swap = next(t for t, r in results if r.user_id == 2)
        self.assertGreater(swap - 1.0, 0.5)  # the bigger face took over only after 0.5 s
        # Track 1 (looking straight in) is now the other face, so it becomes the peeper.
        self.assertEqual((results[-1][1].user_id, results[-1][1].peeper_id), (2, 1))

    def test_two_faces_crossing_keep_their_ids(self):
        rng = random.Random(7)
        fps, frames = 30, 90
        ids_a, ids_b = set(), set()
        tracker = PeeperTracker()
        for n in range(frames):
            xa = 0.2 + 0.6 * n / frames            # near face, left to right
            xb = 0.8 - 0.6 * n / frames            # far face, right to left
            a, b = det(xa, gaze(0, 0), size=0.2, cy=0.5), det(xb, gaze(0, 0), size=0.12, cy=0.42)
            faces = [a, b] if rng.random() < 0.5 else [b, a]
            result = tracker.update(faces, n / fps, ASPECT)
            for face, track in zip(faces, result.tracks):
                (ids_a if face is a else ids_b).add(track.id)
        self.assertEqual((len(ids_a), len(ids_b)), (1, 1))
        self.assertNotEqual(ids_a, ids_b)

    def test_unmatched_track_survives_half_a_second(self):
        tracker = PeeperTracker(TrackerConfig(min_faces=1))
        tracker.update([looking_peeper()], 0.0, ASPECT)
        tracker.update([], 0.45, ASPECT)
        self.assertEqual(tracker.update([looking_peeper()], 0.48, ASPECT).tracks[0].id, 1)
        tracker.update([], 1.0, ASPECT)
        self.assertEqual(tracker.update([looking_peeper()], 1.05, ASPECT).tracks[0].id, 2)


class SmoothingTests(unittest.TestCase):
    def test_single_frame_spike_is_ignored_two_are_not(self):
        tracker = PeeperTracker(TrackerConfig(min_faces=1))
        tracker.update([det(0.5, gaze(0, 5))], 0.0, ASPECT)
        track = tracker.update([det(0.5, gaze(40, 5))], 1 / 30, ASPECT).tracks[0]  # 40 deg jump
        self.assertAlmostEqual(track.gaze_yaw, 5)
        track = tracker.update([det(0.5, gaze(40, 5))], 2 / 30, ASPECT).tracks[0]
        self.assertGreater(track.gaze_yaw, 5)

    def test_blink_holds_the_eye_angle_for_up_to_0_4_s(self):
        tracker = PeeperTracker(TrackerConfig(min_faces=1, smooth_tau_s=0))
        tracker.update([det(0.5, gaze(30, -20))], 0.0, ASPECT)
        held = tracker.update([det(0.5, gaze(30, -20, closed=True))], 0.3, ASPECT).tracks[0]
        self.assertEqual((held.gaze_yaw, held.method, held.looking), (10, "blink-hold", True))
        dropped = tracker.update([det(0.5, gaze(30, -20, closed=True))], 0.5, ASPECT).tracks[0]
        self.assertEqual((dropped.gaze_yaw, dropped.method), (30, "head"))

    def test_looking_has_hysteresis(self):
        tracker = PeeperTracker(TrackerConfig(min_faces=1, smooth_tau_s=0))
        self.assertTrue(tracker.update([det(0.5, gaze(0, 24))], 0.0, ASPECT).tracks[0].looking)
        self.assertTrue(tracker.update([det(0.5, gaze(0, 31))], 0.1, ASPECT).tracks[0].looking)
        self.assertFalse(tracker.update([det(0.5, gaze(0, 33))], 0.2, ASPECT).tracks[0].looking)
        self.assertFalse(tracker.update([det(0.5, gaze(0, 26))], 0.3, ASPECT).tracks[0].looking)
        self.assertTrue(tracker.update([det(0.5, gaze(0, 25))], 0.4, ASPECT).tracks[0].looking)
        # Pitch: range -35..10, margin 7.
        self.assertTrue(tracker.update([det(0.5, gaze(0, 0, pitch=16))], 0.5, ASPECT).tracks[0].looking)
        self.assertFalse(tracker.update([det(0.5, gaze(0, 0, pitch=18))], 0.6, ASPECT).tracks[0].looking)


class FrameRateTests(unittest.TestCase):
    def timing(self, fps):
        frames = ([[user()]] * fps + [[user(), looking_peeper()]] * (2 * fps)
                  + [[user(), away_peeper()]] * (2 * fps))
        actives = [(t, r.decision.active) for t, r in run(PeeperTracker(), frames, fps)]
        on = next(t for t, a in actives if a)
        off = next(t for t, a in actives if t > on and not a)
        return on - 1.0, off - 3.0

    def test_15_and_30_fps_give_the_same_timing(self):
        on30, off30 = self.timing(30)
        on15, off15 = self.timing(15)
        self.assertAlmostEqual(on30, 0.5, delta=1 / 30)
        self.assertAlmostEqual(on15, 0.5, delta=1 / 15)
        self.assertLessEqual(abs(on30 - on15), 1 / 15 + 1e-9)
        self.assertLessEqual(abs(off30 - off15), 1 / 15 + 1e-9)
        self.assertGreater(off30, 0.7)


if __name__ == "__main__":
    unittest.main()
