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


def peeper_at(x, looking):
    return looking_peeper(x) if looking else away_peeper(x)


def two_peepers(fps, seconds, a_looks, b_looks, a_x=0.75, b_x=0.55):
    """User at 0.3 plus peepers A and B; a_looks/b_looks(t) say whether each looks."""
    frames = []
    for n in range(int(fps * seconds)):
        t = n / fps
        frames.append([user(), peeper_at(a_x, a_looks(t)), peeper_at(b_x, b_looks(t))])
    return frames


def switches(results):
    """(time, new peeper id) each time the peeper changes while active."""
    out, last = [], None
    for t, r in results:
        if r.decision.active and r.peeper_id != last:
            out.append((t, r.peeper_id))
            last = r.peeper_id
    return out


class MostRecentPeeperTests(unittest.TestCase):
    """The turret aims at one peeper: the most recent steady look, never flip-flopping."""

    def config(self):
        return TrackerConfig(min_faces=2)  # takeover_s 1.0, min_lock_s 1.5 by default

    def test_a_newer_peeper_takes_over_after_a_steady_takeover_s(self):
        fps = 30
        frames = two_peepers(fps, 6, a_looks=lambda t: True, b_looks=lambda t: t >= 2.0)
        results = run(PeeperTracker(self.config()), frames, fps)
        ids = {r.tracks[1].id: "A" for _, r in results[:1]}
        ids[results[0][1].tracks[2].id] = "B"
        sw = [(t, ids[pid]) for t, pid in switches(results)]
        self.assertEqual([who for _, who in sw], ["A", "B"])  # A never steals back
        self.assertAlmostEqual(sw[1][0], 3.0, delta=0.15)     # B looked from 2.0, +takeover_s

    def test_a_glance_shorter_than_takeover_s_steals_nothing(self):
        fps = 30
        frames = two_peepers(fps, 5, a_looks=lambda t: True, b_looks=lambda t: 2.0 <= t < 2.8)
        results = run(PeeperTracker(self.config()), frames, fps)
        self.assertEqual(len(switches(results)), 1)
        self.assertTrue(all(r.decision.active for _, r in results[30:]))

    def test_repeated_feints_never_add_up_to_a_takeover(self):
        fps = 30
        # B looks 0.7 s, looks away 0.3 s, over and over: each look restarts the count.
        frames = two_peepers(fps, 8, a_looks=lambda t: True,
                             b_looks=lambda t: t >= 1.0 and (t - 1.0) % 1.0 < 0.7)
        results = run(PeeperTracker(self.config()), frames, fps)
        self.assertEqual(len(switches(results)), 1)

    def test_two_people_taking_turns_cannot_flip_faster_than_min_lock_s(self):
        fps = 30
        # Each looks away briefly at its own rhythm, so each keeps starting a "newer" look.
        frames = two_peepers(fps, 12,
                             a_looks=lambda t: (t % 1.3) > 0.25,
                             b_looks=lambda t: t > 0.6 and ((t + 0.6) % 1.1) > 0.25)
        results = run(PeeperTracker(self.config()), frames, fps)
        times = [t for t, _ in switches(results)]
        gaps = [b - a for a, b in zip(times, times[1:])]
        self.assertTrue(all(g >= 1.5 - 1e-6 for g in gaps), gaps)

    def test_when_two_rivals_qualify_together_the_newest_look_wins(self):
        fps = 30
        # A locks at ~0.5 s; B looks from 0.6, C from 0.8. Both qualify when min_lock_s ends
        # (~2.0 s): C, the more recent look, wins, and B (older than C) never takes over.
        frames = []
        for n in range(int(fps * 6)):
            t = n / fps
            frames.append([user(), peeper_at(0.85, True), peeper_at(0.65, t >= 0.6),
                           peeper_at(0.5, t >= 0.8)])
        results = run(PeeperTracker(self.config()), frames, fps)
        a_id, b_id, c_id = (results[0][1].tracks[i].id for i in (1, 2, 3))
        sw = switches(results)
        self.assertEqual([pid for _, pid in sw], [a_id, c_id])
        self.assertAlmostEqual(sw[1][0], sw[0][0] + 1.5, delta=0.1)
        self.assertNotIn(b_id, [pid for _, pid in sw])

    def test_the_old_peeper_leaving_hands_over_to_a_ready_one_at_exit_s(self):
        fps = 30
        frames = two_peepers(fps, 5, a_looks=lambda t: t < 2.0, b_looks=lambda t: t >= 1.5)
        results = run(PeeperTracker(self.config()), frames, fps)
        sw = switches(results)
        self.assertEqual(len(sw), 2)
        self.assertLess(sw[1][0], 2.0 + 0.7 + 0.25)  # B ready, A gone: no 1.0 s wait needed

    def test_with_no_lock_the_newest_ready_look_is_picked(self):
        fps = 30
        # A leaves; when its exit window ends, B (looking since 1.3) and C (since 1.5) are
        # both ready and nobody holds the lock: C, the more recent look, is picked.
        # takeover_s is huge so the pick comes from the no-lock path, not a takeover.
        frames = []
        for n in range(int(fps * 4)):
            t = n / fps
            frames.append([user(), peeper_at(0.85, t < 1.0), peeper_at(0.65, t >= 1.3),
                           peeper_at(0.5, t >= 1.5)])
        tracker = PeeperTracker(TrackerConfig(min_faces=2, exit_s=1.5, takeover_s=99))
        results = run(tracker, frames, fps)
        a_id, b_id, c_id = (results[0][1].tracks[i].id for i in (1, 2, 3))
        self.assertEqual([pid for _, pid in switches(results)], [a_id, c_id])
