"""Laptop-guided fallback aim and the self-learning handoff map."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock

import qnx_ai_service
from medusa.handoff import HandoffLearner, HandoffMap
from medusa.trigger import LaptopTrigger, PanTracker

FPS = 13.0
DT = 1 / FPS
LIVE = {"faces": 1, "match": "single", "armed": True, "gate_closed": True}  # learning allowed


def turret(**kwargs) -> PanTracker:
    """The bench setup: pan 55..125, reversed, proportional mode."""
    return PanTracker(55, 125, 90, reversed=True, mode="proportional", coast_s=0.3, **kwargs)


LINE = HandoffMap([[0.2, 110], [0.8, 70]], extrapolate=True, bounds=(55, 125))  # laptop_x -> pan


class FallbackTests(unittest.TestCase):
    def lose_face(self, pan, laptop_x, start, frames, fallback_map=LINE.angle, **kwargs):
        seen = []
        for k in range(frames):
            pan.update(None, now=start + k * DT, laptop_active=True, laptop_x=laptop_x,
                       fallback_map=fallback_map, **kwargs)
            seen.append((pan.state, pan.command))
        return seen

    def track(self, pan, frames, start=0.0):
        for k in range(frames):
            pan.update(0.5, now=start + k * DT, laptop_active=True, laptop_x=0.5, fallback_map=LINE.angle)
        return start + frames * DT

    def test_engages_after_the_coast_and_settles_on_the_mapped_angle(self) -> None:
        pan = turret()
        now = self.track(pan, 5)
        seen = self.lose_face(pan, 0.7, now, 40)
        states = [state for state, _ in seen]
        self.assertEqual(states[0], "coasting")
        first = states.index("fallback")
        self.assertTrue(set(states[:first]) == {"coasting"} and set(states[first:]) == {"fallback"})
        self.assertLessEqual(first * DT, 0.3 + DT)  # straight after the 0.3 s coast
        self.assertAlmostEqual(seen[-1][1], LINE.angle(0.7), places=3)  # 76.67

    def test_follows_a_moving_laptop_x_smoothly_and_rate_limited(self) -> None:
        pan = turret()
        pan.update(None, now=0.0, laptop_active=True, laptop_x=0.2, fallback_map=LINE.angle)
        self.assertEqual(pan.state, "fallback")
        angles, now = [pan.command], 0.0
        for k in range(1, 60):
            now = k * DT
            x = 0.2 if k < 10 else 0.8 if k < 35 else 0.5  # two jumps in the laptop's x
            pan.update(None, now=now, laptop_active=True, laptop_x=x, fallback_map=LINE.angle)
            angles.append(pan.command)
        steps = [b - a for a, b in zip(angles, angles[1:])]
        self.assertLessEqual(max(abs(s) for s in steps), pan.max_rate * DT + 1e-9)
        self.assertAlmostEqual(angles[9], 110.0, places=3)
        self.assertAlmostEqual(angles[34], 70.0, places=1)
        self.assertAlmostEqual(angles[-1], 90.0, places=1)
        # Smoothed, not stepped: the first move after a jump is a fraction of the 40 degrees.
        self.assertLess(abs(steps[9]), 0.5 * 40)
        # And with no rate limit in the way the move is the EMA share of the jump (tau 0.25 s).
        fast = turret(max_rate=1e6)
        fast.update(None, now=0.0, laptop_active=True, laptop_x=0.2, fallback_map=LINE.angle)
        fast.update(None, now=DT, laptop_active=True, laptop_x=0.8, fallback_map=LINE.angle)
        share = 1 - 2.718281828459045 ** (-DT / 0.25)
        self.assertAlmostEqual(110 - fast.command, share * 40, places=3)

    def test_a_heartbeat_without_target_x_keeps_the_last_aim(self) -> None:
        pan = turret()
        self.lose_face(pan, 0.8, 0.0, 30)
        held = self.lose_face(pan, None, 30 * DT, 5)
        self.assertEqual({state for state, _ in held}, {"fallback"})
        self.assertAlmostEqual(held[-1][1], 70.0, places=2)

    def test_the_face_returning_hands_back_to_tracking(self) -> None:
        pan = turret()
        self.lose_face(pan, 0.8, 0.0, 20)
        self.assertEqual(pan.state, "fallback")
        error = pan.update(0.6, now=20 * DT, laptop_active=True, laptop_x=0.8, fallback_map=LINE.angle)
        self.assertAlmostEqual(error, 0.1)
        self.assertEqual(pan.state, "tracking")

    def test_no_usable_map_keeps_the_search_sweep(self) -> None:
        pan = turret()
        seen = self.lose_face(pan, 0.8, 0.0, 10, fallback_map=HandoffMap([[0.5, 90]]).angle)
        self.assertEqual({state for state, _ in seen}, {"searching"})
        pan = turret()
        seen = self.lose_face(pan, 0.8, 0.0, 3, fallback_map=None)
        self.assertEqual({state for state, _ in seen}, {"searching"})

    def test_laptop_inactive_glides_home(self) -> None:
        pan = turret()
        self.lose_face(pan, 0.8, 0.0, 20)
        for k in range(20, 60):
            pan.update(None, now=k * DT, laptop_active=False, laptop_x=None, fallback_map=LINE.angle)
            self.assertIn(pan.state, ("returning", "home"))
        self.assertEqual((pan.state, pan.command), ("home", 90.0))

    def test_status_reports_fallback_and_learned_points(self) -> None:
        pan = turret()
        self.lose_face(pan, 0.8, 0.0, 3)
        status = qnx_ai_service._trigger_status(LaptopTrigger(Mock()), True, None, pan, None, learned_points=4)
        self.assertEqual((status["pan_state"], status["fallback_aim"], status["learned_points"]),
                         ("fallback", True, 4))
        pan.reset()
        status = qnx_ai_service._trigger_status(LaptopTrigger(Mock()), False, None, pan, None)
        self.assertEqual((status["fallback_aim"], status["learned_points"]), (False, None))


class LearnedMapFitTests(unittest.TestCase):
    def test_interpolates_extrapolates_linearly_and_clamps(self) -> None:
        handoff = HandoffMap([[0.3, 100], [0.5, 90], [0.7, 70]], extrapolate=True, bounds=(55, 112))
        for x, pan in ((0.4, 95), (0.6, 80), (0.2, 105), (0.1, 110), (0.0, 112), (0.8, 60), (1.0, 55)):
            self.assertAlmostEqual(handoff.angle(x), pan)
        self.assertIsNone(HandoffMap([[0.3, 100]], extrapolate=True).angle(0.3))
        # The plain map (manual points, old behaviour) still clamps to the end points.
        self.assertAlmostEqual(HandoffMap([[0.3, 100], [0.5, 90]]).angle(0.0), 100)


class LearnerTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "handoff.json"

    def learner(self, **kwargs) -> HandoffLearner:
        return HandoffLearner(self.path, minimum=55, maximum=125, **kwargs)

    def feed(self, learner, frames, start=0.0, laptop_x=0.42, error=0.0, pan=lambda k: 97.0):
        recorded = []
        for k in range(frames):
            recorded.append(learner.observe(start + k * DT, laptop_x, error, pan(k), **LIVE))
        return recorded

    def test_records_only_when_laptop_active_centred_and_steady_for_0_4_s(self) -> None:
        learner = self.learner()
        self.assertFalse(any(self.feed(learner, 20, laptop_x=None)))                 # laptop inactive
        self.assertFalse(any(self.feed(learner, 20, start=2.0, error=0.04)))         # face off centre
        self.assertFalse(any(self.feed(learner, 20, start=4.0, error=None)))         # no tracked face
        self.assertFalse(any(self.feed(learner, 20, start=6.0, pan=lambda k: None)))  # no real pan
        self.assertFalse(any(self.feed(learner, 20, start=8.0, pan=lambda k: 90 + 0.3 * k)))  # moving
        self.assertEqual(learner.learned_points, 0)
        learner.observe(9.9, None, None, None, **LIVE)  # end the moving run
        recorded = self.feed(learner, 10, start=10.0)
        first = recorded.index(True)
        self.assertGreaterEqual(first * DT, 0.4 - 1e-9)
        self.assertLess((first - 1) * DT, 0.4)
        self.assertTrue(all(recorded[first:]))
        self.assertEqual(learner.learned_points, 1)
        # Any frame off centre restarts the 0.4 s wait.
        learner.observe(15.0, 0.42, 0.05, 97.0, **LIVE)
        self.assertFalse(learner.observe(15.0 + DT, 0.42, 0.0, 97.0, **LIVE))
        # Jitter below 0.5 deg over the window still counts as steady.
        self.assertTrue(any(self.feed(learner, 10, start=20.0, pan=lambda k: 97 + 0.2 * (k % 2))))

    def test_bins_average_with_an_ema_and_two_bins_make_a_usable_map(self) -> None:
        learner = self.learner(alpha=0.5)
        self.assertEqual(self.feed(learner, 7, laptop_x=0.42, pan=lambda k: 100.0)[-2:], [False, True])
        self.assertEqual(learner.learned_points, 1)
        self.assertFalse(learner.current().usable)
        before = learner.bins[8]
        self.assertEqual(before, (0.42, 100.0))
        learner.observe(5.0, 0.44, 0.0, 104.0, **LIVE)  # same bin (0.40..0.45), but a fresh 0.4 s wait
        self.feed(learner, 7, start=5.0 + DT, laptop_x=0.44, pan=lambda k: 104.0)
        x, pan = learner.bins[8]
        self.assertTrue(0.42 < x < 0.44 and 100.0 < pan < 104.0)
        self.assertEqual(learner.learned_points, 1)
        self.feed(learner, 7, start=10.0, laptop_x=0.81, pan=lambda k: 70.0)
        self.assertEqual(learner.learned_points, 2)
        self.assertTrue(learner.current().usable)
        self.assertAlmostEqual(learner.current().angle(0.81), 70.0)
        self.assertEqual(learner.current().angle(1.0), 55.0)  # extrapolated past 0.81, clamped at 55

    def test_loads_at_start_with_manual_points_as_bins(self) -> None:
        HandoffMap([[0.2, 110], [0.21, 108], [0.8, 70]]).save(self.path)  # 0.2 and 0.21 share a bin
        learner = self.learner()
        self.assertEqual(learner.learned_points, 2)
        self.assertAlmostEqual(learner.bins[4][1], 109.0)
        self.assertTrue(learner.current().usable)

    def test_saves_at_most_every_5_s_and_picks_up_cli_edits(self) -> None:
        learner = self.learner()
        self.assertFalse(learner.maybe_save(0.0))  # nothing learned: no write
        self.assertFalse(self.path.exists())
        self.feed(learner, 7, laptop_x=0.3, pan=lambda k: 100.0)
        self.assertTrue(learner.maybe_save(1.0))
        self.assertEqual(HandoffMap.load(self.path).points, [(0.3, 100.0)])
        self.feed(learner, 7, start=2.0, laptop_x=0.7, pan=lambda k: 80.0)
        self.assertFalse(learner.maybe_save(5.9))  # within 5 s of the last write
        self.assertTrue(learner.maybe_save(6.0))
        self.assertEqual(len(HandoffMap.load(self.path).points), 2)
        # Our own write is not reloaded as a change: learning since then is kept.
        self.feed(learner, 7, start=7.0, laptop_x=0.7, pan=lambda k: 84.0)
        moved = learner.bins[14]
        self.assertGreater(moved[1], 80.0)
        learner.file._checked_at = None
        learner.current()
        self.assertEqual(learner.bins[14], moved)
        # A point recorded with the CLI shows up as a bin (it wins over unsaved learning),
        # and a restart loads everything.
        HandoffMap.load(self.path).with_point(0.95, 60).save(self.path)
        learner.file._checked_at = None
        self.assertEqual(learner.current().angle(0.95), 60.0)
        self.assertEqual(learner.learned_points, 3)
        self.assertEqual(learner.bins[14], (0.7, 80.0))
        self.assertFalse(learner.maybe_save(20.0))  # nothing new since the file was read
        self.assertEqual(self.learner().learned_points, 3)
        # force (shutdown) writes inside the 5 s window.
        self.feed(learner, 7, start=21.0, laptop_x=0.5, pan=lambda k: 90.0)
        self.assertTrue(learner.maybe_save(22.0, force=True))
        self.assertEqual(len(HandoffMap.load(self.path).points), 4)

    def test_never_overwrites_a_file_it_could_not_read(self) -> None:
        self.path.write_text("{not json")
        with redirect_stderr(io.StringIO()):
            learner = self.learner()
            self.feed(learner, 7)
            self.assertFalse(learner.maybe_save(0.0, force=True))
        self.assertEqual(self.path.read_text(), "{not json")


if __name__ == "__main__":
    unittest.main()
