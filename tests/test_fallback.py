"""Laptop-guided fallback aim and the self-learning handoff map."""

import io
import json
import random
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
        # Search off and 150 frames: the entry offset now decays over 1 s instead of jumping.
        pan = turret(fallback_search_deg=0)
        now = self.track(pan, 5)
        seen = self.lose_face(pan, 0.7, now, 150)
        states = [state for state, _ in seen]
        self.assertEqual(states[0], "coasting")
        first = states.index("fallback")
        self.assertTrue(set(states[:first]) == {"coasting"} and set(states[first:]) == {"fallback"})
        self.assertLessEqual(first * DT, 0.3 + DT)  # straight after the 0.3 s coast
        self.assertAlmostEqual(seen[-1][1], LINE.angle(0.7), places=3)  # 76.67

    def test_follows_laptop_x_switches_smoothly_and_rate_limited(self) -> None:
        # Rewritten for the damped follow: the old test pinned the EMA-share jump after each switch.
        pan = turret(fallback_search_deg=0)
        pan.update(None, now=0.0, laptop_active=True, laptop_x=0.5, fallback_map=LINE.angle)
        self.assertEqual(pan.state, "fallback")
        angles, now = [pan.command], 0.0
        for k in range(1, 60):
            now = k * DT
            x = 0.5 if k < 10 else 0.8 if k < 35 else 0.2  # two switches in the laptop's x
            pan.update(None, now=now, laptop_active=True, laptop_x=x, fallback_map=LINE.angle)
            angles.append(pan.command)
        steps = [b - a for a, b in zip(angles, angles[1:])]
        self.assertLessEqual(max(abs(s) for s in steps), pan.max_rate * DT + 1e-9)
        self.assertAlmostEqual(angles[9], 90.0, places=3)
        self.assertAlmostEqual(angles[34], 70.0, places=0)
        self.assertAlmostEqual(angles[-1], 110.0, places=0)
        self.assertLess(abs(steps[9]), 0.25 * 20)  # a glide, not a step

    def test_a_heartbeat_without_target_x_keeps_the_last_aim(self) -> None:
        pan = turret(fallback_search_deg=0)  # 150 frames: the entry offset decays over 1 s
        self.lose_face(pan, 0.8, 0.0, 150)
        held = self.lose_face(pan, None, 150 * DT, 5)
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
            self.assertIn(pan.state, ("holding", "returning", "home"))  # holding: lost_hold_s
        self.assertEqual((pan.state, pan.command), ("home", 90.0))

    def test_status_reports_fallback_and_learned_points(self) -> None:
        pan = turret()
        self.lose_face(pan, 0.8, 0.0, 3)
        status = qnx_ai_service._trigger_status(LaptopTrigger(Mock()), True, None, pan, None, learned_points=4)
        self.assertEqual((status["pan_state"], status["fallback_aim"], status["learned_points"]),
                         ("fallback", True, 4))
        self.assertFalse(status["fallback_search"])
        self.lose_face(pan, 0.8, 3 * DT, 30)  # past fallback_search_after_s
        status = qnx_ai_service._trigger_status(LaptopTrigger(Mock()), True, None, pan, None)
        self.assertTrue(status["fallback_search"])
        pan.reset()
        status = qnx_ai_service._trigger_status(LaptopTrigger(Mock()), False, None, pan, None)
        self.assertEqual((status["fallback_aim"], status["learned_points"]), (False, None))

def jittery(start, seconds, seed=7):
    """Frame times at 13 fps with the Pi's jitter (+-25 ms)."""
    rng, now, times = random.Random(seed), start, []
    while now < start + seconds:
        times.append(now)
        now += DT + rng.uniform(-0.025, 0.025)
    return times


class SmoothFallbackTests(unittest.TestCase):
    """The fallback's motion at 13 fps with jittery frame times."""

    def fallback(self, pan, times, x_at, fallback_map=LINE.angle, **kwargs):
        seen = []
        for t in times:
            pan.update(None, now=t, laptop_active=True, laptop_x=x_at(t), fallback_map=fallback_map, **kwargs)
            seen.append((t, pan.state, pan.command))
        return seen

    def assert_smooth(self, pan, seen):
        """Capped at max_rate and fallback_max_accel frame to frame."""
        rate = 0.0
        for (t0, _, a0), (t1, _, a1) in zip(seen, seen[1:]):
            new = (a1 - a0) / (t1 - t0)
            self.assertLessEqual(abs(new), pan.max_rate + 1e-6)
            self.assertLessEqual(abs(new - rate), pan.fallback_max_accel * (t1 - t0) + 1e-6)
            rate = new

    def test_entering_from_a_tracking_angle_off_the_map_does_not_jump(self) -> None:
        pan = turret(fallback_search_deg=0)
        mapped = LINE.angle(0.38)  # 98: the turret tracks a face at 90, 8 degrees off the map
        times = jittery(0.0, 5.0)
        for t in times[:10]:
            pan.update(0.5, now=t, laptop_active=True, laptop_x=0.38, fallback_map=LINE.angle)
        lost = times[10]
        seen = self.fallback(pan, times[10:], lambda t: 0.38)
        steps = [abs(b[2] - a[2]) for a, b in zip(seen, seen[1:])]
        first = next(i for i, (_, state, _) in enumerate(seen) if state == "fallback")
        self.assertLess(abs(seen[first][2] - seen[first - 1][2]), 0.25)  # no jump at the handoff
        self.assertLess(max(steps), 1.0)                                  # a slow, smooth blend
        settled = [angle for t, _, angle in seen if t >= lost + 3.0]
        self.assertTrue(all(abs(angle - mapped) < 1.0 for angle in settled), settled[:3])

    def test_a_moving_peeper_is_followed_without_lag_thanks_to_the_lead(self) -> None:
        pan = turret(fallback_search_deg=0)
        speed, latency = 0.15, 0.15  # laptop widths per second; the laptop's x is 0.15 s old
        true_x = lambda t: 0.5 + speed * (t + latency)
        seen = self.fallback(pan, jittery(0.0, 2.6), lambda t: true_x(t - latency))
        self.assert_smooth(pan, seen)
        lag = [abs(angle - LINE.angle(true_x(t))) for t, _, angle in seen if t >= 1.5]
        self.assertLess(max(lag), 1.0)

    def test_a_target_switch_glides_monotonic_capped_and_without_overshoot(self) -> None:
        for before, after in ((0.5, 0.8), (0.8, 0.2), (0.2, 1.0)):
            with self.subTest(before=before, after=after):
                pan = turret(fallback_search_deg=0)
                times = jittery(0.0, 7.0)
                switch = times[52]  # 4 s: settled on map(before), the entry offset gone
                seen = self.fallback(pan, times, lambda t: before if t < switch else after)
                self.assert_smooth(pan, seen)
                start, goal = seen[51][2], LINE.angle(after)
                self.assertAlmostEqual(start, LINE.angle(before), delta=1.0)
                path = [angle for t, _, angle in seen if t >= switch]
                direction = 1 if goal > start else -1
                steps = [direction * (b - a) for a, b in zip(path, path[1:])]
                self.assertTrue(all(step >= -1e-9 for step in steps), path)     # monotonic
                self.assertTrue(all(direction * (angle - goal) <= 1e-6 for angle in path))  # no overshoot
                self.assertLess(abs(path[0] - start), 4.0)                     # no snap
                self.assertAlmostEqual(path[-1], goal, delta=0.1)

    def test_a_face_after_fallback_blends_back_over_0_3_s(self) -> None:
        pan = PanTracker(55, 125, 90, reversed=True, fallback_search_deg=0)  # predictive: whole degrees
        times = jittery(0.0, 3.0)
        self.fallback(pan, times[:20], lambda t: 0.5)
        self.assertEqual(pan.command, 90.0)
        face = 98.0  # world bearing; tracking alone would jump the command 8 degrees at once
        sent, start = [], times[20]
        for t in times[20:]:
            pan.update(pan.expected_image_x(face), now=t, laptop_active=True, laptop_x=0.5,
                       fallback_map=LINE.angle)
            sent.append((t - start, pan.state, pan.command))
        self.assertEqual(sent[0][1:], ("tracking", 90.0))  # first face frame: still the fallback command
        steps = [b[2] - a[2] for a, b in zip(sent, sent[1:])]
        self.assertTrue(all(step >= 0 for step in steps))
        self.assertLessEqual(max(steps), 4.0)
        reached = next(t for t, _, angle in sent if abs(angle - face) <= 1.0)
        self.assertGreaterEqual(reached, 0.2)
        self.assertLessEqual(reached, 0.45)

    def test_both_lost_holds_about_a_second_then_returns_at_30_deg_s(self) -> None:
        pan = turret(fallback_search_deg=0)
        times = jittery(0.0, 9.0)
        self.fallback(pan, times[:60], lambda t: 0.8)  # settled near map(0.8) = 70
        held, lost = pan.command, times[60]
        seen = []
        for t in times[60:]:
            pan.update(None, now=t, laptop_active=False, fallback_map=LINE.angle)
            seen.append((t, pan.state, pan.command))
        holding = [(t, angle) for t, state, angle in seen if state == "holding"]
        returning = [(t, angle) for t, state, angle in seen if state == "returning"]
        self.assertTrue(all(angle == held for _, angle in holding))
        self.assertAlmostEqual(returning[0][0] - lost, 1.0, delta=DT + 0.03)
        for (t0, a0), (t1, a1) in zip(returning, returning[1:]):
            self.assertLessEqual(abs(a1 - a0), 30.0 * (t1 - t0) + 1e-6)
        self.assertAlmostEqual((returning[-1][1] - returning[0][1]) / (returning[-1][0] - returning[0][0]),
                               30.0, delta=1.0)
        self.assertEqual((seen[-1][1], seen[-1][2]), ("home", 90.0))
        # A face lost with no laptop at all: coast, hold, then the same glide.
        pan = turret()
        for k in range(10):
            pan.update(0.7, now=k * DT)  # moves the turret off home
        states = []
        for t in jittery(10 * DT, 2.0):
            pan.update(None, now=t)
            states.append((t - 10 * DT, pan.state))  # seconds since the face was lost
        first = {state: next(t for t, s in states if s == state) for state in ("holding", "returning")}
        self.assertAlmostEqual(first["holding"], 0.3, delta=0.1)
        self.assertAlmostEqual(first["returning"] - first["holding"], 1.0, delta=0.1)
    def test_a_short_laptop_dropout_never_heads_home(self) -> None:
        # Live: the laptop lost the peeper ~0.5 s and the turret lurched coasting -> home -> fallback.
        # The laptop holds its own exit 0.7 s, so dropouts up to then must leave the turret where it is.
        for face_first in (False, True):
            with self.subTest(face_first=face_first):
                pan = turret(fallback_search_deg=0)
                times = jittery(0.0, 8.0)
                self.fallback(pan, times[:50], lambda t: 0.8)  # settled near map(0.8) = 70
                k = 50
                if face_first:  # the turret camera had the face too, and lost it with the laptop
                    for t in times[k:k + 5]:
                        pan.update(0.5, now=t, laptop_active=True, laptop_x=0.8, fallback_map=LINE.angle)
                    k += 5
                held, drop = pan.command, times[k]
                during = []
                for t in times[k:]:
                    if t >= drop + 0.7:
                        break
                    pan.update(None, now=t, laptop_active=False, fallback_map=LINE.angle)
                    during.append((pan.state, pan.command))
                    k += 1
                self.assertTrue({state for state, _ in during} <= {"fallback", "coasting", "holding"}, during)
                self.assertTrue(all(abs(angle - held) < 0.5 for _, angle in during), during)
                after = self.fallback(pan, times[k:k + 20], lambda t: 0.8)
                self.assertEqual(after[-1][1], "fallback")
                self.assertTrue(all(abs(angle - held) < 1.0 for _, _, angle in after))

    def test_no_face_for_1_5_s_sweeps_around_the_map_estimate_and_widens_once(self) -> None:
        pan = turret()  # 55..125, home 90; laptop x 0.5 -> map 90
        seen, flags = [], []
        for t in jittery(0.0, 9.0):
            pan.update(None, now=t, laptop_active=True, laptop_x=0.5, fallback_map=LINE.angle)
            seen.append((t, pan.state, pan.command))
            flags.append((t, pan.local_search))
        self.assertEqual({state for _, state, _ in seen}, {"fallback"})
        self.assert_smooth(pan, seen)
        early = [angle for t, _, angle in seen if t < 1.5]
        self.assertTrue(all(abs(angle - 90) < 0.5 for angle in early))
        self.assertFalse(any(flag for t, flag in flags if t < 1.4))
        self.assertTrue(all(flag for t, flag in flags if t > 1.6))
        first = [angle for t, _, angle in seen if 1.5 <= t < 1.5 + 80 / 30]  # one +-20 cycle
        self.assertGreaterEqual(max(first), 107)
        self.assertLessEqual(min(first), 73)
        self.assertLess(max(abs(angle - 90) for angle in first), 26)
        wide = [angle for t, _, angle in seen if t >= 1.5 + 80 / 30 + 0.5]
        self.assertGreater(max(wide) - min(wide), 55)  # +-40, clamped to the 55..125 range

    def test_an_extrapolating_map_searches_at_once_and_a_face_ends_it(self) -> None:
        pan = turret()
        domain = (0.3, 0.6)  # learned points only for 0.3..0.6
        self.fallback(pan, jittery(0.0, 1.0), lambda t: 0.62, fallback_domain=domain)
        self.assertFalse(pan.local_search)  # within the 0.05 margin: trust the map for 1.5 s
        pan = turret()
        times = jittery(0.0, 2.0)
        seen = self.fallback(pan, times[:13], lambda t: 0.85, fallback_domain=domain)
        self.assertTrue(pan.local_search)
        entry = seen[0][2]
        self.assertGreater(max(abs(angle - entry) for _, _, angle in seen), 4.0)  # sweeping already
        before = pan.command
        pan.update(0.5, now=times[13], laptop_active=True, laptop_x=0.85, fallback_map=LINE.angle,
                   fallback_domain=domain)
        self.assertEqual((pan.state, pan.local_search, pan.command), ("tracking", False, before))


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
