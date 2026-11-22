"""PeeperMatcher: aim at the laptop's peeper among the turret camera's faces, not at the user."""

import json
import unittest
from unittest.mock import Mock

import qnx_ai_service
from medusa.handoff import HandoffMap
from medusa.tflite_detector import FaceBox
from medusa.trigger import (
    MATCH_REASONS, NO_MATCH_REASON, FaceConfirm, LaptopTrigger, PanTracker, PeeperMatcher, decide_aim,
    parse_peeper_message,
)

W, H = 640, 480


def face(x, size=40, y=0.5, score=0.9):
    """A square box of ``size`` px centred at normalized (x, y)."""
    half = size / 2
    return FaceBox(W * x - half, H * y - half, W * x + half, H * y + half, score)


def picked_x(picked):
    return None if picked is None else round(picked[1], 4)


class ExpectedImageXTests(unittest.TestCase):
    def test_inverts_the_trackers_own_bearing_both_ways(self) -> None:
        for reversed_ in (True, False):
            with self.subTest(reversed=reversed_):
                pan = PanTracker(10, 80, 45, reversed=reversed_, mode="predictive")
                pan.update(0.7, now=0.0)  # First sight: bearing = 45 + sign * 0.2 * 50.
                self.assertAlmostEqual(pan.bearing, 35.0 if reversed_ else 55.0)
                self.assertAlmostEqual(pan.expected_image_x(pan.bearing), 0.7)

    def test_reversed_a_lower_pan_than_the_turret_is_right_of_centre(self) -> None:
        pan = PanTracker(10, 80, 45, reversed=True)
        self.assertAlmostEqual(pan.expected_image_x(35.0), 0.7)  # 10 deg lower: 0.2 of 50 deg right.
        self.assertAlmostEqual(pan.expected_image_x(55.0), 0.3)
        self.assertAlmostEqual(PanTracker(10, 80, 45).expected_image_x(35.0), 0.3)

    def test_uses_the_real_pan_when_fresh(self) -> None:
        pan = PanTracker(10, 80, 45, reversed=True)
        pan.observe_actual(50.0, 0.0)
        pan.update(None, now=0.0)
        self.assertEqual(pan.current_pan(), 50.0)
        self.assertAlmostEqual(pan.expected_image_x(40.0), 0.7)
        pan.observe_actual(None, None)
        pan.update(None, now=0.1)
        self.assertEqual(pan.current_pan(), pan.servo.position)  # Stale: the firmware model.

    def test_service_expected_x_needs_a_usable_map_and_a_laptop_x(self) -> None:
        pan = PanTracker(10, 80, 45, reversed=True)
        learned = HandoffMap([(0.2, 70.0), (0.8, 20.0)])
        self.assertAlmostEqual(qnx_ai_service._expected_x(pan, learned, 0.5), 0.5 + (45 - 45.0) / 50)
        self.assertAlmostEqual(qnx_ai_service._expected_x(pan, learned, 0.65), 0.75)  # map -> 32.5 deg.
        self.assertIsNone(qnx_ai_service._expected_x(pan, learned, None))
        self.assertIsNone(qnx_ai_service._expected_x(pan, HandoffMap([(0.2, 70.0)]), 0.5))
        self.assertIsNone(qnx_ai_service._expected_x(pan, None, 0.5))
        self.assertIsNone(qnx_ai_service._expected_x(None, learned, 0.5))

    def test_service_expected_x_is_none_outside_the_learned_range(self) -> None:
        # Live bug: points only at laptop_x 0.08..0.46; a peeper at 0.85 was gated out by the
        # map's extrapolated guess, so the turret never aimed at the face it saw or learned.
        pan = PanTracker(15, 165, 90, reversed=True)
        learned = HandoffMap([(0.08, 102.6), (0.22, 89.3), (0.46, 66.6)], extrapolate=True, bounds=(15, 165))
        self.assertIsNotNone(qnx_ai_service._expected_x(pan, learned, 0.30))
        self.assertIsNotNone(qnx_ai_service._expected_x(pan, learned, 0.50))   # inside the margin
        self.assertIsNone(qnx_ai_service._expected_x(pan, learned, 0.85))
        self.assertIsNone(qnx_ai_service._expected_x(pan, learned, 0.01))
        # With no expected_x the matcher takes the only face the turret sees.
        matcher = PeeperMatcher()
        picked = matcher.pick([face(0.6)], W, H, expected_x=qnx_ai_service._expected_x(pan, learned, 0.85),
                              laptop_x=0.85)
        self.assertEqual((picked_x(picked), matcher.match), (0.6, "single"))


class MapMatchTests(unittest.TestCase):
    def test_the_face_nearest_expected_x_wins_regardless_of_size(self) -> None:
        for peeper_size, user_size in ((120, 40), (40, 120)):
            with self.subTest(peeper_size=peeper_size):
                matcher = PeeperMatcher()
                boxes = [face(0.25, user_size), face(0.7, peeper_size)]
                self.assertEqual(picked_x(matcher.pick(boxes, W, H, expected_x=0.66)), 0.7)
                self.assertEqual((matcher.match, matcher.expected_x), ("map", 0.66))

    def test_the_gate_rejects_far_faces_so_the_tracker_falls_back_to_the_map(self) -> None:
        matcher = PeeperMatcher()
        boxes = [face(0.2), face(0.3)]
        aim = decide_aim(True, "", boxes, W, H, matcher=matcher, expected_x=0.8, laptop_x=0.6)
        self.assertEqual((aim.active, aim.reason, matcher.match), (False, NO_MATCH_REASON, "none"))
        self.assertIsNone(PeeperMatcher(gate=0.25).pick([face(0.54)], W, H, expected_x=0.8))
        self.assertEqual(picked_x(PeeperMatcher(gate=0.25).pick([face(0.56)], W, H, expected_x=0.8)), 0.56)
        # No aim and a usable map: the tracker aims at map(laptop_x), not at the user.
        learned = HandoffMap([(0.2, 70.0), (0.8, 20.0)], extrapolate=True)
        pan = PanTracker(10, 80, 45, reversed=True, mode="proportional")
        pan.update(None, now=0.0, laptop_active=True, laptop_x=0.6, fallback_map=learned.angle)
        self.assertEqual(pan.state, "fallback")

    def test_map_match_feeds_decide_aim_with_its_reason(self) -> None:
        aim = decide_aim(True, "", [face(0.25, 120), face(0.7)], W, H, matcher=PeeperMatcher(),
                         expected_x=0.7)
        self.assertEqual((aim.active, round(aim.target_x, 4), aim.reason), (True, 0.7, MATCH_REASONS["map"]))


class LockTests(unittest.TestCase):
    def test_keeps_the_peeper_when_the_user_is_close(self) -> None:
        matcher = PeeperMatcher()
        user = face(0.45, 40)  # Smaller: the old rule would pick the user.
        self.assertEqual(picked_x(matcher.pick([user, face(0.58, 120)], W, H, expected_x=0.6)), 0.58)
        # The map drops out (no laptop target_x): the lock, not "smallest", decides.
        for x in (0.56, 0.54, 0.53):
            self.assertEqual(picked_x(matcher.pick([user, face(x, 120)], W, H, laptop_x=None)), x)
        self.assertEqual(matcher.match, "map")  # How the locked face was first chosen.

    def test_rides_out_a_dropout_without_picking_the_user_then_rematches(self) -> None:
        matcher = PeeperMatcher(lock_dropout=2)
        user = face(0.2)
        self.assertEqual(picked_x(matcher.pick([user, face(0.7, 120)], W, H, laptop_x=0.8, user_x=0.3)), 0.7)
        self.assertEqual(matcher.match, "order")
        self.assertIsNone(matcher.pick([user], W, H))
        self.assertIsNone(matcher.pick([user], W, H))
        self.assertEqual(picked_x(matcher.pick([user, face(0.69, 120)], W, H)), 0.69)  # Back in time.
        for _ in range(2):
            matcher.pick([user], W, H)
        self.assertEqual(picked_x(matcher.pick([user], W, H)), 0.2)  # Lock lost: matching runs again.
        self.assertEqual(matcher.match, "single")

    def test_a_locked_face_outside_the_gate_is_dropped_at_once(self) -> None:
        matcher = PeeperMatcher()
        matcher.pick([face(0.5)], W, H)  # Locked by "single" before the map was usable.
        self.assertEqual(picked_x(matcher.pick([face(0.52), face(0.9)], W, H, expected_x=0.88)), 0.9)
        self.assertEqual(matcher.match, "map")

    def test_the_laptop_going_inactive_drops_the_lock(self) -> None:
        matcher = PeeperMatcher()
        matcher.pick([face(0.5, 120), face(0.6)], W, H, expected_x=0.5)
        matcher.pick([face(0.5, 120), face(0.6)], W, H, laptop_active=False)
        self.assertIsNone(matcher.lock_x)
        self.assertIsNone(matcher.expected_x)
        self.assertEqual(picked_x(matcher.pick([face(0.5, 120), face(0.6)], W, H)), 0.6)  # Smallest.


class NoMapTests(unittest.TestCase):
    def test_one_face_is_aimed_at(self) -> None:
        matcher = PeeperMatcher()
        aim = decide_aim(True, "", [face(0.3, 200)], W, H, matcher=matcher, laptop_x=0.7)
        self.assertEqual((aim.active, round(aim.target_x, 4), matcher.match), (True, 0.3, "single"))
        self.assertEqual(aim.reason, MATCH_REASONS["single"])

    def test_two_faces_pick_the_smallest_unless_the_laptop_sends_user_x(self) -> None:
        boxes = [face(0.2, 40), face(0.5, 120), face(0.8, 80)]
        matcher = PeeperMatcher()
        self.assertEqual(picked_x(matcher.pick(boxes, W, H, laptop_x=0.7)), 0.2)
        self.assertEqual(matcher.match, "smallest")
        matcher = PeeperMatcher()
        self.assertEqual(picked_x(matcher.pick(boxes, W, H, laptop_x=0.7, user_x=0.4)), 0.8)  # Right of user.
        self.assertEqual(matcher.match, "order")
        matcher = PeeperMatcher()
        self.assertEqual(picked_x(matcher.pick(boxes, W, H, laptop_x=0.2, user_x=0.5)), 0.2)  # Left.
        self.assertEqual(matcher.match, "order")

    def test_without_a_matcher_decide_aim_keeps_the_smallest_face(self) -> None:
        aim = decide_aim(True, "", [face(0.2, 120), face(0.7, 40)], W, H)
        self.assertEqual((round(aim.target_x, 4), aim.reason), (0.7, MATCH_REASONS["smallest"]))

    def test_matcher_and_face_confirm_together(self) -> None:
        matcher, confirm = PeeperMatcher(), FaceConfirm(confirm_frames=2, dropout_frames=2)
        boxes = [face(0.3), face(0.7, 120)]
        decide_aim(True, "", boxes, W, H, confirm=confirm, matcher=matcher, expected_x=0.7)
        aim = decide_aim(True, "", boxes, W, H, confirm=confirm, matcher=matcher, expected_x=0.7)
        self.assertEqual((aim.active, round(aim.target_x, 4)), (True, 0.7))
        aim = decide_aim(True, "", [face(0.3)], W, H, confirm=confirm, matcher=matcher)  # Peeper dropout.
        self.assertEqual((aim.active, aim.held, round(aim.target_x, 4)), (True, True, 0.7))


class PeeperSwitchTests(unittest.TestCase):
    def test_a_new_laptop_peeper_id_drops_the_lock_and_rematches(self) -> None:
        matcher = PeeperMatcher()
        old, new = face(0.5), face(0.62)  # Both inside the lock radius of each other.
        self.assertEqual(picked_x(matcher.pick([old], W, H, peeper_id=1)), 0.5)
        # Same peeper: the lock keeps the old face even with the newer one in view.
        self.assertEqual(picked_x(matcher.pick([old, new], W, H, expected_x=0.6, peeper_id=1)), 0.5)
        # The laptop switched peepers: the lock drops and the map's expected x picks again.
        self.assertEqual(picked_x(matcher.pick([old, new], W, H, expected_x=0.6, peeper_id=2)), 0.62)

    def test_no_peeper_id_keeps_the_old_lock_behaviour(self) -> None:
        matcher = PeeperMatcher()
        matcher.pick([face(0.5)], W, H)
        self.assertEqual(picked_x(matcher.pick([face(0.5), face(0.62)], W, H, expected_x=0.6)), 0.5)


class PacketAndStatusTests(unittest.TestCase):
    def test_user_x_is_optional(self) -> None:
        def parse(**fields):
            return parse_peeper_message(json.dumps({"type": "medusa-peeper", "active": True, **fields}).encode())

        self.assertEqual(parse(target_x=0.7, user_x=0.3).user_x, 0.3)
        self.assertIsNone(parse(target_x=0.7).user_x)
        self.assertIsNone(parse(target_x=0.7, user_x="left").user_x)
        self.assertEqual(parse(target_x=0.7, peeper_id=4).peeper_id, 4)
        self.assertIsNone(parse(target_x=0.7).peeper_id)
        self.assertIsNone(parse(target_x=0.7, peeper_id=True).peeper_id)
        self.assertIsNone(parse(target_x=0.7, peeper_id="4").peeper_id)

    def test_status_reports_the_match_and_expected_x(self) -> None:
        trigger = LaptopTrigger(Mock())
        matcher = PeeperMatcher()
        aim = decide_aim(True, "", [face(0.3), face(0.7)], W, H, matcher=matcher, expected_x=0.712345)
        status = qnx_ai_service._trigger_status(trigger, True, aim, matcher=matcher)
        self.assertEqual((status["aim_match"], status["expected_x"]), ("map", 0.7123))
        matcher = PeeperMatcher()
        aim = decide_aim(True, "", [face(0.3)], W, H, matcher=matcher, expected_x=0.9)  # Gated out.
        status = qnx_ai_service._trigger_status(trigger, True, aim, matcher=matcher)
        self.assertEqual(status["aim_match"], "none")
        status = qnx_ai_service._trigger_status(trigger, False, None)
        self.assertEqual((status["aim_match"], status["expected_x"]), ("none", None))


if __name__ == "__main__":
    unittest.main()
