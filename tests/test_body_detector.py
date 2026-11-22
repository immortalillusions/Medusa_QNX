"""Head-from-body fallback: estimator geometry, matching heads to the peeper, rate limit, status."""

import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np

import qnx_ai_service
from medusa import body_detector
from medusa.body_detector import (
    BODY_HEAD, UPPER_HEAD, HeadBox, HeadEstimator, HeadFallback, aim_source, dedupe, draw_head_boxes,
    head_from_box, head_status, is_head,
)
from medusa.tflite_detector import FaceBox
from medusa.trigger import AimDecision, LaptopTrigger, PanTracker, PeeperMatcher, decide_aim

W, H = 640, 480
LAPTOP = ("172.20.10.4", 50000)


def face(x, size=40, y=0.5, score=0.9):
    half = size / 2
    return FaceBox(W * x - half, H * y - half, W * x + half, H * y + half, score)


def head(x, size=40, y=0.4, kind="upper"):
    half = size / 2
    return HeadBox(W * x - half, H * y - half, W * x + half, H * y + half, 0.35, kind=kind)


class FakeCascade:
    def __init__(self, normal=(), flipped=()):
        self.normal, self.flipped = list(normal), list(flipped)

    def empty(self):
        return False

    def detectMultiScale(self, image, *args, **kwargs):
        return self.flipped if getattr(image, "flipped", False) else self.normal


class FakeImage:
    def __init__(self, width, height, flipped=False):
        self.shape, self.flipped = (height, width), flipped


def fake_cv2(hog_rects=()):
    hog = Mock()
    hog.detectMultiScale.return_value = (list(hog_rects), [])
    return SimpleNamespace(
        INTER_AREA=0, COLOR_BGR2GRAY=0, HOGDescriptor=Mock(return_value=hog),
        HOGDescriptor_getDefaultPeopleDetector=Mock(),
        resize=lambda frame, size, interpolation=0: FakeImage(*size),
        cvtColor=lambda image, code: image, equalizeHist=lambda image: image,
        flip=lambda image, code: FakeImage(image.shape[1], image.shape[0], flipped=True),
    )


class GeometryTests(unittest.TestCase):
    def test_head_is_the_centred_top_of_the_box(self) -> None:
        left, top, right, bottom = head_from_box(100, 50, 200, 160, UPPER_HEAD)
        self.assertAlmostEqual((left + right) / 2, 200)  # centred
        self.assertAlmostEqual(top, 58)
        self.assertLess(bottom, 50 + 160 / 2 + 1)  # the top half
        left, top, right, bottom = head_from_box(0, 0, 100, 200, BODY_HEAD)
        self.assertAlmostEqual((left + right) / 2, 50)
        self.assertLessEqual(bottom, 0.25 * 200)  # top ~15-20% of a HOG person box

    def test_estimator_scales_back_mirrors_profiles_and_converts_boxes(self) -> None:
        est = HeadEstimator.__new__(HeadEstimator)
        est.cv2, est.work_width, est.hog_width = fake_cv2(hog_rects=[(10, 20, 64, 128)]), 320, 320
        est.profile = FakeCascade(normal=[(10, 10, 30, 30)], flipped=[(10, 10, 30, 30)])
        est.upper = FakeCascade(normal=[(100, 50, 100, 80)])
        est.hog = est.cv2.HOGDescriptor()
        heads = est.detect(FakeImage(640, 480))
        by_kind = {}
        for h in heads:
            by_kind.setdefault(h.kind, []).append(h)
        # 640 -> 320: every box doubles; the mirrored profile maps back to the far side.
        self.assertEqual([(h.left, h.right) for h in by_kind["profile"]], [(20, 80), (560, 620)])
        upper = by_kind["upper"][0]
        self.assertAlmostEqual((upper.left + upper.right) / 2, 300)
        self.assertAlmostEqual(upper.top, 2 * (50 + UPPER_HEAD[1] * 80))
        body = by_kind["body"][0]
        self.assertAlmostEqual((body.left + body.right) / 2, 2 * (10 + 32))
        self.assertTrue(all(is_head(h) and h.score < 0.5 for h in heads))

    def test_real_cascades_load_and_find_nothing_on_a_blank_frame(self) -> None:
        est = HeadEstimator()
        self.assertEqual(est.kinds, ["profile", "upper", "body"])
        self.assertEqual(est.detect(np.zeros((480, 640, 3), np.uint8)), [])

    def test_missing_cascade_only_disables_that_detector(self) -> None:
        est = HeadEstimator("/nonexistent", body=False)
        self.assertEqual(est.kinds, [])

    def test_dedupe_drops_heads_on_faces_and_repeats(self) -> None:
        user_face = face(0.3)
        on_face = head(0.3, y=0.5, kind="body")
        profile, upper = head(0.7, kind="profile"), head(0.71, kind="upper")
        self.assertEqual(dedupe([on_face, upper, profile], [user_face]), [profile])

    def test_draw_head_boxes_runs_on_a_real_frame(self) -> None:
        frame = np.zeros((480, 640, 3), np.uint8)
        target = head(0.5)
        draw_head_boxes(frame, [target, head(0.2, kind="body")], target)
        self.assertGreater(int(frame.sum()), 0)


class HeadMatchTests(unittest.TestCase):
    """Heads go through PeeperMatcher like faces: never the user."""

    def test_order_picks_the_peepers_head_not_the_users_face(self) -> None:
        user, peeper = face(0.3), head(0.7)
        picked = PeeperMatcher().pick([user, peeper], W, H, laptop_x=0.8, user_x=0.2)
        self.assertIs(picked[0], peeper)
        picked = PeeperMatcher().pick([face(0.7), head(0.3)], W, H, laptop_x=0.8, user_x=0.2)
        self.assertFalse(is_head(picked[0]))  # The peeper's own face, right of the user.

    def test_map_gate_rejects_a_head_away_from_the_peeper(self) -> None:
        matcher = PeeperMatcher()
        self.assertIsNone(matcher.pick([head(0.2)], W, H, expected_x=0.7))
        picked = matcher.pick([head(0.2), head(0.65)], W, H, expected_x=0.7)
        self.assertAlmostEqual(picked[1], 0.65)
        self.assertEqual(matcher.match, "map")

    def test_lock_carries_from_head_to_face(self) -> None:
        matcher = PeeperMatcher()
        matcher.pick([head(0.7)], W, H, laptop_x=0.8, user_x=0.2)
        picked = matcher.pick([face(0.3), face(0.72)], W, H, laptop_x=0.8, user_x=0.2)
        self.assertAlmostEqual(picked[1], 0.72)

    def test_decide_aim_returns_the_head_with_its_centre(self) -> None:
        aim = decide_aim(True, "", [head(0.6, y=0.3)], W, H, matcher=PeeperMatcher())
        self.assertTrue(aim.active)
        self.assertAlmostEqual(aim.target_x, 0.6)
        self.assertAlmostEqual(aim.target_y, 0.3)
        self.assertEqual(aim_source(aim), "upper")


class FakeEstimator:
    def __init__(self, heads=()):
        self.heads, self.calls = list(heads), 0

    def detect(self, frame):
        self.calls += 1
        return list(self.heads)


class RateLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 0.0
        self.est = FakeEstimator([head(0.6)])
        self.fallback = HeadFallback(self.est, every=3, ttl_s=0.5, clock=lambda: self.now)

    def step(self, faces=(), active=True, last_aim=None, dt=0.1):
        self.now += dt
        return self.fallback.augment(None, list(faces), laptop_active=active, last_aim=last_aim)

    def test_runs_every_third_frame_and_reuses_in_between(self) -> None:
        results = [self.step() for _ in range(7)]
        self.assertEqual(self.est.calls, 3)  # frames 1, 4, 7
        self.assertTrue(all(len(r) == 1 and is_head(r[0]) for r in results))

    def test_stale_cache_runs_again(self) -> None:
        self.step()
        self.step(dt=0.6)
        self.assertEqual(self.est.calls, 2)

    def test_off_while_a_face_is_aimed_or_the_laptop_is_idle(self) -> None:
        face_aim = AimDecision(True, 0.5, 0.5, "", face(0.5))
        self.assertEqual(self.step(faces=[face(0.5)], last_aim=face_aim), [face(0.5)])
        self.assertEqual(self.step(active=False), [])
        self.assertEqual(self.est.calls, 0)
        self.assertIsNone(self.fallback.augment(None, None, laptop_active=True))
        head_aim = AimDecision(True, 0.6, 0.4, "", head(0.6))
        self.assertEqual(len(self.step(last_aim=head_aim)), 1)  # A head aim keeps it running.

    def test_status_fields(self) -> None:
        self.step()
        status = head_status(AimDecision(True, 0.6, 0.4, "", self.fallback.heads[0]), False, self.fallback)
        self.assertEqual((status["aim_source"], status["head_candidates"], status["head_runs"]), ("upper", 1, 1))
        self.assertIsInstance(status["head_ms"], float)
        self.assertEqual(head_status(AimDecision(False), True), {"aim_source": "laptop"})
        self.assertEqual(head_status(AimDecision(True, 0.5, 0.5, "", face(0.5))), {"aim_source": "face"})
        self.assertEqual(head_status(None), {"aim_source": None})


class ServiceTests(unittest.TestCase):
    def run_loop(self, argv, script, estimator):
        frame = SimpleNamespace(shape=(480, 640, 3))
        index, pending, pan_inputs = {"i": -1}, [], []

        def read():
            index["i"] += 1
            if index["i"] == len(script):
                raise KeyboardInterrupt
            pending.append((json.dumps({"type": "medusa-peeper", "active": True, "target_x": 0.8,
                                        "user_x": 0.2}).encode(), LAPTOP))
            return True, frame

        def recvfrom(_size):
            if pending:
                return pending.pop(0)
            raise BlockingIOError

        real_update = PanTracker.update

        def update(pan, target_x, *args, **kwargs):
            pan_inputs.append(target_x)
            return real_update(pan, target_x, *args, **kwargs)

        trigger_sock = MagicMock()
        trigger_sock.recvfrom.side_effect = recvfrom
        camera, detector, preview = Mock(), Mock(), Mock()
        camera.read.side_effect = read
        detector.face_detector.detect.side_effect = lambda _frame: script[index["i"]]
        args = qnx_ai_service.parse_args(["--trigger-udp-port", "8767", "--camera-on-turret", "--no-search",
                                          *argv])
        with (
            patch.dict(sys.modules, {"cv2": SimpleNamespace(error=RuntimeError)}),
            patch.object(qnx_ai_service, "draw_aim_boxes"),
            patch.object(body_detector, "draw_head_boxes") as draw_heads,
            patch.object(qnx_ai_service, "HeadEstimator", return_value=estimator),
            patch.object(qnx_ai_service.LaptopTrigger, "bind", return_value=LaptopTrigger(trigger_sock)),
            patch.object(PanTracker, "update", update),
        ):
            self.assertEqual(qnx_ai_service.run_laptop_trigger(args, camera, detector, preview, None), 0)
        statuses = [c.kwargs["extra"] | {"active": c.args[0], "face_count": c.kwargs.get("face_count")}
                    for c in preview.state.update_status.call_args_list]
        return statuses, pan_inputs, draw_heads

    def test_head_fallback_aims_at_the_peepers_head_beside_the_users_face(self) -> None:
        user = face(0.3)
        est = FakeEstimator([head(0.7, y=0.3, kind="body"), head(0.3, y=0.5, kind="upper")])  # 2nd: user
        statuses, pan_inputs, draw_heads = self.run_loop(["--head-fallback"], [[user]] * 6, est)
        self.assertEqual(est.calls, 2)  # frames 1 and 4
        self.assertEqual([s["aim_source"] for s in statuses], [None, "body", "body", "body", "body", "body"])
        self.assertEqual([s["head_candidates"] for s in statuses], [1] * 6)  # the user's head is deduped
        self.assertEqual(statuses[0]["face_count"], 1)  # face_count stays faces only
        self.assertAlmostEqual(statuses[1]["aim_target"][0], 0.7)
        self.assertAlmostEqual(statuses[1]["aim_target"][1], 0.3)
        self.assertAlmostEqual(pan_inputs[1], 0.7)
        self.assertTrue(draw_heads.called)

    def test_off_by_default_status_still_has_aim_source(self) -> None:
        est = FakeEstimator([head(0.7)])
        statuses, _, draw_heads = self.run_loop([], [[]] * 3, est)
        self.assertEqual(est.calls, 0)
        self.assertNotIn("head_candidates", statuses[0])
        self.assertEqual({s["aim_source"] for s in statuses}, {None})
        self.assertFalse(draw_heads.called)


if __name__ == "__main__":
    unittest.main()
