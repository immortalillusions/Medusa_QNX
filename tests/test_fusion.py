"""Opt-in --fusion: distance-band handoff map and the bearing fusion filter."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import qnx_ai_service
from medusa.fusion import (
    FACE_VAR, HEAD_VAR, MAP_VAR, BandedHandoffLearner, BearingFusion, distance_band, map_variance,
)
from medusa.handoff import HandoffLearner, HandoffMap
from medusa.tflite_detector import FaceBox
from medusa.trigger import LaptopTrigger, PanTracker

FPS = 15
DT = 1 / FPS
LIVE = {"faces": 1, "match": "single", "armed": True, "gate_closed": True}
SETUP = {"pan_home": 90.0, "pan_min": 55.0, "pan_max": 125.0, "servo_reversed": False, "hfov_deg": 50.0}
LAPTOP = ("172.20.10.4", 50000)


class BandTests(unittest.TestCase):
    def test_band_thresholds(self) -> None:
        for size, band in ((0.2, "near"), (0.101, "near"), (0.10, "mid"), (0.07, "mid"), (0.04, "mid"),
                           (0.039, "far"), (0.01, "far"), (None, None), (0.0, None), (float("nan"), None)):
            with self.subTest(size=size):
                self.assertEqual(distance_band(size), band)

    def test_map_variance_grows_past_the_learned_points(self) -> None:
        domain = (0.3, 0.7)
        self.assertEqual(map_variance(0.5, domain), MAP_VAR)
        self.assertEqual(map_variance(0.3, domain), MAP_VAR)
        self.assertLess(MAP_VAR, map_variance(0.75, domain))
        self.assertLess(map_variance(0.75, domain), map_variance(0.85, domain))
        self.assertAlmostEqual(map_variance(0.2, domain), map_variance(0.8, domain))


class MapFileTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "handoff.json"

    def write(self, data) -> None:
        self.path.write_text(json.dumps(data))

    def test_old_files_still_load(self) -> None:
        self.write({"points": [[0.3, 100], [0.6, 80]], "fingerprint": SETUP})
        old = HandoffMap.load(self.path)
        self.assertEqual((old.points, old.fingerprint, old.bands), ([(0.3, 100), (0.6, 80)], SETUP, {}))
        learner = BandedHandoffLearner(self.path, minimum=55, maximum=125, fingerprint=SETUP)
        self.assertEqual(learner.learned_points, 2)
        self.assertIs(learner.map_for(0.2), learner.current())  # no band points: the single map
        self.write([[0.3, 100], [0.6, 80]])  # the oldest, bare-list format
        self.assertEqual(HandoffMap.load(self.path).points, [(0.3, 100), (0.6, 80)])

    def test_a_map_without_bands_is_saved_exactly_as_before(self) -> None:
        HandoffMap([[0.3, 100], [0.6, 80]], fingerprint=SETUP).save(self.path)
        self.assertEqual(set(json.loads(self.path.read_text())), {"points", "fingerprint"})

    def test_bands_round_trip_and_the_default_learner_keeps_them(self) -> None:
        bands = {"near": [[0.3, 104], [0.6, 76]]}
        self.write({"points": [[0.3, 100], [0.6, 80]], "fingerprint": SETUP, "bands": bands})
        self.assertEqual(HandoffMap.load(self.path).bands, {"near": [(0.3, 104), (0.6, 76)]})
        learner = HandoffLearner(self.path, minimum=55, maximum=125, fingerprint=SETUP)  # --fusion off
        self.assertEqual([p for p in learner.current().points], [(0.3, 100), (0.6, 80)])
        learner.observe(0.0, 0.9, 0.0, 70.0, **LIVE)
        for k in range(1, 10):
            learner.observe(k * DT, 0.9, 0.0, 70.0, **LIVE)
        self.assertTrue(learner.maybe_save(1.0, force=True))
        self.assertEqual(json.loads(self.path.read_text())["bands"], bands)

    def test_malformed_bands_make_the_file_unreadable_like_bad_points(self) -> None:
        self.write({"points": [[0.3, 100], [0.6, 80]], "bands": {"near": [[0.3]]}})
        with self.assertRaises(ValueError):
            HandoffMap.load(self.path)

    def test_another_setups_bands_are_discarded_with_its_points(self) -> None:
        self.write({"points": [[0.3, 100], [0.6, 80]], "fingerprint": dict(SETUP, pan_home=80.0),
                    "bands": {"near": [[0.3, 104], [0.6, 76]]}})
        learner = BandedHandoffLearner(self.path, minimum=55, maximum=125, fingerprint=SETUP)
        self.assertEqual((learner.learned_points, learner.band_points_count("near")), (0, 0))
        self.assertEqual(learner.band_for(0.2), "all")


class BandedLearnerTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "handoff.json"
        self.t = 0.0

    def learner(self) -> BandedHandoffLearner:
        return BandedHandoffLearner(self.path, minimum=55, maximum=125, fingerprint=SETUP)

    def teach(self, learner, laptop_x, pan, size, frames=10) -> None:
        for _ in range(frames):
            learner.observe(self.t, laptop_x, 0.0, pan, target_size=size, **LIVE)
            self.t += DT
        learner.observe(self.t, None, None, None, **LIVE)  # end the steady run
        self.t += 1.0

    def test_each_band_learns_its_own_map_and_falls_back_below_two_points(self) -> None:
        learner = self.learner()
        self.teach(learner, 0.3, 104, 0.15)  # near
        self.teach(learner, 0.6, 76, 0.15)
        self.teach(learner, 0.32, 96, 0.02)  # far
        self.teach(learner, 0.62, 84, 0.02)
        self.teach(learner, 0.45, 90, 0.07)  # mid: one point only
        self.assertEqual((learner.band_for(0.15), learner.band_for(0.02), learner.band_for(0.07),
                          learner.band_for(None)), ("near", "far", "all", "all"))
        self.assertAlmostEqual(learner.map_for(0.15).angle(0.3), 104)
        self.assertAlmostEqual(learner.map_for(0.02).angle(0.32), 96)
        self.assertIs(learner.map_for(0.07), learner.current())  # mid falls back to the single map
        self.assertEqual(learner.learned_points, 3)  # the single map still learns every point (shared bins)
        self.assertNotAlmostEqual(learner.map_for(0.15).angle(0.3), learner.map_for(0.02).angle(0.3))

    def test_band_points_persist_in_the_same_file(self) -> None:
        learner = self.learner()
        self.teach(learner, 0.3, 104, 0.15)
        self.teach(learner, 0.6, 76, 0.15)
        self.assertTrue(learner.maybe_save(self.t, force=True))
        data = json.loads(self.path.read_text())
        self.assertEqual((data["fingerprint"], len(data["points"]), len(data["bands"]["near"])), (SETUP, 2, 2))
        again = self.learner()
        self.assertEqual(again.band_for(0.15), "near")
        self.assertAlmostEqual(again.map_for(0.15).angle(0.6), 76)

    def test_head_aims_do_not_teach_the_map(self) -> None:
        learner = self.learner()
        for k in range(10):  # the service passes match "none" for a head aim
            learner.observe(k * DT, 0.3, 0.0, 100.0, target_size=0.15, **dict(LIVE, match="none"))
        self.assertEqual((learner.learned_points, learner.band_points_count("near")), (0, 0))


class BearingFusionTests(unittest.TestCase):
    def test_converges_to_the_turret_measurement_when_present(self) -> None:
        fusion = BearingFusion()
        for k in range(45):
            fusion.step(k * DT, (100.0, FACE_VAR), (90.0, MAP_VAR))
        self.assertAlmostEqual(fusion.bearing, 100.0, delta=0.5)  # the map is 25x rougher
        self.assertEqual(fusion.sources, ["face", "map"])
        self.assertLess(fusion.var, FACE_VAR)

    def test_converges_to_the_map_when_the_turret_loses_it_without_jumps(self) -> None:
        fusion = BearingFusion()
        for k in range(45):
            fusion.step(k * DT, (100.0, FACE_VAR), (90.0, MAP_VAR))
        trail = [fusion.bearing]
        for k in range(45, 45 + 3 * FPS):
            fusion.step(k * DT, None, (90.0, MAP_VAR))
            trail.append(fusion.bearing)
        self.assertEqual(fusion.sources, ["map"])
        self.assertAlmostEqual(trail[-1], 90.0, delta=0.2)
        self.assertLess(max(abs(b - a) for a, b in zip(trail, trail[1:])), 2.5)  # a glide, not a jump
        self.assertGreater(min(trail), 87.5)  # little overshoot

    def test_a_head_counts_less_than_a_face(self) -> None:
        face, head = BearingFusion(), BearingFusion()
        for k in range(45):
            face.step(k * DT, (100.0, FACE_VAR), (90.0, MAP_VAR))
            head.step(k * DT, (100.0, HEAD_VAR), (90.0, MAP_VAR), turret_source="head")
        self.assertLess(head.bearing, face.bearing)
        self.assertGreater(head.var, face.var)
        self.assertEqual(head.sources, ["head", "map"])

    def test_variance_grows_when_the_map_extrapolates(self) -> None:
        inside, outside = BearingFusion(), BearingFusion()
        domain = (0.3, 0.7)
        for k in range(60):
            inside.step(k * DT, None, (90.0, map_variance(0.5, domain)), start=90.0)
            outside.step(k * DT, None, (90.0, map_variance(0.85, domain)), start=90.0)
        self.assertGreater(outside.var, inside.var)

    def test_variance_grows_with_no_measurement(self) -> None:
        fusion = BearingFusion()
        fusion.step(0.0, (100.0, FACE_VAR))
        before = fusion.var
        fusion.step(0.5)
        self.assertGreater(fusion.var, before)

    def test_a_far_off_map_is_ignored_while_the_turret_sees_the_face(self) -> None:
        fusion = BearingFusion()
        for k in range(30):
            fusion.step(k * DT, (100.0, FACE_VAR), (60.0, MAP_VAR))
        self.assertAlmostEqual(fusion.bearing, 100.0, delta=0.3)
        self.assertEqual(fusion.sources, ["face"])

    def test_a_turret_jump_restarts_on_the_new_face(self) -> None:
        fusion = BearingFusion()
        for k in range(10):
            fusion.step(k * DT, (100.0, FACE_VAR))
        fusion.step(10 * DT, (60.0, FACE_VAR))
        self.assertEqual((fusion.bearing, fusion.rate), (60.0, 0.0))

    def test_nothing_to_fuse_is_none(self) -> None:
        self.assertIsNone(BearingFusion().step(0.0))


class PanTrackerFusionTests(unittest.TestCase):
    def tracker(self, **kwargs) -> PanTracker:
        return PanTracker(55, 125, 90, fusion=BearingFusion(), **kwargs)

    def test_map_only_glides_from_where_the_turret_points(self) -> None:
        for mode in ("proportional", "predictive"):
            with self.subTest(mode=mode):
                pan = self.tracker(mode=mode, fallback_search_after_s=10.0)
                commands = [pan.command]
                for k in range(2 * FPS):
                    pan.update(None, k * DT, laptop_active=True, laptop_x=0.5, map_bearing=110.0, map_var=MAP_VAR)
                    commands.append(pan.command)
                self.assertEqual(pan.state, "fallback")
                self.assertAlmostEqual(commands[-1], 110.0, delta=1.0)
                self.assertLess(max(abs(b - a) for a, b in zip(commands, commands[1:])), 6.0)

    def test_tracks_the_face_and_hands_over_to_the_map(self) -> None:
        pan = self.tracker(mode="predictive", fallback_search_after_s=10.0)
        t = 0.0
        for _ in range(3 * FPS):  # a still face at bearing 100 (the turret starts at 90), the map 2 deg off it
            seen = 0.5 + (100.0 - pan.servo.position_at(t - pan.latency_s)) / pan.hfov_deg
            pan.update(seen, t, laptop_active=True, laptop_x=0.5, map_bearing=98.0, map_var=MAP_VAR)
            t += DT
        self.assertEqual((pan.state, pan.fusion.sources), ("tracking", ["face", "map"]))
        self.assertAlmostEqual(pan.command, 100.0, delta=1.0)
        commands = [pan.command]
        for _ in range(3 * FPS):  # the face is lost; the laptop still sees the peeper
            pan.update(None, t, laptop_active=True, laptop_x=0.5, map_bearing=98.0, map_var=MAP_VAR)
            t += DT
            commands.append(pan.command)
        self.assertEqual(pan.state, "fallback")
        self.assertAlmostEqual(commands[-1], 98.0, delta=1.0)
        self.assertLessEqual(max(abs(b - a) for a, b in zip(commands, commands[1:])), 2.0)  # whole degrees

    def test_laptop_gone_hands_back_to_the_default_hold_and_return(self) -> None:
        pan = self.tracker(lost_hold_s=0.2, fallback_search_after_s=10.0)
        for k in range(FPS):
            pan.update(None, k * DT, laptop_active=True, laptop_x=0.5, map_bearing=100.0, map_var=MAP_VAR)
        pan.update(None, 1.0 + DT, laptop_active=False)
        self.assertEqual(pan.state, "holding")
        self.assertIsNone(pan.fusion.bearing)

    def test_without_fusion_the_fusion_inputs_change_nothing(self) -> None:
        plain, fed = PanTracker(55, 125, 90), PanTracker(55, 125, 90)
        for k in range(3 * FPS):
            x = 0.6 if k < FPS else None
            plain.update(x, k * DT, laptop_active=True, laptop_x=0.5)
            fed.update(x, k * DT, laptop_active=True, laptop_x=0.5, aim_source="head", map_bearing=120.0,
                       map_var=1.0)
            self.assertEqual((plain.command, plain.state), (fed.command, fed.state))

    def test_disarmed_resets_the_fusion(self) -> None:
        pan = self.tracker()
        pan.update(0.6, 0.0, laptop_active=True, map_bearing=98.0)
        pan.observe_actual(90.0, 0.05, armed=False)
        pan.update(0.6, 0.1)
        self.assertIsNone(pan.fusion.bearing)
        self.assertEqual(pan.state, "home")


def _face(x: float) -> FaceBox:
    return FaceBox(x * 640 - 30, 200, x * 640 + 30, 260, 0.9)


class ServiceFlagTests(unittest.TestCase):
    def args(self, *extra):
        return qnx_ai_service.parse_args(["--trigger-udp-port", "8767", "--camera-on-turret", *extra])

    def test_off_by_default_and_needs_the_turret_camera(self) -> None:
        self.assertFalse(self.args().fusion)
        self.assertIsNone(qnx_ai_service._pan_tracker(self.args()).fusion)
        self.assertIsInstance(qnx_ai_service._pan_tracker(self.args("--fusion")).fusion, BearingFusion)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            qnx_ai_service.parse_args(["--trigger-udp-port", "8767", "--fusion"])

    def run_loop(self, argv, script, path):
        frame = MagicMock()
        frame.shape = (480, 640, 3)
        index, pending, learners = {"i": -1}, [], []

        def read():
            index["i"] += 1
            if index["i"] == len(script):
                raise KeyboardInterrupt
            pending.append((json.dumps({"type": "medusa-peeper", "active": True, "target_x": 0.5,
                                        "target_size": 0.15}).encode(), LAPTOP))
            return True, frame

        def recvfrom(_size):
            if pending:
                return pending.pop(0)
            raise BlockingIOError

        trigger_sock = MagicMock()
        trigger_sock.recvfrom.side_effect = recvfrom
        camera, detector, preview = Mock(), Mock(), Mock()
        camera.read.side_effect = read
        detector.face_detector.detect.side_effect = lambda _frame: script[index["i"]]
        args = self.args("--handoff-map", str(path), "--face-confirm-frames", "1", *argv)
        with (
            patch.dict(sys.modules, {"cv2": SimpleNamespace(error=RuntimeError)}),
            patch.object(qnx_ai_service, "draw_aim_boxes"),
            patch.object(qnx_ai_service.LaptopTrigger, "bind", return_value=LaptopTrigger(trigger_sock)),
        ):
            self.assertEqual(qnx_ai_service.run_laptop_trigger(args, camera, detector, preview, None), 0)
        return [c.kwargs["extra"] for c in preview.state.update_status.call_args_list]

    def test_status_reports_fusion_only_when_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "handoff.json"
            HandoffMap([[0.3, 100], [0.7, 80]], fingerprint=SETUP | {"pan_min": 65.0, "pan_max": 115.0}).save(path)
            script = [[_face(0.6)]] * 4 + [[]] * 4
            off = self.run_loop([], script, path)
            self.assertEqual({s["fusion"]["enabled"] for s in off}, {False})
            on = self.run_loop(["--fusion"], script, path)
        self.assertEqual({s["fusion"]["enabled"] for s in on}, {True})
        self.assertEqual(on[3]["fusion"]["sources"], ["face", "map"])
        self.assertEqual(on[3]["fusion"]["band"], "all")  # no near-band points yet: the single map
        self.assertEqual(on[-1]["fusion"]["sources"], ["map"])
        self.assertEqual(on[-1]["pan_state"], "fallback")
        self.assertIsNotNone(on[-1]["fusion"]["bearing"])
        self.assertIsNotNone(on[-1]["fusion"]["var"])


if __name__ == "__main__":
    unittest.main()
