"""Off means off (detached servos, kill gate) and the learned map's safeguards."""

import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock

import qnx_ai_service
from medusa.control import ControlLoop, parse_actual_packet, parse_status_line
from medusa.handoff import HandoffLearner, HandoffMap, main as handoff_main, monotonic_fit, setup_fingerprint
from medusa.trigger import LaptopTrigger, PanTracker
from test_control import INO, FakeClock, FakeSerial, FakeSock, lines, target

DT = 1 / 13.0
LIVE = {"faces": 1, "match": "single", "armed": True, "gate_closed": True}
# The bench map learned before the 45 deg re-seat: the right half rises, which is wrong.
REAL_MAP = [[0.0986, 76.4], [0.1102, 76.4], [0.1692, 62.2], [0.2172, 56.1], [0.266, 64.5], [0.32, 53.7],
            [0.4263, 30.6], [0.8383, 43.0], [0.9398, 56.4], [0.9609, 56.2]]
SETUP = setup_fingerprint(45, 10, 80, False, 50.0)


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    depth, i = 0, source.index("{", start)
    for j in range(i, len(source)):
        depth += {"{": 1, "}": -1}.get(source[j], 0)
        if depth == 0:
            return source[i:j + 1]
    raise AssertionError(f"unbalanced {signature}")


class StatusLineGateTests(unittest.TestCase):
    def test_five_and_six_field_lines(self) -> None:
        self.assertIsNone(parse_status_line("S,1,450,900,7").gate_closed)  # old firmware: unknown
        closed, opened = parse_status_line("S,1,450,900,8,1\r"), parse_status_line("S,0,-12,900,9,0")
        self.assertEqual((closed.armed, closed.pan, closed.seq, closed.gate_closed), (True, 45.0, 8, True))
        self.assertEqual((opened.armed, opened.pan, opened.gate_closed), (False, -1.2, False))
        for line in ("S,1,450,900,8,2", "S,1,450,900,8,", "S,1,450,900,8,1,1"):
            with self.subTest(line=line):
                self.assertIsNone(parse_status_line(line))


class ControlGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial, self.sock, self.actual = FakeSerial(), FakeSock(), FakeSock()
        self.loop = ControlLoop(self.serial, target_sock=self.sock, actual_sock=self.actual,
                                actual_addr=("127.0.0.1", 8769), clock=FakeClock().monotonic)

    def test_gate_open_sends_no_motion_and_reports_safe(self) -> None:
        self.serial.incoming += [b"S,0,450,900,1,0\n"]
        self.sock.queue += [target(True, 70, 95, 1)]
        self.loop.tick(1.0)  # a fresh AI target, but the kill gate is open: heartbeat park line only
        snap = self.loop.snapshot(1.0)
        self.assertEqual((snap["state"], snap["safe_reason"]), ("SAFE", "kill gate open"))
        self.assertEqual((snap["arduino"]["gate_closed"], snap["arduino"]["armed"]), (False, False))
        self.assertIs(parse_actual_packet(self.actual.sent[-1][0]).gate_closed, False)
        # Gate closed but not armed: the old reason; the target goes out (the firmware ignores it).
        self.serial.incoming += [b"S,0,450,900,2,1\n"]
        self.sock.queue += [target(True, 70, 95, 2)]
        self.loop.tick(1.02)
        snap = self.loop.snapshot(1.02)
        self.assertEqual((snap["safe_reason"], snap["arduino"]["gate_closed"]), ("turret disarmed", True))
        self.serial.incoming += [b"S,1,450,900,3,1\n"]
        self.sock.queue += [target(True, 70, 95, 3)]
        self.loop.tick(1.04)
        self.assertEqual(self.loop.snapshot(1.04)["state"], "ACTIVE")
        self.assertEqual(lines(self.serial), ["M,90,90,0,1", "N,700,950,1,2", "N,700,950,1,3"])
        self.assertIs(parse_actual_packet(self.actual.sent[-1][0]).gate_closed, True)
        # A lost link makes the gate unknown, not open.
        snap = self.loop.snapshot(2.0)
        self.assertEqual((snap["safe_reason"], snap["arduino"]["gate_closed"]), ("Arduino link lost", None))


class TrackerGateTests(unittest.TestCase):
    def test_gate_open_holds_like_disarmed(self) -> None:
        pan = PanTracker(10, 80, 45)
        pan.update(0.9, now=0.0)
        self.assertGreater(pan.angle, 55)
        pan.observe_actual(45.0, at=0.1, armed=True, gate_closed=False)
        for k in range(1, 4):
            self.assertIsNone(pan.update(0.9, now=0.1 * k, laptop_active=True, laptop_x=0.5,
                                         fallback_map=lambda _x: 70.0))
            self.assertEqual((pan.state, pan.angle), ("home", 45.0))  # no fallback, no wind-up
            self.assertIsNone(pan.real_pan())
        pan.observe_actual(45.0, at=0.4, armed=True, gate_closed=True)
        pan.update(0.5, now=0.4)
        self.assertEqual((pan.state, pan.real_pan()), ("tracking", 45.0))


class FirmwareOffMeansOffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = INO.read_text()

    def test_disarm_detaches_and_never_eases_home(self) -> None:
        source = self.source
        self.assertNotIn("parkSafely", source)
        self.assertIsNone(re.search(r"target\s*=\s*\w+\.home", source))  # nothing ever targets home
        disarm = function_body(source, "void setArmed(")
        self.assertIn("detachAxis(pan);", disarm)
        self.assertIn("detachAxis(tilt);", disarm)
        detach = function_body(source, "void detachAxis(")
        self.assertIn("axis.servo.detach();", detach)
        self.assertIn("axis.target = axis.position;", detach)
        self.assertIn("digitalWrite(axis.pin, LOW);", detach)
        self.assertIn("setArmed(false);", function_body(source, "void setup("))
        self.assertNotIn(".attach(", function_body(source, "void setup("))  # boot: no pulses

    def test_only_an_attached_axis_gets_pulses_and_arming_starts_at_the_estimate(self) -> None:
        step = function_body(self.source, "void stepAxis(")
        self.assertLess(step.index("if (!axis.servo.attached()) return;"), step.index("writeMicroseconds"))
        attach = function_body(self.source, "void attachAxis(")
        self.assertIn("axis.target = axis.position;", attach)
        self.assertLess(attach.index("attach(axis.pin)"),
                        attach.index("writeMicroseconds(degreesToMicroseconds(axis.position))"))
        self.assertEqual(self.source.count("writeMicroseconds("), 2)  # stepAxis and attachAxis only
        self.assertEqual(self.source.count(".attach("), 1)

    def test_kill_gate_refuses_arming_and_forces_disarm(self) -> None:
        source = self.source
        self.assertIn("const byte KILL_SENSE_PIN = A2;", source)
        self.assertIn("if (!gateClosed) shouldArm = false;", function_body(source, "void setArmed("))
        gate = function_body(source, "void updateKillGate(")
        self.assertIn("analogRead(KILL_SENSE_PIN)", gate)
        self.assertIn("setArmed(false)", gate)
        self.assertIn("updateKillGate();", function_body(source, "void loop("))
        # Thresholds against the core's default 10-bit read of 0..3.3 V: tab in ~1.67 V, tab out 0 V.
        self.assertIsNone(re.search(r"analogReadResolution\(\s*\w", source))  # never changed from 10-bit
        closed = int(re.search(r"GATE_CLOSED_ABOVE = (\d+);", source).group(1))
        opened = int(re.search(r"GATE_OPEN_BELOW = (\d+);", source).group(1))
        counts = lambda volts: volts / 3.3 * 1023
        self.assertTrue(counts(0.05) < opened < closed < counts(1.67) * 0.8)
        self.assertGreaterEqual(opened, round(counts(0.8)))


class LearnerGatingTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "handoff.json"

    def feed(self, learner, start, **flags):
        return [learner.observe(start + k * DT, 0.42, 0.0, 60.0, **{**LIVE, **flags}) for k in range(10)]

    def test_learns_only_from_one_matched_face_on_an_armed_turret_with_the_gate_closed(self) -> None:
        learner = HandoffLearner(self.path, minimum=10, maximum=80, fingerprint=SETUP)
        blocked = [{"faces": 2}, {"faces": 0}, {"faces": None}, {"match": "order"}, {"match": "smallest"},
                   {"match": "none"}, {"armed": False}, {"armed": None}, {"gate_closed": False},
                   {"gate_closed": None}]
        for k, flags in enumerate(blocked):
            with self.subTest(flags=flags):
                self.assertFalse(any(self.feed(learner, 2.0 * k, **flags)))
        self.assertEqual(learner.learned_points, 0)
        self.assertTrue(any(self.feed(learner, 30.0, match="map")))
        self.assertEqual(learner.learned_points, 1)


class MonotonicFitTests(unittest.TestCase):
    def assertMonotonic(self, points, decreasing) -> None:
        pans = [pan for _x, pan in sorted(points)]
        pairs = list(zip(pans, pans[1:]))
        self.assertTrue(all(b <= a for a, b in pairs) if decreasing else all(b >= a for a, b in pairs), pans)

    def test_injected_outlier_is_rejected_and_its_neighbours_kept(self) -> None:
        for outlier in (30.0, 110.0):  # 40 deg low or high at x 0.5 (true pan 70)
            with self.subTest(outlier=outlier):
                points = [(round(0.1 * k, 1), 20.0 + 10 * k) for k in range(1, 9)]
                points[4] = (0.5, outlier)
                fitted, direction, rejected = monotonic_fit(points)
                self.assertEqual((direction, rejected), ("increasing", [4]))
                self.assertEqual(fitted, [p for k, p in enumerate(points) if k != 4])
                self.assertMonotonic(fitted, decreasing=False)

    def test_the_real_map_comes_out_decreasing_with_the_rising_right_end_rejected(self) -> None:
        fitted, direction, rejected = monotonic_fit(REAL_MAP)
        self.assertEqual(direction, "decreasing")
        self.assertMonotonic(fitted, decreasing=True)
        rejected_x = [REAL_MAP[i][0] for i in rejected]
        self.assertEqual(rejected_x, [0.4263, 0.9398, 0.9609])  # 0.8383 (43.0) stays: within 8 deg of the fit
        handoff = HandoffMap(fitted, extrapolate=True, bounds=(10, 80))
        grid = [handoff.angle(k / 50) for k in range(51)]
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(grid, grid[1:])))
        self.assertTrue(all(10 <= a <= 80 for a in grid))

    def test_direction_needs_three_points(self) -> None:
        self.assertEqual(monotonic_fit([(0.2, 70), (0.8, 30)]), ([(0.2, 70.0), (0.8, 30.0)], None, []))
        self.assertEqual(monotonic_fit([(0.2, 70), (0.5, 50), (0.8, 30)])[1], "decreasing")

    def test_learner_reports_direction_and_rejections_in_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "handoff.json"
            HandoffMap(REAL_MAP, fingerprint=SETUP).save(path)
            learner = HandoffLearner(path, minimum=10, maximum=80, fingerprint=SETUP)
            self.assertEqual((learner.learned_points, learner.direction, learner.rejected_points),
                             (10, "decreasing", 3))
            self.assertGreater(learner.current().angle(0.2), learner.current().angle(0.9))
            # A fresh observation in a rejected bin replaces it instead of averaging into it.
            index = learner._bin(0.95)
            self.assertIn(index, learner.rejected)
            for k in range(10):
                learner.observe(k * DT, 0.95, 0.0, 12.0, **LIVE)
            self.assertEqual(learner.bins[index], (0.95, 12.0))
            pan = PanTracker(10, 80, 45)
            status = qnx_ai_service._trigger_status(LaptopTrigger(Mock()), True, None, pan, None,
                                                    learned_points=learner.learned_points, learner=learner)
            self.assertEqual((status["learned_points"], status["map_direction"], status["map_fingerprint"]),
                             (10, "decreasing", SETUP))
            self.assertIsInstance(status["map_rejected"], int)
            status = qnx_ai_service._trigger_status(LaptopTrigger(Mock()), True, None, pan, None)
            self.assertEqual((status["map_direction"], status["map_rejected"]), (None, None))


class FingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.path = self.dir / "handoff.json"

    def learner(self):
        err = io.StringIO()
        with redirect_stderr(err):
            learner = HandoffLearner(self.path, minimum=10, maximum=80, fingerprint=SETUP)
        return learner, err.getvalue()

    def test_matching_fingerprint_is_used(self) -> None:
        HandoffMap(REAL_MAP[:3], fingerprint=dict(SETUP)).save(self.path)
        learner, err = self.learner()
        self.assertEqual((learner.learned_points, err), (3, ""))

    def test_other_setup_or_no_fingerprint_is_discarded_logged_once_and_backed_up_before_overwrite(self) -> None:
        for fingerprint in (setup_fingerprint(90, 65, 115, False, 50.0), setup_fingerprint(45, 10, 80, True, 50.0),
                            None):
            with self.subTest(fingerprint=fingerprint):
                for old in self.dir.glob("handoff.backup-*"):
                    old.unlink()
                HandoffMap(REAL_MAP, fingerprint=fingerprint).save(self.path)
                original = self.path.read_text()
                learner, err = self.learner()
                self.assertEqual(learner.learned_points, 0)
                self.assertFalse(learner.current().usable)
                self.assertEqual(err.count("discarded"), 1)
                with redirect_stderr(io.StringIO()) as again:
                    learner.file._checked_at = None
                    learner.current()  # the same file again: no second warning
                self.assertEqual(again.getvalue(), "")
                for k in range(10):
                    learner.observe(k * DT, 0.3, 0.0, 50.0, **LIVE)
                with redirect_stderr(io.StringIO()):
                    self.assertTrue(learner.maybe_save(10.0))
                backups = list(self.dir.glob("handoff.backup-*.json"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_text(), original)
                saved = HandoffMap.load(self.path)
                self.assertEqual((saved.fingerprint, saved.points), (SETUP, [(0.3, 50.0)]))


class ClearBackupTests(unittest.TestCase):
    def run_cli(self, path, *argv):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(out):
            code = handoff_main(["--map", str(path), *argv])
        return code, out.getvalue()

    def test_clear_backs_up_first_and_keeps_the_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "handoff.json"
            self.assertEqual(self.run_cli(path, "clear"), (0, f"no file to back up\ncleared {path}\n"))
            HandoffMap(REAL_MAP, fingerprint=SETUP).save(path)
            original = path.read_text()
            code, out = self.run_cli(path, "clear")
            backups = sorted(Path(tmp).glob("handoff.backup-*.json"))
            self.assertEqual((code, len(backups)), (0, 1))
            self.assertIn(f"backed up to {backups[0]}", out)
            self.assertEqual(backups[0].read_text(), original)
            cleared = HandoffMap.load(path)
            self.assertEqual((cleared.points, cleared.fingerprint), ([], SETUP))
            self.assertEqual(self.run_cli(path, "clear")[0], 0)  # same second: a second, distinct backup
            self.assertEqual(len(list(Path(tmp).glob("handoff.backup-*.json"))), 2)
            path.write_text("{not json")  # an unreadable file is backed up as it is, then cleared
            self.assertEqual(self.run_cli(path, "clear")[0], 0)
            self.assertIn("{not json", [b.read_text() for b in Path(tmp).glob("handoff.backup-*.json")])
            self.assertEqual(HandoffMap.load(path).points, [])

    def test_record_now_gives_a_fresh_file_the_running_setup(self) -> None:
        status = {"laptop_active": True, "laptop_target_x": 0.3, "pan_state": "tracking", "pan_angle": 50.0,
                  "map_fingerprint": SETUP}
        pages = lambda url: status if "8080" in url else {"arduino": {"link_ok": True, "pan": 51.0}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "handoff.json"
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(out):
                self.assertEqual(handoff_main(["--map", str(path), "record-now"], fetch=pages), 0)
            self.assertNotIn("warning", out.getvalue())
            self.assertEqual(json.loads(path.read_text())["fingerprint"], SETUP)
            # A manual record into a file without a fingerprint warns that the service will discard it.
            bare = Path(tmp) / "bare.json"
            code, out = ClearBackupTests.run_cli(self, bare, "record", "0.5", "45")
            self.assertEqual(code, 0)
            self.assertIn("warning: this file has no setup fingerprint", out)


if __name__ == "__main__":
    unittest.main()
