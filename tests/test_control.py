import io
import json
import socket
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import medusa_control
import qnx_ai_service
from medusa import trigger as trigger_module
from medusa.control import (
    ControlClient, ControlHTTPServer, ControlLoop, StatusReader, StressTest, TickStats, format_n_line,
    make_target_packet, parse_actual_packet, parse_command_line, parse_status_line, parse_target_packet,
    system_clock,
)
from medusa.handoff import HandoffMap, HandoffMapFile, main as handoff_main, record_now, setup_fingerprint
from medusa.servo import ServoMapper
from medusa.tflite_detector import FaceBox
from medusa.trigger import LaptopTrigger, PanTracker

INO = Path(__file__).resolve().parent.parent / "arduino" / "medusa_controller" / "medusa_controller.ino"
LAPTOP = ("172.20.10.4", 50000)
NO_FRAME, RAISE = object(), object()  # ControlModeServiceTests script markers


class FakeSerial:
    def __init__(self) -> None:
        self.incoming: list[bytes] = []
        self.written: list[bytes] = []

    def read(self) -> bytes:
        return self.incoming.pop(0) if self.incoming else b""

    def write(self, data: bytes) -> bool:
        self.written.append(data)
        return True


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now


class FakeSock:
    """A non-blocking UDP socket stand-in: recvfrom pops queued payloads, sendto records."""

    def __init__(self) -> None:
        self.queue: list[bytes] = []
        self.sent: list[tuple[bytes, tuple]] = []

    def recvfrom(self, _size):
        if self.queue:
            return self.queue.pop(0), ("127.0.0.1", 1)
        raise BlockingIOError

    def sendto(self, payload, address):
        self.sent.append((payload, address))


def target(active=True, yaw=80, pitch=90, seq=1, t=0.0) -> bytes:
    return make_target_packet(active, yaw, pitch, seq, t)


def lines(serial: FakeSerial) -> list[str]:
    return [line.decode("ascii").strip() for line in serial.written]


class StatusParsingTests(unittest.TestCase):
    def test_valid_line(self) -> None:
        status = parse_status_line("S,1,1023,900,17\r")
        self.assertEqual((status.armed, status.pan, status.tilt, status.seq), (True, 102.3, 90.0, 17))
        self.assertFalse(parse_status_line("S,0,650,1150,1").armed)

    def test_garbage_and_boot_banner_are_ignored(self) -> None:
        for line in ("", "*** Booting Zephyr OS build v4.1.0 ***", "S,2,900,900,1", "S,1,abc,900,1",
                     "S,1,900,900", "S,1,900,900,1,5", "M,90,90,0,1", " S,1,900,900,1 x", "\x00\xff"):
            with self.subTest(line=line):
                self.assertIsNone(parse_status_line(line))

    def test_reader_handles_split_lines_banner_and_noise(self) -> None:
        reader = StatusReader()
        self.assertEqual(reader.feed(b"\x00\xff*** Booting Zephyr OS ***\r\nS,1,95"), [])
        statuses = reader.feed(b"0,900,2\nS,0,900,900,3\ngarb")
        self.assertEqual([(s.armed, s.pan, s.seq) for s in statuses], [(True, 95.0, 2), (False, 90.0, 3)])
        self.assertEqual(reader.feed(b"x" * 200), [])  # A newline-free flood is dropped, not buffered.
        self.assertEqual([s.seq for s in reader.feed(b"\nS,1,900,900,4\n")], [4])


class TargetPacketTests(unittest.TestCase):
    def test_round_trip_and_garbage(self) -> None:
        self.assertEqual(json.loads(target(True, 92.54, 90, 1))["yaw"], 92.54)  # Floats on the wire.
        self.assertEqual(parse_target_packet(b'{"type":"medusa-target","active":true,"yaw":92,"pitch":90}').yaw, 92)
        packet = parse_target_packet(target(True, 84.6, 90.2, 7, 12.5))
        self.assertEqual((packet.active, packet.yaw, packet.pitch, packet.seq, packet.t), (True, 84.6, 90.2, 7, 12.5))
        for payload in (b"", b"\xff", b"[]", b'{"type":"medusa-peeper","active":true,"yaw":1,"pitch":1}',
                        b'{"type":"medusa-target","active":1,"yaw":90,"pitch":90}',
                        b'{"type":"medusa-target","active":true,"yaw":"90","pitch":90}',
                        b'{"type":"medusa-target","active":true,"yaw":NaN,"pitch":90}'):
            with self.subTest(payload=payload):
                self.assertIsNone(parse_target_packet(payload))


class CommandLineTests(unittest.TestCase):
    def test_n_line_round_trip_in_tenths(self) -> None:
        self.assertEqual(format_n_line(92.5, 90.0, True, 7), b"N,925,900,1,7\n")
        self.assertEqual(format_n_line(84.46, 77.04, False, 8), b"N,845,770,0,8\n")
        for yaw, pitch, led, seq in ((92.5, 90.0, True, 7), (55.0, 125.0, False, 1), (101.3, 64.9, True, 99)):
            with self.subTest(yaw=yaw):
                line = format_n_line(yaw, pitch, led, seq).decode("ascii")
                self.assertEqual(parse_command_line(line, 55, 125), (yaw, pitch, led, seq))

    def test_m_lines_still_parse_and_bad_lines_are_rejected(self) -> None:
        self.assertEqual(parse_command_line("M,92,80,1,3", 55, 125), (92.0, 80.0, True, 3))
        self.assertEqual(parse_command_line("M,92,1,3", 55, 125), (92.0, 90.0, True, 3))  # Legacy 4-field.
        for line in ("N,925,1,3", "N,1260,900,1,3", "N,540,900,1,3", "N,925,900,2,3", "M,126,90,1,3",
                     "N,92.5,90,1,3", "X,925,900,1,3", ""):
            with self.subTest(line=line):
                self.assertIsNone(parse_command_line(line, 55, 125))

    def test_firmware_parses_n_lines_in_tenths(self) -> None:
        source = INO.read_text()
        self.assertIn('sscanf(command, "N,%d,%d,%d,%lu"', source)
        self.assertIn("value >= axis.minimum * scale && value <= axis.maximum * scale", source)
        self.assertIn("pan.target = (float)yaw / scale;", source)


class ControlTickTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial, self.sock, self.actual = FakeSerial(), FakeSock(), FakeSock()
        self.clock = FakeClock()
        self.loop = ControlLoop(self.serial, target_sock=self.sock, actual_sock=self.actual,
                                actual_addr=("127.0.0.1", 8769), clock=self.clock.monotonic)

    def test_parks_without_target_forwards_fresh_target_and_parks_when_stale(self) -> None:
        self.loop.tick(0.00)                                  # no target yet -> park
        self.sock.queue += [target(True, 70, 95, 1), target(True, 80, 90, 2)]
        self.loop.tick(0.02)                                  # latest wins
        self.loop.tick(0.30)                                  # 0.28 s old: still fresh
        self.loop.tick(0.32)                                  # 0.30 s old: stale -> park
        self.sock.queue += [target(False, 100, 90, 3)]
        self.loop.tick(0.34)                                  # fresh, LED off, still aimed
        self.sock.queue += [b"garbage", target(True, 200, -5, 4)]
        self.loop.tick(0.36)                                  # garbage ignored; angles clamped
        self.assertEqual(lines(self.serial), ["M,90,90,0,1", "N,800,900,1,2", "N,800,900,1,3", "M,90,90,0,4",
                                              "N,1000,900,0,5", "N,1150,650,1,6"])

    def test_float_targets_are_sent_as_tenths(self) -> None:
        self.sock.queue += [target(True, 92.5, 90.0, 1)]
        self.loop.tick(0.00)
        self.sock.queue += [b'{"type":"medusa-target","active":true,"yaw":88,"pitch":90,"seq":2}']  # ints too
        self.loop.tick(0.02)
        self.sock.queue += [target(True, 54.96, 125.04, 3)]
        self.loop.tick(0.04)  # still clamped to the safe ranges
        self.assertEqual(lines(self.serial), ["N,925,900,1,1", "N,880,900,1,2", "N,650,1150,1,3"])

    def test_heartbeat_never_waits_for_the_ai(self) -> None:
        for k in range(100):  # Two seconds with no AI at all: one park line every tick.
            self.loop.tick(k * 0.02)
        self.assertEqual(len(self.serial.written), 100)
        self.assertTrue(all(line.startswith("M,90,90,0,") for line in lines(self.serial)))

    def test_status_lines_latest_wins_and_age_is_tracked(self) -> None:
        self.serial.incoming += [b"*** Booting Zephyr ***\nS,0,900,900,1\nS,1,1000,900,2\n"]
        self.loop.tick(1.0)
        self.assertEqual((self.loop.status.armed, self.loop.status.pan, self.loop.status_at), (True, 100.0, 1.0))
        self.assertEqual(self.loop.snapshot(1.2)["arduino"],
                         {"link_ok": True, "armed": True, "parking": False, "gate_closed": None, "pan": 100.0,
                          "tilt": 90.0, "age_s": 0.2})  # A 5-field S line: the gate is unknown.
        self.assertEqual(self.loop.snapshot(1.6)["arduino"]["link_ok"], False)

    def test_each_tick_forwards_the_actual_angle(self) -> None:
        self.loop.tick(1.0)
        first = parse_actual_packet(self.actual.sent[0][0])
        self.assertEqual((first.link_ok, first.armed, first.pan, first.t), (False, None, None, None))
        self.serial.incoming += [b"S,1,1012,900,5\n"]
        self.loop.tick(1.02)
        actual = parse_actual_packet(self.actual.sent[1][0])
        self.assertEqual((actual.link_ok, actual.armed, actual.pan, actual.t), (True, True, 101.2, 1.02))
        self.assertEqual(self.actual.sent[1][1], ("127.0.0.1", 8769))

    def test_state_and_safe_reason(self) -> None:
        def state(now):
            snap = self.loop.snapshot(now)
            return snap["state"], snap["safe_reason"]

        self.assertEqual(state(0.0), ("SAFE", "no Arduino status yet"))
        self.serial.incoming += [b"S,0,900,900,1\n"]
        self.loop.tick(1.0)
        self.assertEqual(state(1.0), ("SAFE", "turret disarmed"))
        self.serial.incoming += [b"S,1,900,900,2\n"]
        self.loop.tick(1.02)
        self.assertEqual(state(1.02), ("SAFE", "no AI target"))
        self.sock.queue += [target(False)]
        self.loop.tick(1.04)
        self.assertEqual(state(1.04), ("SAFE", "AI has no face to aim at"))
        self.sock.queue += [target(True)]
        self.loop.tick(1.06)
        self.assertEqual(state(1.06), ("ACTIVE", ""))
        snap = self.loop.snapshot(1.06)
        self.assertEqual((snap["target_age_s"], snap["ai_alive"]), (0.0, True))
        self.serial.incoming += [b"S,1,900,900,3\n"]
        self.loop.tick(1.40)
        self.assertEqual(state(1.40), ("SAFE", "AI target stale"))
        self.assertEqual(state(2.00), ("SAFE", "Arduino link lost"))
        self.assertFalse(self.loop.snapshot(2.10)["ai_alive"])

    def test_shutdown_sends_park_lines(self) -> None:
        self.sock.queue += [target(True, 70)]
        self.loop.tick(0.0)
        self.loop.shutdown()
        self.assertEqual(lines(self.serial), ["N,700,900,1,1", "M,90,90,0,2", "M,90,90,0,3", "M,90,90,0,4"])


class TickStatsTests(unittest.TestCase):
    def test_mean_jitter_max_and_missed(self) -> None:
        stats = TickStats(0.02, window_s=1.0)
        self.assertEqual(stats.summary()["period_ms_mean"], None)
        for start, deadline in ((0.000, 0.00), (0.020, 0.02), (0.041, 0.04), (0.060, 0.06),
                                (0.090, 0.08), (0.100, 0.10)):
            stats.record(start, deadline)
        summary = stats.summary()
        self.assertAlmostEqual(summary["period_ms_mean"], 20.0, places=6)
        self.assertAlmostEqual(summary["jitter_ms_p99"], 10.0, places=6)   # |30 - 20| and |10 - 20|
        self.assertAlmostEqual(summary["period_ms_max"], 30.0, places=6)
        self.assertEqual(summary["missed_deadlines"], 1)                  # only 0.090 was > 5 ms late
        self.assertAlmostEqual(summary["loop_hz"], 50.0, places=2)

    def test_p99_ignores_one_outlier_in_a_hundred_but_not_two(self) -> None:
        stats = TickStats(0.02, window_s=10.0)
        start = 0.0
        for period in [0.020] * 99 + [0.040]:
            stats.record(start, None)
            start += period
        stats.record(start, None)
        self.assertAlmostEqual(stats.summary()["jitter_ms_p99"], 0.0, places=6)
        stats.record(start + 0.040, None)
        self.assertAlmostEqual(stats.summary()["jitter_ms_p99"], 20.0, places=6)

    def test_window_rolls_but_total_keeps_counting(self) -> None:
        stats = TickStats(0.02, window_s=0.1)  # five ticks
        stats.record(0.0, 0.0)
        stats.record(0.03, 0.02)  # 10 ms late
        for k in range(5):
            stats.record(0.04 + 0.02 * k, 0.04 + 0.02 * k)
        summary = stats.summary()
        self.assertEqual((summary["missed_deadlines"], summary["missed_deadlines_total"]), (0, 1))


class RunLoopTests(unittest.TestCase):
    def test_absolute_deadlines_do_not_drift_and_overruns_skip(self) -> None:
        clock = FakeClock(0.0)
        starts, oversleep = [], {5: 0.050}

        def sleep(seconds):
            clock.now += seconds + 0.001 + oversleep.get(len(starts), 0.0)  # Every wake is 1 ms late.

        serial = FakeSerial()
        loop = ControlLoop(serial, clock=clock.monotonic, sleep=sleep)
        original = loop.tick
        loop.tick = lambda now: (starts.append(now), original(now))[1]
        loop.run(SimpleNamespace(is_set=lambda: False), max_ticks=60)
        # Tick k for deadline k * 20 ms starts 1 ms late: lateness does not add up into drift.
        self.assertAlmostEqual(starts[4], 0.081, places=9)
        # The 5th sleep overslept by 50 ms: deadline 0.10 served late at 0.151, then
        # 0.12 and 0.14 are skipped (missed) and the loop resumes on the 0.16 grid.
        self.assertAlmostEqual(starts[5], 0.151, places=9)
        self.assertAlmostEqual(starts[6], 0.161, places=9)
        self.assertAlmostEqual(starts[59], 0.161 + 53 * 0.02, places=9)
        summary = loop.stats.summary()
        self.assertEqual(summary["missed_deadlines"], 3)
        self.assertEqual(len(serial.written), 60)


class FakeProc:
    def __init__(self, *args, **kwargs) -> None:
        self.args, self.alive = args, True

    def poll(self):
        return None if self.alive else -9

    def kill(self):
        self.alive = False

    def wait(self):
        return -9


class StressTests(unittest.TestCase):
    def test_start_status_stop_with_fake_workers(self) -> None:
        clock = FakeClock(0.0)
        made = []
        stress = StressTest(4, clock=clock.monotonic, popen=lambda *a, **k: made.append(FakeProc(*a)) or made[-1])
        stress.start(20)
        self.addCleanup(stress.stop)
        self.assertEqual(len(made), 4)
        clock.now = 7.5
        self.assertEqual(stress.status(), {"running": True, "seconds_left": 13, "workers": 4})
        stress.stop()
        self.assertTrue(all(not p.alive for p in made))
        self.assertEqual(stress.status(), {"running": False, "seconds_left": 0, "workers": 0})

    def test_workers_run_at_normal_priority_on_qnx(self) -> None:
        stress = StressTest(2, priority=10)
        with patch.object(sys, "platform", "qnx8"), patch("shutil.which", return_value="/usr/bin/on"):
            self.assertEqual(stress.command(5)[:4], ["on", "-p", "10", sys.executable])
        with patch.object(sys, "platform", "darwin"):
            self.assertEqual(stress.command(5)[0], sys.executable)

    def test_real_workers_stop_on_their_own(self) -> None:
        stress = StressTest(1)
        stress.start(0.4)
        self.addCleanup(stress.stop)
        self.assertTrue(stress.status()["running"])
        deadline = time.monotonic() + 5
        while stress.status()["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(stress.status()["running"])


CONTRACT_KEYS = {"state", "safe_reason", "loop_hz", "period_ms_mean", "jitter_ms_p99", "period_ms_max",
                 "missed_deadlines", "target_age_s", "ai_alive", "arduino", "stress"}


class HttpContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serial = FakeSerial()
        self.made = []
        self.stress = StressTest(2, popen=lambda *a, **k: self.made.append(FakeProc(*a)) or self.made[-1])
        self.addCleanup(self.stress.stop)
        self.loop = ControlLoop(self.serial, stress=self.stress)
        now = system_clock()
        self.serial.incoming += [b"S,1,1000,900,1\n"]
        self.loop.tick(now)
        self.loop.stats.record(now, now)
        self.loop.stats.record(now + 0.02, now + 0.02)
        self.server = ControlHTTPServer(self.loop, port=0, host="127.0.0.1")
        self.server.start()
        self.addCleanup(self.server.stop)
        self.base = f"http://127.0.0.1:{self.server.port}"

    def request(self, path, method="GET"):
        request = urllib.request.Request(self.base + path, method=method)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def test_control_json_contract_and_cors(self) -> None:
        status, headers, body = self.request("/control.json")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "*")
        data = json.loads(body)
        self.assertLessEqual(CONTRACT_KEYS, set(data))
        self.assertIn(data["state"], ("SAFE", "ACTIVE"))
        self.assertEqual(data["safe_reason"], "no AI target")
        self.assertEqual(set(data["arduino"]),
                         {"link_ok", "armed", "parking", "gate_closed", "pan", "tilt", "age_s"})
        self.assertEqual((data["arduino"]["armed"], data["arduino"]["pan"]), (True, 100.0))
        self.assertEqual(set(data["stress"]) >= {"running", "seconds_left"}, True)
        for key in ("loop_hz", "period_ms_mean", "jitter_ms_p99", "period_ms_max"):
            self.assertIsInstance(data[key], float)
        self.assertIsInstance(data["missed_deadlines"], int)

    def test_options_and_errors_carry_cors(self) -> None:
        status, headers, _ = self.request("/control.json", "OPTIONS")
        self.assertEqual((status, headers["Access-Control-Allow-Origin"]), (204, "*"))
        self.assertIn("POST", headers["Access-Control-Allow-Methods"])
        for path, method, code in (("/nope", "GET", 404), ("/stress?seconds=61", "POST", 400),
                                   ("/stress?seconds=abc", "POST", 400), ("/stress?seconds=nan", "POST", 400)):
            with self.subTest(path=path):
                status, headers, _ = self.request(path, method)
                self.assertEqual((status, headers["Access-Control-Allow-Origin"]), (code, "*"))

    def test_post_stress_starts_and_stops(self) -> None:
        status, headers, body = self.request("/stress?seconds=20", "POST")
        self.assertEqual((status, headers["Access-Control-Allow-Origin"]), (200, "*"))
        self.assertEqual(json.loads(body)["stress"]["running"], True)
        self.assertEqual(len(self.made), 2)
        stress = json.loads(self.request("/control.json")[2])["stress"]
        self.assertEqual((stress["running"], stress["seconds_left"]), (True, 20))
        self.request("/stress?seconds=0", "POST")
        self.assertTrue(all(not p.alive for p in self.made))
        self.assertFalse(json.loads(self.request("/control.json")[2])["stress"]["running"])


def receive(sock, timeout=2.0):
    sock.settimeout(timeout)
    try:
        return sock.recvfrom(4096)[0]
    finally:
        sock.setblocking(False)


class UdpLinkTests(unittest.TestCase):
    def mappers(self):
        return ServoMapper(smoothing=1.0), ServoMapper(smoothing=1.0)

    def test_ai_targets_reach_the_control_loop_over_localhost(self) -> None:
        control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(control_sock.close)
        control_sock.bind(("127.0.0.1", 0))
        control_sock.setblocking(False)
        client = ControlClient(control_sock.getsockname()[1], *self.mappers(), actual_port=None)
        self.addCleanup(client.close)
        client.send_angles(True, 84.4, 200)       # clamped to the tilt range
        packet = json.loads(receive(control_sock))
        self.assertEqual({k: packet[k] for k in ("type", "active", "yaw", "pitch", "seq")},
                         {"type": "medusa-target", "active": True, "yaw": 84.4, "pitch": 115, "seq": 1})
        self.assertIsInstance(packet["t"], float)
        client.safe_stop()
        client.send(True, 1.0, 0.0)
        serial = FakeSerial()
        loop = ControlLoop(serial, target_sock=control_sock)
        time.sleep(0.05)
        loop.tick(system_clock())
        self.assertEqual(lines(serial), ["N,1150,650,1,1"])  # latest of the two waiting packets

    def test_actual_status_reaches_the_ai_client(self) -> None:
        clock = FakeClock(50.0)
        remote = FakeClock(1050.0)  # The control process's clock: a different base.
        client = ControlClient(9, *self.mappers(), actual_port=0, clock=clock.monotonic, remote_clock=remote.monotonic)
        self.addCleanup(client.close)
        port = client.actual_sock.getsockname()[1]
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sender.close)
        loop = ControlLoop(FakeSerial(), actual_sock=sender, actual_addr=("127.0.0.1", port))
        self.assertIsNone(client.poll_actual())
        loop.serial.incoming += [b"S,0,1023,900,1\n"]
        loop.tick(1049.9)  # 0.1 s ago on the control clock
        time.sleep(0.05)
        actual = client.poll_actual()
        self.assertEqual((actual.armed, actual.pan), (False, 102.3))
        self.assertAlmostEqual(actual.t, 49.9)
        clock.now, remote.now = 50.21, 1050.21  # The S status is now 0.31 s old: not used.
        self.assertIsNone(client.poll_actual())


class TrackerActualAngleTests(unittest.TestCase):
    def test_real_angle_replaces_the_model_when_fresh(self) -> None:
        modelled = PanTracker(65, 115, 90)
        modelled.update(0.5, now=0.0)
        self.assertEqual(modelled.bearing, 90.0)
        real = PanTracker(65, 115, 90)
        real.observe_actual(80.0, at=0.0)  # The turret is really at 80 (the model thinks 90).
        real.update(0.5, now=0.0)
        self.assertEqual((real.bearing, real.angle), (80.0, 80.0))
        self.assertEqual(real.turret_position(-0.07, 0.0), 80.0)

    def test_real_angle_is_carried_forward_by_the_model(self) -> None:
        pan = PanTracker(65, 115, 90)
        pan.update(0.5, now=0.0)
        pan.servo.target = 100.0
        pan.servo.advance(0.25)  # The model has moved 90 -> 100 between 0 and 0.25 s.
        pan.observe_actual(88.0, at=0.0)
        self.assertAlmostEqual(pan.servo.position_at(0.25), 100.0, delta=0.05)
        self.assertAlmostEqual(pan.turret_position(0.25, 0.25), 98.0, delta=0.05)

    def test_stale_real_angle_is_ignored(self) -> None:
        pan = PanTracker(65, 115, 90)
        pan.observe_actual(80.0, at=0.0)
        pan.update(0.5, now=0.31)
        self.assertEqual(pan.bearing, 90.0)

    def test_disarmed_report_resets_home_so_nothing_winds_up(self) -> None:
        pan = PanTracker(65, 115, 90)
        pan.update(0.9, now=0.0)
        self.assertGreater(pan.angle, 100)
        pan.observe_actual(90.0, at=0.1, armed=False)
        for k in range(1, 4):  # while the disarmed report is fresh (0.3 s)
            self.assertIsNone(pan.update(0.9, now=0.1 * k, laptop_active=True))
            self.assertEqual((pan.state, pan.angle, pan.servo.target), ("home", 90.0, 90.0))
        pan.observe_actual(90.0, at=0.4, armed=True)
        pan.update(0.5, now=0.4)
        self.assertEqual(pan.state, "tracking")


class SearchTests(unittest.TestCase):
    def test_sweep_starts_toward_the_laptop_side_bounces_and_stays_in_bounds(self) -> None:
        pan = PanTracker(65, 115, 90, search_speed=60)
        angles = []
        for k in range(40):
            pan.update(None, now=0.1 * k, laptop_active=True, laptop_x=0.8)
            angles.append(round(pan.angle, 6))
            self.assertEqual(pan.state, "searching")
        self.assertEqual(angles[:6], [90, 96, 102, 108, 114, 110])  # right first, bounce off 115
        self.assertTrue(all(65 <= a <= 115 for a in angles))
        self.assertIn(66.0, angles)  # came back down: 110 -> ... -> 68 -> 66 (bounced off 65 at 62)

    def test_reversed_servo_or_left_laptop_side_starts_the_other_way(self) -> None:
        for kwargs, laptop_x in (({"reversed": True}, 0.8), ({}, 0.2)):
            pan = PanTracker(65, 115, 90, search_speed=60, **kwargs)
            pan.update(None, now=0.0, laptop_active=True, laptop_x=laptop_x)
            pan.update(None, now=0.1, laptop_active=True, laptop_x=laptop_x)
            self.assertAlmostEqual(pan.angle, 84.0)

    def test_handoff_then_search_after_the_timeout(self) -> None:
        pan = PanTracker(65, 115, 90, search_speed=60, handoff_timeout=0.6)
        seen = []
        for k in range(9):
            pan.update(None, now=0.1 * k, laptop_active=True, handoff_angle=72.0, laptop_x=0.1)
            seen.append((pan.state, round(pan.angle, 6)))
        self.assertEqual(seen[:6], [("handoff", 72.0)] * 6)
        # At 0.6 s without a face: sweep from 72, toward the mapped side (72 < home 90: left).
        self.assertEqual(seen[6:], [("searching", 66.0), ("searching", 70.0), ("searching", 76.0)])

    def test_a_face_switches_to_tracking_at_once(self) -> None:
        pan = PanTracker(65, 115, 90, search_speed=60)
        for k in range(4):
            pan.update(None, now=0.1 * k, laptop_active=True, laptop_x=0.9)
        self.assertEqual(pan.state, "searching")
        self.assertIsNotNone(pan.update(0.5, now=0.4, laptop_active=True, laptop_x=0.9))
        self.assertEqual(pan.state, "tracking")

    def test_lost_face_coasts_first_then_searches_and_laptop_off_glides_home(self) -> None:
        pan = PanTracker(65, 115, 90, search_speed=60, coast_s=0.3)
        pan.update(0.5, now=0.0)
        states = []
        for k in range(1, 8):
            pan.update(None, now=0.1 * k, laptop_active=True, laptop_x=0.9)
            states.append(pan.state)
        self.assertEqual(states, ["coasting"] * 3 + ["searching"] * 4)  # hold 0.3 s, then sweep
        self.assertAlmostEqual(pan.angle, 108.0)
        pan.update(None, now=0.8, laptop_active=False)
        self.assertEqual((pan.state, round(pan.angle, 6)), ("returning", 102.0))

    def test_search_speed_zero_keeps_coast_then_home(self) -> None:
        pan = PanTracker(65, 115, 90, search_speed=0)
        pan.update(None, now=0.0, laptop_active=True, laptop_x=0.9)
        self.assertEqual((pan.state, pan.angle), ("home", 90.0))


class HandoffMapTests(unittest.TestCase):
    def test_piecewise_linear_and_clamped(self) -> None:
        handoff = HandoffMap([[0.8, 70], [0.2, 110], [0.5, 95]])
        self.assertEqual(handoff.points, [(0.2, 110.0), (0.5, 95.0), (0.8, 70.0)])
        for x, pan in ((0.0, 110), (0.2, 110), (0.35, 102.5), (0.5, 95), (0.65, 82.5), (1.0, 70)):
            self.assertAlmostEqual(handoff.angle(x), pan)
        self.assertIsNone(handoff.angle(None))
        self.assertIsNone(HandoffMap([[0.5, 90]]).angle(0.5))  # one point is not a map
        self.assertEqual(handoff.with_point(0.5, 90).points[1], (0.5, 90.0))  # re-record replaces
        with self.assertRaises(ValueError):
            HandoffMap([[1.5, 90]])

    def test_file_load_save_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "handoff.json"
            self.assertEqual(HandoffMap.load(path).points, [])
            HandoffMap([[0.2, 100], [0.8, 80]]).save(path)
            self.assertEqual(json.loads(path.read_text()), {"points": [[0.2, 100.0], [0.8, 80.0]]})
            clock = FakeClock(0.0)
            live = HandoffMapFile(path, clock=clock.monotonic)
            self.assertEqual(live.current().angle(0.5), 90.0)
            HandoffMap([[0.2, 110], [0.8, 70], [0.9, 65]]).save(path)
            self.assertEqual(live.current().angle(0.5), 90.0)  # checked at most once a second
            clock.now = 1.0
            self.assertEqual(live.current().angle(0.5), 90.0 + 0.0)
            self.assertEqual(live.current().points[-1], (0.9, 65.0))
            path.write_text("{not json")
            clock.now = 2.0
            with redirect_stderr(io.StringIO()):
                self.assertFalse(live.current().usable)
            self.assertIsNotNone(live.error)

    def test_cli_record_show_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "handoff.json")

            def run(*argv):
                out = io.StringIO()
                with redirect_stdout(out), redirect_stderr(out):
                    code = handoff_main(["--map", path, *argv])
                return code, out.getvalue()

            code, out = run("record", "0.2", "105")
            self.assertEqual(code, 0)
            self.assertIn("recorded laptop_x 0.200 -> pan 105.0", out)
            self.assertIn("needs at least 2 points", run("show")[1])
            self.assertEqual(run("record", "0.8", "75")[0], 0)
            self.assertAlmostEqual(HandoffMap.load(path).angle(0.5), 90.0)
            code, out = run("record", "2", "75")
            self.assertEqual(code, 1)
            self.assertIn("laptop_x must be 0..1", out)
            self.assertIn("laptop_x 0.800 -> pan 75.0", run("show")[1])
            self.assertEqual(run("clear")[0], 0)
            self.assertEqual(HandoffMap.load(path).points, [])

    def test_record_now_uses_live_status_and_real_pan(self) -> None:
        status = {"laptop_active": True, "laptop_target_x": 0.3, "pan_state": "tracking", "pan_angle": 99.0}
        control = {"arduino": {"link_ok": True, "pan": 97.4}}
        pages = {"S": status, "C": control}
        self.assertEqual(record_now("S", "C", fetch=pages.__getitem__), (0.3, 97.4, "Arduino actual pan"))
        pages["C"] = {"arduino": {"link_ok": False, "pan": 97.4}}
        self.assertEqual(record_now("S", "C", fetch=pages.__getitem__)[:2], (0.3, 99.0))
        for change in ({"laptop_active": False}, {"pan_state": "searching"}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                record_now("S", "C", fetch={"S": {**status, **change}, "C": control}.__getitem__)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "handoff.json")
            pages["C"] = control
            with redirect_stdout(io.StringIO()):
                self.assertEqual(handoff_main(["--map", path, "record-now"],
                                              fetch=lambda url: pages["S" if "8080" in url else "C"]), 0)
            self.assertEqual(HandoffMap.load(path).points, [(0.3, 97.4)])


class FirmwareStatusTests(unittest.TestCase):
    def test_firmware_sends_s_lines_every_100_ms_in_the_parsed_format(self) -> None:
        source = INO.read_text()
        self.assertIn("const unsigned long STATUS_PERIOD_MS = 100;", source)
        self.assertIn('"S,%d,%ld,%ld,%lu,%d,%d\\n"', source)
        self.assertIn("parking ? 1 : 0);", source)
        self.assertIn("Serial1.print(status);", source)
        status = parse_status_line("S,%d,%ld,%ld,%lu,%d,%d" % (1, 1015, 900, 4294967295, 1, 0))
        self.assertEqual((status.seq, status.gate_closed, status.parking), (4294967295, True, False))


class ControlFlagTests(unittest.TestCase):
    def test_control_process_defaults(self) -> None:
        args = medusa_control.parse_args([])
        self.assertEqual((args.port, args.target_port, args.actual_port, args.http_port, args.period_ms,
                          args.target_timeout, args.stress_workers), ("/dev/ser1", 8768, 8769, 8082, 20.0, 0.3, 4))
        for argv in (["--tilt-min", "100", "--tilt-home", "90"], ["--period-ms", "0"]):
            with self.subTest(argv=argv), patch("sys.stderr"), self.assertRaises(SystemExit):
                medusa_control.parse_args(argv)

    def test_ai_service_flags(self) -> None:
        args = qnx_ai_service.parse_args(["--trigger-udp-port", "8767", "--camera-on-turret"])
        self.assertEqual((args.control_port, args.actual_port, args.search_speed, args.handoff_timeout,
                          args.no_search), (None, 8769, 60.0, 0.6, False))
        self.assertTrue(args.handoff_map.endswith("Medusa_QNX-bench/handoff.json"))
        self.assertEqual(qnx_ai_service._pan_tracker(args).search_speed, 60.0)
        args = qnx_ai_service.parse_args(["--trigger-udp-port", "8767", "--camera-on-turret", "--no-search"])
        self.assertEqual(qnx_ai_service._pan_tracker(args).search_speed, 0.0)

    def test_control_main_refuses_without_serial(self) -> None:
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(medusa_control.main(["--port", "/nonexistent/ser9"]), 2)
        self.assertIn("serial /nonexistent/ser9", err.getvalue())


class ControlModeServiceTests(unittest.TestCase):
    """qnx_ai_service --control-port: no serial port; search, handoff, light and disarm end to end."""

    def setUp(self) -> None:
        self.clock = FakeClock()
        patcher = patch.object(trigger_module, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_service(self, script, map_points, extra_argv=()):
        """Run qnx_ai_service over ``script``; return (sent (active, yaw) packets, tracker states).

        Each frame is (laptop, turret faces, send a disarmed S status first?), with laptop
        None (no packet), False (an inactive packet) or the laptop's target_x (active), and
        turret faces NO_FRAME (camera.read fails) or RAISE (the face model raises).
        Every packet, with its pitch, is kept in ``self.packets``.
        """
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(receiver.close)
        receiver.bind(("127.0.0.1", 0))
        actual_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(actual_out.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        handoff = Path(tmp.name) / "handoff.json"
        # The service's setup (default --servo-* and --camera-hfov): a file for another one is discarded.
        HandoffMap(map_points, fingerprint=setup_fingerprint(90, 65, 115, False, 50.0)).save(handoff)
        frame = SimpleNamespace(shape=(480, 640, 3))
        pending, index, clients, states = [], {"i": -1}, [], []

        def read():
            index["i"] += 1
            if index["i"] == len(script):
                raise KeyboardInterrupt
            laptop, _faces, disarm = script[index["i"]]
            self.clock.now += 0.1
            if laptop is not None:
                message = {"type": "medusa-peeper", "active": laptop is not False, "seq": index["i"]}
                if laptop is not False:
                    message.update(target_x=laptop, target_y=0.5)
                pending.append((json.dumps(message).encode(), LAPTOP))
            if disarm:
                port = clients[0].actual_sock.getsockname()[1]
                actual_out.sendto(json.dumps({"type": "medusa-actual", "link_ok": True, "armed": False,
                                              "pan": 90.0, "tilt": 90.0, "t": self.clock.now}).encode(),
                                  ("127.0.0.1", port))
                time.sleep(0.05)
            return _faces is not NO_FRAME, frame

        def recvfrom(_size):
            if pending:
                return pending.pop(0)
            raise BlockingIOError

        def make_client(port, mapper, tilt_mapper, actual_port=None):
            clients.append(ControlClient(port, mapper, tilt_mapper, actual_port=0, clock=self.clock.monotonic,
                                         remote_clock=self.clock.monotonic))
            return clients[-1]

        trigger_sock = MagicMock()
        trigger_sock.recvfrom.side_effect = recvfrom
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = read
        cv2 = SimpleNamespace(VideoCapture=Mock(return_value=camera), error=RuntimeError)
        detector = Mock()
        def detect(_frame):
            faces = script[index["i"]][1]
            if faces is RAISE:
                raise RuntimeError("TFLite invoke failed")
            return faces

        detector.face_detector.detect.side_effect = detect
        real_update = PanTracker.update

        def update(pan, *args, **kwargs):
            result = real_update(pan, *args, **kwargs)
            states.append(pan.state)
            return result

        with (
            patch.dict(sys.modules, {"cv2": cv2}),
            patch.object(sys, "argv", [
                "qnx_ai_service.py", "--preview-port", "0", "--control-port", str(receiver.getsockname()[1]),
                "--trigger-udp-port", "8767", "--trigger-from", LAPTOP[0], "--camera-on-turret",
                "--handoff-map", str(handoff), "--handoff-timeout", "0.15",
                "--face-confirm-frames", "1", "--face-dropout-frames", "0",  # Debounce: test_trigger.py.
                *extra_argv,
            ]),
            patch.object(qnx_ai_service, "TFLiteFaceGazeDetector", return_value=detector),
            patch.object(qnx_ai_service, "ArduinoController", side_effect=AssertionError("no serial port")),
            patch.object(qnx_ai_service, "ControlClient", side_effect=make_client),
            patch.object(qnx_ai_service.LaptopTrigger, "bind",
                         return_value=LaptopTrigger(trigger_sock, allowed_sender=LAPTOP[0])),
            patch.object(PanTracker, "update", update),
        ):
            self.assertEqual(qnx_ai_service.main(), 0)

        packets = []
        receiver.setblocking(False)
        while True:
            try:
                packets.append(json.loads(receiver.recvfrom(4096)[0]))
            except BlockingIOError:
                break
        self.assertTrue(all(p["type"] == "medusa-target" for p in packets))
        self.packets = packets
        return [(p["active"], p["yaw"]) for p in packets], states

    def test_camera_frame_failure_and_inference_error_command_home_exactly(self) -> None:
        face = FaceBox(480, 100, 520, 140, 0.9)  # right of and above centre: yaw and pitch leave home
        _sent, states = self.run_service([
            (0.8, [face], False),     # 1: tracking, light on
            (0.8, [face], False),     # 2
            (0.8, NO_FRAME, False),   # 3: camera frame fails -> home, light off
            (0.8, [face], False),     # 4: tracking again
            (0.8, [face], False),     # 5
            (0.8, RAISE, False),      # 6: the face model raises -> home, light off
        ], [[0.5, 90]])
        angles = [(p["active"], p["yaw"], p["pitch"]) for p in self.packets]
        home = (False, 90.0, 90.0)  # default --servo-home and --tilt-home
        self.assertEqual(angles[0], home)  # startup safe_stop
        for k in (1, 2, 4, 5):
            with self.subTest(frame=k):
                self.assertTrue(angles[k][0])
                self.assertNotEqual(angles[k][1:], home[1:])  # aimed away from home
        self.assertEqual(angles[3], home)   # camera fault: exactly home, not a smoothed step
        self.assertEqual(angles[6], home)   # inference error: exactly home
        self.assertEqual(angles[7], home)   # shutdown
        self.assertEqual(len(angles), 8)
        self.assertEqual(states, ["tracking"] * 4)  # the error frames never reach the tracker

    def test_error_home_uses_the_configured_homes(self) -> None:
        face = FaceBox(480, 100, 520, 140, 0.9)
        with redirect_stderr(io.StringIO()):  # the test map's setup does not match: discarded, logged
            self.run_service([(0.8, [face], False), (0.8, NO_FRAME, False), (0.8, [face], False),
                              (0.8, RAISE, False)], [[0.5, 90]],
                             ["--servo-min", "0", "--servo-max", "70", "--servo-home", "0",
                              "--tilt-min", "55", "--tilt-max", "125", "--tilt-home", "90"])
        angles = [(p["active"], p["yaw"], p["pitch"]) for p in self.packets]
        self.assertEqual([angles[k] for k in (0, 2, 4, 5)], [(False, 0.0, 90.0)] * 4)
        self.assertTrue(angles[1][0] and angles[3][0])
        self.assertNotEqual(angles[1][1:], (0.0, 90.0))

    def test_fallback_still_aims_from_the_laptop_without_errors(self) -> None:
        # No error: the turret camera merely loses the face, so laptop-guided fallback aims (unchanged).
        face = FaceBox(300, 220, 340, 260, 0.9)
        sent, states = self.run_service([(0.8, [face], False)] + [(0.8, [], False)] * 6,
                                        [[0.0, 70], [1.0, 110]])
        self.assertIn("fallback", states)
        self.assertTrue(all(active for active, _yaw in sent[1:8]))

    def test_without_a_usable_map_search_keeps_the_light_off_and_disarm_resets(self) -> None:
        face = FaceBox(300, 220, 340, 260, 0.9)  # image centre
        sent, states = self.run_service([
            (None, [], False),        # 0: no laptop -> home, light off
            (0.8, [], False),         # 1: laptop sees a peeper, turret does not -> search, from 90
            (0.8, [], False),         # 2: 96 (right first: laptop_x 0.8)
            (0.8, [], False),         # 3: 102
            (0.8, [], False),         # 4: 108
            (0.8, [face], False),     # 5: turret camera sees a face -> tracking, light ON
            (0.8, [face], True),      # 6: Arduino reports disarmed -> tracker home
        ], [[0.5, 90]])               # one point is not a usable map
        # sent[0] is the startup safe_stop; then one packet per frame; the last is shutdown's.
        self.assertEqual(sent[:6], [(False, 90), (False, 90), (False, 90), (False, 96), (False, 102),
                                    (False, 108)])
        self.assertEqual(sent[6][0], True)          # light only once the turret camera sees the face
        self.assertEqual(sent[7], (True, 90))       # disarmed: tracker reset home (firmware ignores it)
        self.assertEqual(sent[8], (False, 90))      # safe_stop at shutdown
        self.assertEqual(states, ["home"] + ["searching"] * 4 + ["tracking", "home"])

    def test_usable_map_aims_from_the_laptop_with_the_light_on_until_the_face_returns(self) -> None:
        face = FaceBox(300, 220, 340, 260, 0.9)  # image centre
        sent, states = self.run_service([
            (None, [], False),        # 0: no laptop -> home, light off
            (0.8, [], False),         # 1: laptop only -> fallback: toward map(0.8) = 102, light ON
            (0.8, [face], False),     # 2: turret face -> tracking
            (0.8, [], False),         # 3: face lost -> coasting (0.3 s), light stays on
            (0.8, [], False),         # 4
            (0.8, [], False),         # 5
            (0.8, [], False),         # 6: coast over (at 6 or 7: float time) -> fallback
            (0.8, [], False),         # 7: fallback at 102
            (0.2, [], False),         # 8: the peeper moves left: map(0.2) = 78, smoothed
            (0.2, [], False),         # 9
            (0.2, [], False),         # 10
            (0.2, [face], False),     # 11: face again -> tracking
            (False, [], False),       # 12: laptop sees no peeper -> usual coast then home, light off
        ], [[0.0, 70], [1.0, 110]])
        self.assertEqual(sent[:3], [(False, 90), (False, 90), (True, 102)])  # 12 deg = 120 deg/s * 0.1 s
        self.assertEqual(states[:4], ["home", "fallback", "tracking", "coasting"])
        self.assertEqual(states[-2:], ["tracking", "coasting"])
        self.assertEqual(states[7:11], ["fallback"] * 4)
        self.assertTrue(all(active for active, _yaw in sent[2:13]))  # on from fallback to the face
        self.assertFalse(sent[13][0])                               # laptop inactive: light off
        self.assertEqual(sent[8], (True, 102))
        yaws = [yaw for _active, yaw in sent[9:12]]
        steps = [b - a for a, b in zip([102] + yaws, yaws)]
        self.assertTrue(all(-12 <= step < 0 for step in steps), yaws)  # heading for 78, rate-limited
        self.assertGreater(yaws[-1], 78)  # smoothed (tau 0.25 s): not there in three frames

    def test_learner_gets_face_count_match_and_turret_flags(self) -> None:
        face, other = FaceBox(300, 220, 340, 260, 0.9), FaceBox(500, 220, 540, 260, 0.9)
        calls = []
        real_observe = qnx_ai_service.HandoffLearner.observe

        def observe(learner, *args, **kwargs):
            calls.append(kwargs)
            return real_observe(learner, *args, **kwargs)

        with patch.object(qnx_ai_service.HandoffLearner, "observe", observe):
            self.run_service([
                (0.8, [face], False),          # one face, no usable map: "single"
                (0.8, [face, other], False),   # two faces (locked on the first)
                (0.8, [face], True),           # the Arduino reports disarmed
            ], [[0.5, 90]])
        self.assertEqual([(c["faces"], c["match"]) for c in calls], [(1, "single"), (2, "single"), (1, "single")])
        self.assertEqual([(c["armed"], c["gate_closed"]) for c in calls], [(None, None), (None, None),
                                                                            (False, None)])

    def test_no_search_turns_fallback_off(self) -> None:
        sent, states = self.run_service([(0.8, [], False)] * 2, [[0.0, 70], [1.0, 110]], ["--no-search"])
        self.assertEqual(states, ["home", "home"])
        self.assertFalse(any(active for active, _yaw in sent))

if __name__ == "__main__":
    unittest.main()
