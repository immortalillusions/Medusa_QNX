"""Errors send the turret home: firmware parking on link loss, Pi-side home targets."""

import io
import re
import sys
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import Mock, patch

import medusa_control
import qnx_ai_service
from medusa.control import ControlClient, ControlLoop, parse_command_line
from medusa.servo import ServoMapper
from test_control import INO, FakeClock, FakeSerial, FakeSock, lines, target
from test_turret_safety import function_body


def firmware_constant(source: str, name: str) -> int:
    return int(re.search(rf"\b{name} = (\d+)", source).group(1))


class FirmwareParkingTests(unittest.TestCase):
    """Link loss while armed glides home, then goes limp; deliberate off stays stop-in-place."""

    def setUp(self) -> None:
        self.source = INO.read_text()
        self.loop = function_body(self.source, "void loop(")

    def test_heartbeat_timeout_starts_parking_instead_of_detaching(self) -> None:
        timeout = re.search(r"if \(armed && millis\(\) - lastCommandMs > HEARTBEAT_TIMEOUT_MS\) \{(.*?)\}",
                            self.loop, re.S).group(1)
        self.assertIn("startParking();", timeout)
        self.assertNotIn("setArmed", timeout)

    def test_parking_targets_home_and_stays_attached(self) -> None:
        park = function_body(self.source, "void startParking(")
        self.assertIn("pan.target = pan.home;", park)
        self.assertIn("tilt.target = tilt.home;", park)
        self.assertIn("armed = false;", park)  # status reads 0; applyCommand ignores motion
        self.assertIn("parking = true;", park)
        self.assertIn("digitalWrite(LED_PIN, LOW);", park)
        for detaching in ("detach", "setArmed"):
            self.assertNotIn(detaching, park)  # still attached, so stepAxis keeps easing
        apply = function_body(self.source, "void applyCommand(")
        self.assertIn("if (armed) {", apply)  # a parking turret takes no new aim

    def test_detach_only_once_home_or_after_the_cap(self) -> None:
        self.assertEqual(firmware_constant(self.source, "PARK_TIMEOUT_MS"), 2000)
        done = re.search(r"if \(parking && \(\(reachedHome\(pan\) && reachedHome\(tilt\)\) \|\| "
                         r"millis\(\) - parkingSinceMs >= PARK_TIMEOUT_MS\)\) \{(.*?)\}", self.loop, re.S)
        self.assertIsNotNone(done)
        self.assertIn("setArmed(false);", done.group(1))
        reached = function_body(self.source, "bool reachedHome(")
        self.assertIn("axis.position == (float)axis.home && axis.velocity == 0", reached)
        # The heartbeat check and the park check run before the motion step, every loop.
        self.assertLess(self.loop.index("startParking();"), self.loop.index("stepAxis(pan"))
        self.assertLess(self.loop.index("reachedHome(pan)"), self.loop.index("stepAxis(pan"))
        self.assertIn("parking = false;", function_body(self.source, "void setArmed("))

    def test_button_or_gate_during_parking_cuts_to_limp(self) -> None:
        button = function_body(self.source, "void updateArmButton(")
        # A press while parking cuts to limp; a press while armed parks first (glide home,
        # then limp, so the next arm starts at home); a press while off arms.
        self.assertIn("if (parking) setArmed(false);", button)
        self.assertIn("else if (armed) startParking();", button)
        self.assertIn("else setArmed(true);", button)
        gate = function_body(self.source, "void updateKillGate(")
        self.assertIn("if (!gateClosed && (armed || parking)) setArmed(false);", gate)
        # setArmed(false) is the stop-in-place detach (target = position), not a glide.
        self.assertIn("detachAxis(pan);", function_body(self.source, "void setArmed("))
        self.assertIn("axis.target = axis.position;", function_body(self.source, "void detachAxis("))

    def test_status_reports_parking(self) -> None:
        self.assertIn('"S,%d,%ld,%ld,%lu,%d,%d\\n"', self.source)
        self.assertIn("parking ? 1 : 0);", function_body(self.source, "void sendStatus("))


class ControlParkLineTests(unittest.TestCase):
    """The control loop's stale-target line is the configured home, which the firmware accepts."""

    def test_main_park_line_uses_the_configured_homes(self) -> None:
        serial = FakeSerial()
        serial.close = Mock()
        with (
            patch.object(medusa_control.SerialPort, "open", return_value=serial),
            patch.object(medusa_control.signal, "signal"),
            patch.object(medusa_control, "ControlHTTPServer"),  # binding 0.0.0.0 costs ~5 s of getfqdn
            patch.object(medusa_control.ControlLoop, "run", lambda loop, _stop: loop.tick(loop.clock())),
            patch("builtins.print"),
        ):
            self.assertEqual(medusa_control.main([
                "--servo-min", "0", "--servo-max", "70", "--servo-home", "0", "--tilt-min", "55",
                "--tilt-max", "125", "--tilt-home", "90", "--target-port", "0", "--actual-port", "0",
                "--http-port", "0"]), 0)
        # One tick with no AI target, then shutdown's three park lines.
        self.assertEqual(lines(serial), ["M,0,90,0,1", "M,0,90,0,2", "M,0,90,0,3", "M,0,90,0,4"])

    def test_stale_target_parks_at_home_inside_the_firmware_range(self) -> None:
        serial, sock = FakeSerial(), FakeSock()
        loop = ControlLoop(serial, target_sock=sock, yaw_range=(0, 70), pitch_range=(55, 125), home=(0, 90),
                           clock=FakeClock().monotonic)
        serial.incoming += [b"S,1,500,1000,1,1,0\n"]
        sock.queue += [target(True, 50, 100, 1)]
        loop.tick(1.0)
        loop.tick(1.5)  # target 0.5 s old: stale
        self.assertEqual(lines(serial), ["N,500,1000,1,1", "M,0,90,0,2"])
        # The firmware's own range check (applyCommand) accepts the park line, so an armed turret moves.
        source = INO.read_text()
        pan = [firmware_constant(source, k) for k in ("PAN_MIN", "PAN_MAX", "PAN_HOME")]
        tilt = [firmware_constant(source, k) for k in ("TILT_MIN", "TILT_MAX", "TILT_HOME")]
        self.assertEqual((pan[2], tilt[2]), (0, 90))
        yaw, pitch, led, _seq = parse_command_line("M,0,90,0,2", 0, 180)
        self.assertTrue(pan[0] <= yaw <= pan[1] and tilt[0] <= pitch <= tilt[1])
        self.assertFalse(led)

    def test_parking_status_is_reported(self) -> None:
        serial = FakeSerial()
        loop = ControlLoop(serial, home=(0, 90), clock=FakeClock().monotonic)
        serial.incoming += [b"S,0,350,900,1,1,1\n"]
        loop.tick(1.0)
        snap = loop.snapshot(1.1)
        self.assertEqual((snap["state"], snap["safe_reason"]), ("SAFE", "returning home, then limp"))
        self.assertEqual((snap["arduino"]["parking"], snap["arduino"]["armed"]), (True, False))
        serial.incoming += [b"S,0,0,900,2,1,0\n"]  # home and limp
        loop.tick(1.2)
        snap = loop.snapshot(1.3)
        self.assertEqual((snap["safe_reason"], snap["arduino"]["parking"]), ("turret disarmed", False))
        serial.incoming += [b"S,0,0,900,3,0,1\n"]  # gate open wins over a parking flag
        loop.tick(1.4)
        self.assertEqual(loop.snapshot(1.5)["safe_reason"], "kill gate open")


class SafeStopTests(unittest.TestCase):
    def test_control_client_safe_stop_is_exactly_home_after_aiming(self) -> None:
        mapper = ServoMapper(0, 70, 0)
        tilt = ServoMapper(55, 125, 90)
        client = ControlClient(9, mapper, tilt, actual_port=None)
        self.addCleanup(client.close)
        sent = []
        client._send = lambda active, yaw, pitch: sent.append((active, yaw, pitch))
        for _ in range(5):
            mapper.command(True, 1.0)
            tilt.command(True, 0.0)
        client.safe_stop()
        self.assertEqual(sent, [(False, 0, 90)])
        self.assertEqual(mapper.command(False, None), (0, False))  # smoothing restarts from home


class EventModeInferenceErrorTests(unittest.TestCase):
    def test_inference_error_sends_home_not_a_smoothed_step(self) -> None:
        frames = iter([True, True])

        def read():
            try:
                return next(frames), SimpleNamespace(shape=(480, 640, 3))
            except StopIteration:
                raise KeyboardInterrupt

        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = read
        detector = Mock()
        detector.face_detector.detect.side_effect = RuntimeError("TFLite invoke failed")
        client = Mock()
        cv2 = SimpleNamespace(VideoCapture=Mock(return_value=camera), error=RuntimeError)
        with (
            patch.dict(sys.modules, {"cv2": cv2}),
            patch.object(sys, "argv", ["qnx_ai_service.py", "--preview-port", "0", "--event-host", "127.0.0.1",
                                       "--event-port", "9", "--control-port", "9"]),
            patch.object(qnx_ai_service, "TFLiteFaceGazeDetector", return_value=detector),
            patch.object(qnx_ai_service, "ControlClient", return_value=client),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(qnx_ai_service.main(), 0)
        # Startup, one per failed frame, shutdown; never a smoothed send().
        self.assertEqual(client.safe_stop.call_count, 4)
        client.send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
