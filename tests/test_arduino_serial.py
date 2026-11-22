from __future__ import annotations

import os
import unittest

import qnx_ai_service
from medusa.arduino_serial import ArduinoController
from medusa.servo import ServoMapper


def controller(tilt: ServoMapper | None = None) -> tuple[ArduinoController, int]:
    # Skip the serial-port setup in __init__ and write into a pipe instead.
    arduino = ArduinoController.__new__(ArduinoController)
    arduino.mapper = ServoMapper(smoothing=1.0)
    arduino.tilt_mapper = tilt or ServoMapper(smoothing=1.0)
    arduino.sequence = 0
    read_fd, arduino.fd = os.pipe()
    return arduino, read_fd


class ArduinoSerialTests(unittest.TestCase):
    def sent(self, arduino: ArduinoController, read_fd: int) -> str:
        arduino.close()
        try:
            return os.read(read_fd, 1024).decode("ascii")
        finally:
            os.close(read_fd)

    def test_send_writes_five_field_line_with_pitch(self) -> None:
        arduino, read_fd = controller()
        arduino.send(True, 1.0, 0.0)
        self.assertEqual(self.sent(arduino, read_fd), "M,115,65,1,1\n")

    def test_reversed_tilt_flips_pitch(self) -> None:
        arduino, read_fd = controller(ServoMapper(reversed=True, smoothing=1.0))
        arduino.send(True, 0.5, 0.0)
        self.assertEqual(self.sent(arduino, read_fd), "M,90,115,1,1\n")

    def test_tilt_goes_home_without_target_y_or_when_inactive(self) -> None:
        arduino, read_fd = controller()
        arduino.send(True, 0.0)
        arduino.send(False, 0.0, 0.0)
        self.assertEqual(self.sent(arduino, read_fd), "M,65,90,1,1\nM,90,90,0,2\n")

    def test_send_angles_writes_exact_clamped_angles_without_smoothing(self) -> None:
        arduino, read_fd = controller(ServoMapper(smoothing=0.35))
        arduino.mapper = ServoMapper(smoothing=0.35)
        arduino.send_angles(True, 84.4, 77.6)
        arduino.send_angles(False, 200, -5)
        self.assertEqual(self.sent(arduino, read_fd), "N,844,776,1,1\nN,1150,650,0,2\n")

    def test_safe_stop_sends_both_axes_home(self) -> None:
        arduino, read_fd = controller()
        arduino.safe_stop()
        self.assertEqual(self.sent(arduino, read_fd), "M,90,90,0,1\n")


class ServiceFlagTests(unittest.TestCase):
    def test_defaults(self) -> None:
        args = qnx_ai_service.build_parser().parse_args(["--event-host", "127.0.0.1"])
        self.assertEqual(args.hold_frames, 8)
        self.assertEqual((args.tilt_min, args.tilt_max, args.tilt_home), (65, 115, 90))
        self.assertFalse(args.tilt_reversed)

    def test_tilt_reversed_flag(self) -> None:
        args = qnx_ai_service.build_parser().parse_args(
            ["--event-host", "127.0.0.1", "--tilt-reversed"])
        self.assertTrue(args.tilt_reversed)


if __name__ == "__main__":
    unittest.main()
