import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np

import qnx_ai_service
from medusa import camera_select
from medusa.camera_select import (
    AUTO_UNITS, BLANK, REAL, TEST_PATTERN, AutoCamera, NoCameraError, classify_frame, find_camera,
    probe_unit,
)

# BGR values read from QNX unit 2 on the Pi (2026-11-22): white, yellow, cyan, green, magenta, red,
# blue, black at 75% bars.
BARS = [(235, 235, 235), (16, 180, 180), (180, 180, 16), (16, 180, 16), (180, 16, 180), (16, 16, 180),
        (180, 16, 16), (16, 16, 16)]


def colour_bars(width=1280, height=720):
    frame = np.zeros((height, width, 3), np.uint8)
    for index, colour in enumerate(BARS):
        frame[:, index * width // len(BARS):(index + 1) * width // len(BARS)] = colour
    return frame


def real_scene(width=640, height=480, seed=0):
    """A dim, low-contrast scene: a soft gradient plus sensor noise, the hardest real case."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = 60 + 20 * (y / height) + 10 * (x / width)
    frame = base[:, :, None] + rng.normal(0, 2.0, (height, width, 3))
    return np.clip(frame, 0, 255).astype(np.uint8)


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)
        self.reads = 0
        self.released = False

    def read(self):
        self.reads += 1
        if not self.frames:
            return False, None
        frame = self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]
        return (False, None) if frame is None else (True, frame)

    def release(self):
        self.released = True


class Opener:
    """Fake QnxCamera: units maps a unit to a frame list; missing units raise like error 2."""

    def __init__(self, units):
        self.units = units
        self.opened = []
        self.cameras = {}

    def __call__(self, unit):
        self.opened.append(unit)
        if unit not in self.units:
            raise RuntimeError(f"QNX camera unit {unit} failed to open or start (error 2)")
        self.cameras[unit] = FakeCamera(self.units[unit])
        return self.cameras[unit]


NO_WAIT = {"sleep": lambda _s: None}


class ClassifyFrameTests(unittest.TestCase):
    def test_colour_bars_are_a_test_pattern(self):
        kind, why = classify_frame(colour_bars())
        self.assertEqual(kind, TEST_PATTERN)
        self.assertIn("constant top to bottom", why)

    def test_scrolled_bars_and_a_small_frame_are_still_a_test_pattern(self):
        self.assertEqual(classify_frame(np.roll(colour_bars(), 77, axis=1))[0], TEST_PATTERN)
        self.assertEqual(classify_frame(colour_bars(320, 240))[0], TEST_PATTERN)

    def test_a_dim_noisy_gradient_is_real(self):
        for seed in range(5):
            self.assertEqual(classify_frame(real_scene(seed=seed))[0], REAL)

    def test_one_colour_frames_are_blank(self):
        self.assertEqual(classify_frame(np.zeros((480, 640, 3), np.uint8))[0], BLANK)
        self.assertEqual(classify_frame(np.full((480, 640, 3), 255, np.uint8))[0], BLANK)

    def test_bars_with_a_real_bottom_half_are_not_a_test_pattern(self):
        frame = real_scene(1280, 720)
        frame[:200] = colour_bars(1280, 200)
        self.assertEqual(classify_frame(frame)[0], REAL)


class ProbeUnitTests(unittest.TestCase):
    def test_black_frames_while_exposure_settles_then_real_is_accepted(self):
        black = np.zeros((480, 640, 3), np.uint8)
        camera = FakeCamera([black, black, real_scene()])
        self.assertEqual(probe_unit(camera, **NO_WAIT)[0], REAL)
        self.assertEqual(camera.reads, 3)

    def test_a_test_pattern_stops_after_one_frame(self):
        camera = FakeCamera([colour_bars()])
        self.assertEqual(probe_unit(camera, **NO_WAIT)[0], TEST_PATTERN)
        self.assertEqual(camera.reads, 1)

    def test_waits_to_settle_before_reading(self):
        sleeps = []
        probe_unit(FakeCamera([real_scene()]), sleep=sleeps.append)
        self.assertEqual(sleeps, [camera_select.SETTLE_S])

    def test_no_frames(self):
        self.assertEqual(probe_unit(FakeCamera([None]), **NO_WAIT)[0], "no frames")


class FindCameraTests(unittest.TestCase):
    def test_order_is_unit_5_then_1_to_10(self):
        self.assertEqual(AUTO_UNITS, (5, 1, 2, 3, 4, 6, 7, 8, 9, 10))

    def test_skips_missing_and_test_patterns_and_takes_the_first_real_camera(self):
        opener = Opener({1: [colour_bars()], 2: [colour_bars()], 3: [real_scene()], 4: [real_scene()]})
        log = []
        camera, unit = find_camera(opener, log=log.append, **NO_WAIT)
        self.assertEqual(unit, 3)
        self.assertEqual(opener.opened, [5, 1, 2, 3])
        self.assertIs(camera, opener.cameras[3])
        self.assertFalse(camera.released)
        self.assertTrue(opener.cameras[1].released and opener.cameras[2].released)
        self.assertIn("camera unit 5: rejected, did not open", log[0])
        self.assertIn("camera unit 1: rejected, test pattern", log[1])
        self.assertIn("camera unit 3: using it, real camera", log[-1])

    def test_unit_5_first_when_present(self):
        opener = Opener({1: [colour_bars()], 5: [real_scene()]})
        self.assertEqual(find_camera(opener, log=lambda _l: None, **NO_WAIT)[1], 5)
        self.assertEqual(opener.opened, [5])

    def test_tonights_pi_none_found_message(self):
        opener = Opener({1: [colour_bars()], 2: [colour_bars()]})
        with self.assertRaises(NoCameraError) as caught:
            find_camera(opener, log=lambda _l: None, **NO_WAIT)
        self.assertEqual(str(caught.exception), "no real camera found (units tried: 5, 1, 2, 3, 4, 6, 7, "
                                                "8, 9, 10; test patterns at: 1, 2)")
        self.assertEqual(caught.exception.test_patterns, (1, 2))

    def test_blank_and_frameless_units_are_named(self):
        opener = Opener({3: [np.zeros((48, 64, 3), np.uint8)], 4: [None]})
        with self.assertRaises(NoCameraError) as caught:
            find_camera(opener, log=lambda _l: None, **NO_WAIT)
        self.assertIn("test patterns at: none; blank at: 3; no frames at: 4", str(caught.exception))
        self.assertTrue(opener.cameras[3].released and opener.cameras[4].released)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


class AutoCameraTests(unittest.TestCase):
    def make(self, opener):
        clock, log = Clock(), []
        return AutoCamera(opener, clock=clock, sleep=clock.sleep, log=log.append), clock, log

    def test_waits_for_the_camera_then_comes_back_without_a_restart(self):
        opener = Opener({1: [colour_bars()], 2: [colour_bars()]})
        camera, clock, log = self.make(opener)
        self.assertEqual(camera.read(), (False, None))
        self.assertEqual(camera.unavailable_reason, "waiting for the turret camera")
        self.assertIn("no real camera found", log[-1])
        logged = len(log)
        opener.opened.clear()
        self.assertEqual(camera.read(), (False, None))
        self.assertEqual(clock.sleeps[-1], camera_select.RETRY_S)  # retries every 3 s
        self.assertNotIn(1, opener.opened)  # built-in test patterns are not reopened
        self.assertEqual(len(log), logged)  # the same failure is not logged again
        frame = real_scene()
        opener.units[5] = [frame]  # the C270 is plugged in
        ok, got = camera.read()
        self.assertTrue(ok)
        self.assertIs(got, frame)
        self.assertEqual(camera.unit, 5)
        self.assertEqual(camera.unavailable_reason, "camera frame unavailable")
        self.assertIn("camera unit 5: using it", log[-1])

    def test_a_camera_that_disappears_is_released_and_searched_for_again(self):
        opener = Opener({5: [real_scene(), real_scene()]})
        camera, _clock, log = self.make(opener)
        self.assertTrue(camera.read()[0])  # probe used the first frame, read gets a frame too
        first = opener.cameras[5]
        del opener.units[5]  # unplugged: reads fail from now on
        first.frames = [None]
        for _ in range(camera_select.LOST_AFTER - 1):
            self.assertEqual(camera.read(), (False, None))
            self.assertEqual(camera.unavailable_reason, "camera frame unavailable")
        self.assertEqual(camera.read(), (False, None))
        self.assertTrue(first.released)
        self.assertEqual(camera.unavailable_reason, "waiting for the turret camera")
        self.assertIn("searching again", log[-1])
        opener.units[5] = [real_scene()]  # plugged back in: the next read finds it again
        self.assertTrue(camera.read()[0])
        self.assertIsNot(opener.cameras[5], first)

    def test_one_good_frame_resets_the_failure_count(self):
        opener = Opener({5: [real_scene()]})
        camera, _clock, _log = self.make(opener)
        camera.read()
        cam = opener.cameras[5]
        for _ in range(3):
            cam.frames = [None]
            camera.read()
            camera.read()
            cam.frames = [real_scene()]
            self.assertTrue(camera.read()[0])
        self.assertFalse(cam.released)

    def test_release(self):
        opener = Opener({5: [real_scene()]})
        camera, _clock, _log = self.make(opener)
        camera.read()
        camera.release()
        self.assertTrue(opener.cameras[5].released)
        camera.release()


class ServiceCameraTests(unittest.TestCase):
    def test_unit_flag_takes_auto_or_an_integer(self):
        parse = qnx_ai_service.build_parser().parse_args
        self.assertEqual(parse(["--qnx-camera-unit", "auto"]).qnx_camera_unit, "auto")
        self.assertEqual(parse(["--qnx-camera-unit", "5"]).qnx_camera_unit, 5)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse(["--qnx-camera-unit", "five"])

    def test_start_script_uses_auto(self):
        with open("pi/start_medusa.sh") as script:
            self.assertIn("--qnx-camera-unit auto ", script.read())

    def test_integer_unit_opens_that_unit_only(self):
        args = SimpleNamespace(qnx_camera_unit=5, camera=0)
        with patch("medusa.qnx_camera.QnxCamera") as qnx:
            self.assertIs(qnx_ai_service._open_camera(args), qnx.return_value)
        qnx.assert_called_once_with(5)

    def test_integer_unit_missing_still_exits(self):
        error = RuntimeError("QNX camera unit 5 failed to open or start (error 2)")
        with (
            patch("medusa.qnx_camera.QnxCamera", side_effect=error),
            patch.object(sys, "argv", ["qnx_ai_service.py", "--event-host", "127.0.0.1",
                                       "--qnx-camera-unit", "5", "--preview-port", "0"]),
            patch.object(qnx_ai_service, "TFLiteFaceGazeDetector"),
            patch.dict(sys.modules, {"cv2": SimpleNamespace(error=RuntimeError)}),
            patch("sys.stderr"),
        ):
            self.assertEqual(qnx_ai_service.main(), 2)

    def test_auto_opens_an_auto_camera_without_opening_a_unit(self):
        with patch("medusa.qnx_camera.QnxCamera") as qnx:
            camera = qnx_ai_service._open_camera(SimpleNamespace(qnx_camera_unit="auto", camera=0))
        self.assertIsInstance(camera, AutoCamera)
        qnx.assert_not_called()  # the search happens on the first read, after the service is up

    def test_service_waits_homes_and_recovers_when_the_camera_is_plugged_in(self):
        opener = Opener({1: [colour_bars()], 2: [colour_bars()]})
        clock = Clock()
        camera = AutoCamera(opener, clock=clock, sleep=clock.sleep, log=lambda _l: None)
        reads = []
        real_read = camera.read

        def read():
            reads.append(1)
            if len(reads) == 3:
                opener.units[5] = [real_scene()]
            if len(reads) == 5:
                raise KeyboardInterrupt
            return real_read()

        camera.read = read
        detector = Mock()
        detector.face_detector.detect.return_value = []
        detector.detect.return_value = []
        socket = MagicMock()
        arduino = Mock()
        with (
            patch.object(sys, "argv", ["qnx_ai_service.py", "--event-host", "127.0.0.1",
                                       "--qnx-camera-unit", "auto", "--preview-port", "0",
                                       "--arduino-port", "/dev/ser1"]),
            patch.object(qnx_ai_service, "TFLiteFaceGazeDetector", return_value=detector),
            patch.object(qnx_ai_service, "_open_camera", return_value=camera),
            patch.object(qnx_ai_service, "ArduinoController", return_value=arduino),
            patch.object(qnx_ai_service.socket, "socket", return_value=socket),
        ):
            self.assertEqual(qnx_ai_service.main(), 0)
        reasons = [json.loads(sent.args[0])["reason"]
                   for sent in socket.__enter__.return_value.sendto.call_args_list]
        self.assertEqual(reasons[:2], ["waiting for the turret camera"] * 2)
        self.assertNotIn("waiting for the turret camera", reasons[2:])
        self.assertEqual(detector.detect.call_count, 2)  # frames 3 and 4 reached the AI
        self.assertGreaterEqual(arduino.safe_stop.call_count, 3)  # setup + home while waiting


if __name__ == "__main__":
    unittest.main()
