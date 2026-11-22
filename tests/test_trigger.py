import json
import math
import os
import random
import re
import socket as socket_module
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import qnx_ai_service
from medusa import trigger as trigger_module
from medusa.arduino_serial import ArduinoController
from medusa.preview import PreviewState
from medusa.tflite_detector import FaceBox
from medusa.trigger import (
    NO_FACE_REASON, FirmwareAxisModel, LaptopTrigger, PanTracker, decide_aim, draw_aim_boxes, parse_peeper_message,
    pick_aim_target,
)

LAPTOP = ("172.20.10.4", 50000)
OTHER = ("172.20.10.9", 50000)


def packet(active=True, **fields):
    return json.dumps({"type": "medusa-peeper", "active": active, "target_x": 0.4,
                       "target_y": 0.5, "seq": 1, **fields}).encode()


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


class TriggerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        patcher = patch.object(trigger_module, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)


class ParseTests(unittest.TestCase):
    def test_valid_message(self) -> None:
        message = parse_peeper_message(packet(seq=7))
        self.assertEqual((message.active, message.target_x, message.target_y, message.seq),
                         (True, 0.4, 0.5, 7))

    def test_null_targets_are_allowed(self) -> None:
        message = parse_peeper_message(packet(False, target_x=None, target_y=None))
        self.assertEqual((message.active, message.target_x, message.target_y), (False, None, None))

    def test_target_size_is_optional(self) -> None:
        self.assertAlmostEqual(parse_peeper_message(packet(target_size=0.07)).target_size, 0.07)
        self.assertIsNone(parse_peeper_message(packet()).target_size)
        self.assertIsNone(parse_peeper_message(packet(target_size="big")).target_size)

    def test_garbage_is_ignored(self) -> None:
        for payload in (
            b"", b"\xff\xfe", b"not json", b"[]", b"7",
            b'{"type":"medusa-event","active":true}',
            b'{"type":"medusa-peeper"}',
            b'{"type":"medusa-peeper","active":"true"}',
            b'{"type":"medusa-peeper","active":1}',
        ):
            with self.subTest(payload=payload):
                self.assertIsNone(parse_peeper_message(payload))


class LaptopTriggerTests(TriggerTestCase):
    def test_accepts_valid_and_ignores_garbage_without_overwriting(self) -> None:
        trigger = LaptopTrigger(Mock())
        self.assertEqual(trigger.status(), (False, "waiting for laptop trigger"))
        self.assertIsNone(trigger.age())
        self.assertTrue(trigger.handle(packet(True), LAPTOP))
        self.assertFalse(trigger.handle(b"garbage", LAPTOP))
        self.assertFalse(trigger.handle(b'{"type":"medusa-peeper","active":"no"}', LAPTOP))
        self.assertEqual(trigger.status(), (True, ""))
        self.assertTrue(trigger.handle(packet(False), LAPTOP))
        self.assertEqual(trigger.status(), (False, "laptop sees no peeper"))

    def test_goes_inactive_after_timeout(self) -> None:
        trigger = LaptopTrigger(Mock(), timeout=0.5)
        trigger.handle(packet(True), LAPTOP)
        self.clock.now += 0.5
        self.assertEqual(trigger.status(), (True, ""))
        self.assertAlmostEqual(trigger.age(), 0.5)
        self.clock.now += 0.01
        self.assertEqual(trigger.status(), (False, "laptop trigger stale"))

    def test_sender_filter(self) -> None:
        trigger = LaptopTrigger(Mock(), allowed_sender=LAPTOP[0])
        self.assertFalse(trigger.handle(packet(True), OTHER))
        self.assertEqual(trigger.status(), (False, "waiting for laptop trigger"))
        self.assertTrue(trigger.handle(packet(True), LAPTOP))
        self.assertEqual(trigger.status(), (True, ""))

    def test_poll_drains_socket_and_keeps_latest(self) -> None:
        sock = Mock()
        sock.recvfrom.side_effect = [(packet(False), LAPTOP), (packet(True), LAPTOP),
                                     (b"junk", LAPTOP), BlockingIOError]
        trigger = LaptopTrigger(sock)
        trigger.poll()
        self.assertEqual(sock.recvfrom.call_count, 4)
        self.assertEqual(trigger.status(), (True, ""))

    def test_bind_is_non_blocking_on_all_interfaces(self) -> None:
        sock = Mock()
        with patch.object(trigger_module.socket, "socket", return_value=sock) as factory:
            trigger = LaptopTrigger.bind(8767, timeout=0.25, allowed_sender="172.20.10.4")
        factory.assert_called_once_with(socket_module.AF_INET, socket_module.SOCK_DGRAM)
        sock.setblocking.assert_called_once_with(False)
        sock.bind.assert_called_once_with(("0.0.0.0", 8767))
        self.assertEqual((trigger.timeout, trigger.allowed_sender), (0.25, "172.20.10.4"))

    def test_real_socket_round_trip(self) -> None:
        trigger = LaptopTrigger.bind(0, host="127.0.0.1")
        self.addCleanup(trigger.close)
        trigger.poll()  # Nothing waiting: must return at once.
        with socket_module.socket(socket_module.AF_INET, socket_module.SOCK_DGRAM) as sender:
            sender.sendto(packet(True), trigger.sock.getsockname())
        for _ in range(100):
            trigger.poll()
            if trigger.latest is not None:
                break
            __import__("time").sleep(0.01)
        self.assertEqual(trigger.status(), (True, ""))


class PickAimTargetTests(unittest.TestCase):
    def test_no_faces(self) -> None:
        self.assertIsNone(pick_aim_target([], 640, 480))

    def test_only_face_is_normalized(self) -> None:
        face = FaceBox(100, 40, 200, 160, 0.9)
        self.assertEqual(pick_aim_target([face], 640, 480), (face, 150 / 640, 100 / 480))

    def test_smallest_of_several_faces(self) -> None:
        big = FaceBox(0, 0, 200, 200, 0.99)
        small = FaceBox(500, 300, 540, 340, 0.8)
        medium = FaceBox(300, 100, 400, 200, 0.9)
        box, x, y = pick_aim_target([big, small, medium], 640, 480)
        self.assertIs(box, small)
        self.assertEqual((x, y), (520 / 640, 320 / 480))


class DecideAimTests(unittest.TestCase):
    face = FaceBox(100, 40, 200, 160, 0.9)

    def test_active_only_with_laptop_and_face(self) -> None:
        aim = decide_aim(True, "", [self.face], 640, 480)
        self.assertTrue(aim.active)
        self.assertIs(aim.target_box, self.face)
        self.assertEqual(decide_aim(True, "", [], 640, 480).reason, NO_FACE_REASON)
        self.assertFalse(decide_aim(True, "", [], 640, 480).active)
        idle = decide_aim(False, "laptop trigger stale", [self.face], 640, 480)
        self.assertEqual((idle.active, idle.target_x, idle.reason, idle.target_box),
                         (False, None, "laptop trigger stale", None))
        self.assertEqual(decide_aim(True, "", None, 640, 480).reason, "AI inference unavailable")


try:
    import cv2 as _cv2
    import numpy as _np
except ImportError:  # the Pi's test runs may lack OpenCV
    _cv2 = _np = None


def _pixel(frame, x, y):
    return tuple(int(v) for v in frame[y, x])


def _near(pixel, colour, tolerance=40):  # anti-aliased 1 px lines blend toward the background
    return all(abs(a - b) <= tolerance for a, b in zip(pixel, colour)) and any(pixel)


@unittest.skipIf(_cv2 is None, "needs OpenCV and numpy")
class DrawAimBoxesTests(unittest.TestCase):
    def test_target_amber_with_aim_tag_others_blue_grey_and_a_centre_crosshair(self) -> None:
        from medusa.overlay import AMBER, BLUE_GREY, INK, TEXT
        for height, width in ((480, 640), (1080, 1920)):
            with self.subTest(size=(width, height)):
                frame = _np.zeros((height, width, 3), _np.uint8)
                s = height / 480
                target = FaceBox(int(100 * s), int(150 * s), int(200 * s), int(250 * s), 0.9)
                other = FaceBox(int(400 * s), int(150 * s), int(500 * s), int(250 * s), 0.9)
                draw_aim_boxes(frame, [target, other], target)
                self.assertEqual(_pixel(frame, int(150 * s), int(249 * s)), AMBER)  # target's bottom edge
                self.assertEqual(_pixel(frame, int(450 * s), int(249 * s)), BLUE_GREY)
                self.assertTrue(_near(_pixel(frame, width // 2 + int(8 * s), height // 2), BLUE_GREY))  # cross arm
                self.assertEqual(_pixel(frame, width // 2, height // 2), (0, 0, 0))  # open centre
                tag = frame[int(150 * s) - int(30 * s):int(150 * s) - 2, int(100 * s):int(160 * s)]
                colours = {tuple(int(v) for v in px) for row in tag for px in row}
                self.assertTrue({INK, TEXT, AMBER} <= colours)  # ink box, off-white text, amber bar
                self.assertNotIn((0, 0, 255), colours)

    def test_label_stays_inside_the_frame_at_the_top_edge(self) -> None:
        from medusa.overlay import INK
        frame = _np.zeros((480, 640, 3), _np.uint8)
        target = FaceBox(600, 0, 639, 40, 0.9)
        draw_aim_boxes(frame, [target], target)
        self.assertEqual(_pixel(frame, 636, 2), INK)  # pushed left and down, still drawn


class PreviewStatusTests(unittest.TestCase):
    def test_extra_fields_extend_status_and_default_is_unchanged(self) -> None:
        state = PreviewState()
        state.update_status(False, "x", face_count=1)
        self.assertEqual(set(state.status), {"active", "reason", "face_count", "snooper_detected",
                                             "snooper_count", "nose_offset"})
        state.update_status(True, "y", extra={"mode": "laptop-trigger", "aim_target": [0.1, 0.2]})
        self.assertEqual(state.status["mode"], "laptop-trigger")
        self.assertEqual(state.status["aim_target"], [0.1, 0.2])
        self.assertTrue(state.status["active"])


class ServiceTriggerFlagTests(unittest.TestCase):
    def test_trigger_flags_default_off(self) -> None:
        args = qnx_ai_service.parse_args(["--event-host", "127.0.0.1"])
        self.assertIsNone(args.trigger_udp_port)
        self.assertIsNone(args.trigger_from)
        self.assertEqual(args.trigger_timeout, 0.5)

    def test_event_host_required_only_without_trigger(self) -> None:
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            qnx_ai_service.parse_args([])
        args = qnx_ai_service.parse_args(["--trigger-udp-port", "8767", "--trigger-from", "172.20.10.4"])
        self.assertEqual((args.trigger_udp_port, args.trigger_from, args.event_host),
                         (8767, "172.20.10.4", None))


class TriggerModeServiceTests(TriggerTestCase):
    def test_main_aims_only_with_fresh_laptop_trigger_and_a_face_and_sends_no_events(self) -> None:
        far, near = FaceBox(500, 300, 540, 340, 0.8), FaceBox(0, 0, 200, 200, 0.99)
        # Each frame: (packets arriving before that frame's poll, faces seen by the turret camera).
        script = [
            ([], [far, near]),                               # 0: no laptop message yet
            ([(packet(True), LAPTOP)], [far, near]),         # 1: fresh trigger, 2 faces -> aim at far
            ([], [near]),                                    # 2: still fresh, one face -> aim at it
            ([(packet(True), OTHER)], [near]),               # 3: wrong sender ignored; 0.2 s old, fresh
            ([], [near]),                                    # 4: 0.6 s old -> stale
            ([(b"garbage", LAPTOP), (packet(True), LAPTOP)], []),  # 5: fresh but turret sees no face
            ([(packet(False), LAPTOP)], [far]),              # 6: laptop says no peeper
            ([(packet(True), LAPTOP)], [far]),               # 7: active again
        ]
        steps = [0.0, 0.0, 0.1, 0.1, 0.4, 0.0, 0.0, 0.0]
        frame = SimpleNamespace(shape=(480, 640, 3))
        pending = []
        index = {"i": -1}

        def read():
            index["i"] += 1
            if index["i"] == len(script):
                raise KeyboardInterrupt
            self.clock.now += steps[index["i"]]
            pending.extend(script[index["i"]][0])
            return True, frame

        def recvfrom(_size):
            if pending:
                return pending.pop(0)
            raise BlockingIOError

        def detect(_frame):
            return script[index["i"]][1]

        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = read
        cv2 = SimpleNamespace(VideoCapture=Mock(return_value=camera), error=RuntimeError)
        detector = Mock()
        detector.face_detector.detect.side_effect = detect
        trigger_sock = MagicMock()
        trigger_sock.recvfrom.side_effect = recvfrom
        with (
            patch.dict(sys.modules, {"cv2": cv2}),
            patch.object(sys, "argv", [
                "qnx_ai_service.py", "--preview-port", "0", "--arduino-port", "/dev/ser1",
                "--trigger-udp-port", "8767", "--trigger-from", LAPTOP[0],
                "--face-confirm-frames", "1", "--face-dropout-frames", "0",  # Debounce: FaceConfirmTests.
            ]),
            patch.object(qnx_ai_service, "TFLiteFaceGazeDetector", return_value=detector),
            patch.object(qnx_ai_service, "ArduinoController") as arduino_class,
            patch.object(qnx_ai_service.socket, "socket", return_value=trigger_sock) as socket_factory,
        ):
            self.assertEqual(qnx_ai_service.main(), 0)

        # The only socket is the trigger listener: no medusa-event socket, no sendto at all.
        socket_factory.assert_called_once()
        trigger_sock.bind.assert_called_once_with(("0.0.0.0", 8767))
        trigger_sock.setblocking.assert_called_once_with(False)
        trigger_sock.sendto.assert_not_called()
        trigger_sock.close.assert_called_once()
        detector.detect.assert_not_called()  # Box detector only; no screen-facing requirement.

        arduino = arduino_class.return_value
        self.assertEqual(arduino.send.call_args_list, [
            call(False, None, None),
            call(True, 520 / 640, 320 / 480),
            call(True, 100 / 640, 100 / 480),
            call(True, 100 / 640, 100 / 480),
            call(False, None, None),
            call(False, None, None),
            call(False, None, None),
            call(True, 520 / 640, 320 / 480),
        ])
        self.assertEqual(arduino.safe_stop.call_count, 2)  # At startup and at shutdown.
        arduino.close.assert_called_once()
        camera.release.assert_called_once()



INO = Path(__file__).resolve().parent.parent / "arduino" / "medusa_controller" / "medusa_controller.ino"
FPS = 15.0


def simulate(pan, face_at, seconds, *, latency=1 / FPS, noise=0.0, visible=lambda t: True, seed=1, fps=FPS):
    """Closed loop: the camera rides on a simulated turret that eases exactly like the firmware.

    Each frame was captured ``latency`` before the tracker sees it, and the face's
    image x comes from the simulated turret's actual pan at capture time. Returns
    (samples, frames): the aim error face - actual pan every 5 ms firmware step, and
    per frame (time, state, commanded angle, angle sent to the firmware: whole degrees in
    predictive mode, float in proportional mode).
    """
    rng = random.Random(seed)
    turret = FirmwareAxisModel(pan.home, pan.minimum, pan.maximum)
    turret.advance(0.0)
    samples, frames, now = [], [], 0.0
    while now < seconds:
        last = turret.time
        turret.advance(now)
        samples.extend((t, face_at(t) - p) for t, p in turret.history if t > last)
        captured = now - latency
        x = 0.5 + pan.sign * (face_at(captured) - turret.position_at(captured)) / pan.hfov_deg
        x += rng.uniform(-noise, noise)
        pan.update(x if visible(captured) and 0 <= x <= 1 else None, now=now)
        turret.target = pan.command
        frames.append((now, pan.state, pan.angle, turret.target))
        now += 1 / fps + rng.uniform(-0.004, 0.004)  # The Pi's frame interval jitters a little.
    return samples, frames


def sweep(start, end, speed, begin):
    """A face that waits at ``start``, moves at ``speed`` deg/s, then stops dead at ``end``."""
    def face_at(t):
        moved = max(0.0, t - begin) * speed
        return min(end, start + moved) if end > start else max(end, start - moved)
    return face_at, begin + abs(end - start) / speed


def rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


class FirmwareAxisModelTests(unittest.TestCase):
    def test_constants_match_the_firmware(self) -> None:
        source = INO.read_text()

        def constant(name):
            return float(re.search(rf"const \w+(?: \w+)? {name} = ([0-9.]+);", source).group(1))

        self.assertEqual(constant("MAX_SPEED"), trigger_module.FIRMWARE_MAX_SPEED)
        self.assertEqual(constant("MAX_ACCEL"), trigger_module.FIRMWARE_MAX_ACCEL)
        self.assertEqual(constant("MOTION_STEP_MS") / 1000, trigger_module.FIRMWARE_STEP_S)

    def test_eases_with_the_speed_and_acceleration_caps_and_lands_on_target(self) -> None:
        axis = FirmwareAxisModel(65, 65, 115)
        axis.advance(0.0)
        axis.target = 115.0
        axis.advance(0.05)  # 10 steps of 5 ms at 900 deg/s^2.
        self.assertAlmostEqual(axis.velocity, 45.0)
        positions = []
        for i in range(1, 100):
            axis.advance(0.05 + i * 0.005)
            positions.append(axis.position)
            self.assertLessEqual(abs(axis.velocity), 180.0)
        self.assertEqual(positions, sorted(positions))  # No overshoot and bounce back.
        self.assertEqual((axis.position, axis.velocity), (115.0, 0.0))
        self.assertAlmostEqual(axis.position_at(0.0), 65.0)
        mid = axis.position_at(0.0525)  # Between the 0.050 and 0.055 steps.
        self.assertGreater(mid, axis.position_at(0.05))
        self.assertLess(mid, axis.position_at(0.055))

    def test_a_long_gap_ends_at_the_target(self) -> None:
        axis = FirmwareAxisModel(90, 65, 115)
        axis.advance(0.0)
        axis.target = 70.0
        axis.advance(5.0)
        self.assertEqual((axis.position, axis.velocity), (70.0, 0.0))


class PanTrackerTests(unittest.TestCase):
    def test_first_sight_aims_at_the_bearing_and_reversed_flips_direction(self) -> None:
        normal, flipped = PanTracker(65, 115, 90), PanTracker(65, 115, 90, reversed=True)
        self.assertAlmostEqual(normal.update(0.7, now=0.0), 0.2)
        flipped.update(0.7, now=0.0)
        self.assertAlmostEqual(normal.angle, 100.0)  # 90 + 0.2 * 50.
        self.assertAlmostEqual(flipped.angle, 80.0)  # Reversed: a face right of centre lowers yaw.
        self.assertEqual((normal.state, normal.bearing, normal.velocity), ("tracking", 100.0, 0.0))

    def test_bearing_uses_where_the_turret_actually_points_not_the_command(self) -> None:
        pan = PanTracker(65, 115, 90, latency_s=0.0)
        pan.update(0.7, now=0.0)  # Commands 100; the turret has not moved yet.
        pan.servo.advance(0.03)  # Where the eased turret has got to 30 ms later.
        pan.update(0.5 + (100 - pan.servo.position) / 50, now=0.03)
        self.assertLess(pan.servo.position, 100.0)  # Still easing toward 100 ...
        self.assertAlmostEqual(pan.bearing, 100.0, places=6)  # ... yet the face's bearing is exact.

    def test_angle_is_clamped_to_servo_limits(self) -> None:
        right, left = PanTracker(65, 115, 90), PanTracker(65, 115, 90, reversed=True)
        right.update(1.0, now=0.0)
        left.update(1.0, now=0.0)
        self.assertEqual((right.angle, left.angle), (115.0, 65.0))
        self.assertEqual((right.servo.target, left.servo.target), (115.0, 65.0))

    def test_an_absurd_jump_is_a_new_target(self) -> None:
        pan = PanTracker(0, 180, 90)
        face, _ = sweep(90, 180, 60, 0.0)
        simulate(pan, face, 0.8)
        self.assertGreater(pan.velocity, 40)
        bearing = pan.servo.position_at(pan._last_now + 0.0667 - pan.latency_s)
        pan.update(0.5 - 30 / 50, now=pan._last_now + 0.0667)  # 30 deg left of the camera axis.
        self.assertEqual(pan.velocity, 0.0)
        self.assertAlmostEqual(pan.bearing, bearing - 30, delta=0.01)
        self.assertAlmostEqual(pan.angle, pan.bearing)

    def test_tracks_moving_faces_closely(self) -> None:
        # Wide limits so there is a steady state to measure; the pan stage itself is 65..115.
        # 100 deg/s: the gentler firmware (900 deg/s^2) takes longer to reach a fast face.
        for speed, rms_limit, max_limit in ((60, 2.0, 4.0), (100, 3.5, 7.5)):
            for reversed_ in (False, True):
                with self.subTest(speed=speed, reversed=reversed_):
                    start, end = (160, 30) if reversed_ else (20, 150)
                    pan = PanTracker(0, 180, start, reversed=reversed_)
                    face, stop = sweep(start, end, speed, 0.5)
                    samples, frames = simulate(pan, face, stop + 1.5, noise=0.005)
                    steady = [e for t, e in samples if 0.9 <= t <= stop]
                    self.assertLessEqual(rms(steady), rms_limit)
                    self.assertLessEqual(max(abs(e) for e in steady), max_limit)
                    self.assertTrue(all(state == "tracking" for _, state, _, _ in frames))

    def test_a_stopping_face_overshoots_once_then_settles_still(self) -> None:
        for speed, overshoot_limit in ((60, 10.0), (150, 22.0)):
            with self.subTest(speed=speed):
                pan = PanTracker(0, 180, 20)
                face, stop = sweep(20, 130, speed, 0.5)
                samples, frames = simulate(pan, face, stop + 2.0, noise=0.005)
                past = [(t, -e) for t, e in samples if t > stop]  # Positive: turret beyond the face.
                peak_t, peak = max(past, key=lambda s: s[1])
                # One frame of latency means the turret learns of the stop late; the bound is that.
                self.assertLessEqual(peak, overshoot_limit)
                self.assertGreaterEqual(min(o for t, o in past if t > peak_t), -1.0)  # No ringing back.
                self.assertLessEqual(max(abs(o) for t, o in past if t > stop + 0.7), 1.0)
                settled = {sent for t, _, _, sent in frames if t > stop + 1.0}
                self.assertEqual(len(settled), 1)  # Still, within the 1-degree start hysteresis.
                self.assertLessEqual(abs(settled.pop() - 130.0), 1.0)

    def test_a_still_face_gives_a_perfectly_still_command(self) -> None:
        pan = PanTracker(65, 115, 90)
        samples, frames = simulate(pan, lambda t: 97.3, 6.0, noise=0.005)
        settled = [(angle, sent) for t, _, angle, sent in frames if t > 1.0]
        self.assertEqual(len(set(settled)), 1)  # Not even the float command moves.
        self.assertEqual(settled[0][1], 97.0)
        self.assertLess(abs(pan.velocity), 1.0)

    def test_losing_the_face_coasts_then_glides_home_and_a_face_resumes_at_once(self) -> None:
        pan = PanTracker(0, 180, 20, lost_hold_s=0, return_speed=60)  # The old glide.
        face, _ = sweep(20, 180, 60, 0.3)
        lost_at = 1.3
        samples, frames = simulate(pan, face, 3.5, visible=lambda t: t < lost_at)
        states = [state for _, state, _, _ in frames]
        coast = [(t, angle) for t, state, angle, _ in frames if state == "coasting"]
        glide = [(t, angle) for t, state, angle, _ in frames if state == "returning"]
        self.assertEqual(states[-1], "home")
        self.assertEqual(pan.angle, 20.0)
        self.assertAlmostEqual(coast[-1][0] - coast[0][0], 0.3, delta=1 / FPS)
        angles = [angle for _, angle in coast]
        self.assertEqual(angles, sorted(angles))  # Keeps going the way the face went ...
        steps = [b - a for a, b in zip(angles, angles[1:])]
        self.assertLess(steps[-1], steps[0])  # ... slowing down, not stopping dead.
        for (t0, a0), (t1, a1) in zip(glide, glide[1:]):
            self.assertLess(a1, a0)
            self.assertLessEqual(a0 - a1, 60.0 * (t1 - t0) + 1e-6)  # Glides; never jumps home.
        # A face seen mid-glide is tracked on that very frame.
        pan = PanTracker(0, 180, 20, lost_hold_s=0, return_speed=60)
        simulate(pan, face, 2.0, visible=lambda t: t < lost_at)
        self.assertEqual(pan.state, "returning")
        before = pan.angle
        pan.update(0.6, now=pan._last_now + 0.0667)
        self.assertEqual(pan.state, "tracking")
        self.assertGreater(pan.angle, before)

    def test_hold_keeps_the_turret_still_before_gliding_home(self) -> None:
        pan = PanTracker(65, 115, 90, hold_s=1.0, lost_hold_s=0, return_speed=60)  # The old glide.
        pan.update(0.7, now=0.0)
        held = pan.angle
        pan.update(None, now=0.1)  # Lost here; nothing moves, so the coast changes nothing.
        pan.update(None, now=1.05)
        self.assertEqual((pan.state, pan.angle), ("coasting", held))
        pan.update(None, now=1.15)
        self.assertEqual(pan.state, "returning")
        self.assertAlmostEqual(pan.angle, held - 60 * 0.1)
        pan.reset()
        self.assertEqual((pan.angle, pan.state, pan.bearing, pan.servo.target), (90.0, "home", None, 90.0))


def reversals(values, threshold):
    """Direction changes whose swing back exceeds ``threshold`` degrees."""
    count, direction, extreme = 0, 0, values[0]
    for v in values[1:]:
        if direction >= 0 and v < extreme - threshold:
            count, direction, extreme = count + (direction > 0), -1, v
        elif direction <= 0 and v > extreme + threshold:
            count, direction, extreme = count + (direction < 0), 1, v
        elif (direction >= 0 and v > extreme) or (direction <= 0 and v < extreme):
            extreme = v
    return count


def bench_pan(**kwargs):
    """The bench setup: pan 55..125, reversed servo, proportional mode with its defaults."""
    return PanTracker(55, 125, 90, reversed=True, mode="proportional", **kwargs)


def step_response(latency, fps=FPS):
    """Face jumps 90 -> 75 at 0.5 s; returns (overshoot past 75, seconds until within 1 degree for good)."""
    samples, _ = simulate(bench_pan(), lambda t: 90.0 if t < 0.5 else 75.0, 4.0, latency=latency, fps=fps, seed=3)
    after = [(t, e) for t, e in samples if t >= 0.5]  # e = face - turret: positive once past 75.
    return max(0.0, max(e for _, e in after)), next(t for t, e in reversed(after) if abs(e) > 1.0) - 0.5


class ProportionalModeTests(unittest.TestCase):
    def test_soft_dead_zone_ramps_the_gain_from_zero_with_the_sign_kept(self) -> None:
        def first_step(error):
            pan = bench_pan()
            pan.update(0.5 + error, now=0.0)
            pan.update(0.5 + error, now=1 / 15)  # Same error: the average is exact.
            return pan.angle - 90.0

        self.assertEqual(first_step(0.011), 0.0)  # Inside the dead zone: no motion at all.
        full = 0.12 * 0.05 * 50  # Gain x error x hfov once past full_gain_error.
        self.assertAlmostEqual(first_step(0.05), -full)  # Reversed servo: right of centre lowers yaw.
        self.assertAlmostEqual(first_step(-0.05), full)
        ramp = [-first_step(e) / (0.12 * e * 50) for e in (0.015, 0.025, 0.035, 0.045)]
        self.assertEqual(ramp, sorted(ramp))  # The share of full gain rises smoothly ...
        self.assertTrue(0 < ramp[0] < 0.05 and 0.9 < ramp[-1] < 1)  # ... from near zero to near full.

    def test_still_face_with_noise_gives_a_still_command(self) -> None:
        for latency in (0.067, 0.15):
            with self.subTest(latency=latency):
                _, frames = simulate(bench_pan(), lambda t: 95.0, 5.0, latency=latency, noise=0.01)
                sent = [s for t, _, _, s in frames if t >= 2.0]  # 3 s after settling.
                self.assertLess(max(sent) - min(sent), 0.3)
                self.assertEqual(reversals(sent, 0.2), 0)
                self.assertLess(abs(sent[-1] - 95.0), 1.0)

    def test_slow_drift_gives_a_smooth_monotonic_command(self) -> None:
        face = lambda t: 80.0 + 5.0 * max(0.0, t - 1.0)  # noqa: E731 - 5 deg/s from 1 s.
        _, frames = simulate(bench_pan(), face, 5.0, latency=0.15, noise=0.003)
        sent = [s for t, _, _, s in frames if t >= 1.5]
        steps = [b - a for a, b in zip(sent, sent[1:])]
        self.assertGreaterEqual(min(steps), 0.0)  # Never steps back.
        self.assertLess(max(steps), 1.0)
        self.assertNotEqual({round(s, 1) for s in sent}, {round(s) for s in sent})  # Sub-degree steps.

    def test_a_step_settles_without_overshoot(self) -> None:
        for latency in (0.067, 0.15):
            with self.subTest(latency=latency):
                overshoot, settle = step_response(latency)
                self.assertLessEqual(overshoot, 1.0)
                self.assertLess(settle, 1.5)

    def test_settle_time_does_not_depend_on_the_frame_rate(self) -> None:
        for latency in (0.067, 0.15):
            with self.subTest(latency=latency):
                (over15, settle15), (over30, settle30) = step_response(latency, 15.0), step_response(latency, 30.0)
                self.assertLess(abs(settle15 - settle30), 0.25)
                self.assertLessEqual(max(over15, over30), 1.0)


class CameraOnTurretFlagTests(unittest.TestCase):
    base = ["--trigger-udp-port", "8767"]

    def test_defaults_and_off_by_default(self) -> None:
        args = qnx_ai_service.parse_args(self.base)
        self.assertFalse(args.camera_on_turret)
        self.assertEqual(
            (args.camera_hfov, args.camera_latency, args.pan_lead, args.pan_coast, args.pan_hold,
             args.pan_return_speed, args.pan_lost_hold),
            (50.0, 0.07, 0.15, 0.3, None, 30.0, 1.0))
        self.assertTrue(qnx_ai_service.parse_args(self.base + ["--camera-on-turret"]).camera_on_turret)
        pan = qnx_ai_service._pan_tracker(qnx_ai_service.parse_args(
            self.base + ["--camera-on-turret", "--servo-reversed", "--pan-lead", "0.1", "--pan-hold", "2"]))
        self.assertEqual((pan.sign, pan.lead_s, pan.coast_s, pan.hold_s, pan.return_speed, pan.lost_hold_s),
                         (-1.0, 0.1, 0.3, 2.0, 30.0, 1.0))

    def test_old_proportional_flags_are_accepted_but_ignored(self) -> None:
        argv = self.base + ["--camera-on-turret", "--pan-gain", "0", "--pan-deadband", "0.5",
                            "--pan-max-step", "6"]
        with patch("sys.stderr") as stderr:
            args = qnx_ai_service.parse_args(argv)
        warnings = "".join(c.args[0] for c in stderr.write.call_args_list)
        for flag in ("--pan-gain", "--pan-deadband", "--pan-max-step"):
            self.assertIn(f"warning: {flag} is deprecated and ignored", warnings)
        self.assertIsInstance(qnx_ai_service._pan_tracker(args), PanTracker)

    def test_requires_trigger_mode_and_sane_values(self) -> None:
        for argv in (
            ["--event-host", "127.0.0.1", "--camera-on-turret"],
            self.base + ["--camera-hfov", "-1"],
            self.base + ["--pan-return-speed", "0"],
            self.base + ["--pan-lead", "-0.1"],
            self.base + ["--pan-coast", "-0.1"],
            self.base + ["--camera-latency", "-0.1"],
            self.base + ["--pan-hold", "-0.1"],
        ):
            with self.subTest(argv=argv), patch("sys.stderr"), self.assertRaises(SystemExit):
                qnx_ai_service.parse_args(argv)

    def test_status_adds_pan_fields_only_in_this_mode(self) -> None:
        trigger = LaptopTrigger(Mock())
        self.assertNotIn("pan_angle", qnx_ai_service._trigger_status(trigger, False, None))
        pan = PanTracker(65, 115, 90)
        error = pan.update(0.61234, now=0.0)
        status = qnx_ai_service._trigger_status(trigger, True, None, pan, error)
        self.assertEqual(
            {k: status[k] for k in ("pan_angle", "pan_error", "pan_bearing", "pan_velocity", "pan_state")},
            {"pan_angle": 95.6, "pan_error": 0.1123, "pan_bearing": 95.6, "pan_velocity": 0.0,
             "pan_state": "tracking"})
        pan.update(None, now=0.1)
        status = qnx_ai_service._trigger_status(trigger, False, None, pan, None)
        self.assertEqual((status["pan_error"], status["pan_state"]), (None, "coasting"))
        pan.reset()
        status = qnx_ai_service._trigger_status(trigger, False, None, pan, None)
        self.assertEqual((status["pan_bearing"], status["pan_state"]), (None, "home"))


class CameraOnTurretServiceTests(TriggerTestCase):
    def test_main_closes_the_pan_loop_and_sends_exact_angles(self) -> None:
        def face(x, y=0.5):
            return FaceBox(640 * x - 20, 480 * y - 20, 640 * x + 20, 480 * y + 20, 0.9)

        # Each frame: (seconds since the previous frame, laptop packet before it?, faces).
        # The face stands still at yaw 75. Its image x is where a camera on the eased
        # turret sees it 70 ms before each frame, so x drifts while the bearing does not.
        script = [
            (0.0, False, [face(0.8)]),           # 0: no laptop trigger yet -> home, LED off
            (0.1, True, [face(0.8, 0.25)]),      # 1: first sight, reversed: 90 - 0.3 * 50 -> 75
            (0.1, True, [face(0.7905, 0.25)]),   # 2: turret at 89.5 when captured: still 75
            (0.1, True, [face(0.6420, 0.25)]),   # 3: turret at 82.1: still 75, no hunting
            (0.1, True, [face(0.5014, 0.25)]),   # 4: turret at 75.1, settled: still 75
            (0.1, True, []),                     # 5: face lost -> coast (no velocity), LED off
            (0.1, True, []),                     # 6: 0.1 s lost -> coasting
            (0.1, True, []),                     # 7: 0.2 s lost -> coasting
            (0.15, True, []),                    # 8: 0.35 s lost -> glide home at 60 deg/s: 84
            (0.1, True, []),                     # 9: home 90
        ]
        frame = SimpleNamespace(shape=(480, 640, 3))
        pending, index = [], {"i": -1}

        def read():
            index["i"] += 1
            if index["i"] == len(script):
                raise KeyboardInterrupt
            step, laptop, _faces = script[index["i"]]
            self.clock.now += step
            if laptop:
                pending.append((packet(True), LAPTOP))
            return True, frame

        def recvfrom(_size):
            if pending:
                return pending.pop(0)
            raise BlockingIOError

        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)

        def make_controller(_device, mapper, baud=115200, tilt_mapper=None):
            arduino = ArduinoController.__new__(ArduinoController)
            arduino.mapper, arduino.tilt_mapper, arduino.sequence, arduino.fd = mapper, tilt_mapper, 0, write_fd
            return arduino

        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = read
        cv2 = SimpleNamespace(VideoCapture=Mock(return_value=camera), error=RuntimeError)
        detector = Mock()
        detector.face_detector.detect.side_effect = lambda _frame: script[index["i"]][2]
        trigger_sock = MagicMock()
        trigger_sock.recvfrom.side_effect = recvfrom
        with (
            patch.dict(sys.modules, {"cv2": cv2}),
            patch.object(sys, "argv", [
                "qnx_ai_service.py", "--preview-port", "0", "--arduino-port", "/dev/ser1",
                "--servo-reversed", "--trigger-udp-port", "8767", "--trigger-from", LAPTOP[0],
                "--camera-on-turret", "--no-search", "--pan-mode", "predictive",  # Coast-then-home path; search: tests/test_control.py.
                "--pan-lost-hold", "0", "--pan-return-speed", "60",  # The old glide: no hold, 60 deg/s.
                "--face-confirm-frames", "1", "--face-dropout-frames", "0",  # Debounce: FaceConfirmTests.
            ]),
            patch.object(qnx_ai_service, "TFLiteFaceGazeDetector", return_value=detector),
            patch.object(qnx_ai_service, "ArduinoController", side_effect=make_controller),
            patch.object(qnx_ai_service.socket, "socket", return_value=trigger_sock),
        ):
            self.assertEqual(qnx_ai_service.main(), 0)

        sent = os.read(read_fd, 4096).decode("ascii").splitlines()
        # Pitch follows the unchanged smoothed tilt mapper: face y 0.25 -> 77.5, eased by 0.35.
        # send_angles writes N lines (tenths); predictive mode still commands whole degrees.
        self.assertEqual(sent, [
            "M,90,90,0,1",   # safe_stop at startup
            "N,900,900,0,2",
            "N,750,860,1,3",
            "N,750,830,1,4",
            "N,750,810,1,5",
            "N,750,800,1,6",
            "N,750,830,0,7",
            "N,750,860,0,8",
            "N,750,870,0,9",
            "N,840,880,0,10",
            "N,900,890,0,11",
            "M,90,90,0,12",  # safe_stop at shutdown: exactly home, not a smoothed step toward it
        ])
        camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
