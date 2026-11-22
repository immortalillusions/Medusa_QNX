import ctypes
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from medusa.decision import SnooperSelector
from medusa.windows_desktop import DesktopSwitchTrigger, SnooperSignal, send_taskbar_shortcut
from qnx import face_webcam


class DesktopSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.action = Mock()
        self.trigger = DesktopSwitchTrigger(3, clear_frames=3, switch_action=self.action)
        self.addCleanup(self.trigger.close)

    def update(self, signal: SnooperSignal):
        return self.trigger.update(signal, "a test scene")

    def confirm(self) -> bool:
        attempt = self.update(SnooperSignal.CONFIRMED)
        self.assertIsNotNone(attempt)
        return attempt.result(timeout=2)

    def test_first_confirmation_sends_configured_slot_immediately(self) -> None:
        self.assertIsNone(self.update(SnooperSignal.UNCONFIRMED))
        self.action.assert_not_called()
        with self.assertLogs("medusa.windows_desktop", level="INFO") as log:
            self.assertTrue(self.confirm())
        self.action.assert_called_once_with(3)
        self.assertIn("because a test scene", "\n".join(log.output))
        self.assertIn("Foreground focus was not verified", "\n".join(log.output))

    def test_sustained_confirmation_does_not_switch_again(self) -> None:
        self.assertTrue(self.confirm())
        for _ in range(4):
            self.assertIsNone(self.update(SnooperSignal.CONFIRMED))
        self.action.assert_called_once_with(3)

    def test_brief_clear_dropout_does_not_rearm(self) -> None:
        self.assertTrue(self.confirm())
        for _ in range(2):
            self.update(SnooperSignal.CLEAR)
        self.assertIsNone(self.update(SnooperSignal.CONFIRMED))
        self.action.assert_called_once_with(3)

    def test_consecutive_clear_frames_rearm_for_a_new_encounter(self) -> None:
        self.assertTrue(self.confirm())
        for _ in range(2):
            self.update(SnooperSignal.CLEAR)
        self.update(SnooperSignal.UNCONFIRMED)
        self.update(SnooperSignal.CLEAR)
        self.assertIsNone(self.update(SnooperSignal.CONFIRMED))
        for _ in range(3):
            self.update(SnooperSignal.CLEAR)
        self.assertTrue(self.confirm())
        self.assertEqual(self.action.call_count, 2)

    def test_failed_switch_stays_latched_and_later_encounter_can_switch(self) -> None:
        self.action.side_effect = [OSError("input blocked"), None]
        with self.assertLogs("medusa.windows_desktop", level="ERROR") as log:
            self.assertFalse(self.confirm())
        self.assertIn("input blocked", "\n".join(log.output))
        self.assertIsNone(self.update(SnooperSignal.CONFIRMED))
        for _ in range(3):
            self.update(SnooperSignal.CLEAR)
        self.assertTrue(self.confirm())
        self.assertEqual(self.action.call_count, 2)

    def test_slow_switch_does_not_block_detection_updates(self) -> None:
        started = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def slow_switch(slot):
            started.set()
            release.wait(timeout=2)

        self.action.side_effect = slow_switch
        attempt = self.update(SnooperSignal.CONFIRMED)
        self.assertIsNotNone(attempt)
        self.assertTrue(started.wait(timeout=2))
        self.assertFalse(attempt.done())
        self.assertIsNone(self.update(SnooperSignal.CONFIRMED))
        release.set()
        self.assertTrue(attempt.result(timeout=2))


class WindowsShortcutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.user32 = Mock()
        self.user32.GetAsyncKeyState.return_value = 0
        self.user32.SendInput.side_effect = lambda count, inputs, size: count
        for patcher in (
            patch.object(sys, "platform", "win32"),
            patch.object(ctypes, "WinDLL", return_value=self.user32, create=True),
            patch.object(ctypes, "set_last_error", create=True),
            patch.object(ctypes, "get_last_error", return_value=5, create=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def events(call):
        _, inputs, _ = call.args
        return [(entry.type, entry.data.ki.wVk, entry.data.ki.dwFlags) for entry in inputs]

    def test_shortcut_uses_tenth_slot_digit_and_balanced_key_events(self) -> None:
        send_taskbar_shortcut(10)
        self.user32.SendInput.assert_called_once()
        self.assertEqual(self.events(self.user32.SendInput.call_args), [
            (1, 0x5B, 1), (1, 0xA2, 0), (1, ord("0"), 0),
            (1, ord("0"), 2), (1, 0xA2, 2), (1, 0x5B, 3),
        ])
        expected_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        self.assertEqual(self.user32.SendInput.call_args.args[2], expected_size)

    def test_partial_send_releases_only_keys_that_were_pressed(self) -> None:
        self.user32.SendInput.side_effect = [2, 2]
        with self.assertRaisesRegex(RuntimeError, "accepted 2/6"):
            send_taskbar_shortcut(1)
        self.assertEqual(self.events(self.user32.SendInput.call_args), [
            (1, 0xA2, 2), (1, 0x5B, 3),
        ])

    def test_held_shift_prevents_sending_a_different_shortcut(self) -> None:
        self.user32.GetAsyncKeyState.side_effect = lambda key: -32768 if key == 0x10 else 0
        with self.assertRaisesRegex(RuntimeError, "Shift"):
            send_taskbar_shortcut(1)
        self.user32.SendInput.assert_not_called()


def detection(origin_x: int, width: int, *, score: float = 0.9, points=()):
    return SimpleNamespace(
        bounding_box=SimpleNamespace(origin_x=origin_x, origin_y=60, width=width, height=width),
        categories=[SimpleNamespace(score=score)],
        keypoints=[SimpleNamespace(x=x) for x in points],
    )


class WebcamSwitchTests(unittest.TestCase):
    def test_raw_faces_need_valid_landmarks_and_sustained_confirmation(self) -> None:
        selector = SnooperSelector(hold_frames=2)
        front = detection(200, 240, points=(0.22, 0.28, 0.25))
        raw_face = detection(840, 100)
        turned_face = detection(840, 100, points=(0.68, 0.71, 0.73))
        snooper = detection(840, 100, points=(0.68, 0.71, 0.695))
        scenes = [
            ([front], SnooperSignal.CLEAR),
            ([front, raw_face], SnooperSignal.UNCONFIRMED),
            ([front, turned_face], SnooperSignal.UNCONFIRMED),
            ([front, snooper], SnooperSignal.UNCONFIRMED),
            ([front, snooper], SnooperSignal.CONFIRMED),
        ]
        for faces, expected in scenes:
            signal, _ = face_webcam.confirmed_snooper_signal(selector, faces, 1280, 720)
            self.assertIs(signal, expected)

    def test_camera_free_mode_uses_the_same_trigger_and_default_slot(self) -> None:
        action = Mock()
        with (
            patch.dict(sys.modules, {"cv2": None, "mediapipe": None}),
            patch.object(face_webcam.time, "sleep") as countdown,
            patch.object(face_webcam, "DesktopSwitchTrigger",
                         side_effect=lambda slot: DesktopSwitchTrigger(slot, switch_action=action)),
        ):
            self.assertEqual(face_webcam.main(["--test-switch"]), 0)
        countdown.assert_called_once_with(15)
        action.assert_called_once_with(1)

    def test_existing_camera_command_runs_with_switching_disabled(self) -> None:
        cv2 = Mock()
        cv2.VideoCapture.return_value.isOpened.return_value = True
        cv2.VideoCapture.return_value.read.return_value = (True, object())
        cv2.waitKey.return_value = ord("q")
        mediapipe = MagicMock()
        with (
            patch.dict(sys.modules, {"cv2": cv2, "mediapipe": mediapipe}),
            patch.object(face_webcam, "detect_faces", return_value=[]),
            patch.object(face_webcam, "draw_faces"),
            patch.object(face_webcam, "DesktopSwitchTrigger") as trigger,
        ):
            self.assertEqual(face_webcam.main(["--camera", "0"]), 0)
        cv2.VideoCapture.assert_called_once_with(0, cv2.CAP_DSHOW)
        cv2.VideoCapture.return_value.release.assert_called_once()
        trigger.assert_not_called()

    def test_live_opt_in_keeps_reading_after_a_failed_switch(self) -> None:
        cv2 = Mock()
        frame = SimpleNamespace(shape=(720, 1280, 3))
        cv2.VideoCapture.return_value.isOpened.return_value = True
        cv2.VideoCapture.return_value.read.return_value = (True, frame)
        front = detection(200, 240, points=(0.22, 0.28, 0.25))
        snooper = detection(840, 100, points=(0.68, 0.71, 0.695))
        scenes = [[front], [front, detection(840, 100)]] + [[front, snooper]] * 17
        cv2.waitKey.side_effect = [-1] * (len(scenes) - 1) + [ord("q")]
        action = Mock(side_effect=OSError("input blocked"))
        with (
            patch.dict(sys.modules, {"cv2": cv2, "mediapipe": MagicMock()}),
            patch.object(face_webcam, "detect_faces", side_effect=scenes),
            patch.object(face_webcam, "draw_faces"),
            patch.object(face_webcam, "DesktopSwitchTrigger",
                         side_effect=lambda slot: DesktopSwitchTrigger(slot, switch_action=action)),
            self.assertLogs("medusa.windows_desktop", level="ERROR"),
        ):
            self.assertEqual(face_webcam.main(["--camera", "0", "--switch-desktop", "--taskbar-slot", "4"]), 0)
        action.assert_called_once_with(4)
        self.assertEqual(cv2.imshow.call_count, len(scenes))
