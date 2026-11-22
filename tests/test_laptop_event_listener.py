import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import laptop_event_listener
import qnx_ai_service
from medusa.decision import SCENE_CLEAR_FRAMES
from medusa.windows_desktop import SnooperSignal


class LaptopListenerTests(unittest.TestCase):
    def test_desktop_mode_routes_only_confirmations_and_explicit_clear_from_the_selected_pi(self):
        pi = ("172.20.10.2", 57000)
        other = ("172.20.10.4", 57000)
        socket = MagicMock()
        connection = socket.__enter__.return_value
        connection.recvfrom.side_effect = [
            (b"[]", pi),
            (b'{"type":"face-detection","active":true}', pi),
            (b'{"type":"medusa-event","active":"true"}', pi),
            (b'{"type":"medusa-event","active":true}', other),
            (b'{"type":"medusa-event","active":true,"reason":"confirmed"}', pi),
            (b'{"type":"medusa-event","active":false,"reason":"still watching"}', pi),
            (b'{"type":"medusa-event","active":false,"scene_clear":true,"reason":"clear"}', pi),
            KeyboardInterrupt,
        ]
        with (
            patch.object(sys, "platform", "win32"),
            patch.object(laptop_event_listener.socket, "socket", return_value=socket),
            patch.object(laptop_event_listener, "DesktopSwitchTrigger") as trigger,
            patch.object(laptop_event_listener, "PrivacyAction") as browser,
        ):
            result = laptop_event_listener.main([
                "--switch-desktop", "--pi-ip", pi[0], "--taskbar-slot", "2",
            ])
        self.assertEqual(result, 0)
        connection.bind.assert_called_once_with(("", 8766))
        browser.assert_called_once_with(enabled=False)
        browser.return_value.update.assert_not_called()
        trigger.assert_called_once_with(2, clear_frames=1)
        self.assertEqual(trigger.return_value.update.call_args_list, [
            call(SnooperSignal.CONFIRMED, "confirmed"),
            call(SnooperSignal.UNCONFIRMED, "still watching"),
            call(SnooperSignal.CLEAR, "clear"),
        ])
        trigger.return_value.close.assert_called_once()

    def test_browser_is_the_default_for_existing_pi_events(self):
        socket = MagicMock()
        socket.__enter__.return_value.recvfrom.side_effect = [
            (b'{"type":"medusa-event","active":true}', ("172.20.10.2", 57000)),
            KeyboardInterrupt,
        ]
        with (
            patch.object(laptop_event_listener.socket, "socket", return_value=socket),
            patch.object(laptop_event_listener, "DesktopSwitchTrigger") as trigger,
            patch.object(laptop_event_listener, "PrivacyAction") as browser,
            patch("builtins.print"),
        ):
            self.assertEqual(laptop_event_listener.main([]), 0)
        trigger.assert_not_called()
        browser.assert_called_once_with(enabled=True)
        browser.return_value.update.assert_called_once_with(True)


class PiClearSignalTests(unittest.TestCase):
    def test_camera_dropout_resets_clear_streak_before_the_pi_sends_clear(self):
        camera = Mock()
        camera.isOpened.return_value = True
        clear_frame = (True, object())
        camera.read.side_effect = (
            [clear_frame] * (SCENE_CLEAR_FRAMES - 1) + [(False, None)]
            + [clear_frame] * SCENE_CLEAR_FRAMES + [KeyboardInterrupt]
        )
        cv2 = SimpleNamespace(VideoCapture=Mock(return_value=camera), error=RuntimeError)
        detector = Mock()
        detector.face_detector.detect.return_value = []
        detector.detect.return_value = []
        socket = MagicMock()
        with (
            patch.dict(sys.modules, {"cv2": cv2}),
            patch.object(sys, "argv", [
                "qnx_ai_service.py", "--event-host", "172.20.10.3", "--preview-port", "0",
            ]),
            patch.object(qnx_ai_service, "TFLiteFaceGazeDetector", return_value=detector),
            patch.object(qnx_ai_service.socket, "socket", return_value=socket),
        ):
            self.assertEqual(qnx_ai_service.main(), 0)
        events = [
            json.loads(sent.args[0])
            for sent in socket.__enter__.return_value.sendto.call_args_list
        ]
        self.assertTrue(all(not event["active"] for event in events))
        self.assertTrue(all(not event["scene_clear"] for event in events[:-2]))
        self.assertTrue(events[-2]["scene_clear"])
        self.assertFalse(events[-1]["scene_clear"])
