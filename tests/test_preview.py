import contextlib
import http.client
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from medusa.nose_offset import NoseOffset
from medusa.preview import PreviewServer, RuntimeSettings, parse_settings


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = RuntimeSettings()
        self.preview = PreviewServer(0, self.settings)
        self.preview.start()
        self.addCleanup(self.preview.stop)
        self.port = self.preview.server.server_port

    def post(self, path, body, port=None):
        with contextlib.closing(http.client.HTTPConnection("127.0.0.1", port or self.port, timeout=2)) as conn:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            conn.request("POST", path, data, {"Content-Type": "application/json"})
            with conn.getresponse() as response:
                return response.status, response.getheader("Access-Control-Allow-Origin"), response.read()

    def test_post_settings_only_queues_partial_updates_that_merge(self) -> None:
        status, cors, body = self.post("/settings", {"peeper_mode": "one"})
        self.assertEqual((status, cors, json.loads(body)), (202, "*", {"requested": {"peeper_mode": "one"}}))
        self.post("/settings", {"screen_switch": False})
        self.post("/settings", {"peeper_mode": "two"})
        self.assertEqual(self.settings.take(), {"peeper_mode": "two", "screen_switch": False})
        self.assertEqual(self.settings.take(), {})

    def test_bad_bodies_are_400_and_other_paths_404(self) -> None:
        for body in ({}, {"peeper_mode": "three"}, {"screen_switch": "yes"}, {"volume": 3}, [1], b"{nope", b""):
            with self.subTest(body=body):
                self.assertEqual(self.post("/settings", body)[:2], (400, "*"))
        self.assertEqual(self.post("/nope", {"screen_switch": True})[0], 404)
        self.assertEqual(self.settings.take(), {})

    def test_preflight_allows_json_posts(self) -> None:
        with contextlib.closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)) as conn:
            conn.request("OPTIONS", "/settings")
            with conn.getresponse() as response:
                self.assertEqual(response.status, 204)
                self.assertIn("POST", response.getheader("Access-Control-Allow-Methods"))
                self.assertIn("Content-Type", response.getheader("Access-Control-Allow-Headers"))

    def test_without_settings_the_endpoint_does_not_exist(self) -> None:
        plain = PreviewServer(0)
        plain.start()
        self.addCleanup(plain.stop)
        self.assertEqual(self.post("/settings", {"screen_switch": True}, plain.server.server_port)[0], 404)

    def test_parse_settings(self) -> None:
        self.assertEqual(parse_settings({"peeper_mode": "one", "screen_switch": True}),
                         {"peeper_mode": "one", "screen_switch": True})
        with self.assertRaises(ValueError):
            parse_settings({"screen_switch": 1})


class PreviewTests(unittest.TestCase):
    def test_stream_waits_for_the_next_frame_without_blocking_status(self) -> None:
        preview = PreviewServer(0)
        preview.start()
        self.addCleanup(preview.stop)
        port = preview.server.server_port
        fake_cv2 = SimpleNamespace(
            IMWRITE_JPEG_QUALITY=1,
            imencode=lambda extension, frame, options: (True, memoryview(frame)),
        )

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            preview.state.update_frame(b"first-frame")
            preview.state.update_status(
                False, "fewer than two screen-facing faces", face_count=3,
                snooper_count=2, nose_offset=NoseOffset(3, -4, 5),
            )

            with contextlib.closing(http.client.HTTPConnection("127.0.0.1", port, timeout=2)) as stream:
                stream.request("GET", "/stream.mjpg")
                with stream.getresponse() as response:
                    self.assertEqual(response.status, 200)
                    self._assert_frame(response, b"first-frame")

                    with contextlib.closing(http.client.HTTPConnection("127.0.0.1", port, timeout=2)) as status:
                        status.request("GET", "/status.json")
                        with status.getresponse() as status_response:
                            self.assertEqual(status_response.status, 200)
                            self.assertEqual(json.load(status_response), {
                                "active": False,
                                "reason": "fewer than two screen-facing faces",
                                "face_count": 3,
                                "snooper_detected": True,
                                "snooper_count": 2,
                                "nose_offset": {"dx_px": 3, "dy_px": -4, "distance_px": 5},
                            })

                    preview.state.update_frame(b"second-frame")
                    self._assert_frame(response, b"second-frame")

    def _assert_frame(self, response: http.client.HTTPResponse, frame: bytes) -> None:
        self.assertEqual(response.readline(), b"--frame\r\n")
        self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
        self.assertEqual(response.readline(), f"Content-Length: {len(frame)}\r\n".encode())
        self.assertEqual(response.readline(), b"\r\n")
        self.assertEqual(response.read(len(frame) + 2), frame + b"\r\n")
