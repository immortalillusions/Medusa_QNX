import contextlib
import http.client
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from medusa.nose_offset import NoseOffset
from medusa.preview import PreviewServer


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
