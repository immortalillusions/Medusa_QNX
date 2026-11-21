from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .nose_offset import NoseOffset


class PreviewState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.frame_number = 0
        self.closed = False
        self.update_status(False, "starting")

    def update_frame(self, frame) -> None:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.condition:
                self.jpeg = encoded.tobytes()
                self.frame_number += 1
                self.condition.notify_all()

    def update_status(
        self, active: bool, reason: str, *, face_count: int | None = None,
        snooper_detected: bool = False, nose_offset: NoseOffset | None = None,
    ) -> None:
        with self.condition:
            self.status = {
                "active": active,
                "reason": reason,
                "face_count": face_count,
                "snooper_detected": snooper_detected,
                "nose_offset": asdict(nose_offset) if nose_offset is not None else None,
            }


class PreviewServer:
    """Small LAN-only MJPEG and JSON-status server for setup/debugging."""

    def __init__(self, port: int) -> None:
        self.state = PreviewState()
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                return

            def _headers(self, content_type: str) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Type", content_type)

            def do_GET(self) -> None:
                if self.path.startswith("/status.json"):
                    body = json.dumps(state.status).encode()
                    self.send_response(HTTPStatus.OK)
                    self._headers("application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if not self.path.startswith("/stream.mjpg"):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_response(HTTPStatus.OK)
                self._headers("multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_frame_number = 0
                try:
                    while True:
                        with state.condition:
                            if not state.condition.wait_for(
                                lambda: state.closed or state.frame_number != last_frame_number,
                                timeout=1,
                            ):
                                continue
                            if state.closed:
                                return
                            frame = state.jpeg
                            last_frame_number = state.frame_number
                        if frame is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame + b"\r\n")
                except ConnectionError:
                    return

        self.server = ThreadingHTTPServer(("", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        with self.state.condition:
            self.state.closed = True
            self.state.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()

