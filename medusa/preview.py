from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class PreviewState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.status = {"active": False, "reason": "starting"}

    def update_frame(self, frame) -> None:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.condition:
                self.jpeg = encoded.tobytes()
                self.condition.notify_all()

    def update_status(self, active: bool, reason: str) -> None:
        with self.condition:
            self.status = {"active": active, "reason": reason}


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
                try:
                    while True:
                        with state.condition:
                            state.condition.wait_for(lambda: state.jpeg is not None, timeout=1)
                            frame = state.jpeg
                        if frame is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.server = ThreadingHTTPServer(("", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

