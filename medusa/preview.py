from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .nose_offset import NoseOffset


class PreviewState:
    def __init__(self, jpeg_quality: int = 75, max_width: int | None = None) -> None:
        self.jpeg_quality, self.max_width = jpeg_quality, max_width
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.frame_number = 0
        self.closed = False
        self.update_status(False, "starting")

    def update_frame(self, frame) -> None:
        import cv2

        if self.max_width and frame.shape[1] > self.max_width:
            # A smaller, lighter stream for a slow link (the Pi's preview over a phone hotspot).
            height = round(frame.shape[0] * self.max_width / frame.shape[1])
            frame = cv2.resize(frame, (self.max_width, height), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            with self.condition:
                self.jpeg = encoded.tobytes()
                self.frame_number += 1
                self.condition.notify_all()

    def update_status(
        self, active: bool, reason: str, *, face_count: int | None = None,
        snooper_count: int = 0, nose_offset: NoseOffset | None = None,
        extra: dict | None = None,
    ) -> None:
        """Replace the status; ``extra`` adds mode-specific fields after the standard ones."""
        status = {
            "active": active,
            "reason": reason,
            "face_count": face_count,
            "snooper_detected": snooper_count > 0,
            "snooper_count": snooper_count,
            "nose_offset": asdict(nose_offset) if nose_offset is not None else None,
        }
        if extra:
            status.update(extra)
        with self.condition:
            self.status = status


PEEPER_MODES = ("one", "two")
_MAX_SETTINGS_BODY = 4096


def parse_settings(payload) -> dict:
    """Validate a POST /settings body: {"peeper_mode": "one"|"two", "screen_switch": bool}, any subset.

    Raises ValueError on anything else (unknown keys, wrong types, empty)."""
    if not isinstance(payload, dict) or not payload:
        raise ValueError("expected a JSON object with peeper_mode and/or screen_switch")
    unknown = set(payload) - {"peeper_mode", "screen_switch"}
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(sorted(unknown))}")
    if "peeper_mode" in payload and payload["peeper_mode"] not in PEEPER_MODES:
        raise ValueError('peeper_mode must be "one" or "two"')
    if "screen_switch" in payload and not isinstance(payload["screen_switch"], bool):
        raise ValueError("screen_switch must be true or false")
    return dict(payload)


class RuntimeSettings:
    """Settings changed from the dashboard, handed from the HTTP thread to the detector loop.

    The HTTP thread only records a request under the lock; the detector takes it on its next
    frame and applies it. Runtime only: a restart goes back to the command-line flags.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict = {}

    def request(self, update: dict) -> None:
        with self._lock:
            self._pending.update(update)  # later keys win; partial updates merge

    def take(self) -> dict:
        with self._lock:
            pending, self._pending = self._pending, {}
        return pending


class PreviewServer:
    """Small LAN-only MJPEG and JSON-status server for setup/debugging.

    With ``settings`` it also takes POST /settings (see parse_settings), answering 202.
    """

    def __init__(self, port: int, settings: RuntimeSettings | None = None, host: str = "", *,
                 jpeg_quality: int = 75, max_width: int | None = None) -> None:
        self.state = PreviewState(jpeg_quality, max_width)
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                return

            def _headers(self, content_type: str) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Type", content_type)

            def _json(self, status: HTTPStatus, body: dict | None) -> None:
                data = json.dumps(body).encode() if body is not None else b""
                self.send_response(status)
                self._headers("application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_OPTIONS(self) -> None:
                self._json(HTTPStatus.NO_CONTENT, None)

            def do_POST(self) -> None:
                if self.path.split("?")[0] != "/settings" or settings is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if not 0 < length <= _MAX_SETTINGS_BODY:
                        raise ValueError("body missing or too large")
                    update = parse_settings(json.loads(self.rfile.read(length)))
                except (ValueError, UnicodeDecodeError) as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    return
                settings.request(update)
                self._json(HTTPStatus.ACCEPTED, {"requested": update})

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

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        with self.state.condition:
            self.state.closed = True
            self.state.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()

