"""Real-time control process: the only owner of the Arduino serial port.

The AI process (qnx_ai_service.py --control-port 8768) runs at default priority
(10) and may stall under load. This process runs a fixed 50 Hz loop at a higher
priority (QNX: ``on -p 21 python3 -u medusa_control.py ...``; every thread it
starts inherits that priority) so the Arduino heartbeat never depends on the
AI loop. Each tick it:

1. reads the firmware's ``S,<armed>,<pan_x10>,<tilt_x10>,<seq>[,<gate>[,<parking>]]``
   status lines (gate 1: kill gate closed, servo power on; parking 1: the firmware
   lost the heartbeat while armed and is gliding home before it goes limp; older
   firmware sends neither; latest wins; anything else, such as the Zephyr boot
   banner, is ignored);
2. drains ``medusa-target`` packets from the AI process (latest wins);
3. sends ``N,yaw_x10,pitch_x10,led,seq`` (tenths of a degree) with the AI's
   target while it is fresh and the kill gate is not reported open, otherwise an
   ``M`` park line (the configured ``home`` angles, LED off): an armed turret
   glides home on it, and a disarmed one only takes it as the heartbeat (its
   servos are detached then);
4. sends a ``medusa-actual`` packet back to the AI process, so its pan tracker
   can use where the turret really points.

IPC is localhost UDP JSON: simple and works on QNX's io-sock today. QNX native
message passing (MsgSend/MsgReceive, priority-inheriting) would be the
production choice, but has no binding in the stock Python build.

The loop runs in the main thread; the HTTP status server and the stress-test
timer run in daemon threads that are almost always blocked. The GIL switch
interval is shortened so a waking loop thread gets the interpreter quickly.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

TARGET_MESSAGE_TYPE = "medusa-target"
ACTUAL_MESSAGE_TYPE = "medusa-actual"
DEFAULT_TARGET_PORT = 8768
DEFAULT_ACTUAL_PORT = 8769
DEFAULT_HTTP_PORT = 8082
PERIOD_S = 0.020
TARGET_TIMEOUT_S = 0.3
AI_ALIVE_S = 1.0
LINK_TIMEOUT_S = 0.5      # S lines come every 100 ms; five missed means the link is down
LATE_S = 0.005            # a tick starting more than 5 ms after its deadline missed it
WINDOW_S = 10.0
STRESS_MAX_S = 60
_MAX_PACKETS_PER_TICK = 64
_MAX_LINE = 128
_STATUS_LINE = re.compile(r"S,([01]),(-?\d{1,5}),(-?\d{1,5}),(\d{1,10})(?:,([01])(?:,([01]))?)?")
_COMMAND_LINE = re.compile(r"([MN]),(-?\d+),(-?\d+),(-?\d+)(?:,(\d+))?")


def system_clock() -> float:
    """CLOCK_MONOTONIC: one clock for every process on the host, so timestamps can cross the IPC.

    (time.monotonic() is CLOCK_MONOTONIC on QNX and Linux, but process-relative on macOS.)
    """
    return time.clock_gettime(time.CLOCK_MONOTONIC) if hasattr(time, "CLOCK_MONOTONIC") else time.monotonic()


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


# --- Arduino command lines -----------------------------------------------------------------

def tenths(degrees: float) -> int:
    """Degrees to the firmware's integer tenths, rounding half away from zero like its tenths()."""
    return int(math.floor(abs(degrees) * 10 + 0.5)) * (1 if degrees >= 0 else -1)


def format_n_line(yaw: float, pitch: float, led: bool, seq: int) -> bytes:
    """``N,<yaw_x10>,<pitch_x10>,<led>,<seq>\\n``: angles in tenths of a degree (92.5 -> 925)."""
    return f"N,{tenths(yaw)},{tenths(pitch)},{int(bool(led))},{seq}\n".encode("ascii")


def parse_command_line(line: str, servo_min: int = 0, servo_max: int = 180):
    """Python twin of applyCommand() in the firmware: (yaw, pitch, led, seq) in degrees, or None.

    N lines carry tenths; M lines whole degrees, and the old 4-field M line has no
    pitch (home 90). Out-of-range angles or an LED other than 0/1 are rejected.
    """
    match = _COMMAND_LINE.fullmatch(line.strip())
    if match is None:
        return None
    kind, a, b, c, d = match.groups()
    if d is None:
        if kind != "M":
            return None
        yaw, pitch, led, seq = int(a), 90, int(b), int(c)
    else:
        yaw, pitch, led, seq = int(a), int(b), int(c), int(d)
    scale = 10 if kind == "N" else 1
    if led not in (0, 1) or not all(servo_min * scale <= v <= servo_max * scale for v in (yaw, pitch)):
        return None
    return yaw / scale, pitch / scale, bool(led), seq


# --- Arduino status -------------------------------------------------------------------------

@dataclass(frozen=True)
class ArduinoStatus:
    armed: bool
    pan: float   # degrees, eased actual position
    tilt: float
    seq: int
    gate_closed: bool | None = None  # None: a 5-field line from firmware without the kill-gate sense
    parking: bool = False  # link lost while armed: gliding home, then limp (7-field line only)


def parse_status_line(line: str) -> ArduinoStatus | None:
    """``S,armed,pan_x10,tilt_x10,seq``, the 6-field form with ``,gate`` (1 closed, 0 open),
    or the 7-field form with ``,gate,parking`` (1: gliding home after a lost heartbeat)."""
    match = _STATUS_LINE.fullmatch(line.strip())
    if match is None:
        return None
    armed, pan, tilt, seq, gate, parking = match.groups()
    return ArduinoStatus(armed == "1", int(pan) / 10, int(tilt) / 10, int(seq),
                         None if gate is None else gate == "1", parking == "1")


class StatusReader:
    """Split serial bytes into lines and keep the S, ones; partial lines wait for the next read."""

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, data: bytes) -> list[ArduinoStatus]:
        self._buffer += data
        *lines, self._buffer = self._buffer.split(b"\n")
        if len(self._buffer) > _MAX_LINE:
            self._buffer = b""  # No newline in a long run of bytes: noise, not a status line.
        statuses = []
        for raw in lines:
            status = parse_status_line(raw.decode("ascii", "replace"))
            if status is not None:
                statuses.append(status)
        return statuses


class SerialPort:
    """Non-blocking fd wrapper; a full TX buffer drops the line (the next tick sends a fresh one)."""

    def __init__(self, fd: int) -> None:
        self.fd = fd

    @classmethod
    def open(cls, device: str, baud: int = 115200) -> "SerialPort":
        from .arduino_serial import open_serial

        return cls(open_serial(device, baud, read=True))

    def read(self) -> bytes:
        chunks = []
        for _ in range(16):
            try:
                chunk = os.read(self.fd, 512)
            except (BlockingIOError, InterruptedError):
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def write(self, data: bytes) -> bool:
        try:
            return os.write(self.fd, data) == len(data)
        except (BlockingIOError, InterruptedError):
            return False

    def close(self) -> None:
        os.close(self.fd)


# --- AI target packets ----------------------------------------------------------------------

@dataclass(frozen=True)
class TargetPacket:
    active: bool
    yaw: float
    pitch: float
    seq: int | None = None
    t: float | None = None


def make_target_packet(active: bool, yaw: float, pitch: float, seq: int, t: float) -> bytes:
    """Float angles (degrees); the control process sends them on as tenths."""
    return json.dumps({"type": TARGET_MESSAGE_TYPE, "active": bool(active), "yaw": round(float(yaw), 3),
                       "pitch": round(float(pitch), 3), "seq": seq, "t": t}).encode()


def parse_target_packet(payload: bytes) -> TargetPacket | None:
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(message, dict) or message.get("type") != TARGET_MESSAGE_TYPE:
        return None
    active, yaw, pitch = message.get("active"), _number(message.get("yaw")), _number(message.get("pitch"))
    if not isinstance(active, bool) or yaw is None or pitch is None:
        return None
    seq = message.get("seq")
    return TargetPacket(active, yaw, pitch, seq if isinstance(seq, int) and not isinstance(seq, bool) else None,
                        _number(message.get("t")))


@dataclass(frozen=True)
class ActualStatus:
    """The control process's view of the Arduino, as forwarded to the AI process."""
    link_ok: bool
    armed: bool | None
    pan: float | None
    tilt: float | None
    t: float | None  # when the S line was read, on the receiver's clock (ControlClient converts)
    gate_closed: bool | None = None  # kill gate: True closed, False open, None unknown


def parse_actual_packet(payload: bytes) -> ActualStatus | None:
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(message, dict) or message.get("type") != ACTUAL_MESSAGE_TYPE:
        return None
    link_ok, armed, gate = message.get("link_ok"), message.get("armed"), message.get("gate_closed")
    if not isinstance(link_ok, bool) or not all(v is None or isinstance(v, bool) for v in (armed, gate)):
        return None
    return ActualStatus(link_ok, armed, _number(message.get("pan")), _number(message.get("tilt")),
                        _number(message.get("t")), gate)


def _drain(sock, limit: int = _MAX_PACKETS_PER_TICK):
    for _ in range(limit):
        try:
            payload, _sender = sock.recvfrom(4096)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            return  # e.g. an ICMP error from a previous send; try again next time.
        yield payload


def _udp_socket(bind_port: int | None = None, host: str = "127.0.0.1") -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setblocking(False)
        if bind_port is not None:
            sock.bind((host, bind_port))
    except OSError:
        sock.close()
        raise
    return sock


# --- timing statistics ----------------------------------------------------------------------

class TickStats:
    """Rolling window of tick periods and missed deadlines."""

    def __init__(self, period_s: float = PERIOD_S, window_s: float = WINDOW_S, late_s: float = LATE_S) -> None:
        self.period_s, self.late_s = period_s, late_s
        self._ticks: deque[tuple[float | None, int]] = deque(maxlen=max(1, int(round(window_s / period_s))))
        self._last_start: float | None = None
        self.missed_total = 0
        self.lock = threading.Lock()

    def record(self, start: float, deadline: float | None, skipped: int = 0) -> None:
        """Log a tick that began at ``start`` for ``deadline``; ``skipped`` deadlines passed unserved."""
        period = None if self._last_start is None else start - self._last_start
        self._last_start = start
        missed = skipped + int(deadline is not None and start - deadline > self.late_s)
        self.missed_total += missed
        with self.lock:
            self._ticks.append((period, missed))

    def summary(self) -> dict:
        with self.lock:
            ticks = list(self._ticks)
        periods = [p for p, _ in ticks if p is not None]
        result = {"missed_deadlines": sum(m for _, m in ticks), "missed_deadlines_total": self.missed_total}
        if not periods:
            return {**result, "loop_hz": None, "period_ms_mean": None, "jitter_ms_p99": None,
                    "period_ms_max": None}
        mean = sum(periods) / len(periods)
        jitter = sorted(abs(p - self.period_s) for p in periods)
        p99 = jitter[max(0, math.ceil(0.99 * len(jitter)) - 1)]
        return {**result, "loop_hz": round(1 / mean, 2) if mean > 0 else None,
                "period_ms_mean": round(mean * 1000, 3), "jitter_ms_p99": round(p99 * 1000, 3),
                "period_ms_max": round(max(periods) * 1000, 3)}


# --- CPU stress test ------------------------------------------------------------------------

_BUSY_LOOP = "import sys, time\nend = time.monotonic() + float(sys.argv[1])\nwhile time.monotonic() < end:\n    pass\n"


class StressTest:
    """Busy-loop worker processes at NORMAL priority, to show the control loop is not disturbed.

    The workers would inherit this process's priority (21), so on QNX they are
    started through ``on -p <priority>`` (default 10, the AI process's level).
    """

    def __init__(self, workers: int = 4, priority: int = 10, *, clock=time.monotonic,
                 popen=subprocess.Popen) -> None:
        self.workers, self.priority = workers, priority
        self.clock, self.popen = clock, popen
        self.procs: list = []
        self.ends_at: float | None = None
        self._timer: threading.Timer | None = None
        self.lock = threading.Lock()

    def command(self, seconds: float) -> list[str]:
        command = [sys.executable, "-c", _BUSY_LOOP, str(seconds)]
        on_qnx = sys.platform.startswith(("qnx", "nto")) or os.uname().sysname.upper().startswith("QNX")
        if on_qnx and shutil.which("on"):
            command = ["on", "-p", str(self.priority)] + command
        return command

    def start(self, seconds: float) -> None:
        self.stop()
        with self.lock:
            self.procs = [self.popen(self.command(seconds), stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                          for _ in range(self.workers)]
            self.ends_at = self.clock() + seconds
            self._timer = threading.Timer(seconds, self.stop)
            self._timer.daemon = True
            self._timer.start()

    def stop(self) -> None:
        with self.lock:
            procs, self.procs, self.ends_at = self.procs, [], None
            if self._timer is not None and self._timer is not threading.current_thread():
                self._timer.cancel()
            self._timer = None
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def status(self) -> dict:
        with self.lock:
            ends_at = self.ends_at
            running = ends_at is not None and any(p.poll() is None for p in self.procs)
        left = max(0, math.ceil(ends_at - self.clock())) if running else 0
        return {"running": running, "seconds_left": left, "workers": self.workers if running else 0}


# --- the control loop -----------------------------------------------------------------------

class ControlLoop:
    def __init__(
        self, serial, *, target_sock=None, actual_sock=None, actual_addr=None,
        period_s: float = PERIOD_S, target_timeout: float = TARGET_TIMEOUT_S,
        link_timeout: float = LINK_TIMEOUT_S, yaw_range=(65, 115), pitch_range=(65, 115),
        home=(90, 90), window_s: float = WINDOW_S, stress: StressTest | None = None,
        clock=system_clock, sleep=time.sleep,
    ) -> None:
        self.serial, self.target_sock = serial, target_sock
        self.actual_sock, self.actual_addr = actual_sock, actual_addr
        self.period_s, self.target_timeout, self.link_timeout = period_s, target_timeout, link_timeout
        self.yaw_range, self.pitch_range = yaw_range, pitch_range
        self.home_yaw, self.home_pitch = home
        self.stats = TickStats(period_s, window_s)
        self.stress = stress
        self.clock, self.sleep = clock, sleep
        self.reader = StatusReader()
        self.status: ArduinoStatus | None = None
        self.status_at: float | None = None
        self.target: TargetPacket | None = None
        self.target_at: float | None = None
        self.sequence = 0
        self.write_failures = 0
        self.ticks = 0

    @staticmethod
    def _clamp(value: float, bounds) -> float:
        return float(max(bounds[0], min(bounds[1], value)))

    def park_line(self) -> bytes:
        """``M,<home yaw>,<home pitch>,0,<seq>``: the configured homes (--servo-home, --tilt-home)."""
        return f"M,{self.home_yaw},{self.home_pitch},0,{self.sequence}\n".encode("ascii")

    def target_fresh(self, now: float) -> bool:
        return self.target_at is not None and now - self.target_at < self.target_timeout

    def gate_open(self, now: float) -> bool:
        """The firmware's fresh status says the kill gate is open (no servo power)."""
        return self.link_ok(now) and self.status is not None and self.status.gate_closed is False

    def tick(self, now: float) -> bytes:
        """One control step at ``now``; returns the line written to the Arduino."""
        for status in self.reader.feed(self.serial.read()):
            self.status, self.status_at = status, now  # Latest wins.
        if self.target_sock is not None:
            for payload in _drain(self.target_sock):
                target = parse_target_packet(payload)
                if target is not None:
                    self.target, self.target_at = target, now
        self.sequence += 1
        if self.target_fresh(now) and not self.gate_open(now):
            target = self.target
            line = format_n_line(self._clamp(target.yaw, self.yaw_range),
                                 self._clamp(target.pitch, self.pitch_range), target.active, self.sequence)
        else:
            line = self.park_line()  # Heartbeat only (no AI target, or the kill gate open): LED off.
        if not self.serial.write(line):
            self.write_failures += 1
        if self.actual_sock is not None and self.actual_addr is not None:
            try:
                self.actual_sock.sendto(self.actual_packet(now), self.actual_addr)
            except OSError:
                pass  # The AI process is not listening yet; it is optional.
        self.ticks += 1
        return line

    def link_ok(self, now: float) -> bool:
        return self.status_at is not None and now - self.status_at < self.link_timeout

    def actual_packet(self, now: float) -> bytes:
        status = self.status
        return json.dumps({
            "type": ACTUAL_MESSAGE_TYPE, "link_ok": self.link_ok(now),
            "armed": status.armed if status else None, "pan": status.pan if status else None,
            "tilt": status.tilt if status else None, "seq": status.seq if status else None,
            "gate_closed": status.gate_closed if status else None,
            "t": self.status_at,  # on this loop's clock: system_clock() in production
        }).encode()

    def snapshot(self, now: float | None = None) -> dict:
        """The /control.json contract."""
        now = self.clock() if now is None else now
        status, status_at = self.status, self.status_at
        target, target_at = self.target, self.target_at
        link_ok = self.link_ok(now)
        target_age = None if target_at is None else round(now - target_at, 3)
        fresh = self.target_fresh(now)
        armed = bool(link_ok and status and status.armed)
        parking = bool(link_ok and status and status.parking)
        gate_closed = status.gate_closed if link_ok and status else None
        if not link_ok:
            reason = "Arduino link lost" if status_at is not None else "no Arduino status yet"
        elif gate_closed is False:
            reason = "kill gate open"
        elif parking:
            reason = "returning home after link loss"
        elif not armed:
            reason = "turret disarmed"
        elif target is None:
            reason = "no AI target"
        elif not fresh:
            reason = "AI target stale"
        elif not target.active:
            reason = "AI has no face to aim at"
        else:
            reason = ""
        result = {
            "state": "ACTIVE" if reason == "" else "SAFE",
            "safe_reason": reason,
            **self.stats.summary(),
            "target_age_s": target_age,
            "ai_alive": target_age is not None and target_age < AI_ALIVE_S,
            "arduino": {
                "link_ok": link_ok, "armed": armed, "parking": parking, "gate_closed": gate_closed,
                "pan": status.pan if status else None, "tilt": status.tilt if status else None,
                "age_s": None if status_at is None else round(now - status_at, 3),
            },
            "stress": self.stress.status() if self.stress else {"running": False, "seconds_left": 0},
            "write_failures": self.write_failures,
        }
        return result

    def run(self, stop: threading.Event, max_ticks: int | None = None) -> None:
        """Tick on absolute deadlines start + k * period: lateness never accumulates into drift."""
        deadline = self.clock()
        skipped = 0
        while not stop.is_set() and (max_ticks is None or self.ticks < max_ticks):
            start = self.clock()
            self.stats.record(start, deadline, skipped)
            self.tick(start)
            deadline += self.period_s
            now = self.clock()
            skipped = 0
            if now - deadline > self.stats.late_s:
                # Already missed the next deadline: skip to the next one still ahead
                # (the skipped ones count as missed) rather than bursting to catch up.
                skipped = int((now - deadline) // self.period_s) + 1
                deadline += skipped * self.period_s
            if deadline > now:
                self.sleep(deadline - now)

    def shutdown(self, lines: int = 3) -> None:
        for _ in range(lines):
            self.sequence += 1
            self.serial.write(self.park_line())


# --- HTTP status server ---------------------------------------------------------------------

class ControlHTTPServer:
    """GET /control.json, POST /stress?seconds=N; CORS on every response. Runs in a daemon thread."""

    def __init__(self, loop: ControlLoop, port: int = DEFAULT_HTTP_PORT, host: str = "0.0.0.0",
                 stress_max: int = STRESS_MAX_S) -> None:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                return

            def _reply(self, status: HTTPStatus, body: dict | None = None) -> None:
                data = json.dumps(body).encode() if body is not None else b""
                self.send_response(status)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_OPTIONS(self) -> None:
                self._reply(HTTPStatus.NO_CONTENT)

            def do_GET(self) -> None:
                if urlparse(self.path).path == "/control.json":
                    self._reply(HTTPStatus.OK, loop.snapshot())
                else:
                    self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:
                url = urlparse(self.path)
                if url.path != "/stress":
                    self._reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                if loop.stress is None:
                    self._reply(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "stress test disabled"})
                    return
                try:
                    seconds = float(parse_qs(url.query).get("seconds", ["20"])[0])
                except ValueError:
                    seconds = math.nan
                if not 0 <= seconds <= stress_max:  # Also rejects NaN.
                    self._reply(HTTPStatus.BAD_REQUEST, {"error": f"seconds must be 0..{stress_max}"})
                    return
                if seconds == 0:
                    loop.stress.stop()
                else:
                    loop.stress.start(seconds)
                self._reply(HTTPStatus.OK, {"stress": loop.stress.status()})

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="control-http")

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


# --- AI-process side ------------------------------------------------------------------------

class ControlClient:
    """Drop-in for ArduinoController in the AI process: sends medusa-target packets, no serial port.

    It also listens for the control process's medusa-actual packets so the pan
    tracker can use the real turret angle (poll_actual).
    """

    def __init__(self, port: int, mapper, tilt_mapper, *, actual_port: int | None = DEFAULT_ACTUAL_PORT,
                 host: str = "127.0.0.1", actual_timeout: float = TARGET_TIMEOUT_S, clock=time.monotonic,
                 remote_clock=system_clock) -> None:
        self.mapper, self.tilt_mapper = mapper, tilt_mapper
        self.address = (host, port)
        # ``clock`` is the AI process's (the pan tracker's time.monotonic); ``remote_clock``
        # is the control process's. Packet times are converted from one to the other.
        self.actual_timeout, self.clock, self.remote_clock = actual_timeout, clock, remote_clock
        self.sequence = 0
        self.latest_actual: ActualStatus | None = None
        self.sock = _udp_socket()
        try:
            self.actual_sock = _udp_socket(actual_port, host) if actual_port is not None else None
        except OSError:
            self.sock.close()
            raise

    def _send(self, active: bool, yaw: float, pitch: float) -> None:
        self.sequence += 1
        try:
            self.sock.sendto(make_target_packet(active, yaw, pitch, self.sequence, self.remote_clock()),
                             self.address)
        except OSError:
            pass  # Control process not up yet: it parks the turret on its own.

    def send(self, active: bool, target_x: float | None, target_y: float | None = None) -> None:
        yaw, led = self.mapper.command(active, target_x)
        pitch, _ = self.tilt_mapper.command(active, target_y)
        self._send(led, yaw, pitch)

    def send_angles(self, active: bool, yaw: float, pitch: float) -> None:
        """Float angles, clamped to the safe ranges (the control process clamps again)."""
        yaw = max(self.mapper.minimum, min(self.mapper.maximum, float(yaw)))
        pitch = max(self.tilt_mapper.minimum, min(self.tilt_mapper.maximum, float(pitch)))
        self._send(active, yaw, pitch)

    def safe_stop(self) -> None:
        """Home on both axes, light off: exactly the configured homes, not a smoothed step toward them."""
        self._send(False, self.mapper.park(), self.tilt_mapper.park())

    def poll_actual(self) -> ActualStatus | None:
        """Latest medusa-actual whose S status is fresh, else None (the tracker falls back to its model)."""
        if self.actual_sock is not None:
            for payload in _drain(self.actual_sock):
                actual = parse_actual_packet(payload)
                if actual is not None:
                    if actual.t is not None:  # Control-process time -> this process's clock.
                        actual = ActualStatus(actual.link_ok, actual.armed, actual.pan, actual.tilt,
                                              actual.t + (self.clock() - self.remote_clock()),
                                              actual.gate_closed)
                    self.latest_actual = actual
        actual = self.latest_actual
        if actual is None or not actual.link_ok or actual.t is None or actual.pan is None:
            return None
        if self.clock() - actual.t > self.actual_timeout:
            return None
        return actual

    def close(self) -> None:
        self.sock.close()
        if self.actual_sock is not None:
            self.actual_sock.close()
