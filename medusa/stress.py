"""CPU stress test, run by medusa_sysmon.py (priority 10), never by the control process.

POST :8082/stress?seconds=N on the control process only sends one localhost UDP datagram
(``{"type": "medusa-stress", "seconds": N}``, 0 stops) to medusa_sysmon.py, which spawns or
kills the busy-loop workers and writes their state to a small JSON file that control.json's
``stress`` block reads. Spawning from the priority-21 control process (fork/exec of a threaded
Python process while it holds the GIL) stalled the 50 Hz loop for 140-165 ms at every stress
start; the control process now does no fork/exec at all.

Workers run through ``on -p <priority>`` (default 10, the AI process's level) and each one
also stops itself at the deadline, so they never outlive a killed medusa_sysmon.py by more
than the requested duration.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

from .sysmon import on_qnx

STRESS_MESSAGE_TYPE = "medusa-stress"
DEFAULT_STRESS_PORT = 8770
DEFAULT_STRESS_FILE = "/tmp/medusa_stress.json"
STRESS_STALE_S = 5.0   # the runner rewrites its file every second; older means medusa_sysmon is down
_BUSY_LOOP = "import sys, time\nend = time.monotonic() + float(sys.argv[1])\nwhile time.monotonic() < end:\n    pass\n"


def make_stress_request(seconds: float) -> bytes:
    return json.dumps({"type": STRESS_MESSAGE_TYPE, "seconds": float(seconds)}).encode()


def parse_stress_request(payload: bytes, stress_max: float) -> float | None:
    """Seconds (0 = stop) from a request datagram, or None for anything malformed or out of range."""
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(message, dict) or message.get("type") != STRESS_MESSAGE_TYPE:
        return None
    seconds = message.get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 <= seconds <= stress_max:
        return None  # Also rejects NaN.
    return float(seconds)


class StressRunner:
    """Busy-loop worker processes at the AI's priority, to show the control loop is not disturbed."""

    def __init__(self, workers: int = 4, priority: int = 10, *, clock=time.monotonic, wall=time.time,
                 popen=subprocess.Popen) -> None:
        self.workers, self.priority = workers, priority
        self.clock, self.wall, self.popen = clock, wall, popen
        self.procs: list = []
        self.ends_at: float | None = None  # on self.clock
        self.lock = threading.Lock()

    def command(self, seconds: float) -> list[str]:
        command = [sys.executable, "-c", _BUSY_LOOP, str(seconds)]
        if on_qnx() and shutil.which("on"):
            command = ["on", "-p", str(self.priority)] + command
        return command

    def start(self, seconds: float) -> None:
        self.stop()
        with self.lock:
            self.procs = [self.popen(self.command(seconds), stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                          for _ in range(self.workers)]
            self.ends_at = self.clock() + seconds

    def stop(self) -> None:
        with self.lock:
            procs, self.procs, self.ends_at = self.procs, [], None
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def expire(self) -> None:
        """Stop the workers once their time is up (they also stop themselves)."""
        with self.lock:
            due = self.ends_at is not None and self.clock() >= self.ends_at
        if due:
            self.stop()

    def status(self) -> dict:
        """What the control process serves: {running, seconds_left, workers} plus the wall-clock end."""
        with self.lock:
            ends_at = self.ends_at
            running = ends_at is not None and any(p.poll() is None for p in self.procs)
        left = max(0.0, ends_at - self.clock()) if running else 0.0
        return {"running": running, "seconds_left": math.ceil(left), "workers": self.workers if running else 0,
                "ends_at_wall": round(self.wall() + left, 3) if running else None}


class StressService:
    """medusa_sysmon.py's thread: UDP requests in, workers spawned/killed, status file out."""

    def __init__(self, runner: StressRunner, port: int = DEFAULT_STRESS_PORT, out: str = DEFAULT_STRESS_FILE,
                 stress_max: float = 60, host: str = "127.0.0.1") -> None:
        self.runner, self.out, self.stress_max = runner, out, stress_max
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((host, port))
            self.sock.settimeout(0.25)
        except OSError:
            self.sock.close()
            raise
        self.port = self.sock.getsockname()[1]

    def handle(self, payload: bytes) -> None:
        seconds = parse_stress_request(payload, self.stress_max)
        if seconds is None:
            return
        if seconds == 0:
            self.runner.stop()
        else:
            self.runner.start(seconds)
        self.write()

    def write(self) -> None:
        write_json_atomic(self.out, self.runner.status())

    def serve(self, stop: threading.Event) -> None:
        last_write = 0.0
        try:
            while not stop.is_set():
                try:
                    payload, _sender = self.sock.recvfrom(1024)
                except (socket.timeout, BlockingIOError, InterruptedError):
                    payload = None
                except OSError:
                    payload = None
                if payload is not None:
                    self.handle(payload)
                    last_write = time.monotonic()
                self.runner.expire()
                if time.monotonic() - last_write >= 1.0:  # heartbeat: the control process sees we are up
                    try:
                        self.write()
                    except OSError:
                        pass
                    last_write = time.monotonic()
        finally:
            self.runner.stop()
            self.sock.close()
            try:
                write_json_atomic(self.out, self.runner.status())
            except OSError:
                pass


def write_json_atomic(path: str, payload: dict) -> None:
    """Write next to ``path`` and os.replace() it, so a reader never sees half a file."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, temp = tempfile.mkstemp(prefix=".medusa_stress.", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


class StressClient:
    """The control process's side: one UDP datagram per request, status read from the runner's file.

    No fork, no exec, no waiting: the HTTP thread only sends and reads a small cached file.
    """

    def __init__(self, port: int = DEFAULT_STRESS_PORT, path: str = DEFAULT_STRESS_FILE,
                 stale_s: float = STRESS_STALE_S, wall=time.time, host: str = "127.0.0.1") -> None:
        self.address, self.path, self.stale_s, self.wall = (host, port), path, stale_s, wall
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self._lock = threading.Lock()
        self._cached: tuple[int, dict] | None = None  # (mtime_ns, payload)

    def _payload(self) -> dict | None:
        """The runner's file if it is fresh, else None (medusa_sysmon.py not running)."""
        try:
            stat = os.stat(self.path)
        except OSError:
            return None
        if not -1 <= self.wall() - stat.st_mtime <= self.stale_s:
            return None
        with self._lock:
            cached = self._cached
            if cached is None or cached[0] != stat.st_mtime_ns:
                try:
                    with open(self.path, "rb") as handle:
                        payload = json.loads(handle.read(4096))
                except (OSError, ValueError):
                    return None
                if not isinstance(payload, dict):
                    return None
                cached = self._cached = (stat.st_mtime_ns, payload)
        return cached[1]

    def available(self) -> bool:
        return self._payload() is not None

    def request(self, seconds: float) -> bool:
        try:
            self.sock.sendto(make_stress_request(seconds), self.address)
            return True
        except OSError:
            return False

    def status(self) -> dict:
        payload = self._payload() or {}
        ends_at = payload.get("ends_at_wall")
        running = payload.get("running") is True and isinstance(ends_at, (int, float)) and ends_at > self.wall()
        workers = payload.get("workers")
        return {"running": running, "seconds_left": math.ceil(ends_at - self.wall()) if running else 0,
                "workers": workers if running and isinstance(workers, int) else 0}

    def close(self) -> None:
        self.sock.close()
