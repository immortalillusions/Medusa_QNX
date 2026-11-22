"""Pi usage sampler (CPU, memory, temperature, uptime) for medusa_sysmon.py.

It runs in its own process at priority 10, below the control process (21): sampling
inside the control process cost the 50 Hz loop its deadlines (GIL and spawns). Every
``period_s`` it spawns one shell running ``pidin info``, ``pidin arg`` and ``pidin
times``, and medusa_sysmon.py writes the result to a JSON file that the control
process's HTTP thread reads (medusa.control.SystemStatusFile).

CPU % is of the whole chip (all cores), from utime + stime deltas. The control and AI
processes are found by their command lines (medusa_control.py, qnx_ai_service.py).
Every source is optional: a failed command, file or parse leaves its fields null.
"""

from __future__ import annotations

import calendar
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque

SYSTEM_PERIOD_S = 2.0
DEFAULT_SYSTEM_FILE = "/tmp/medusa_system.json"
SYSTEM_KEYS = ("cpu_total_pct", "cpu_control_pct", "cpu_ai_pct", "mem_free_mb", "mem_total_mb",
               "temp_c", "uptime_s", "sampled_age_s")
CONTROL_SCRIPT = "medusa_control.py"
AI_SCRIPT = "qnx_ai_service.py"
CPU_WINDOW_S = 10.0  # CPU % is averaged over up to this long: pidin prints 1 s steps past a minute
_ARG_MARK, _TIMES_MARK = "@@medusa-arg", "@@medusa-times"
# One spawn per sample. Root sees every process's times (hogs was dropped: as root its SYS
# column read per core and its rows came back N/A or empty).
SYSTEM_COMMAND = (f"pidin info; echo {_ARG_MARK}; sudo -n pidin arg; "
                  f"echo {_TIMES_MARK}; sudo -n pidin times")
_TIMES_HEADER = re.compile(r"^\s*pid\s+name\b.*\butime\s+stime\b.*$", re.M | re.I)
_DURATION = re.compile(r"^(?:\d+(?:\.\d+)?|(?:\d+(?:\.\d+)?[dhms])+)$")
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)([dhms])")
_UNIT_S = {"d": 86400, "h": 3600, "m": 60, "s": 1}
_MEM = re.compile(r"FreeMem:\s*(\d+(?:\.\d+)?)\s*([KMG]?)B?\s*/\s*(\d+(?:\.\d+)?)\s*([KMG]?)B?", re.I)
_BOOT = re.compile(r"BootTime:\s*(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\w+\s+\d{4})")
_MB = {"K": 1 / 1024, "": 1 / (1024 * 1024), "M": 1, "G": 1024}


def on_qnx() -> bool:
    return sys.platform.startswith(("qnx", "nto")) or os.uname().sysname.upper().startswith("QNX")


def split_output(text: str) -> tuple[str, str, str]:
    """The one spawn's output -> (pidin info, pidin arg, pidin times) sections."""
    info, _, rest = text.partition(_ARG_MARK)
    args, _, times = rest.partition(_TIMES_MARK)
    return info, args, times


def parse_cpu_seconds(token: str) -> float | None:
    """A pidin CPU time: ``68.071`` (seconds), ``14m40s``, ``2h05m``, ``1d03h`` ... or None."""
    token = token.strip()
    if not _DURATION.match(token):
        return None
    if token[-1].isdigit():
        return float(token)
    return sum(float(value) * _UNIT_S[unit] for value, unit in _DURATION_PART.findall(token))


def parse_pidin_times(text: str) -> dict[int, tuple[str, float]]:
    """``pidin times`` rows (pid name sid start-time utime stime cutime cstime) after the header
    -> {pid: (name, utime + stime seconds)}. The start time has spaces, so the four times are
    read from the right. Rows that do not parse are skipped."""
    header = _TIMES_HEADER.search(text)
    if header is None:
        return {}
    times = {}
    for line in text[header.end():].splitlines():
        fields = line.split()
        if len(fields) < 6 or not fields[0].isdigit():
            continue
        utime, stime = parse_cpu_seconds(fields[-4]), parse_cpu_seconds(fields[-3])
        if utime is None or stime is None or parse_cpu_seconds(fields[-2]) is None:
            continue
        name = fields[1] if len(fields) > 6 and not fields[1].isdigit() else ""
        times[int(fields[0])] = (name, utime + stime)
    return times


def find_pid(pidin_arg: str, script: str) -> int | None:
    """``pidin arg`` rows are ``<pid> <argv...>``: the first python process running ``script``.

    argv[0] must be python, so ``on -p 21 python3 medusa_control.py`` (the launcher) and shells
    that merely mention the script are not taken for it."""
    for line in pidin_arg.splitlines():
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit() or "python" not in os.path.basename(fields[1]):
            continue
        if any(arg == script or arg.endswith("/" + script) for arg in fields[2:]):
            return int(fields[0])
    return None


def cpu_from_times(before: dict, after: dict, elapsed_s: float, cores: int, control_pid: int | None,
                   ai_pid: int | None) -> tuple[float | None, float | None, float | None]:
    """(total, control, AI) as % of the whole chip (all ``cores``), from two pidin times samples.

    Total sums every process but procnto (PID 1). Processes that started, ended or reused a PID
    between the samples are skipped. An unknown or missing control/AI pid gives None."""
    if not before or not after or elapsed_s <= 0 or cores < 1:
        return None, None, None
    capacity = elapsed_s * cores

    def pct(seconds: float | None) -> float | None:
        return None if seconds is None else round(max(0.0, min(100.0, seconds / capacity * 100)), 1)

    deltas = {}
    for pid, (name, seconds) in after.items():
        if pid in before and before[pid][0] == name and seconds >= before[pid][1]:
            deltas[pid] = seconds - before[pid][1]
    total = sum(delta for pid, delta in deltas.items() if pid != 1 and "procnto" not in after[pid][0])
    return pct(total), pct(deltas.get(control_pid)), pct(deltas.get(ai_pid))


def parse_pidin_info(text: str) -> tuple[float | None, float | None, float | None]:
    """``pidin info`` -> (free MB, total MB, boot time as a Unix time); None where absent."""
    free = total = boot = None
    mem = _MEM.search(text)
    if mem:
        free = round(float(mem.group(1)) * _MB[mem.group(2).upper()], 1)
        total = round(float(mem.group(3)) * _MB[mem.group(4).upper()], 1)
    when = _BOOT.search(text)
    if when:
        try:
            boot = calendar.timegm(time.strptime(" ".join(when.group(1).split()), "%b %d %H:%M:%S %Z %Y"))
        except (ValueError, OverflowError):
            boot = None
    return free, total, boot


def parse_thermal(text: str) -> float | None:
    """``/dev/thermal`` prints the SoC temperature in degrees C, e.g. ``51.57``."""
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    value = float(match.group(0))
    return round(value, 1) if -40 <= value <= 150 else None


def _run_system_command(timeout: float = 6.0) -> str:
    if not on_qnx():
        return ""  # pidin is a QNX tool: elsewhere every field stays null.
    command = ["sh", "-c", SYSTEM_COMMAND]
    if shutil.which("on"):
        command = ["on", "-p", "10"] + command
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, timeout=timeout, check=False)
    return result.stdout.decode("utf-8", "replace")


def _read_thermal(path: str = "/dev/thermal") -> str:
    with open(path, "rb") as handle:
        return handle.read(64).decode("ascii", "replace")


class SystemSampler:
    """One sample per sample_once(); medusa_sysmon.py calls it every ``period_s``."""

    def __init__(self, *, period_s: float = SYSTEM_PERIOD_S, run=_run_system_command,
                 read_thermal=_read_thermal, clock=time.monotonic, wall=time.time,
                 cores: int | None = None, window_s: float = CPU_WINDOW_S) -> None:
        self.period_s, self.run, self.read_thermal = period_s, run, read_thermal
        self.clock, self.wall = clock, wall
        self.cores = cores or os.cpu_count() or 4
        self.window_s = window_s
        self._history: deque[tuple[float, dict]] = deque()  # (when, pidin times), oldest first
        self._stop = threading.Event()

    def sample_once(self) -> dict:
        try:
            text = self.run()
        except Exception:  # missing tool, timeout, anything: null fields, never an exception
            text = ""
        now = self.clock()
        info, args, times_text = split_output(text)
        total = control = ai = None
        try:
            times = parse_pidin_times(times_text)
            if times:
                # Against the oldest sample inside the window: a longer base smooths pidin's
                # whole-second steps. Null until there are two samples.
                while self._history and now - self._history[0][0] > self.window_s:
                    self._history.popleft()
                if self._history:
                    then, before = self._history[0]
                    total, control, ai = cpu_from_times(before, times, now - then, self.cores,
                                                        find_pid(args, CONTROL_SCRIPT), find_pid(args, AI_SCRIPT))
                self._history.append((now, times))
        except Exception:
            total = control = ai = None
        free, mem_total, boot = parse_pidin_info(info)
        try:
            temp = parse_thermal(self.read_thermal())
        except Exception:
            temp = None
        uptime = None
        if boot is not None and 0 <= self.wall() - boot < 10 * 365 * 86400:
            uptime = round(self.wall() - boot)
        elif hasattr(time, "CLOCK_MONOTONIC") and on_qnx():
            uptime = round(time.clock_gettime(time.CLOCK_MONOTONIC))  # QNX: seconds since boot
        return {"cpu_total_pct": total, "cpu_control_pct": control, "cpu_ai_pct": ai,
                "mem_free_mb": free, "mem_total_mb": mem_total, "temp_c": temp, "uptime_s": uptime}

    def run_forever(self, out: str, stop: threading.Event | None = None, max_samples: int | None = None) -> None:
        stop = stop or self._stop
        count = 0
        while not stop.is_set() and (max_samples is None or count < max_samples):
            started = self.clock()
            write_atomic(out, self.sample_once())
            count += 1
            stop.wait(max(0.1, self.period_s - (self.clock() - started)))


def write_atomic(path: str, sample: dict) -> None:
    """Write next to ``path`` and os.replace() it, so a reader never sees half a file."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, temp = tempfile.mkstemp(prefix=".medusa_system.", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(sample, handle)
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
