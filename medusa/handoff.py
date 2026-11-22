"""Laptop-to-turret handoff map: where to point the turret when only the laptop sees the peeper.

The laptop heartbeat carries target_x (0..1 across the laptop image). The map
turns that into a pan angle, piecewise-linear between recorded points and
clamped to the end points outside them. It needs at least two points.

The running service also learns the map itself (HandoffLearner): whenever the
laptop sees the peeper and the turret camera holds the face centred on a
steady turret, it records (laptop_x, real pan) into 20 bins across laptop_x,
each an average of what it saw there. The learned map extrapolates linearly
past its outermost bins (clamped to the pan range) and is saved to the same
file at most every 5 s; points recorded by hand below count as bins too.

    python3 -m medusa.handoff record <laptop_x> <pan_deg>   # add or replace a point
    python3 -m medusa.handoff record-now                    # laptop_x and pan from the live services
    python3 -m medusa.handoff show
    python3 -m medusa.handoff clear

``record-now``: stand where a peeper would, let the turret track you until it
is centred, then run it. It reads laptop_target_x from the AI preview
(/status.json, port 8080) and the real pan angle from the control process
(/control.json, port 8082), falling back to the AI's commanded pan_angle.
The running AI service picks up a changed file within a second.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

DEFAULT_PATH = "~/Medusa_QNX-bench/handoff.json"
DEFAULT_STATUS_URL = "http://127.0.0.1:8080/status.json"
DEFAULT_CONTROL_URL = "http://127.0.0.1:8082/control.json"


class HandoffMap:
    """Piecewise-linear laptop_x -> pan; ``extrapolate`` continues the end segments, clamped to ``bounds``."""

    def __init__(self, points=(), *, extrapolate: bool = False,
                 bounds: tuple[float, float] | None = None) -> None:
        self.extrapolate, self.bounds = extrapolate, bounds
        cleaned = {}
        for point in points:
            x, pan = (float(v) for v in point)
            if not (math.isfinite(x) and math.isfinite(pan) and 0.0 <= x <= 1.0 and 0.0 <= pan <= 180.0):
                raise ValueError(f"bad handoff point {point!r}: laptop_x must be 0..1, pan 0..180")
            cleaned[round(x, 4)] = pan  # One pan per laptop_x: a re-record replaces it.
        self.points = sorted(cleaned.items())

    @property
    def usable(self) -> bool:
        return len(self.points) >= 2

    def angle(self, laptop_x: float | None) -> float | None:
        """Pan angle for a laptop image x, or None without a usable map or an x."""
        if not self.usable or laptop_x is None:
            return None
        points = self.points
        if laptop_x <= points[0][0]:
            pan = self._line(points[0], points[1], laptop_x) if self.extrapolate else points[0][1]
        elif laptop_x >= points[-1][0]:
            pan = self._line(points[-2], points[-1], laptop_x) if self.extrapolate else points[-1][1]
        else:
            pan = next(self._line(a, b, laptop_x) for a, b in zip(points, points[1:]) if laptop_x <= b[0])
        if self.bounds is not None:
            pan = max(self.bounds[0], min(self.bounds[1], pan))
        return pan

    @staticmethod
    def _line(a, b, x: float) -> float:
        return a[1] + (x - a[0]) / (b[0] - a[0]) * (b[1] - a[1])

    def with_point(self, laptop_x: float, pan: float) -> "HandoffMap":
        return HandoffMap(self.points + [(laptop_x, pan)])

    @classmethod
    def load(cls, path) -> "HandoffMap":
        """Read a map file; a missing file is an empty map, a malformed one raises ValueError."""
        path = Path(os.path.expanduser(str(path)))
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return cls()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read {path}: {error}") from error
        points = data.get("points") if isinstance(data, dict) else data
        if not isinstance(points, list) or not all(isinstance(p, list) and len(p) == 2 for p in points):
            raise ValueError(f"{path}: expected {{\"points\": [[laptop_x, pan_deg], ...]}}")
        try:
            return cls(points)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error

    def save(self, path) -> None:
        path = Path(os.path.expanduser(str(path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps({"points": [list(p) for p in self.points]}, indent=1) + "\n")
        os.replace(temporary, path)  # Atomic: the running service never reads half a file.


def _file_stamp(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


class HandoffMapFile:
    """The map file as the running service sees it: reloaded when it changes, checked once a second."""

    def __init__(self, path, *, check_every: float = 1.0, clock=time.monotonic) -> None:
        self.path = Path(os.path.expanduser(str(path)))
        self.check_every, self.clock = check_every, clock
        self._checked_at: float | None = None
        self._stamp = None
        self.map = HandoffMap()
        self.error: str | None = None

    def current(self) -> HandoffMap:
        now = self.clock()
        if self._checked_at is not None and now - self._checked_at < self.check_every:
            return self.map
        self._checked_at = now
        stamp = _file_stamp(self.path)
        if stamp != self._stamp:
            self._stamp = stamp
            try:
                self.map, self.error = HandoffMap.load(self.path), None
            except ValueError as error:
                self.map, self.error = HandoffMap(), str(error)
                print(f"warning: handoff map ignored: {error}", file=sys.stderr)
        return self.map

    def save(self, handoff: HandoffMap) -> None:
        """Write the map and remember the result, so our own write is not reloaded as a change."""
        handoff.save(self.path)
        self.map, self.error, self._stamp = handoff, None, _file_stamp(self.path)


class HandoffLearner:
    """The handoff map, learned while the turret camera tracks a face the laptop also sees.

    observe() each frame with the laptop's target_x (None while the laptop is
    inactive), the turret image error (None without a tracked face) and the
    real pan the Arduino reports (None when stale). Once the face has stayed
    within ``centred`` of image centre for ``steady_s`` with the real pan moving
    less than ``steady_deg`` over that time, each frame folds (laptop_x, pan)
    into the bin for that laptop_x (an EMA with weight ``alpha``). current()
    fits the bins (at least 2 for a usable map); maybe_save() writes them at
    most every ``save_every`` s. A file changed by someone else (the CLI's
    record or clear) replaces the bins, so a manual point is a bin like any other.
    """

    def __init__(self, path, *, minimum: float = 0.0, maximum: float = 180.0, bins: int = 20,
                 alpha: float = 0.2, centred: float = 0.03, steady_deg: float = 0.5, steady_s: float = 0.4,
                 save_every: float = 5.0, clock=time.monotonic) -> None:
        self.file = HandoffMapFile(path, clock=clock)
        self.minimum, self.maximum, self.bin_count = float(minimum), float(maximum), bins
        self.alpha, self.centred, self.steady_deg, self.steady_s = alpha, centred, steady_deg, steady_s
        self.save_every = save_every
        self.bins: dict[int, tuple[float, float]] = {}  # bin index -> (laptop_x, pan)
        self.dirty = False
        self._saved_at: float | None = None
        self._since: float | None = None
        self._window: deque[tuple[float, float]] = deque()  # (time, real pan) while centred
        self._seen = None
        self._map = HandoffMap()
        self._sync()

    def _bin(self, laptop_x: float) -> int:
        return min(self.bin_count - 1, max(0, int(laptop_x * self.bin_count)))

    def _sync(self) -> None:
        """Take the file's points as the bins when the file changed under us."""
        loaded = self.file.current()
        if loaded is self._seen:
            return
        self._seen, grouped = loaded, {}
        for x, pan in loaded.points:
            grouped.setdefault(self._bin(x), []).append((x, pan))
        self.bins = {i: (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))
                     for i, ps in grouped.items()}
        self.dirty = False  # The file is the truth now; learning since our last save is dropped.
        self._refit()

    def _refit(self) -> None:
        self._map = HandoffMap(sorted(self.bins.values()), extrapolate=True,
                               bounds=(self.minimum, self.maximum))

    def current(self) -> HandoffMap:
        self._sync()
        return self._map

    @property
    def learned_points(self) -> int:
        return len(self.bins)

    def observe(self, now: float, laptop_x: float | None, image_error: float | None,
                real_pan: float | None) -> bool:
        """Feed one frame; return True when it recorded a point."""
        if (laptop_x is None or image_error is None or real_pan is None or abs(image_error) >= self.centred
                or not 0.0 <= laptop_x <= 1.0):
            self._since = None
            self._window.clear()
            return False
        if self._since is None:
            self._since = now
        self._window.append((now, float(real_pan)))
        while self._window[0][0] < now - self.steady_s:
            self._window.popleft()
        pans = [pan for _t, pan in self._window]
        if now - self._since < self.steady_s or max(pans) - min(pans) >= self.steady_deg:
            return False
        pan = max(self.minimum, min(self.maximum, float(real_pan)))
        index = self._bin(laptop_x)
        old = self.bins.get(index)
        self.bins[index] = (laptop_x, pan) if old is None else (
            old[0] + self.alpha * (laptop_x - old[0]), old[1] + self.alpha * (pan - old[1]))
        self.dirty = True
        self._refit()
        return True

    def maybe_save(self, now: float, *, force: bool = False) -> bool:
        """Write the bins when they changed and ``save_every`` has passed (or ``force``)."""
        if not self.dirty or (not force and self._saved_at is not None and now - self._saved_at < self.save_every):
            return False
        self._sync()  # A point recorded by hand since the last check wins over unsaved learning.
        if not self.dirty or self.file.error is not None:
            return False  # Never overwrite a file we could not read: the user fixes or clears it.
        self._saved_at = now
        try:
            self.file.save(HandoffMap(sorted(self.bins.values())))
        except OSError as error:
            print(f"warning: handoff map not saved: {error}", file=sys.stderr)
            return False
        self._seen, self.dirty = self.file.map, False
        return True


def _fetch_json(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def record_now(status_url: str, control_url: str, fetch=_fetch_json) -> tuple[float, float, str]:
    """Return (laptop_x, pan, pan source) from the live services, or raise RuntimeError."""
    try:
        status = fetch(status_url)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read {status_url}: {error}") from error
    laptop_x = status.get("laptop_target_x")
    if not status.get("laptop_active") or not isinstance(laptop_x, (int, float)):
        raise RuntimeError("the laptop does not see a peeper right now (no laptop_target_x)")
    if status.get("pan_state") != "tracking":
        raise RuntimeError(f"the turret is not tracking a face (pan_state {status.get('pan_state')!r})")
    try:
        arduino = fetch(control_url).get("arduino") or {}
    except (OSError, ValueError):
        arduino = {}
    if arduino.get("link_ok") and isinstance(arduino.get("pan"), (int, float)):
        return float(laptop_x), float(arduino["pan"]), "Arduino actual pan"
    pan = status.get("pan_angle")
    if not isinstance(pan, (int, float)):
        raise RuntimeError("no pan angle from the control process or the AI service")
    return float(laptop_x), float(pan), "AI commanded pan (control process unavailable)"


def describe(handoff: HandoffMap) -> str:
    if not handoff.points:
        return "handoff map is empty"
    lines = [f"  laptop_x {x:.3f} -> pan {pan:.1f}" for x, pan in handoff.points]
    if not handoff.usable:
        lines.append("  (needs at least 2 points before the turret uses it)")
    return "\n".join(lines)


def main(argv=None, *, fetch=_fetch_json) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m medusa.handoff", description="Edit the handoff map.")
    parser.add_argument("--map", default=DEFAULT_PATH, help=f"map file (default {DEFAULT_PATH})")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="add or replace a point")
    record.add_argument("laptop_x", type=float)
    record.add_argument("pan_deg", type=float)
    now = commands.add_parser("record-now", help="record the live laptop_x and turret pan")
    now.add_argument("--status-url", default=DEFAULT_STATUS_URL)
    now.add_argument("--control-url", default=DEFAULT_CONTROL_URL)
    commands.add_parser("show")
    commands.add_parser("clear")
    args = parser.parse_args(argv)
    try:
        handoff = HandoffMap.load(args.map)
        if args.command == "show":
            print(f"{args.map}:\n{describe(handoff)}")
            return 0
        if args.command == "clear":
            HandoffMap().save(args.map)
            print(f"cleared {args.map}")
            return 0
        if args.command == "record":
            laptop_x, pan, source = args.laptop_x, args.pan_deg, "command line"
        else:
            laptop_x, pan, source = record_now(args.status_url, args.control_url, fetch)
        handoff = handoff.with_point(laptop_x, pan)
        handoff.save(args.map)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"recorded laptop_x {laptop_x:.3f} -> pan {pan:.1f} ({source})\n{describe(handoff)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
