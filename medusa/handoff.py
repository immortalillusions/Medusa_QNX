"""Laptop-to-turret handoff map: where to point the turret when only the laptop sees the peeper.

The laptop heartbeat carries target_x (0..1 across the laptop image). The map
turns that into a pan angle, piecewise-linear between recorded points and
clamped to the end points outside them. It needs at least two points.

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
from pathlib import Path

DEFAULT_PATH = "~/Medusa_QNX-bench/handoff.json"
DEFAULT_STATUS_URL = "http://127.0.0.1:8080/status.json"
DEFAULT_CONTROL_URL = "http://127.0.0.1:8082/control.json"


class HandoffMap:
    def __init__(self, points=()) -> None:
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
            return points[0][1]
        for (x0, p0), (x1, p1) in zip(points, points[1:]):
            if laptop_x <= x1:
                return p0 + (laptop_x - x0) / (x1 - x0) * (p1 - p0)
        return points[-1][1]

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
        try:
            stat = self.path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = None
        if stamp != self._stamp:
            self._stamp = stamp
            try:
                self.map, self.error = HandoffMap.load(self.path), None
            except ValueError as error:
                self.map, self.error = HandoffMap(), str(error)
                print(f"warning: handoff map ignored: {error}", file=sys.stderr)
        return self.map


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
