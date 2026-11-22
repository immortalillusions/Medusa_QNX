"""Laptop-to-turret handoff map: where to point the turret when only the laptop sees the peeper.

The laptop heartbeat carries target_x (0..1 across the laptop image). The map
turns that into a pan angle, piecewise-linear between recorded points and
clamped to the end points outside them. It needs at least two points.

The running service also learns the map itself (HandoffLearner): whenever the
laptop sees the peeper, the turret is armed with its kill gate closed, and the
turret camera sees exactly one face, matched as the peeper, held centred on a
steady turret, it records (laptop_x, real pan) into 20 bins across laptop_x,
each an average of what it saw there. The map it uses is a monotonic fit of the
bins (monotonic_fit: isotonic regression in the data's dominant direction, bins
more than 8 deg off the fit rejected); it extrapolates past its outermost points
along the fit's end slopes (clamped to the pan range) and is saved to the same
file at most every 5 s; points recorded by hand below count as bins too.

The file carries a setup fingerprint (pan home/min/max, --servo-reversed, camera
hfov). The service discards the points of a file made for another setup, or
with no fingerprint (files from before it), backing the file up before its
first save over it.

    python3 -m medusa.handoff record <laptop_x> <pan_deg>   # add or replace a point
    python3 -m medusa.handoff record-now                    # laptop_x and pan from the live services
    python3 -m medusa.handoff show
    python3 -m medusa.handoff clear                         # backs the file up next to it first

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
import shutil
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

DEFAULT_PATH = "~/Medusa_QNX-bench/handoff.json"
DEFAULT_STATUS_URL = "http://127.0.0.1:8080/status.json"
DEFAULT_CONTROL_URL = "http://127.0.0.1:8082/control.json"
MAX_FIT_DEVIATION_DEG = 8.0
MIN_DIRECTION_POINTS = 3


def setup_fingerprint(pan_home: float, pan_min: float, pan_max: float, servo_reversed: bool,
                      hfov_deg: float) -> dict:
    """What a learned map depends on: change any of these and the old points are wrong."""
    return {"pan_home": float(pan_home), "pan_min": float(pan_min), "pan_max": float(pan_max),
            "servo_reversed": bool(servo_reversed), "hfov_deg": round(float(hfov_deg), 2)}


def _ranks(values) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # Ties share their average rank.
        i = j + 1
    return ranks


def _slope_sign(xs, ys) -> int:
    """Sign of the Pearson correlation of ``xs`` and ``ys`` (0 when flat)."""
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    covariance = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return (covariance > 1e-12) - (covariance < -1e-12)


def _isotonic(ys, increasing: bool) -> list[float]:
    """Least-squares monotonic fit (pool adjacent violators)."""
    sign = 1.0 if increasing else -1.0
    blocks: list[list[float]] = []  # [sum, count]
    for y in ys:
        blocks.append([sign * y, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    fitted = []
    for total, count in blocks:
        fitted += [sign * total / count] * int(count)
    return fitted


def monotonic_fit(points, *, max_deviation: float = MAX_FIT_DEVIATION_DEG,
                  min_points: int = MIN_DIRECTION_POINTS):
    """Fit (laptop_x, pan) points with a monotonic curve; return (fitted points, direction, rejected).

    The direction ("increasing"/"decreasing") is the sign of the Spearman rank
    correlation (Pearson on the raw values breaks a tie), trusted only with at
    least ``min_points`` points; without one the points come back as they are.
    Points more than ``max_deviation`` deg from the isotonic fit of all points
    are rejected and the rest refitted; then every point is checked again against
    that curve (interpolated, end slopes extended), so a good neighbour that a big
    outlier dragged off the first fit comes back, until the kept set is stable.
    The fitted points are monotonic in the found direction; the rejected indices
    are in x order.
    """
    points = sorted((float(x), float(pan)) for x, pan in points)
    if len(points) < min_points:
        return points, None, []
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    sign = _slope_sign(_ranks(xs), _ranks(ys)) or _slope_sign(xs, ys)
    if sign == 0:
        return points, None, []
    increasing = sign > 0
    first = _isotonic(ys, increasing)
    kept = [i for i in range(len(points)) if abs(ys[i] - first[i]) <= max_deviation]
    for _ in range(len(points)):
        if len(kept) < 2:
            kept = list(range(len(points)))  # Nothing consistent to keep: the plain fit of everything.
            break
        fitted = _isotonic([ys[i] for i in kept], increasing)
        curve = HandoffMap([(xs[i], pan) for i, pan in zip(kept, fitted)], extrapolate=True)
        again = [i for i in range(len(points)) if abs(ys[i] - curve.angle(xs[i])) <= max_deviation]
        if again == kept:
            break
        kept = again
    fitted = _isotonic([ys[i] for i in kept], increasing)
    rejected = [i for i in range(len(points)) if i not in kept]
    return ([(xs[i], pan) for i, pan in zip(kept, fitted)], "increasing" if increasing else "decreasing",
            rejected)


def backup_file(path) -> Path | None:
    """Copy ``path`` to ``<stem>.backup-YYYYmmdd-HHMMSS<suffix>`` beside it; None if there is no file."""
    path = Path(os.path.expanduser(str(path)))
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup, n = path.with_name(f"{path.stem}.backup-{stamp}{path.suffix}"), 1
    while backup.exists():
        backup, n = path.with_name(f"{path.stem}.backup-{stamp}-{n}{path.suffix}"), n + 1
    shutil.copy2(path, backup)
    return backup


class HandoffMap:
    """Piecewise-linear laptop_x -> pan; ``extrapolate`` continues the end segments, clamped to ``bounds``.

    ``fingerprint`` (setup_fingerprint) is the setup the points were made for; it
    is kept through with_point and save/load. ``bands`` ({band name: points}) are the
    per-distance-band points --fusion learns (medusa/fusion.py); they ride along in the
    same file under "bands", which older code ignores, and a file without them is
    written exactly as before.
    """

    def __init__(self, points=(), *, extrapolate: bool = False,
                 bounds: tuple[float, float] | None = None, fingerprint: dict | None = None,
                 bands: dict | None = None) -> None:
        self.extrapolate, self.bounds, self.fingerprint = extrapolate, bounds, fingerprint
        self.bands = {str(name): HandoffMap(band).points for name, band in (bands or {}).items() if band}
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
        return HandoffMap(self.points + [(laptop_x, pan)], fingerprint=self.fingerprint, bands=self.bands)

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
        fingerprint = data.get("fingerprint") if isinstance(data, dict) else None
        bands = data.get("bands") if isinstance(data, dict) else None
        if bands is not None and not (isinstance(bands, dict) and all(
                isinstance(band, list) and all(isinstance(p, list) and len(p) == 2 for p in band)
                for band in bands.values())):
            raise ValueError(f"{path}: expected \"bands\": {{name: [[laptop_x, pan_deg], ...]}}")
        try:
            return cls(points, fingerprint=fingerprint if isinstance(fingerprint, dict) else None, bands=bands)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error

    def save(self, path) -> None:
        path = Path(os.path.expanduser(str(path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        data = {"points": [list(p) for p in self.points]}
        if self.fingerprint is not None:
            data["fingerprint"] = self.fingerprint
        if self.bands:
            data["bands"] = {name: [list(p) for p in band] for name, band in sorted(self.bands.items())}
        temporary.write_text(json.dumps(data, indent=1) + "\n")
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
    inactive), the turret image error (None without a tracked face), the real
    pan the Arduino reports (None when stale), and the frame's turret face count,
    how the aimed face was matched (PeeperMatcher.match, "none" without one) and
    the Arduino's armed and kill-gate flags. Only while the turret is armed with
    the gate closed and the camera sees exactly one face, matched as the peeper
    ("map" or "single"), once the face has stayed within ``centred`` of image
    centre for ``steady_s`` with the real pan moving less than ``steady_deg``
    over that time, each frame folds (laptop_x, pan) into the bin for that
    laptop_x (an EMA with weight ``alpha``; a bin the fit rejected is replaced
    outright). current() is monotonic_fit of the bins (at least 2 for a usable
    map); maybe_save() writes the bins at most every ``save_every`` s. A file
    changed by someone else (the CLI's record or clear) replaces the bins, so a
    manual point is a bin like any other, unless its fingerprint is not
    ``fingerprint`` (when one is given): then its points are discarded.
    """

    LEARN_MATCHES = ("map", "single")

    def __init__(self, path, *, minimum: float = 0.0, maximum: float = 180.0, bins: int = 20,
                 alpha: float = 0.2, centred: float = 0.03, steady_deg: float = 0.5, steady_s: float = 0.4,
                 save_every: float = 5.0, fingerprint: dict | None = None, clock=time.monotonic) -> None:
        self.file = HandoffMapFile(path, clock=clock)
        self.fingerprint = fingerprint
        self.direction: str | None = None
        self.rejected: set[int] = set()  # bin indices the monotonic fit rejected
        self._backup_first = False  # the file's points were discarded: keep a copy before overwriting it
        self.minimum, self.maximum, self.bin_count = float(minimum), float(maximum), bins
        self.alpha, self.centred, self.steady_deg, self.steady_s = alpha, centred, steady_deg, steady_s
        self.save_every = save_every
        self.bins: dict[int, tuple[float, float]] = {}  # bin index -> (laptop_x, pan)
        self.band_points: dict[str, list] = {}  # the file's --fusion band points, saved back untouched
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
        points, self.band_points = loaded.points, dict(loaded.bands)
        if self.fingerprint is not None and (points or self.band_points) and loaded.fingerprint != self.fingerprint:
            print(f"warning: handoff map {self.file.path} discarded: made for setup "
                  f"{loaded.fingerprint or '(no fingerprint)'}, running {self.fingerprint}; "
                  f"it is backed up before the first save over it", file=sys.stderr)
            points, self.band_points, self._backup_first = [], {}, True  # bands belong to the setup too
        for x, pan in points:
            grouped.setdefault(self._bin(x), []).append((x, pan))
        self.bins = {i: (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))
                     for i, ps in grouped.items()}
        self.dirty = False  # The file is the truth now; learning since our last save is dropped.
        self._refit()

    def _bands_to_save(self) -> dict:
        return self.band_points

    def _refit(self) -> None:
        items = sorted(self.bins.items(), key=lambda item: item[1])
        fitted, self.direction, rejected = monotonic_fit([point for _i, point in items])
        self.rejected = {items[k][0] for k in rejected}
        self._map = HandoffMap(fitted, extrapolate=True, bounds=(self.minimum, self.maximum))

    def current(self) -> HandoffMap:
        self._sync()
        return self._map

    @property
    def learned_points(self) -> int:
        return len(self.bins)

    @property
    def rejected_points(self) -> int:
        return len(self.rejected)

    def observe(self, now: float, laptop_x: float | None, image_error: float | None,
                real_pan: float | None, *, faces: int | None, match: str, armed: bool | None,
                gate_closed: bool | None) -> bool:
        """Feed one frame; return True when it recorded a point."""
        if (laptop_x is None or image_error is None or real_pan is None or abs(image_error) >= self.centred
                or not 0.0 <= laptop_x <= 1.0 or faces != 1 or match not in self.LEARN_MATCHES
                or armed is not True or gate_closed is not True):
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
        old = None if index in self.rejected else self.bins.get(index)
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
            if self._backup_first:
                backup = backup_file(self.file.path)
                if backup is not None:
                    print(f"handoff map: the discarded file is kept as {backup}", file=sys.stderr)
                self._backup_first = False
            self.file.save(HandoffMap(sorted(self.bins.values()), fingerprint=self.fingerprint,
                                      bands=self._bands_to_save()))
        except OSError as error:
            print(f"warning: handoff map not saved: {error}", file=sys.stderr)
            return False
        self._seen, self.dirty = self.file.map, False
        return True


def _fetch_json(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def record_now(status_url: str, control_url: str, fetch=_fetch_json, *, status_out: dict | None = None,
               ) -> tuple[float, float, str]:
    """Return (laptop_x, pan, pan source) from the live services, or raise RuntimeError.

    ``status_out`` receives the AI service's status.json (for its map_fingerprint).
    """
    try:
        status = fetch(status_url)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read {status_url}: {error}") from error
    if status_out is not None:
        status_out.update(status)
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
        if args.command == "clear":
            try:
                fingerprint = HandoffMap.load(args.map).fingerprint  # The setup is still the same one.
            except ValueError:
                fingerprint = None  # An unreadable file is still backed up as it is.
            backup = backup_file(args.map)
            HandoffMap(fingerprint=fingerprint).save(args.map)
            print(f"backed up to {backup}" if backup else "no file to back up")
            print(f"cleared {args.map}")
            return 0
        handoff = HandoffMap.load(args.map)
        if args.command == "show":
            print(f"{args.map}:\n{describe(handoff)}")
            return 0
        live = {}
        if args.command == "record":
            laptop_x, pan, source = args.laptop_x, args.pan_deg, "command line"
        else:
            laptop_x, pan, source = record_now(args.status_url, args.control_url, fetch, status_out=live)
        running = live.get("map_fingerprint")
        if handoff.fingerprint is None and not handoff.points and isinstance(running, dict):
            handoff.fingerprint = running  # A fresh file takes the running service's setup.
        handoff = handoff.with_point(laptop_x, pan)
        handoff.save(args.map)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"recorded laptop_x {laptop_x:.3f} -> pan {pan:.1f} ({source})\n{describe(handoff)}")
    if handoff.fingerprint is None or (isinstance(running, dict) and handoff.fingerprint != running):
        print("warning: this file has no setup fingerprint matching the running service, which discards "
              "its points; clear it and use record-now (or let the service learn)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
