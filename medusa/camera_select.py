"""Find the real turret camera among QNX Sensor Framework units.

QNX renumbers units when the USB camera is replugged, and units 1 and 2 are built-in test-pattern
cameras that open fine and return 1280x720 vertical colour bars. A fixed unit can therefore select
a test pattern silently, so ``--qnx-camera-unit auto`` opens each unit and keeps the first one whose
frames look like a real scene.
"""

from __future__ import annotations

import time

import numpy as np

# The C270 was unit 5 before the replug, so try it first.
AUTO_UNITS = (5, 1, 2, 3, 4, 6, 7, 8, 9, 10)
SETTLE_S = 0.5
FRAMES = 40  # ~2-3 s at the C270's rate: black frames while exposure settles must not reject it
# A column is "constant" when its vertical spread is under this (0-255 scale, worst channel).
# Probed on the Pi: QNX's test patterns read exactly 0; sensor noise alone is well above 1.
COLUMN_STD_MAX = 1.0
CONSTANT_COLUMN_SHARE = 0.9
BLANK_STD_MAX = 2.0

REAL = "real"
TEST_PATTERN = "test pattern"
BLANK = "blank"


class NoCameraError(RuntimeError):
    pass


def classify_frame(frame) -> tuple[str, str]:
    """Return (kind, why): kind is REAL, TEST_PATTERN or BLANK."""
    pixels = np.asarray(frame, dtype=np.float32)[::4, ::4]
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    if pixels.reshape(-1, pixels.shape[2]).std(axis=0).max() < BLANK_STD_MAX:
        return BLANK, "all one colour"
    column_std = pixels.std(axis=0).max(axis=1)
    share = float((column_std < COLUMN_STD_MAX).mean())
    if share > CONSTANT_COLUMN_SHARE:
        return TEST_PATTERN, f"{share:.0%} of columns constant top to bottom (colour bars)"
    return REAL, f"{share:.0%} of columns constant"


def probe_unit(camera, *, frames: int = FRAMES, settle_s: float = SETTLE_S, sleep=time.sleep) -> tuple[str, str]:
    """Read up to ``frames`` frames after ``settle_s``; REAL on the first real-looking frame.

    A real camera can start with black frames while exposure settles, so one real frame is enough.
    """
    sleep(settle_s)
    kind, why = None, "no frames"
    for _ in range(frames):
        ok, frame = camera.read()
        if not ok or frame is None:
            continue
        kind, why = classify_frame(frame)
        if kind in (REAL, TEST_PATTERN):  # A test pattern never turns into a scene: stop reading.
            return kind, why
    return kind or "no frames", why


def find_camera(open_camera, units=AUTO_UNITS, *, log=print, skip=(), **probe) -> tuple[object, int]:
    """Open each unit in order and return (camera, unit) for the first real camera.

    Units in ``skip`` (test patterns found earlier) are counted as test patterns without opening.
    Every other opened unit is released. Raises NoCameraError naming the units tried and where the
    test patterns were."""
    patterns, blank, failed, quiet = [], [], [], []
    for unit in units:
        if unit in skip:
            patterns.append(unit)
            continue
        try:
            camera = open_camera(unit)
        except (RuntimeError, OSError, ValueError) as error:
            failed.append(unit)
            log(f"camera unit {unit}: rejected, did not open ({error})")
            continue
        try:
            kind, why = probe_unit(camera, **probe)
        except BaseException:
            camera.release()
            raise
        if kind == REAL:
            log(f"camera unit {unit}: using it, real camera ({why})")
            return camera, unit
        camera.release()
        {TEST_PATTERN: patterns, BLANK: blank}.get(kind, quiet).append(unit)
        log(f"camera unit {unit}: rejected, {kind} ({why})")
    detail = f"units tried: {_units(units)}; test patterns at: {_units(patterns) or 'none'}"
    if blank:
        detail += f"; blank at: {_units(blank)}"
    if quiet:
        detail += f"; no frames at: {_units(quiet)}"
    error = NoCameraError(f"no real camera found ({detail})")
    error.test_patterns = tuple(patterns)
    raise error


WAITING_REASON = "waiting for the turret camera"
FRAME_REASON = "camera frame unavailable"
RETRY_S = 3.0
LOST_AFTER = 3  # failed reads (about 1 s each on QnxCamera) before the camera counts as gone


class AutoCamera:
    """A camera that finds itself and comes back: ``read()`` never raises for a missing camera.

    With no camera it re-runs find_camera every ``retry_s`` and returns (False, None) meanwhile, so
    the service loop homes the turret and keeps serving status. After ``lost_after`` failed reads in
    a row it releases the camera and starts searching again. ``unavailable_reason`` says which."""

    def __init__(self, open_camera, units=AUTO_UNITS, *, retry_s: float = RETRY_S,
                 lost_after: int = LOST_AFTER, log=print, clock=time.monotonic, sleep=time.sleep,
                 **probe) -> None:
        self._open, self._units, self._probe = open_camera, tuple(units), probe
        self._retry_s, self._lost_after = retry_s, lost_after
        self._log, self._clock, self._sleep = log, clock, sleep
        self._camera, self.unit = None, None
        self._failures = 0
        self._next_search = None  # None: search on the next read
        self._last_message = None
        self._patterns: set[int] = set()  # built-in test patterns stay test patterns: skip them later

    @property
    def unavailable_reason(self) -> str:
        return WAITING_REASON if self._camera is None else FRAME_REASON

    def read(self):
        if self._camera is None and not self._search():
            return False, None
        ok, frame = self._camera.read()
        if ok:
            self._failures = 0
            return ok, frame
        self._failures += 1
        if self._failures >= self._lost_after:
            self._log(f"camera unit {self.unit}: no frames for {self._failures} reads, searching again")
            self._drop()
        return False, None

    def _search(self) -> bool:
        if self._next_search is not None:
            wait = self._next_search - self._clock()
            if wait > 0:
                self._sleep(wait)
        quiet = []
        try:
            self._camera, self.unit = find_camera(self._open, self._units, log=quiet.append,
                                                  skip=self._patterns, sleep=self._sleep, **self._probe)
        except NoCameraError as error:
            self._patterns.update(error.test_patterns)
            message = f"{error}; {WAITING_REASON}, retrying every {self._retry_s:g} s"
            if message != self._last_message:  # Log changes only: this repeats every few seconds.
                for line in quiet:
                    self._log(line)
                self._log(message)
                self._last_message = message
            self._next_search = self._clock() + self._retry_s
            return False
        for line in quiet:
            self._log(line)
        self._failures, self._last_message, self._next_search = 0, None, None
        return True

    def _drop(self) -> None:
        camera, self._camera = self._camera, None
        self._failures = 0
        self._next_search = None
        try:
            camera.release()
        except (RuntimeError, OSError):
            pass

    def release(self) -> None:
        if self._camera is not None:
            self._drop()


def _units(units) -> str:
    return ", ".join(str(unit) for unit in units)
