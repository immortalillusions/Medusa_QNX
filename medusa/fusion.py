"""Opt-in (--fusion) two-camera fusion: a distance-aware handoff map and a bearing estimator.

Nothing here runs unless qnx_ai_service.py is started with --fusion; the default
tracking paths in medusa/trigger.py and medusa/handoff.py do not use it.

Distance bands: the laptop heartbeat's optional target_size (peeper face width /
laptop frame width) picks a band, and BandedHandoffLearner learns laptop_x -> pan
separately per band (the laptop and turret cameras sit apart, so the same laptop x
is a different pan for a near and a far peeper). A band with fewer than 2 points
falls back to the single map, which is still learned from every point as before.

BearingFusion: a 1D constant-velocity Kalman filter on the peeper's pan bearing and
bearing rate. Each frame it takes the turret camera's measurement (a face is
precise, a head less so) and the laptop map's estimate (rough, and rougher the
further laptop_x lies past the points the map has learned); the turret aims at the
fused bearing (PanTracker with ``fusion=``).
"""

from __future__ import annotations

import math

from .handoff import HandoffLearner, HandoffMap, monotonic_fit

# Face-size bands, from the face sizes measured on the bench clips.
NEAR_SIZE = 0.10  # target_size above this: near
FAR_SIZE = 0.04   # target_size below this: far; in between: mid
BANDS = ("near", "mid", "far")
MIN_BAND_POINTS = 2  # a band map with fewer points falls back to the single map

# Measurement variances, deg^2.
FACE_VAR = 1.0    # turret camera face: about 1 deg
HEAD_VAR = 9.0    # turret camera head (profile/body box): about 3 deg
MAP_VAR = 25.0    # laptop map inside its learned points: about 5 deg
MAP_EXTRAPOLATE_SD = 100.0  # extra deg of map sd per laptop image width past the learned points
ACCEL_VAR = 400.0           # process noise: white bearing acceleration, (deg/s^2)^2 per s
MAX_RATE = 90.0             # deg/s: a faster fused rate is noise
JUMP_DEG = 25.0             # a turret measurement this far off is a new target: restart on it
GATE_SIGMA = 3.0            # a map estimate this many sd off the fusion is ignored while a turret measurement is in
START_VAR = 4.0             # bearing variance when fusion starts from the turret's own angle (map only)
START_RATE_VAR = 25.0
GIVE_UP_VAR = 400.0         # with no measurement at all, stop coasting on the fusion past 20 deg sd


def distance_band(target_size: float | None) -> str | None:
    """The band for a face size (face width / frame width), or None without one."""
    if target_size is None or not math.isfinite(target_size) or target_size <= 0:
        return None
    if target_size > NEAR_SIZE:
        return "near"
    return "far" if target_size < FAR_SIZE else "mid"


def map_variance(laptop_x: float, domain: tuple[float, float] | None) -> float:
    """Variance (deg^2) of the map's pan for ``laptop_x``: MAP_VAR inside the learned
    x range ``domain``, its sd growing linearly with the distance past it."""
    past = 0.0 if domain is None else max(0.0, domain[0] - laptop_x, laptop_x - domain[1])
    return (math.sqrt(MAP_VAR) + MAP_EXTRAPOLATE_SD * past) ** 2


def turret_variance(aim_source: str | None) -> float:
    return HEAD_VAR if aim_source not in (None, "face") else FACE_VAR


class BandedHandoffLearner(HandoffLearner):
    """HandoffLearner that also learns one map per distance band.

    observe() takes ``target_size``; a frame the base learner records also folds into
    that band's bins (same bins and EMA). map_for(target_size) is the band's monotonic
    fit when it has MIN_BAND_POINTS bins, else the single map. Band bins are saved under
    "bands" in the same file, with the same fingerprint rules.
    """

    def __init__(self, path, **kwargs) -> None:
        self.band_bins: dict[str, dict[int, tuple[float, float]]] = {band: {} for band in BANDS}
        self._band_maps: dict[str, HandoffMap] = {}
        super().__init__(path, **kwargs)

    def _sync(self) -> None:
        seen = self._seen
        super()._sync()
        if self._seen is seen:
            return
        self.band_bins = {band: {} for band in BANDS}
        for band, points in self.band_points.items():
            if band not in self.band_bins:
                continue
            grouped = {}
            for x, pan in points:
                grouped.setdefault(self._bin(x), []).append((x, pan))
            self.band_bins[band] = {i: (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))
                                    for i, ps in grouped.items()}
        self._refit_bands()

    def _refit_bands(self) -> None:
        self._band_maps = {}
        for band, bins in self.band_bins.items():
            fitted, _direction, _rejected = monotonic_fit(list(bins.values()))
            self._band_maps[band] = HandoffMap(fitted, extrapolate=True, bounds=(self.minimum, self.maximum))

    def _bands_to_save(self) -> dict:
        return {band: sorted(bins.values()) for band, bins in self.band_bins.items() if bins}

    def band_points_count(self, band: str) -> int:
        return len(self.band_bins.get(band, {}))

    def observe(self, now, laptop_x, image_error, real_pan, *, target_size: float | None = None,
                **kwargs) -> bool:
        recorded = super().observe(now, laptop_x, image_error, real_pan, **kwargs)
        band = distance_band(target_size)
        if recorded and band is not None:
            pan = max(self.minimum, min(self.maximum, float(real_pan)))
            bins, index = self.band_bins[band], self._bin(laptop_x)
            old = bins.get(index)
            bins[index] = (laptop_x, pan) if old is None else (
                old[0] + self.alpha * (laptop_x - old[0]), old[1] + self.alpha * (pan - old[1]))
            self._refit_bands()
        return recorded

    def band_for(self, target_size: float | None) -> str:
        """The band whose map map_for() uses, or "all" for the single map."""
        self.current()
        band = distance_band(target_size)
        band_map = self._band_maps.get(band) if band is not None else None
        return band if band_map is not None and len(band_map.points) >= MIN_BAND_POINTS else "all"

    def map_for(self, target_size: float | None) -> HandoffMap:
        band = self.band_for(target_size)
        return self._map if band == "all" else self._band_maps[band]


class BearingFusion:
    """Constant-velocity Kalman filter on pan bearing (deg) and bearing rate (deg/s).

    step(t, turret, mapped, start) once per frame: ``turret`` is (bearing, variance) from
    the turret camera or None, ``mapped`` (bearing, variance) from the laptop map or None,
    ``start`` the angle to begin from when the first measurement is the map's (so fusion
    glides from where the turret points instead of jumping), ``turret_source`` the label
    ("face"/"head") listed in ``sources``. Returns the fused bearing, or None when it has
    nothing.
    """

    def __init__(self, *, accel_var: float = ACCEL_VAR, max_rate: float = MAX_RATE,
                 jump_deg: float = JUMP_DEG, gate_sigma: float = GATE_SIGMA) -> None:
        self.accel_var, self.max_rate, self.jump_deg, self.gate_sigma = accel_var, max_rate, jump_deg, gate_sigma
        self.reset()

    def reset(self) -> None:
        self.x: list[float] | None = None  # [bearing, rate]
        self.p: list[list[float]] | None = None
        self.t: float | None = None
        self.sources: list[str] = []

    @property
    def bearing(self) -> float | None:
        return None if self.x is None else self.x[0]

    @property
    def rate(self) -> float:
        return 0.0 if self.x is None else self.x[1]

    @property
    def var(self) -> float | None:
        return None if self.p is None else self.p[0][0]

    def _start(self, bearing: float, var: float, rate_var: float, t: float) -> None:
        self.x, self.p, self.t = [float(bearing), 0.0], [[float(var), 0.0], [0.0, float(rate_var)]], t

    def predict(self, t: float) -> None:
        dt = max(0.0, t - self.t)
        self.t = max(self.t, t)
        if dt <= 0:
            return
        (p00, p01), (p10, p11) = self.p
        q = self.accel_var
        self.x[0] += self.x[1] * dt
        self.p = [[p00 + dt * (p01 + p10) + dt * dt * p11 + q * dt ** 3 / 3, p01 + dt * p11 + q * dt * dt / 2],
                  [p10 + dt * p11 + q * dt * dt / 2, p11 + q * dt]]

    def innovation(self, z: float, r: float) -> tuple[float, float]:
        return z - self.x[0], self.p[0][0] + r

    def correct(self, z: float, r: float) -> None:
        y, s = self.innovation(z, r)
        (p00, p01), (p10, p11) = self.p
        k0, k1 = p00 / s, p10 / s
        self.x[0] += k0 * y
        self.x[1] = max(-self.max_rate, min(self.max_rate, self.x[1] + k1 * y))
        self.p = [[(1 - k0) * p00, (1 - k0) * p01], [p10 - k1 * p00, p11 - k1 * p01]]

    def step(self, t: float, turret: tuple[float, float] | None = None,
             mapped: tuple[float, float] | None = None, start: float | None = None,
             turret_source: str = "face") -> float | None:
        self.sources = []
        if self.x is None:
            if turret is not None:
                self._start(turret[0], turret[1], START_RATE_VAR, t)
                self.sources.append(turret_source)
                turret = None
            elif mapped is not None and start is not None:
                self._start(start, START_VAR, START_RATE_VAR, t)
            elif mapped is not None:
                self._start(mapped[0], mapped[1], START_RATE_VAR, t)
                self.sources.append("map")
                return self.bearing
            else:
                return None
        self.predict(t)
        if turret is not None:
            if abs(turret[0] - self.x[0]) > self.jump_deg:
                self._start(turret[0], turret[1], START_RATE_VAR, t)  # A new target: the face wins outright.
            else:
                self.correct(*turret)
            self.sources.append(turret_source)
        if mapped is not None:
            y, s = self.innovation(*mapped)
            if not (self.sources and y * y > self.gate_sigma ** 2 * s):
                self.correct(*mapped)
                self.sources.append("map")
        return self.bearing
