"""Stable peeper decision for the laptop camera: face tracks, smoothed gaze, time hysteresis.

SnooperSelector (decision.py) keys its streak on a 0.1-wide position bucket and resets on any
single not-looking frame, so a still viewer flickers. This module instead:

1. keeps an identity per face across frames (greedy IoU match, nearest-centre fallback),
   dropping a track only after ``track_ttl_s`` without a match;
2. smooths each track's gaze with a time-based EMA, holds the last eye angles through a
   blink, and ignores a single-frame jump larger than ``outlier_deg``;
3. judges "looking" with hysteresis on the smoothed angles;
4. keeps the laptop user (largest face) sticky, swapping only when another face is
   ``swap_ratio`` times larger for ``swap_hold_s``;
5. goes ACTIVE after one peeper track has looked for ``enter_s`` and stays ACTIVE until that
   track has not been seen looking for ``exit_s``.

Everything is in seconds, not frames, so 15 fps and 30 fps behave the same. Pure Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .gaze import GazeEstimate
from .models import SnooperDecision

EPS = 1e-6


@dataclass(frozen=True)
class TrackerConfig:
    yaw_max: float = 25.0
    pitch_min: float = -35.0
    pitch_max: float = 10.0
    yaw_margin: float = 7.0       # exit looking only past yaw_max + this
    pitch_margin: float = 7.0     # ... or this far outside the pitch range
    enter_s: float = 0.5
    exit_s: float = 0.7
    min_faces: int = 2            # 1 = any face may be the peeper (bench calibration)
    smooth_tau_s: float = 0.15
    blink_hold_s: float = 0.4
    outlier_deg: float = 30.0
    track_ttl_s: float = 0.5
    min_iou: float = 0.1
    match_distance: float = 0.15  # fraction of frame width
    swap_ratio: float = 1.25
    swap_hold_s: float = 0.5


@dataclass(frozen=True)
class Detection:
    """One face this frame, normalized to the frame (0..1)."""

    center_x: float
    center_y: float
    width: float
    height: float
    looking: bool                      # the frame's own verdict, used when gaze is None
    gaze: GazeEstimate | None = None
    source: str = "mac-camera"


@dataclass
class Track:
    id: int
    center_x: float
    center_y: float
    width: float
    height: float
    last_seen: float
    looking: bool = False              # hysteresis on the smoothed gaze
    looking_since: float | None = None
    frame_looking: bool = False        # this frame's de-spiked gaze, within the exit band
    gaze_yaw: float | None = None      # smoothed
    gaze_pitch: float | None = None
    method: str = "head"
    source: str = "mac-camera"
    _gaze_time: float | None = None
    _eye: tuple[float, float] | None = None
    _eye_time: float | None = None
    _last_outlier: bool = False

    @property
    def area(self) -> float:
        return self.width * self.height

    def looking_s(self, now: float) -> float:
        return now - self.looking_since if self.looking and self.looking_since is not None else 0.0


@dataclass(frozen=True)
class TrackerResult:
    decision: SnooperDecision
    tracks: list[Track]                # aligned with this frame's detections
    peeper_id: int | None
    user_id: int | None


def _alpha(dt: float, tau: float) -> float:
    return 1.0 if tau <= 0 else 1.0 - math.exp(-max(dt, 0.0) / tau)


def _iou(track: Track, det: Detection) -> float:
    ax0, ax1 = track.center_x - track.width / 2, track.center_x + track.width / 2
    ay0, ay1 = track.center_y - track.height / 2, track.center_y + track.height / 2
    bx0, bx1 = det.center_x - det.width / 2, det.center_x + det.width / 2
    by0, by1 = det.center_y - det.height / 2, det.center_y + det.height / 2
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = track.area + det.width * det.height - inter
    return inter / union if union > 0 else 0.0


@dataclass
class PeeperTracker:
    config: TrackerConfig = field(default_factory=TrackerConfig)
    tracks: dict[int, Track] = field(default_factory=dict)
    user_id: int | None = None
    peeper_id: int | None = None
    _next_id: int = 1
    _challenger: tuple[int, float] | None = None
    _peeper_ok_at: float = 0.0           # last time the active peeper was seen looking
    _peeper_center: tuple[float, float] = (0.0, 0.0)
    _peeper_source: str = "mac-camera"
    _aspect: float = 1.0

    # ---- per-frame entry point -------------------------------------------------------
    def update(self, detections: list[Detection], now: float, aspect: float = 1.0) -> TrackerResult:
        """``aspect`` = frame height / width, so the centre-distance fallback is in widths."""
        self._aspect = aspect
        matched = self._match(detections, aspect)
        aligned: list[Track] = []
        for index, det in enumerate(detections):
            track = matched.get(index)
            if track is None:
                track = self._new_track(det, now)
            else:
                self._observe(track, det, now)
            aligned.append(track)
        ttl = self.config.track_ttl_s
        self.tracks = {tid: t for tid, t in self.tracks.items() if now - t.last_seen <= ttl + EPS}
        self._pick_user(now)
        decision = self._decide(now, {t.id for t in aligned})
        return TrackerResult(decision, aligned, self.peeper_id, self.user_id)

    # ---- identity --------------------------------------------------------------------
    def _match(self, detections: list[Detection], aspect: float) -> dict[int, Track]:
        cfg = self.config
        pairs = [(_iou(t, d), tid, i) for tid, t in self.tracks.items()
                 for i, d in enumerate(detections)]
        matched: dict[int, Track] = {}
        used: set[int] = set()
        for score, tid, i in sorted(pairs, key=lambda p: -p[0]):
            if score < cfg.min_iou:
                break
            if tid not in used and i not in matched:
                matched[i] = self.tracks[tid]
                used.add(tid)
        near = []
        for tid, t in self.tracks.items():
            if tid in used:
                continue
            for i, d in enumerate(detections):
                if i in matched:
                    continue
                dist = math.hypot(d.center_x - t.center_x, (d.center_y - t.center_y) * aspect)
                if dist <= cfg.match_distance:
                    near.append((dist, tid, i))
        for _, tid, i in sorted(near):
            if tid not in used and i not in matched:
                matched[i] = self.tracks[tid]
                used.add(tid)
        return matched

    def _new_track(self, det: Detection, now: float) -> Track:
        track = Track(self._next_id, det.center_x, det.center_y, det.width, det.height, now,
                      source=det.source)
        self._next_id += 1
        self.tracks[track.id] = track
        self._observe(track, det, now, first=True)
        return track

    # ---- smoothing + looking ---------------------------------------------------------
    def _observe(self, track: Track, det: Detection, now: float, first: bool = False) -> None:
        cfg = self.config
        if not first:
            a = _alpha(now - track.last_seen, cfg.smooth_tau_s)
            track.center_x += a * (det.center_x - track.center_x)
            track.center_y += a * (det.center_y - track.center_y)
            track.width += a * (det.width - track.width)
            track.height += a * (det.height - track.height)
        track.last_seen = now
        gaze = det.gaze
        if gaze is None:  # head-direction-only detector: no angles to smooth
            track.method = "head"
            track.frame_looking = det.looking
            self._set_looking(track, det.looking, now)
            return
        if gaze.method == "eyes":
            eye = (gaze.eye_yaw, gaze.eye_pitch)
            if track._eye is not None:  # smoothed eye contribution, held through a blink
                a = _alpha(now - track._eye_time, cfg.smooth_tau_s)
                eye = (track._eye[0] + a * (eye[0] - track._eye[0]),
                       track._eye[1] + a * (eye[1] - track._eye[1]))
            track._eye, track._eye_time = eye, now
            raw, track.method = (gaze.gaze_yaw, gaze.gaze_pitch), "eyes"
        elif track._eye is not None and now - track._eye_time <= cfg.blink_hold_s + EPS:
            raw = (gaze.head_yaw + track._eye[0], gaze.head_pitch + track._eye[1])
            track.method = "blink-hold"
        else:
            raw, track.method = (gaze.head_yaw, gaze.head_pitch), "head"
        if track.gaze_yaw is None:
            track.gaze_yaw, track.gaze_pitch = raw
            track.frame_looking = self._in_band(*raw, wide=False)
        else:
            jump = max(abs(raw[0] - track.gaze_yaw), abs(raw[1] - track.gaze_pitch))
            if jump > cfg.outlier_deg and not track._last_outlier:
                track._last_outlier = True  # one wild frame: skip it, keep the old estimate
                return
            track._last_outlier = False
            track.frame_looking = self._in_band(*raw, wide=True)
            a = _alpha(now - track._gaze_time, cfg.smooth_tau_s)
            track.gaze_yaw += a * (raw[0] - track.gaze_yaw)
            track.gaze_pitch += a * (raw[1] - track.gaze_pitch)
        track._gaze_time = now
        self._set_looking(track, self._in_band(track.gaze_yaw, track.gaze_pitch,
                                               wide=track.looking), now)

    def _in_band(self, yaw: float, pitch: float, wide: bool) -> bool:
        """Enter band, or the wider exit band (thresholds + margins) once looking."""
        cfg = self.config
        dy, dp = (cfg.yaw_margin, cfg.pitch_margin) if wide else (0.0, 0.0)
        return abs(yaw) <= cfg.yaw_max + dy and cfg.pitch_min - dp <= pitch <= cfg.pitch_max + dp

    @staticmethod
    def _set_looking(track: Track, looking: bool, now: float) -> None:
        if looking and not track.looking:
            track.looking_since = now
        elif not looking:
            track.looking_since = None
        track.looking = looking

    # ---- user / peeper ---------------------------------------------------------------
    def _pick_user(self, now: float) -> None:
        cfg = self.config
        if not self.tracks:
            self.user_id, self._challenger = None, None
            return
        if self.user_id not in self.tracks:
            self.user_id = max(self.tracks.values(), key=lambda t: t.area).id
            self._challenger = None
            return
        user = self.tracks[self.user_id]
        others = [t for t in self.tracks.values() if t.id != user.id]
        biggest = max(others, key=lambda t: t.area, default=None)
        if biggest is None or biggest.area < cfg.swap_ratio * user.area:
            self._challenger = None
        elif self._challenger is None or self._challenger[0] != biggest.id:
            self._challenger = (biggest.id, now)
        elif now - self._challenger[1] >= cfg.swap_hold_s - EPS:
            self.user_id, self._challenger = biggest.id, None

    def _candidates(self, seen: set[int]) -> list[Track]:
        if len(self.tracks) < max(1, self.config.min_faces):
            return []
        exclude_user = self.config.min_faces >= 2
        return [t for t in self.tracks.values()
                if t.looking and t.id in seen and not (exclude_user and t.id == self.user_id)]

    def _decide(self, now: float, seen: set[int]) -> SnooperDecision:
        cfg = self.config
        candidates = self._candidates(seen)
        # The smoothed gaze lags a glance's end by ~tau; frame_looking stops that lag
        # stretching a short glance past enter_s.
        ready = [t for t in candidates
                 if t.looking_s(now) >= cfg.enter_s - EPS and t.frame_looking]
        if self.peeper_id is not None:
            current = next((t for t in candidates if t.id == self.peeper_id), None)
            if current is not None:
                self._remember(current, now)
            elif now - self._peeper_ok_at > cfg.exit_s + EPS:
                self.peeper_id = None  # gone too long; a ready candidate may take over below
            elif self.peeper_id not in self.tracks and self._reacquire(candidates, now):
                return self._active(f"peeper {self.peeper_id} looking (re-acquired)")
            else:
                away = now - self._peeper_ok_at
                return self._active(f"peeper {self.peeper_id} away {away:.1f}/{cfg.exit_s:g} s")
            if self.peeper_id is not None:
                return self._active(f"peeper {self.peeper_id} looking")
        if ready:
            pick = max(ready, key=lambda t: (t.looking_s(now), -t.area))
            self.peeper_id = pick.id
            self._remember(pick, now)
            return self._active(f"peeper {pick.id} looking")
        if len(self.tracks) < max(1, cfg.min_faces):
            return SnooperDecision(False, reason="no face" if cfg.min_faces <= 1
                                   else f"fewer than {cfg.min_faces} faces")
        if not candidates:
            return SnooperDecision(False, reason="nobody looking" if cfg.min_faces <= 1
                                   else "nobody else looking")
        best = max(candidates, key=lambda t: t.looking_s(now))
        return SnooperDecision(False, reason=f"face {best.id} looking "
                                             f"{best.looking_s(now):.1f}/{cfg.enter_s:g} s")

    def _reacquire(self, candidates: list[Track], now: float) -> bool:
        """The peeper's track expired inside the exit window: a looking face where it was
        is the same person back (a detector dropout longer than track_ttl_s)."""
        x, y = self._peeper_center
        near = [(math.hypot(t.center_x - x, (t.center_y - y) * self._aspect), t)
                for t in candidates]
        near = [pair for pair in near if pair[0] <= self.config.match_distance]
        if not near:
            return False
        track = min(near, key=lambda pair: pair[0])[1]
        self.peeper_id = track.id
        self._remember(track, now)
        return True

    def _remember(self, track: Track, now: float) -> None:
        self._peeper_ok_at = now
        self._peeper_center = (track.center_x, track.center_y)
        self._peeper_source = track.source

    def _active(self, reason: str) -> SnooperDecision:
        x, y = self._peeper_center
        return SnooperDecision(True, target_x=x, target_y=y, source=self._peeper_source,
                               reason=reason)
