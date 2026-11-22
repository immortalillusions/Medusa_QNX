from __future__ import annotations

from dataclasses import dataclass

from .models import FaceObservation, SnooperDecision


SCENE_CLEAR_FRAMES = 30


def screen_facing_from_landmarks(nose_x: float, eye_a_x: float, eye_b_x: float) -> bool | None:
    """Estimate head direction from x coordinates normalized to the face region."""
    eye_span = abs(eye_b_x - eye_a_x)
    if eye_span < 0.01:
        return None
    yaw = abs(nose_x - (eye_a_x + eye_b_x) / 2) / eye_span
    return yaw <= 0.20


@dataclass
class SnooperSelector:
    """Selects a sustained farthest screen-facing person.

    This deliberately uses no biometric identity. The small-face heuristic is
    useful for a demo but should be replaced by calibrated depth/triangulation
    for a production system.
    """

    hold_frames: int = 15
    min_viewers: int = 2  # 1 = aim at any single face (bench calibration only)
    min_confidence: float = 0.65
    _last_bucket: tuple[str, int] | None = None
    _streak: int = 0

    def update(self, faces: list[FaceObservation]) -> SnooperDecision:
        viewers = [
            face for face in faces
            if face.screen_facing and face.confidence >= self.min_confidence
        ]
        if len(viewers) < self.min_viewers:
            self._reset()
            return SnooperDecision(False, reason="fewer than two screen-facing faces")

        # Largest face is presumed to be the laptop user; smallest is the
        # probable more-distant onlooker.
        snooper = min(viewers, key=lambda face: face.relative_size)
        bucket = (snooper.source, round(snooper.center_x * 10))
        if bucket == self._last_bucket:
            self._streak += 1
        else:
            self._last_bucket, self._streak = bucket, 1

        if self._streak < self.hold_frames:
            return SnooperDecision(
                False,
                reason=f"waiting for stable detection ({self._streak}/{self.hold_frames})",
            )
        return SnooperDecision(
            True,
            target_x=snooper.center_x,
            target_y=snooper.center_y,
            source=snooper.source,
            reason="two or more stable screen-facing faces",
        )

    def _reset(self) -> None:
        self._last_bucket = None
        self._streak = 0

