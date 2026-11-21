"""Measure the snooper's nose position relative to the center of the camera frame."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class NoseOffset:
    dx_px: float
    dy_px: float
    distance_px: float


def measure_nose_offset(
    nose: tuple[float, float], frame_width: int, frame_height: int,
) -> NoseOffset:
    """Return offsets in camera pixels, with positive x right and positive y down."""
    dx = nose[0] - frame_width / 2
    dy = nose[1] - frame_height / 2
    return NoseOffset(dx, dy, hypot(dx, dy))
