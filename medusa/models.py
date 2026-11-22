from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    LAPTOP_ONLY = "laptop-only"
    FULL_HARDWARE = "full-hardware"


@dataclass(frozen=True)
class FaceObservation:
    """One face found in a frame.

    All positions are normalized: 0 is the left/top edge and 1 is the
    right/bottom edge. relative_size is face width divided by frame width,
    so smaller values are a rough proxy for greater distance.
    """

    source: str
    center_x: float
    center_y: float
    relative_size: float
    screen_facing: bool
    confidence: float = 1.0


@dataclass(frozen=True)
class SnooperDecision:
    active: bool
    target_x: float | None = None
    target_y: float | None = None
    source: str | None = None
    reason: str = ""

