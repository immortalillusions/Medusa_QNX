"""Identify snoopers by apparent face size in the camera preview."""

from __future__ import annotations

from collections.abc import Sequence

from .tflite_detector import FaceBox


def select_snoopers(faces: Sequence[FaceBox]) -> list[FaceBox]:
    """Exclude the largest face and return the other faces, largest first."""
    ranked = sorted(
        faces,
        key=lambda face: (
            (face.right - face.left) * (face.bottom - face.top),
            face.score,
        ),
        reverse=True,
    )
    return ranked[1:]
