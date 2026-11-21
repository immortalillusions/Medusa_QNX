"""Choose the smallest detected face in the camera preview."""

from __future__ import annotations

from collections.abc import Sequence

from .tflite_detector import FaceBox


def select_snooper(faces: Sequence[FaceBox]) -> FaceBox | None:
    if len(faces) < 2:
        return None
    return min(
        faces,
        key=lambda face: (
            (face.right - face.left) * (face.bottom - face.top),
            -face.score,
        ),
    )
