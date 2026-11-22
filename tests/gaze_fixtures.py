"""Synthetic FaceLandmarker output for gaze tests (no MediaPipe needed)."""

import math
from types import SimpleNamespace

from medusa.gaze import H_TRAVEL, LEFT_EYE, NOSE_TIP, RIGHT_EYE, V_CENTRE, V_TRAVEL, GazeConfig

GAIN = GazeConfig().eye_gain_deg


def pose(yaw=0.0, pitch=0.0, translation=(0.0, 0.0, -50.0)):
    """4x4 canonical-face-to-camera matrix whose forward axis has this yaw/pitch (degrees)."""
    y, p = math.radians(yaw), math.radians(pitch)
    # R = Ry(yaw) @ Rx(-pitch): forward (0, 0, 1) -> (sin y cos p, sin p, cos y cos p).
    rotation = [
        [math.cos(y), math.sin(y) * math.sin(p), math.sin(y) * math.cos(p)],
        [0.0, math.cos(p), math.sin(p)],
        [-math.sin(y), math.cos(y) * math.sin(p), math.cos(y) * math.cos(p)],
    ]
    return [row + [t] for row, t in zip(rotation, translation)] + [[0.0, 0.0, 0.0, 1.0]]


def landmarks(eye_yaw=0.0, eye_pitch=0.0, *, cx=0.5, cy=0.5, size=0.2,
              open_ratio=0.3, closed=(), count=478):
    """A face whose iris sits where an eye turned by eye_yaw/eye_pitch degrees would put it.

    ``closed`` names eyes ("right", "left") with nearly shut lids.
    """
    points = [SimpleNamespace(x=cx, y=cy) for _ in range(count)]
    # Face outline so the bounds are the face box: forehead, chin, cheeks.
    for index, (x, y) in {10: (cx, cy - size * 0.6), 152: (cx, cy + size * 0.6),
                          234: (cx - size / 2, cy), 454: (cx + size / 2, cy)}.items():
        points[index] = SimpleNamespace(x=x, y=y)
    points[NOSE_TIP] = SimpleNamespace(x=cx, y=cy + size * 0.05)
    if count < 478:
        return points
    h = 0.5 + eye_yaw / GAIN * H_TRAVEL
    v = V_CENTRE - eye_pitch / GAIN * V_TRAVEL
    eye_w = size * 0.25
    for name, eye, centre_x in (("right", RIGHT_EYE, cx - size * 0.2), ("left", LEFT_EYE, cx + size * 0.2)):
        gap = eye_w * (0.03 if name in closed else open_ratio)
        left_x, eye_y = centre_x - eye_w / 2, cy - size * 0.1
        upper, lower = eye_y - gap / 2, eye_y + gap / 2
        points[eye["corner_a"]] = SimpleNamespace(x=left_x, y=eye_y)
        points[eye["corner_b"]] = SimpleNamespace(x=left_x + eye_w, y=eye_y)
        points[eye["upper"]] = SimpleNamespace(x=centre_x, y=upper)
        points[eye["lower"]] = SimpleNamespace(x=centre_x, y=lower)
        points[eye["iris"]] = SimpleNamespace(x=left_x + h * eye_w, y=upper + v * gap)
    return points


def result(*faces):
    """A FaceLandmarkerResult-shaped object from (landmarks, matrix) pairs."""
    return SimpleNamespace(face_landmarks=[lms for lms, _ in faces],
                           facial_transformation_matrixes=[matrix for _, matrix in faces])
