"""Where a face is looking, from MediaPipe FaceLandmarker output (pure Python, no numpy).

gaze = head pose (facial transformation matrix) + eye-in-head angle (iris position).
That catches a "side-eye": the head turned away from the laptop while the eyes look back
at the screen.

Angles are degrees in camera-image terms, as the camera sees the viewer (no mirroring):
yaw > 0 points toward image right, pitch > 0 points up. (0, 0) points straight back at the
camera. The camera sits at the top centre of the screen, so the screen is a band below it:
a gaze counts as "looking" when |yaw| <= yaw_max and pitch_min <= pitch <= pitch_max.

Measured on the bench image (FaceLandmarker float16 v1): the matrix is mostly
appearance-based, but a face near a frame edge reads ~8 deg of yaw toward the frame centre
when it looks at the camera; the default thresholds absorb that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

# MediaPipe face mesh indices (478-point model). The iris ring 469-472 / 474-477 follows
# each centre. "Right"/"left" are the subject's own sides, MediaPipe's naming.
RIGHT_EYE = {"iris": 468, "corner_a": 33, "corner_b": 133, "upper": 159, "lower": 145}
LEFT_EYE = {"iris": 473, "corner_a": 362, "corner_b": 263, "upper": 386, "lower": 374}
NOSE_TIP = 1
IRIS_LANDMARK_COUNT = 478

# Iris-centre ratio offset from 0.5 that counts as full deflection. The iris centre cannot
# get nearer a corner than its own radius (~0.2 of the eye width), so horizontally it only
# travels 0.2..0.8. Lids follow the eye, so vertically the full 0..1 is used.
H_TRAVEL = 0.3
V_TRAVEL = 0.5
V_CENTRE = 0.5


@dataclass(frozen=True)
class GazeConfig:
    eye_gain_deg: float = 35.0     # eye-in-head angle at full iris deflection
    yaw_max_deg: float = 25.0      # |gaze yaw| that still lands on the screen
    pitch_min_deg: float = -35.0   # looking down at the bottom of the screen
    pitch_max_deg: float = 10.0    # a little above the camera still counts
    closed_ratio: float = 0.12     # lid gap / eye width below this = eye closed
    # Iris trust by eye width in pixels: the eye-in-head angle is scaled from 0 at eye_px_min
    # (head pose alone) to 1 at eye_px_full. (0, 0) = always trusted.
    eye_px_min: float = 0.0
    eye_px_full: float = 0.0
    # Head pose source: "matrix" (the landmarker's, fitted to all 468 points), "upper"
    # (medusa.upper_pose: solvePnP on eyes/brows/forehead/bridge only) or "auto": the matrix
    # unless there is none or the lower face is implausible (jaw_ratio outside the range).
    # Measured on synthetic masks the matrix stays within 0.4-2.4 deg of the unmasked pose
    # while upper-only pitch drifts 3-7 deg, so "auto" keeps the matrix whenever it can.
    head_pose: str = "auto"
    jaw_ratio_min: float = 1.3     # chin-to-eyes / eyes-to-forehead; ~2.2 on a real face,
    jaw_ratio_max: float = 3.5     # 1.5-2.4 with a mask the landmarker filled in


@dataclass(frozen=True)
class GazeEstimate:
    head_yaw: float
    head_pitch: float
    eye_yaw: float
    eye_pitch: float
    gaze_yaw: float
    gaze_pitch: float
    looking: bool
    method: str  # "eyes" (head + iris) or "head" (both eyes closed or no iris landmarks)
    pose: str = "matrix"  # where head yaw/pitch came from: "matrix" or "upper"


def head_pose_from_matrix(matrix) -> tuple[float, float]:
    """Yaw/pitch of the face's forward axis from a 4x4 canonical-face-to-camera matrix.

    The camera frame is x right, y up, z toward the viewer (column-vector convention:
    translation in the last column), so forward = third column of the rotation.
    """
    fx, fy, fz = matrix[0][2], matrix[1][2], matrix[2][2]
    norm = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    yaw = math.degrees(math.atan2(fx, fz))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fy / norm))))
    return yaw, pitch


def _ratio(point, start, end) -> tuple[float, float]:
    """Position of ``point`` along start->end (0..1) and |end - start|."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return 0.5, 0.0
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    return t, math.sqrt(length_sq)


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def eye_angles(points: Sequence[tuple[float, float]], config: GazeConfig) -> tuple[float, float] | None:
    """Eye-in-head (yaw, pitch) averaged over the open eyes, or None if both are closed.

    ``points`` are pixel-proportional (x, y) with y down. Each eye's ratio runs from the
    image-left corner to the image-right one, so iris-right means looking image-right.
    """
    yaws, pitches = [], []
    for eye in (RIGHT_EYE, LEFT_EYE):
        a, b = points[eye["corner_a"]], points[eye["corner_b"]]
        if a[0] > b[0]:
            a, b = b, a
        iris = points[eye["iris"]]
        h, width = _ratio(iris, a, b)
        v, gap = _ratio(iris, points[eye["upper"]], points[eye["lower"]])
        if width <= 0 or gap / width < config.closed_ratio:
            continue
        yaws.append(_clamp((h - 0.5) / H_TRAVEL) * config.eye_gain_deg)
        pitches.append(_clamp((V_CENTRE - v) / V_TRAVEL) * config.eye_gain_deg)
    if not yaws:
        return None
    return sum(yaws) / len(yaws), sum(pitches) / len(pitches)


def eye_width_px(points: Sequence[tuple[float, float]]) -> float:
    """Mean corner-to-corner width of the two eyes, in the units of ``points``."""
    widths = []
    for eye in (RIGHT_EYE, LEFT_EYE):
        a, b = points[eye["corner_a"]], points[eye["corner_b"]]
        widths.append(math.hypot(b[0] - a[0], b[1] - a[1]))
    return sum(widths) / len(widths)


def iris_trust(eye_px: float, config: GazeConfig) -> float:
    """0..1 weight for the eye-in-head angle: 0 below eye_px_min, 1 from eye_px_full up."""
    if config.eye_px_full <= 0 or eye_px >= config.eye_px_full:
        return 1.0
    if eye_px <= config.eye_px_min:
        return 0.0
    return (eye_px - config.eye_px_min) / (config.eye_px_full - config.eye_px_min)


def looking_at_screen(gaze_yaw: float, gaze_pitch: float, config: GazeConfig) -> bool:
    return (abs(gaze_yaw) <= config.yaw_max_deg
            and config.pitch_min_deg <= gaze_pitch <= config.pitch_max_deg)


CHIN, FOREHEAD = 152, 10


def jaw_ratio(points: Sequence[tuple[float, float]]) -> float:
    """|chin - eye midpoint| / |eye midpoint - forehead|: ~2.2 on the canonical face."""
    a, b = points[RIGHT_EYE["corner_a"]], points[LEFT_EYE["corner_b"]]
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    up = math.dist(mid, points[FOREHEAD])
    return math.dist(points[CHIN], mid) / up if up > 0 else float("inf")


def head_pose(landmarks, matrix, config: GazeConfig, width: float, height: float):
    """(yaw, pitch, source) from the matrix or the upper face, per config.head_pose."""
    use_upper = config.head_pose == "upper" or matrix is None
    points = None
    if not use_upper and config.head_pose == "auto" and len(landmarks) >= IRIS_LANDMARK_COUNT:
        points = [(p.x * width, p.y * height) for p in landmarks]
        ratio = jaw_ratio(points)
        use_upper = not config.jaw_ratio_min <= ratio <= config.jaw_ratio_max
    if use_upper:
        from .upper_pose import upper_face_pose

        if points is None:
            points = [(p.x * width, p.y * height) for p in landmarks]
        upper = upper_face_pose(points, width, height)
        if upper is not None:
            return upper[0], upper[1], "upper"
    if matrix is None:
        raise ValueError("no transformation matrix and no upper-face pose")
    return (*head_pose_from_matrix(matrix), "matrix")


def estimate_gaze(landmarks, matrix, config: GazeConfig = GazeConfig(),
                  width: float = 1.0, height: float = 1.0) -> GazeEstimate:
    """landmarks: MediaPipe normalized landmarks (.x, .y); matrix: 4x4 transformation or None."""
    head_yaw, head_pitch, source = head_pose(landmarks, matrix, config, width, height)
    eyes = None
    if len(landmarks) >= IRIS_LANDMARK_COUNT:
        points = [(p.x * width, p.y * height) for p in landmarks]
        eyes = eye_angles(points, config)
        trust = iris_trust(eye_width_px(points), config) if eyes is not None else 0.0
        # A few-pixel eye: its iris is noise, so lean on head pose (all of it below eye_px_min).
        eyes = None if trust <= 0 else (eyes[0] * trust, eyes[1] * trust)
    eye_yaw, eye_pitch = eyes if eyes is not None else (0.0, 0.0)
    gaze_yaw, gaze_pitch = head_yaw + eye_yaw, head_pitch + eye_pitch
    return GazeEstimate(
        head_yaw=head_yaw, head_pitch=head_pitch, eye_yaw=eye_yaw, eye_pitch=eye_pitch,
        gaze_yaw=gaze_yaw, gaze_pitch=gaze_pitch,
        looking=looking_at_screen(gaze_yaw, gaze_pitch, config),
        method="eyes" if eyes is not None else "head", pose=source,
    )
