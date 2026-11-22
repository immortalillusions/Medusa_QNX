"""Head yaw/pitch from the upper face only: eyes, brows, forehead, nose bridge.

A mask or scarf hides the nose tip, mouth and jaw, and the FaceLandmarker then guesses those
landmarks (and the pose matrix it fits to all 468) from the cloth. These points stay visible:
cv2.solvePnP fits them to the same points of MediaPipe's canonical face model.

UPPER_FACE is the canonical model's vertices for those indices, in cm, x right / y up /
z toward the viewer (the matrix's convention), read from the canonical mesh in
qnx/face_landmarker.task (geometry_pipeline_metadata_landmarks.binarypb; see
canonical_vertices, which the tests use to check these numbers).

Angles match medusa.gaze.head_pose_from_matrix (yaw > 0 toward image right, pitch > 0 up),
measured against the ray from the face to the camera: a face anywhere in the frame looking
straight at the lens reads (0, 0), as the zoomed crops' matrices do.
"""

from __future__ import annotations

import math
import struct
import zipfile
from typing import Sequence

UPPER_FACE = {
    33: (-4.4459, 2.6640, 3.1734),   # eye corners
    133: (-1.8564, 2.5852, 3.7579),
    362: (1.8564, 2.5852, 3.7579),
    263: (4.4459, 2.6640, 3.1734),
    70: (-5.7210, 4.2546, 2.8309),   # brows
    63: (-4.9859, 4.8025, 3.7520),
    105: (-3.9866, 5.1095, 4.4663),
    66: (-2.7603, 5.1010, 5.0160),
    107: (-1.3956, 5.0120, 5.3160),
    336: (1.3956, 5.0120, 5.3160),
    296: (2.7603, 5.1010, 5.0160),
    334: (3.9866, 5.1095, 4.4663),
    293: (4.9859, 4.8025, 3.7520),
    300: (5.7210, 4.2546, 2.8309),
    10: (0.0000, 8.2618, 4.4815),    # forehead
    109: (-1.8914, 8.2364, 4.2750),
    67: (-3.5240, 8.0060, 3.7292),
    338: (1.8914, 8.2364, 4.2750),
    297: (3.5240, 8.0060, 3.7292),
    151: (0.0000, 6.5454, 5.0273),
    21: (-7.0880, 5.4348, 0.0996),   # temples
    251: (7.0880, 5.4348, 0.0996),
    127: (-7.7431, 2.3650, -2.0052),
    356: (7.7431, 2.3650, -2.0052),
    168: (0.0000, 3.2710, 5.2360),   # nose bridge (above where a mask sits)
    6: (0.0000, 2.4733, 5.7886),
    197: (0.0000, 1.7284, 6.3168),
}
FOCAL_PER_WIDTH = 0.8  # focal length / image width: ~64 deg horizontal field of view


def canonical_vertices(task_path: str) -> list[tuple[float, float, float]]:
    """The 468 canonical-mesh vertices (x, y, z) stored in a face_landmarker.task bundle."""
    with zipfile.ZipFile(task_path) as bundle:
        data = bundle.read("geometry_pipeline_metadata_landmarks.binarypb")

    def varint(i):
        value = shift = 0
        while True:
            byte = data[i]
            i += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if byte < 0x80:
                return value, i

    def fields(lo, hi):
        i = lo
        while i < hi:
            key, i = varint(i)
            number, wire = key >> 3, key & 7
            if wire == 0:
                value, i = varint(i)
            elif wire == 2:
                length, i = varint(i)
                value, i = (i, i + length), i + length
            elif wire == 5:
                value, i = struct.unpack("<f", data[i:i + 4])[0], i + 4
            elif wire == 1:
                value, i = None, i + 8
            else:
                raise ValueError(f"unexpected protobuf wire type {wire}")
            yield number, wire, value

    for number, wire, span in fields(0, len(data)):
        if number == 1 and wire == 2:  # canonical_mesh; its field 3 = vertex_buffer x,y,z,u,v
            floats = [v for n, w, v in fields(*span) if n == 3 and w == 5]
            return [tuple(floats[i * 5:i * 5 + 3]) for i in range(len(floats) // 5)]
    raise ValueError("no canonical mesh in the bundle")


def upper_face_pose(points: Sequence[tuple[float, float]], width: float, height: float,
                    focal: float | None = None) -> tuple[float, float] | None:
    """(yaw, pitch) in degrees from pixel (x, y) landmarks (y down), or None if the fit fails."""
    import cv2
    import numpy as np

    if len(points) <= max(UPPER_FACE):
        return None
    indices = list(UPPER_FACE)
    # Canonical (x right, y up, z to viewer) -> OpenCV object frame (x right, y down, z away).
    model = np.array([(x, -y, -z) for x, y, z in (UPPER_FACE[i] for i in indices)], np.float64)
    image = np.array([points[i] for i in indices], np.float64)
    f = focal if focal is not None else FOCAL_PER_WIDTH * width
    camera = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(model, image, camera, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None  # Degenerate points (e.g. a tiny or edge face): SQPnP asserts instead of failing.
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    forward = rotation @ np.array([0.0, 0.0, -1.0])  # the face's +z, in OpenCV camera axes
    t = tvec.ravel()
    if t[2] <= 0:
        return None
    # To gaze axes (x right, y up, z toward the viewer), then relative to the face->camera ray.
    fx, fy, fz = forward[0], -forward[1], -forward[2]
    rx, ry, rz = -t[0], t[1], t[2]  # face -> camera, gaze axes
    yaw = math.degrees(math.atan2(fx, fz) - math.atan2(rx, rz))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, fy / (math.hypot(fx, fy, fz) or 1.0))))
                         - math.asin(max(-1.0, min(1.0, ry / (math.hypot(rx, ry, rz) or 1.0)))))
    return yaw, pitch
