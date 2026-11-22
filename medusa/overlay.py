"""Preview-stream labels in the dashboard's palette (BGR), legible from 640x480 to 1920x1080.

Ink label backgrounds with warm off-white text; the accent colour marks the role:
blue-grey for ordinary faces and the USER, amber for AIM and looking faces, red only
for the PEEPER, muted grey for faces looking away.
"""

from __future__ import annotations

INK = (22, 17, 14)          # #0e1116
TEXT = (228, 237, 242)      # #f2ede4 warm off-white
BLUE_GREY = (184, 164, 138)  # #8aa4b8 ordinary / USER
AMBER = (61, 163, 232)      # #e8a33d AIM, looking
RED = (77, 72, 229)         # #e5484d PEEPER only
MUTED = (150, 162, 168)     # #a8a296 looking away


def frame_scale(frame) -> float:
    """1.0 at 480 px tall, 2.25 at 1080: line widths and text grow with the frame."""
    return max(0.75, frame.shape[0] / 480)


def thickness(scale: float, base: float = 2.0) -> int:
    return max(1, round(base * scale))


def label(cv2, frame, text: str, x: int, y: int, color, scale: float) -> None:
    """A tag whose bottom-left sits at (x, y) (moved inside the frame if needed): ink box,
    an accent bar on the left, off-white text."""
    height, width = frame.shape[:2]
    font, font_scale, weight = cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, thickness(scale, 1.0)
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, weight)
    pad, bar = round(4 * scale), max(3, round(3 * scale))
    box_w, box_h = text_w + bar + 2 * pad, text_h + baseline + 2 * pad
    x0 = int(min(max(0, x), max(0, width - box_w)))
    y0 = int(y - box_h if y - box_h >= 0 else y)
    y0 = min(max(0, y0), max(0, height - box_h))
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), INK, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + bar - 1, y0 + box_h), color, -1)
    cv2.putText(frame, text, (x0 + bar + pad, y0 + pad + text_h), font, font_scale, TEXT, weight,
                cv2.LINE_AA)


def crosshair(cv2, frame, scale: float, color=BLUE_GREY) -> None:
    """A small open cross at the frame centre (where the camera, and so the turret, points)."""
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    gap, arm, weight = round(4 * scale), round(10 * scale), thickness(scale, 1.0)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(frame, (cx + dx * gap, cy + dy * gap), (cx + dx * (gap + arm), cy + dy * (gap + arm)),
                 color, weight, cv2.LINE_AA)
