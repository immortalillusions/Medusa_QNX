"""Head-from-body fallback: aim at a peeper whose face the turret camera cannot see.

BlazeFace needs a frontal-ish face. A peeper turned toward the laptop, side-on, masked or
far away has none, so the turret falls back to the laptop map alone. HeadEstimator
looks for the head another way, with detectors OpenCV ships (no model download):

* "profile": the profile-face Haar cascade, run on the frame and on its mirror image
  (the cascade only knows one direction). The box is the head.
* "upper": the head-and-shoulders Haar cascade; the head is the centred top of the box.
* "body": OpenCV's built-in HOG people detector; the head is the centred top of the box.

Head candidates are HeadBox (a FaceBox with a ``kind``), so PeeperMatcher matches them to the
laptop's peeper exactly as it matches faces (map gate / single / order / lock). HeadFallback
runs the estimator only while the laptop is active and the face path has no face target,
every few frames, reusing the last result for a short TTL in between.

Pi-safe: cv2 and numpy only. The cascades are qnx/cascades/*.xml (the Pi's cv2 has no
cv2.data); a missing cascade file only disables that detector.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .tflite_detector import FaceBox

CASCADE_DIR = Path(__file__).resolve().parent.parent / "qnx" / "cascades"
PROFILE_CASCADE = "haarcascade_profileface.xml"
UPPER_CASCADE = "haarcascade_upperbody.xml"
# Head-from-box geometry, as shares of the detector's box (x0, y0 of the head's top-left, w, h).
UPPER_HEAD = (0.30, 0.05, 0.40, 0.45)   # head-and-shoulders box: head is the centred top ~half
BODY_HEAD = (0.36, 0.06, 0.28, 0.16)    # HOG box (padded person): head is the centred top ~15%
# Lower than BlazeFace scores (which are >= 0.5), so a real face always outranks a head guess.
KIND_SCORES = {"profile": 0.45, "upper": 0.35, "body": 0.30}
KIND_PRIORITY = {"profile": 0, "upper": 1, "body": 2}
SOURCES = ("face", "profile", "upper", "body", "laptop")


@dataclass(frozen=True)
class HeadBox(FaceBox):
    """A head estimated without a face: ``kind`` is "profile", "upper" or "body"."""

    kind: str = "body"


def is_head(box) -> bool:
    return isinstance(box, HeadBox)


def head_from_box(x: float, y: float, w: float, h: float, geometry) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) of the head inside a detector box, by the share ``geometry``."""
    gx, gy, gw, gh = geometry
    left, top = x + gx * w, y + gy * h
    return left, top, left + gw * w, top + gh * h


def _overlaps(a: FaceBox, b: FaceBox) -> bool:
    """True when the centre of either box lies inside the other: the same head."""
    def inside(p, q):
        cx, cy = (p.left + p.right) / 2, (p.top + p.bottom) / 2
        return q.left <= cx <= q.right and q.top <= cy <= q.bottom
    return inside(a, b) or inside(b, a)


def dedupe(heads: Sequence[HeadBox], faces: Sequence[FaceBox] = ()) -> list[HeadBox]:
    """Drop heads on a detected face (that person already has a face box) and repeats of one
    head across detectors, keeping profile over upper over body, then the higher score."""
    kept: list[HeadBox] = []
    for head in sorted(heads, key=lambda h: (KIND_PRIORITY.get(h.kind, 9), -h.score)):
        if any(_overlaps(head, f) for f in faces) or any(_overlaps(head, k) for k in kept):
            continue
        kept.append(head)
    return kept


class HeadEstimator:
    """Head candidates (HeadBox, full-frame pixels) from profile faces, upper bodies and people.

    The frame is downscaled to ``work_width`` for the cascades; HOG runs at ``hog_width``
    (its 64x128 window needs a person >= 128 px tall there).
    """

    def __init__(self, cascade_dir: Path | str = CASCADE_DIR, *, work_width: int = 320,
                 hog_width: int = 320, profile: bool = True, upper: bool = True, body: bool = True,
                 cv2=None) -> None:
        if cv2 is None:
            import cv2  # noqa: PLC0415 - kept lazy so tests can pass a fake.
        self.cv2 = cv2
        self.work_width, self.hog_width = work_width, hog_width
        cascade_dir = Path(cascade_dir)
        self.profile = self._cascade(cascade_dir / PROFILE_CASCADE) if profile else None
        self.upper = self._cascade(cascade_dir / UPPER_CASCADE) if upper else None
        self.hog = None
        if body:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _cascade(self, path: Path):
        if not path.is_file():
            return None
        cascade = self.cv2.CascadeClassifier(str(path))
        return None if cascade.empty() else cascade

    @property
    def kinds(self) -> list[str]:
        return [k for k, d in (("profile", self.profile), ("upper", self.upper), ("body", self.hog)) if d]

    def _small(self, frame, width: int):
        height = round(frame.shape[0] * width / frame.shape[1])
        if width >= frame.shape[1]:
            return frame, 1.0
        return self.cv2.resize(frame, (width, height), interpolation=self.cv2.INTER_AREA), frame.shape[1] / width

    def detect(self, frame) -> list[HeadBox]:
        cv2 = self.cv2
        heads: list[HeadBox] = []
        if self.profile is not None or self.upper is not None:
            small, k = self._small(frame, self.work_width)
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            gw = gray.shape[1]
            if self.profile is not None:
                for flipped in (False, True):
                    image = cv2.flip(gray, 1) if flipped else gray
                    for x, y, w, h in self.profile.detectMultiScale(image, 1.15, 3, minSize=(20, 20)):
                        if flipped:
                            x = gw - x - w
                        heads.append(HeadBox(x * k, y * k, (x + w) * k, (y + h) * k,
                                             KIND_SCORES["profile"], kind="profile"))
            if self.upper is not None:
                for x, y, w, h in self.upper.detectMultiScale(gray, 1.1, 3, minSize=(30, 24)):
                    l, t, r, b = head_from_box(x * k, y * k, w * k, h * k, UPPER_HEAD)
                    heads.append(HeadBox(l, t, r, b, KIND_SCORES["upper"], kind="upper"))
        if self.hog is not None:
            small, k = self._small(frame, self.hog_width)
            rects, _ = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.1)
            for x, y, w, h in rects:
                l, t, r, b = head_from_box(x * k, y * k, w * k, h * k, BODY_HEAD)
                heads.append(HeadBox(l, t, r, b, KIND_SCORES["body"], kind="body"))
        return heads


class HeadFallback:
    """Rate-limited head candidates for the frames where the face path has no face target.

    ``augment`` returns the face boxes plus this frame's head candidates (deduped against the
    faces) while the laptop is active and the last aim was not a face; otherwise the face boxes
    unchanged. The estimator runs on the first such frame and then every ``every`` frames;
    in between the last result is reused while younger than ``ttl_s``.
    """

    def __init__(self, estimator: HeadEstimator, *, every: int = 3, ttl_s: float = 0.5,
                 clock=time.monotonic) -> None:
        self.estimator, self.every, self.ttl_s, self.clock = estimator, max(1, every), ttl_s, clock
        self.heads: list[HeadBox] = []
        self.cached: list[HeadBox] = []
        self.cached_at: float | None = None
        self.frames = 0
        self.runs = 0
        self.last_ms: float | None = None

    def reset(self) -> None:
        self.heads, self.cached, self.cached_at, self.frames = [], [], None, 0

    @staticmethod
    def wanted(last_aim) -> bool:
        """The face path has no face target: no aim yet, an inactive aim, or a head aim."""
        return last_aim is None or not last_aim.active or is_head(last_aim.target_box)

    def augment(self, frame, boxes: Sequence[FaceBox] | None, *, laptop_active: bool, last_aim=None):
        if boxes is None or not laptop_active or not self.wanted(last_aim):
            self.reset()
            return boxes
        now = self.clock()
        fresh = self.cached_at is not None and now - self.cached_at <= self.ttl_s
        if self.frames % self.every == 0 or not fresh:
            start = time.perf_counter()
            self.cached, self.cached_at = self.estimator.detect(frame), now
            self.last_ms = (time.perf_counter() - start) * 1000
            self.runs += 1
            self.frames = 0
        self.frames += 1
        self.heads = dedupe(self.cached, boxes)
        return list(boxes) + self.heads

    def status(self, aim, fallback: bool = False) -> dict:
        """status.json fields: where the aim came from and the head estimator's last run."""
        if aim is not None and aim.active:
            source = aim.target_box.kind if is_head(aim.target_box) else "face"
        else:
            source = "laptop" if fallback else None
        return {
            "aim_source": source,
            "head_candidates": len(self.heads),
            "head_ms": round(self.last_ms, 1) if self.last_ms is not None else None,
        }

    def draw(self, frame, target=None) -> None:
        draw_head_boxes(frame, self.heads, target)


def _dashed_rect(cv2, frame, p0, p1, color, weight: int, dash: int) -> None:
    (x0, y0), (x1, y1) = p0, p1
    for a, b, horizontal, fixed in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0),
                                    (y0, y1, False, x1)):
        for s in range(a, b, 2 * dash):
            e = min(s + dash, b)
            pts = ((s, fixed), (e, fixed)) if horizontal else ((fixed, s), (fixed, e))
            cv2.line(frame, pts[0], pts[1], color, weight, cv2.LINE_AA)


def draw_head_boxes(frame, heads: Sequence[HeadBox], target=None) -> None:
    """Head guesses as dashed boxes tagged "HEAD (kind)": amber "AIM HEAD" for the target,
    muted grey otherwise, so they never read as a detected face."""
    import cv2

    from .overlay import AMBER, MUTED, frame_scale, label, thickness

    scale = frame_scale(frame)
    for head in heads:
        is_target = head is target
        color = AMBER if is_target else MUTED
        left, top = int(head.left), int(head.top)
        _dashed_rect(cv2, frame, (left, top), (int(head.right) - 1, int(head.bottom) - 1), color,
                     thickness(scale, 2.5 if is_target else 1.5), max(3, round(6 * scale)))
        text = f"{'AIM ' if is_target else ''}HEAD ({head.kind})"
        label(cv2, frame, text, left, top - thickness(scale), color, scale)
