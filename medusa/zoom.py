"""Find small faces, then zoom: landmarks for faces the full-frame landmarker is too coarse for.

FaceLandmarker's own detector is BlazeFace short-range at 128x128 on the whole frame, so a
face 3-4 m from a 1920x1080 camera is a few pixels there and is never found. The default
finder here is OpenCV's YuNet (medusa.yunet_detector), one half-resolution pass over the
frame, which finds 23 px faces (10 m). The alternative (finder="blazeface") is the
256x256 BlazeFace "back" model on native-resolution tiles: a 2x1 grid finds
faces down to ~30 px (7 m), a 4x2 grid down to ~23 px (10 m). The tiles are shared out
round-robin, ``tiles_per_frame`` a frame, and a face once found is followed by re-cropping
around its last landmarks every frame, so a new face is found within a few frames while the
cost per frame stays bounded. Each face the full-frame landmarker did not already
return is cut out as a square ``margin`` times its box, upscaled to ``crop_size``, cleaned
up by the switchable ``enhance`` steps (see ENHANCE_STEPS / PRESETS), and run through a
FaceLandmarker in IMAGE mode with num_faces=1. Its landmarks are mapped back to full-frame normalized coordinates and its
transformation matrix kept for head pose, so the result looks like an ordinary
FaceLandmarker result to ``laptop_detector.landmarker_faces``.

The crop is centred on the face, so the matrix reads pose relative to the camera's ray to
the face (a face looking straight at the camera reads ~0 yaw), not the off-centre bias the
full-frame matrix shows near a frame edge.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field

DEFAULT_CROP_SIZE = 256
DEFAULT_MARGIN = 1.8
DEFAULT_TILES = (2, 1)
DEFAULT_GRIDS = ((2, 1), (4, 2))  # 2x1 finds >= ~30 px faces, 4x2 >= ~23 px (10 m at 1080p)
DEFAULT_TILES_PER_FRAME = 3       # of the 10 tiles: a full sweep every 4 frames
FINDERS = ("yunet", "blazeface")
DEFAULT_FINDER = "yunet"
DEFAULT_OVERLAP = 0.1
DEFAULT_SCORE = 0.5
ZOOM_MODES = ("off", "far", "all")

# Cheap per-crop steps, applied in this order whatever order they are named in:
#   lanczos    upscale with INTER_LANCZOS4 instead of INTER_CUBIC
#   temporal   average the same window cut from this and the previous 2 frames (sensor noise)
#   bilateral  edge-preserving denoise (fast)          nlmeans  fastNlMeans denoise (slower)
#   grey       luminance replicated to 3 channels (no colour noise or white-balance cast)
#   clahe      CLAHE on luminance                      gamma    auto gamma from mean luminance
#   unsharp    unsharp-mask sharpening
ENHANCE_STEPS = ("lanczos", "temporal", "bilateral", "nlmeans", "grey", "clahe", "gamma", "unsharp")
PRESETS = {
    "none": (),
    "grey-clahe": ("grey", "clahe"),
    "sharp": ("bilateral", "unsharp"),
    "full": ("lanczos", "temporal", "bilateral", "grey", "clahe", "unsharp"),
}
DEFAULT_ENHANCE = "none"
TEMPORAL_FRAMES = 3


def parse_enhance(spec: str) -> tuple[str, ...]:
    """A preset name or a '+'/','-separated list of steps -> steps in pipeline order."""
    spec = spec.strip()
    if spec in PRESETS:
        return PRESETS[spec]
    names = [name.strip() for name in spec.replace("+", ",").split(",") if name.strip()]
    unknown = [name for name in names if name not in ENHANCE_STEPS]
    if unknown:
        raise ValueError(f"unknown enhance step(s) {unknown}; presets {sorted(PRESETS)}, "
                         f"steps {list(ENHANCE_STEPS)}")
    return tuple(step for step in ENHANCE_STEPS if step in names)


def auto_gamma_lut(mean: float):
    """LUT mapping mean luminance toward mid-grey; gamma clamped to 0.5..2 (dark crops brighten)."""
    import math

    import numpy as np

    mean = min(max(mean, 1.0), 254.0) / 255.0
    gamma = min(2.0, max(0.5, math.log(0.5) / math.log(mean)))
    return np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0 + 0.5, 0, 255).astype(np.uint8)


class CropEnhancer:
    """Applies the chosen steps (except upscale/temporal, done by the caller) to a BGR crop."""

    def __init__(self, steps: Sequence[str] = ()) -> None:
        self.steps = tuple(steps)
        self._clahe = None
        if "clahe" in self.steps:
            import cv2
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def __call__(self, crop):
        import cv2

        steps = self.steps
        if "bilateral" in steps:
            crop = cv2.bilateralFilter(crop, 5, 25, 5)
        if "nlmeans" in steps:
            crop = cv2.fastNlMeansDenoisingColored(crop, None, 5, 5, 5, 15)
        grey = "grey" in steps
        if grey or "clahe" in steps or "gamma" in steps:
            if grey:
                luma = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
                luma = ycc[..., 0]
            if "gamma" in steps:
                luma = cv2.LUT(luma, auto_gamma_lut(float(luma.mean())))
            if self._clahe is not None:
                luma = self._clahe.apply(luma)
            if grey:
                crop = cv2.cvtColor(luma, cv2.COLOR_GRAY2BGR)
            else:
                ycc[..., 0] = luma
                crop = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        if "unsharp" in steps:
            blurred = cv2.GaussianBlur(crop, (0, 0), 1.5)
            crop = cv2.addWeighted(crop, 1.6, blurred, -0.6, 0)
        return crop


@dataclass(frozen=True)
class Window:
    """A square crop in full-frame pixels; may hang over the frame edge (padded black)."""

    left: int
    top: int
    side: int


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float = 0.0


@dataclass
class ZoomResult:
    """Duck-types mediapipe's FaceLandmarkerResult for landmarker_faces()."""

    face_landmarks: list = field(default_factory=list)
    facial_transformation_matrixes: list = field(default_factory=list)
    zoomed: int = 0  # how many of the faces came from a zoomed crop


def square_window(box: tuple[float, float, float, float], margin: float = DEFAULT_MARGIN) -> Window:
    """box = (left, top, right, bottom) pixels -> square window ``margin`` x its longer side."""
    left, top, right, bottom = box
    side = max(1, round(max(right - left, bottom - top) * margin))
    cx, cy = (left + right) / 2, (top + bottom) / 2
    return Window(round(cx - side / 2), round(cy - side / 2), side)


def cut(frame, window: Window, size: int, lanczos: bool = False):
    """The window's pixels resized to size x size; outside the frame is black."""
    import cv2

    height, width = frame.shape[:2]
    x0, y0 = window.left, window.top
    x1, y1 = x0 + window.side, y0 + window.side
    inner = frame[max(0, y0):min(height, y1), max(0, x0):min(width, x1)]
    pad = (max(0, -y0), max(0, y1 - height), max(0, -x0), max(0, x1 - width))
    if any(pad):
        inner = cv2.copyMakeBorder(inner, *pad, cv2.BORDER_CONSTANT, value=0)
    if window.side >= size:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_CUBIC
    return cv2.resize(inner, (size, size), interpolation=interp)


def map_landmarks(landmarks: Sequence, window: Window, width: int, height: int) -> list[Point]:
    """Crop-normalized landmarks -> full-frame normalized. z (in crop widths) -> frame widths."""
    side = window.side
    return [Point((window.left + p.x * side) / width, (window.top + p.y * side) / height,
                  getattr(p, "z", 0.0) * side / width) for p in landmarks]


def landmark_box(landmarks: Sequence, width: int, height: int) -> tuple[float, float, float, float]:
    xs = [p.x * width for p in landmarks]
    ys = [p.y * height for p in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def covered(box: tuple[float, float, float, float],
            others: Sequence[tuple[float, float, float, float]]) -> bool:
    """True when the box's centre lies inside one of ``others`` (same face already handled)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(o[0] <= cx <= o[2] and o[1] <= cy <= o[3] for o in others)


def tile_windows(width: int, height: int, tiles: tuple[int, int] = DEFAULT_TILES,
                 overlap: float = DEFAULT_OVERLAP) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) tiles covering the frame, each grown by ``overlap`` of a tile per side."""
    nx, ny = tiles
    tw, th = width / nx, height / ny
    out = []
    for j in range(ny):
        for i in range(nx):
            out.append((max(0, int(i * tw - overlap * tw)), max(0, int(j * th - overlap * th)),
                        min(width, int((i + 1) * tw + overlap * tw)),
                        min(height, int((j + 1) * th + overlap * th))))
    return out


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_boxes(boxes: Sequence[tuple[float, float, float, float, float]],
                threshold: float = 0.3) -> list[tuple[float, float, float, float, float]]:
    """NMS over (left, top, right, bottom, score): a face seen by two overlapping tiles counts once."""
    kept = []
    for box in sorted(boxes, key=lambda b: -b[4]):
        if all(_iou(box, old) < threshold for old in kept):
            kept.append(box)
    return kept


def detect_in(detector, frame, windows) -> list[tuple[float, float, float, float, float]]:
    """Run a TFLiteFaceDetector per (x0, y0, x1, y1) tile; full-frame pixel boxes, merged."""
    found = []
    for x0, y0, x1, y1 in windows:
        for b in detector.detect(frame[y0:y1, x0:x1]):
            found.append((b.left + x0, b.top + y0, b.right + x0, b.bottom + y0, b.score))
    return merge_boxes(found)


def detect_tiled(detector, frame, tiles: tuple[int, int] = DEFAULT_TILES,
                 overlap: float = DEFAULT_OVERLAP) -> list[tuple[float, float, float, float, float]]:
    """Run a TFLiteFaceDetector on every tile of one grid."""
    height, width = frame.shape[:2]
    return detect_in(detector, frame, tile_windows(width, height, tiles, overlap))


def grid_schedule(width: int, height: int, grids=DEFAULT_GRIDS,
                  overlap: float = DEFAULT_OVERLAP) -> list[tuple[int, int, int, int]]:
    """Every tile of every grid, coarse grids first: the round-robin order."""
    return [tile for grid in grids for tile in tile_windows(width, height, grid, overlap)]


class FaceZoomer:
    """Adds zoomed-crop landmarks to a full-frame FaceLandmarker result.

    ``crop_landmarker`` is a FaceLandmarker in IMAGE mode with num_faces=1 and transformation
    matrices on; ``detector`` a TFLiteFaceDetector (BlazeFace back model). ``max_faces``
    caps the total faces returned.
    """

    def __init__(self, crop_landmarker, detector, *, crop_size: int = DEFAULT_CROP_SIZE,
                 margin: float = DEFAULT_MARGIN, grids=DEFAULT_GRIDS,
                 tiles_per_frame: int | None = DEFAULT_TILES_PER_FRAME,
                 enhance: Sequence[str] = (), max_faces: int = 4,
                 whole_frame: bool = False) -> None:
        self.crop_landmarker, self.detector = crop_landmarker, detector
        self.whole_frame = whole_frame  # also detect on the whole frame (big faces split by tiles)
        self.crop_size, self.margin, self.max_faces = crop_size, margin, max_faces
        self.grids, self.tiles_per_frame = tuple(grids), tiles_per_frame  # None = every tile
        self._schedule: list = []
        self._schedule_shape = None
        self._next_tile = 0
        self._tracked: list = []  # landmark boxes (pixels) of last frame's zoomed faces
        self.enhance = CropEnhancer(enhance)
        self._lanczos = "lanczos" in self.enhance.steps
        # Earlier frames for "temporal": the current window is cut from each, so they align
        # as long as the face holds still (it barely moves at distance).
        self._previous: list = []
        self.last_crops: list = []  # (raw upscaled crop, enhanced crop) per zoomed face, for debugging

    def _tiles_now(self, width: int, height: int) -> list:
        """This frame's share of the round-robin tile schedule."""
        if self._schedule_shape != (width, height):
            self._schedule = grid_schedule(width, height, self.grids)
            self._schedule_shape, self._next_tile = (width, height), 0
        count = len(self._schedule)
        if self.tiles_per_frame is None or self.tiles_per_frame >= count:
            return list(self._schedule)
        start = self._next_tile
        self._next_tile = (start + self.tiles_per_frame) % count
        return [self._schedule[(start + i) % count] for i in range(self.tiles_per_frame)]

    def crop(self, frame, window: Window):
        """The enhanced, upscaled crop the crop landmarker sees."""
        import numpy as np

        raw = cut(frame, window, self.crop_size, self._lanczos)
        crop = raw
        if "temporal" in self.enhance.steps and self._previous:
            stack = [raw.astype(np.float32)] + [cut(old, window, self.crop_size, self._lanczos)
                                                .astype(np.float32) for old in self._previous]
            crop = (sum(stack) / len(stack) + 0.5).astype(np.uint8)
        crop = self.enhance(crop)
        self.last_crops.append((raw, crop))
        return crop

    def landmarks_in(self, frame, window: Window):
        """(landmarks mapped to the full frame, matrix) for the one face in ``window``, or None."""
        import cv2
        import mediapipe as mp

        crop = self.crop(frame, window)
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        result = self.crop_landmarker.detect(image)
        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return None
        height, width = frame.shape[:2]
        return (map_landmarks(result.face_landmarks[0], window, width, height),
                result.facial_transformation_matrixes[0])

    def __call__(self, frame, full_result=None) -> ZoomResult:
        height, width = frame.shape[:2]
        result = ZoomResult()
        self.last_crops = []
        if full_result is not None:
            for lms, matrix in zip(full_result.face_landmarks or [],
                                   full_result.facial_transformation_matrixes or []):
                if lms:
                    result.face_landmarks.append(lms)
                    result.facial_transformation_matrixes.append(matrix)
        handled = [landmark_box(lms, width, height) for lms in result.face_landmarks]
        boxes = detect_in(self.detector, frame, self._tiles_now(width, height))
        if self.whole_frame:
            boxes = merge_boxes(boxes + [(b.left, b.top, b.right, b.bottom, b.score)
                                         for b in self.detector.detect(frame)])
        # Faces zoomed last frame come first (re-cropped on their last landmarks), then new
        # detections; largest first within each, so max_faces keeps the near faces.
        tracked, self._tracked = self._tracked, []
        candidates = (sorted(tracked, key=lambda b: -(b[2] - b[0]))
                      + sorted(boxes, key=lambda b: -(b[2] - b[0])))
        for box in candidates:
            if len(result.face_landmarks) >= self.max_faces:
                break
            if covered(box[:4], handled):
                continue
            found = self.landmarks_in(frame, square_window(box[:4], self.margin))
            if found is None:
                continue
            lms, matrix = found
            result.face_landmarks.append(lms)
            result.facial_transformation_matrixes.append(matrix)
            result.zoomed += 1
            handled.append(landmark_box(lms, width, height))
            self._tracked.append(handled[-1])
        if "temporal" in self.enhance.steps:
            # A copy: the live loop draws the preview overlay onto ``frame`` afterwards.
            self._previous = ([frame.copy()] + self._previous)[:TEMPORAL_FRAMES - 1]
        return result


class LandmarkPipeline:
    """Frame -> FaceLandmarker-shaped result, for --zoom-faces off / far / all.

    off: the full-frame FaceLandmarker (VIDEO mode) alone, the old path.
    far: that, plus zoomed crops for faces it missed (the distant ones).
    all: no full-frame landmarker; every detected face is zoomed (one pose convention for all).
    Everything is local: the models are the checked-in qnx/ files.
    """

    def __init__(self, landmarker_model: str, detector_model: str | None = None, *,
                 zoom: str = "far", max_faces: int = 4, crop_size: int = DEFAULT_CROP_SIZE,
                 enhance: Sequence[str] = (), num_threads: int = 4,
                 finder: str = "blazeface") -> None:
        import mediapipe as mp

        if zoom not in ZOOM_MODES:
            raise ValueError(f"zoom must be one of {ZOOM_MODES}")
        vision = mp.tasks.vision
        self.zoom = zoom
        self._last_ms = -1
        self._stack = ExitStack()

        def landmarker(mode, faces):
            return self._stack.enter_context(vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(landmarker_model)),
                    running_mode=mode, num_faces=faces,
                    output_facial_transformation_matrixes=True)))

        self.full = landmarker(vision.RunningMode.VIDEO, max_faces) if zoom != "all" else None
        self.zoomer = None
        if zoom != "off":
            if detector_model is None:
                raise ValueError("zoom needs a face detector model")
            if finder == "yunet":
                from .yunet_detector import YuNetFaceDetector

                # One half-resolution pass over the whole frame finds 23 px faces.
                detector = YuNetFaceDetector(str(detector_model))
                grids, whole = ((1, 1),), False
            elif finder == "blazeface":
                from .tflite_detector import TFLiteFaceDetector

                detector = TFLiteFaceDetector(str(detector_model), score_threshold=DEFAULT_SCORE,
                                              num_threads=num_threads)
                grids, whole = DEFAULT_GRIDS, zoom == "all"
            else:
                raise ValueError(f"finder must be one of {FINDERS}")
            self.zoomer = FaceZoomer(landmarker(vision.RunningMode.IMAGE, 1), detector,
                                     crop_size=crop_size, enhance=enhance, max_faces=max_faces,
                                     grids=grids, tiles_per_frame=None if finder == "yunet"
                                     else DEFAULT_TILES_PER_FRAME, whole_frame=whole)

    def __call__(self, frame, timestamp_ms: int):
        import cv2
        import mediapipe as mp

        result = None
        if self.full is not None:
            # VIDEO mode needs strictly increasing timestamps.
            self._last_ms = max(self._last_ms + 1, int(timestamp_ms))
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.full.detect_for_video(image, self._last_ms)
        if self.zoomer is not None:
            result = self.zoomer(frame, result)
        return result

    def close(self) -> None:
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
