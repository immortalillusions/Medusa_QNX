"""QNX AI model adapter.

This module loads Medusa's checked-in MediaPipe-derived TFLite models with
QNX's ``tflite_runtime`` package, runs face/landmark inference on Pi camera
frames, and turns the results into screen-facing face observations. It is the
code path that uses the AI models on QNX; the QNX image must still have the
matching TensorFlow Lite runtime installed first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .decision import screen_facing_from_landmarks
from .models import FaceObservation

_NOSE_KEYPOINT_OFFSET = 4 + 2 * 2  # Four box values, then the third x/y keypoint.
DEFAULT_SCORE_THRESHOLD = 0.55
DEFAULT_NUM_THREADS = 4


@dataclass(frozen=True)
class BlazeFaceSpec:
    """One BlazeFace variant: its square input size and MediaPipe's SSD anchor strides."""

    name: str
    input_size: int
    strides: tuple[int, ...]


# MediaPipe face_detection_short_range / face_detection_back SsdAnchorsCalculator options.
MODEL_SPECS = {
    128: BlazeFaceSpec("short", 128, (8, 16, 16, 16)),
    256: BlazeFaceSpec("back", 256, (16, 32, 32, 32)),
}


def model_spec_for_input(shape) -> BlazeFaceSpec | None:
    """Pick the BlazeFace variant from the input tensor shape [1, S, S, 3]."""
    shape = tuple(int(v) for v in shape)
    if len(shape) != 4 or shape[0] != 1 or shape[1] != shape[2] or shape[3] != 3:
        return None
    return MODEL_SPECS.get(shape[1])


def ssd_anchors(input_size: int, strides: Sequence[int], *, anchor_offset: float = 0.5,
                aspect_ratios: Sequence[float] = (1.0,),
                interpolated_scale_aspect_ratio: float = 1.0) -> list[tuple[float, float]]:
    """Anchor centres (0..1) in the order MediaPipe's SsdAnchorsCalculator emits them.

    Consecutive layers with the same stride share one feature map; each layer adds
    one anchor per aspect ratio plus one interpolated-scale anchor per cell. With
    fixed_anchor_size every anchor is 1x1, so only the centres matter here.
    """
    anchors = []
    layer = 0
    while layer < len(strides):
        per_cell, last = 0, layer
        while last < len(strides) and strides[last] == strides[layer]:
            per_cell += len(aspect_ratios) + (1 if interpolated_scale_aspect_ratio > 0 else 0)
            last += 1
        grid = math.ceil(input_size / strides[layer])
        for y in range(grid):
            for x in range(grid):
                anchors.extend([((x + anchor_offset) / grid, (y + anchor_offset) / grid)] * per_cell)
        layer = last
    return anchors


def order_output_heads(shapes) -> tuple[list[int], list[int]]:
    """Indices of the box-regressor ([1, N, 16]) and score ([1, N, 1]) outputs, larger N first.

    The back model splits each into a 512-anchor (16x16 grid) and a 384-anchor
    (8x8 grid) head; anchors list the 16x16 grid first.
    """
    shapes = [tuple(int(v) for v in shape) for shape in shapes]

    def heads(width):
        found = [i for i, shape in enumerate(shapes) if len(shape) == 3 and shape[0] == 1 and shape[2] == width]
        return sorted(found, key=lambda i: -shapes[i][1])

    return heads(16), heads(1)


def load_interpreter():
    """The TFLite Interpreter class: tflite_runtime (the Pi/QNX build) first, else
    ai_edge_litert (the Mac's mediapipe venv). Raises ImportError when neither exists."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from ai_edge_litert.interpreter import Interpreter
    return Interpreter


@dataclass(frozen=True)
class FaceBox:
    left: float
    top: float
    right: float
    bottom: float
    score: float
    nose: tuple[float, float] | None = None


def draw_face_boxes(
    frame, boxes: Sequence[FaceBox], *, snoopers: Sequence[FaceBox] = (),
) -> None:
    """Label the snoopers and draw the nose offset for the first one."""
    import cv2

    snooper_ids = {id(box) for box in snoopers}
    closest_snooper = snoopers[0] if snoopers else None
    for box in boxes:
        is_snooper = id(box) in snooper_ids
        color = (64, 80, 255) if is_snooper else (0, 255, 0)
        left, top = int(box.left), int(box.top)
        cv2.rectangle(frame, (left, top), (int(box.right) - 1, int(box.bottom) - 1), color, 2)
        if is_snooper:
            height, width = frame.shape[:2]
            label = "SNOOPER (CLOSEST)" if box is closest_snooper else "SNOOPER"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, _), _ = cv2.getTextSize(label, font, 0.6, 2)
            x = min(left, max(0, width - text_width - 2))
            cv2.putText(frame, label, (x, min(height - 1, max(20, top - 6))),
                        font, 0.6, color, 2, cv2.LINE_AA)
            if box is closest_snooper and box.nose is not None:
                nose = (round(box.nose[0]), round(box.nose[1]))
                center = (width // 2, height // 2)
                cv2.line(frame, center, nose, color, 2, cv2.LINE_AA)
                cv2.drawMarker(frame, center, color, markerSize=14, thickness=2)
                cv2.circle(frame, nose, 4, color, -1, cv2.LINE_AA)


class TFLiteFaceDetector:
    """Run a BlazeFace model (short-range 128 or back 256) and report boxes in photo coordinates."""

    input_size = 128
    score_threshold = DEFAULT_SCORE_THRESHOLD

    def __init__(self, model_path: str, *, score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                 num_threads: int = DEFAULT_NUM_THREADS) -> None:
        try:
            import cv2
            import numpy as np
            Interpreter = load_interpreter()
        except ImportError as error:
            raise RuntimeError("QNX TFLite runtime, NumPy, and OpenCV are required.") from error
        self.cv2, self.np, self.Interpreter = cv2, np, Interpreter
        self.num_threads = num_threads
        self.face = Interpreter(model_path=model_path, num_threads=num_threads)
        self.face.allocate_tensors()
        self.face_in = self.face.get_input_details()[0]
        spec = model_spec_for_input(self.face_in["shape"])
        if spec is None or self.face_in["dtype"] != np.float32:
            raise RuntimeError("Expected a float BlazeFace model: 128×128 short-range or 256×256 back.")
        self.model, self.input_size = spec.name, spec.input_size
        self.anchors = ssd_anchors(spec.input_size, spec.strides)
        self.score_threshold = score_threshold
        details = self.face.get_output_details()
        box_heads, score_heads = order_output_heads(detail["shape"] for detail in details)
        self._box_heads = [details[i]["index"] for i in box_heads]
        self._score_heads = [details[i]["index"] for i in score_heads]
        anchors = len(self.anchors)
        if (sum(int(details[i]["shape"][1]) for i in box_heads) != anchors
                or sum(int(details[i]["shape"][1]) for i in score_heads) != anchors):
            raise RuntimeError(f"Expected {anchors} anchors in the {spec.name} BlazeFace outputs.")

    def detect(self, frame) -> list[FaceBox]:
        np, size = self.np, self.input_size
        height, width = frame.shape[:2]
        scale = size / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        pad_x = (size - resized_width) // 2
        pad_y = (size - resized_height) // 2
        resized = self.cv2.resize(frame, (resized_width, resized_height))
        rgb = self.cv2.cvtColor(resized, self.cv2.COLOR_BGR2RGB)
        image = np.full((1, size, size, 3), -1, dtype=np.float32)
        image[0, pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = rgb / 127.5 - 1
        self.face.set_tensor(self.face_in["index"], image)
        self.face.invoke()
        raw_boxes = np.concatenate([self.face.get_tensor(i)[0] for i in self._box_heads])
        logits = np.concatenate([self.face.get_tensor(i)[0, :, 0] for i in self._score_heads])
        # Sigmoid is monotonic: drop anchors below the threshold before the per-box Python loop.
        threshold = min(max(self.score_threshold, 1e-6), 1 - 1e-6)
        keep = np.nonzero(logits >= math.log(threshold / (1 - threshold)))[0]
        return self._decode(raw_boxes[keep], logits[keep], width, height,
                            pad_x, pad_y, resized_width, resized_height,
                            anchors=[self.anchors[i] for i in keep])

    def _decode(self, raw_boxes, logits, width, height, pad_x, pad_y,
                resized_width, resized_height, *, anchors=None) -> list[FaceBox]:
        size = self.input_size
        x_scale, y_scale = width / resized_width, height / resized_height
        boxes = []
        for raw, logit, (anchor_x, anchor_y) in zip(raw_boxes, logits, self.anchors if anchors is None else anchors):
            if not math.isfinite(logit):
                continue
            score = 1 / (1 + math.exp(-max(-100, min(100, float(logit)))))
            if score < self.score_threshold:
                continue
            offset_x, offset_y, box_width, box_height = map(float, raw[:4])
            if not all(map(math.isfinite, (offset_x, offset_y, box_width, box_height))):
                continue
            cx = offset_x + anchor_x * size - pad_x
            cy = offset_y + anchor_y * size - pad_y
            half_width, half_height = box_width / 2, box_height / 2
            left, top = max(0, (cx - half_width) * x_scale), max(0, (cy - half_height) * y_scale)
            right = min(width, (cx + half_width) * x_scale)
            bottom = min(height, (cy + half_height) * y_scale)
            if right - left >= 1 and bottom - top >= 1:
                nose_x, nose_y = map(float, raw[_NOSE_KEYPOINT_OFFSET:_NOSE_KEYPOINT_OFFSET + 2])
                nose_x = (nose_x + anchor_x * size - pad_x) * x_scale
                nose_y = (nose_y + anchor_y * size - pad_y) * y_scale
                nose = (nose_x, nose_y) if 0 <= nose_x < width and 0 <= nose_y < height else None
                boxes.append(FaceBox(left, top, right, bottom, score, nose))
        return self._nms(boxes)

    def _nms(self, boxes, threshold=.3):
        kept = []
        for box in sorted(boxes, key=lambda box: box.score, reverse=True):
            if all(self._iou(box, old) < threshold for old in kept):
                kept.append(box)
        return kept

    @staticmethod
    def _iou(a, b):
        left, top = max(a.left, b.left), max(a.top, b.top)
        right, bottom = min(a.right, b.right), min(a.bottom, b.bottom)
        overlap = max(0, right - left) * max(0, bottom - top)
        area_a = (a.right - a.left) * (a.bottom - a.top)
        area_b = (b.right - b.left) * (b.bottom - b.top)
        return overlap / (area_a + area_b - overlap)


class TFLiteFaceGazeDetector:
    """Local BlazeFace and landmark inference through the QNX TFLite runtime."""

    def __init__(self, detector_path: str, landmark_path: str, *,
                 score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                 num_threads: int = DEFAULT_NUM_THREADS) -> None:
        self.face_detector = TFLiteFaceDetector(detector_path, score_threshold=score_threshold,
                                                num_threads=num_threads)
        self.cv2, self.np = self.face_detector.cv2, self.face_detector.np
        self.landmark = self.face_detector.Interpreter(model_path=landmark_path, num_threads=num_threads)
        self.landmark.allocate_tensors()
        self.landmark_in = self.landmark.get_input_details()[0]

    def detect(self, frame, *, boxes: Sequence[FaceBox] | None = None) -> list[FaceObservation]:
        if boxes is None:
            boxes = self.face_detector.detect(frame)
        height, width = frame.shape[:2]
        result = []
        for box in boxes:
            crop, left, top = self._crop(frame, box)
            points = self._landmarks(crop)
            nose, left_eye, right_eye = points[1], points[33], points[263]
            screen_facing = screen_facing_from_landmarks(nose[0], left_eye[0], right_eye[0])
            if screen_facing is None:
                continue
            result.append(FaceObservation(
                source="qnx-fixed-camera",
                center_x=(box.left + box.right) / (2 * width),
                center_y=(box.top + box.bottom) / (2 * height),
                relative_size=(box.right - box.left) / width,
                screen_facing=screen_facing,
                confidence=box.score,
            ))
        return result

    def _landmarks(self, crop):
        self.landmark.set_tensor(self.landmark_in["index"], self._image(crop, self.landmark_in))
        self.landmark.invoke()
        values = [self.landmark.get_tensor(x["index"]).reshape(-1)
                  for x in self.landmark.get_output_details()]
        raw = next((x for x in values if x.size >= 468 * 3), None)
        if raw is None:
            raise RuntimeError("Expected 468-point landmark output.")
        points = raw[:1404].reshape(468, 3)[:, :2]
        if points.max() > 2:
            ih, iw = self.landmark_in["shape"][1:3]
            points[:, 0] /= iw
            points[:, 1] /= ih
        return points

    def _image(self, frame, detail):
        ih, iw = detail["shape"][1:3]
        image = self.cv2.resize(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB), (iw, ih))
        if detail["dtype"] == self.np.float32:
            image = image.astype(self.np.float32) / 255
        return image.astype(detail["dtype"])[None, ...]

    def _crop(self, frame, box):
        h, w = frame.shape[:2]
        pad_x, pad_y = (box.right-box.left)*.15, (box.bottom-box.top)*.15
        left, top = max(0, int(box.left-pad_x)), max(0, int(box.top-pad_y))
        right, bottom = min(w, int(box.right+pad_x)), min(h, int(box.bottom+pad_y))
        crop = frame[top:bottom, left:right]
        # Return landmark coordinates in the original normalized frame.
        points_space = crop
        return points_space, left, top

