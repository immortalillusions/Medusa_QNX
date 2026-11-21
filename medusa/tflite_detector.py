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

from .models import FaceObservation

_NOSE_KEYPOINT_OFFSET = 4 + 2 * 2  # Four box values, then the third x/y keypoint.


@dataclass(frozen=True)
class FaceBox:
    left: float
    top: float
    right: float
    bottom: float
    score: float
    nose: tuple[float, float] | None = None


def draw_face_boxes(
    frame, boxes: Sequence[FaceBox], *, snooper: FaceBox | None = None,
) -> None:
    """Annotate faces on a BGR frame and label the selected snooper."""
    import cv2

    for box in boxes:
        selected = box is snooper
        color = (64, 80, 255) if selected else (0, 255, 0)
        left, top = int(box.left), int(box.top)
        cv2.rectangle(frame, (left, top), (int(box.right) - 1, int(box.bottom) - 1), color, 2)
        if selected:
            height, width = frame.shape[:2]
            label = "SNOOPER"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, _), _ = cv2.getTextSize(label, font, 0.6, 2)
            x = min(left, max(0, width - text_width - 2))
            cv2.putText(frame, label, (x, min(height - 1, max(20, top - 6))),
                        font, 0.6, color, 2, cv2.LINE_AA)
            if box.nose is not None:
                nose = (round(box.nose[0]), round(box.nose[1]))
                center = (width // 2, height // 2)
                cv2.line(frame, center, nose, color, 2, cv2.LINE_AA)
                cv2.drawMarker(frame, center, color, markerSize=14, thickness=2)
                cv2.circle(frame, nose, 4, color, -1, cv2.LINE_AA)


class TFLiteFaceDetector:
    """Run the short-range BlazeFace model and report boxes in photo coordinates."""

    def __init__(self, model_path: str) -> None:
        try:
            import cv2
            import numpy as np
            from tflite_runtime.interpreter import Interpreter
        except ImportError as error:
            raise RuntimeError("QNX TFLite runtime, NumPy, and OpenCV are required.") from error
        self.cv2, self.np, self.Interpreter = cv2, np, Interpreter
        self.face = Interpreter(model_path=model_path, num_threads=1)
        self.face.allocate_tensors()
        self.face_in = self.face.get_input_details()[0]
        if (tuple(self.face_in["shape"]) != (1, 128, 128, 3)
                or self.face_in["dtype"] != np.float32):
            raise RuntimeError("Expected the checked-in 128×128 float BlazeFace model.")
        self.anchors = self._anchors(128, 128)

    def detect(self, frame) -> list[FaceBox]:
        height, width = frame.shape[:2]
        scale = 128 / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        pad_x = (128 - resized_width) // 2
        pad_y = (128 - resized_height) // 2
        resized = self.cv2.resize(frame, (resized_width, resized_height))
        rgb = self.cv2.cvtColor(resized, self.cv2.COLOR_BGR2RGB)
        image = self.np.full((1, 128, 128, 3), -1, dtype=self.np.float32)
        image[0, pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = rgb / 127.5 - 1
        self.face.set_tensor(self.face_in["index"], image)
        self.face.invoke()
        outputs = {tuple(detail["shape"]): self.face.get_tensor(detail["index"])
                   for detail in self.face.get_output_details()}
        raw_boxes = outputs.get((1, 896, 16))
        logits = outputs.get((1, 896, 1))
        if raw_boxes is None or logits is None:
            raise RuntimeError("Expected the checked-in 896-anchor BlazeFace model.")

        return self._decode(raw_boxes[0], logits[0, :, 0], width, height,
                            pad_x, pad_y, resized_width, resized_height)

    def _decode(self, raw_boxes, logits, width, height, pad_x, pad_y,
                resized_width, resized_height) -> list[FaceBox]:
        x_scale, y_scale = width / resized_width, height / resized_height
        boxes = []
        for raw, logit, (anchor_x, anchor_y) in zip(raw_boxes, logits, self.anchors):
            if not math.isfinite(logit):
                continue
            score = 1 / (1 + math.exp(-max(-100, min(100, float(logit)))))
            if score < .65:
                continue
            offset_x, offset_y, box_width, box_height = map(float, raw[:4])
            if not all(map(math.isfinite, (offset_x, offset_y, box_width, box_height))):
                continue
            cx = offset_x + anchor_x * 128 - pad_x
            cy = offset_y + anchor_y * 128 - pad_y
            half_width, half_height = box_width / 2, box_height / 2
            left, top = max(0, (cx - half_width) * x_scale), max(0, (cy - half_height) * y_scale)
            right = min(width, (cx + half_width) * x_scale)
            bottom = min(height, (cy + half_height) * y_scale)
            if right - left >= 1 and bottom - top >= 1:
                nose_x, nose_y = map(float, raw[_NOSE_KEYPOINT_OFFSET:_NOSE_KEYPOINT_OFFSET + 2])
                nose_x = (nose_x + anchor_x * 128 - pad_x) * x_scale
                nose_y = (nose_y + anchor_y * 128 - pad_y) * y_scale
                nose = (nose_x, nose_y) if 0 <= nose_x < width and 0 <= nose_y < height else None
                boxes.append(FaceBox(left, top, right, bottom, score, nose))
        return self._nms(boxes)

    @staticmethod
    def _anchors(width, height):
        anchors = []
        for stride, repeats in ((8, 2), (16, 6)):
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    anchors.extend([((x + stride / 2) / width, (y + stride / 2) / height)] * repeats)
        return anchors

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

    def __init__(self, detector_path: str, landmark_path: str) -> None:
        self.face_detector = TFLiteFaceDetector(detector_path)
        self.cv2, self.np = self.face_detector.cv2, self.face_detector.np
        self.landmark = self.face_detector.Interpreter(model_path=landmark_path, num_threads=1)
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
            eye_span = abs(right_eye[0] - left_eye[0])
            if eye_span < 0.01:
                continue
            yaw = abs(nose[0] - (left_eye[0] + right_eye[0]) / 2) / eye_span
            result.append(FaceObservation(
                source="qnx-fixed-camera",
                center_x=(box.left + box.right) / (2 * width),
                center_y=(box.top + box.bottom) / (2 * height),
                relative_size=(box.right - box.left) / width,
                screen_facing=yaw <= 0.20,
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

