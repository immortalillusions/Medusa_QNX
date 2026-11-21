"""QNX AI model adapter.

This module loads Medusa's checked-in MediaPipe-derived TFLite models with
QNX's ``tflite_runtime`` package, runs face/landmark inference on Pi camera
frames, and turns the results into screen-facing face observations. It is the
code path that uses the AI models on QNX; the QNX image must still have the
matching TensorFlow Lite runtime installed first.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import FaceObservation


@dataclass(frozen=True)
class _Box:
    left: float
    top: float
    right: float
    bottom: float
    score: float


class TFLiteFaceGazeDetector:
    """Local BlazeFace + landmark inference through QNX TFLite runtime."""

    def __init__(self, detector_path: str, landmark_path: str) -> None:
        try:
            import cv2
            import numpy as np
            from tflite_runtime.interpreter import Interpreter
        except ImportError as error:
            raise RuntimeError("QNX TFLite runtime, NumPy, and OpenCV are required.") from error
        self.cv2, self.np, self.Interpreter = cv2, np, Interpreter
        self.face = Interpreter(model_path=detector_path, num_threads=1)
        self.landmark = Interpreter(model_path=landmark_path, num_threads=1)
        self.face.allocate_tensors()
        self.landmark.allocate_tensors()
        self.face_in = self.face.get_input_details()[0]
        self.landmark_in = self.landmark.get_input_details()[0]

    def detect(self, frame) -> list[FaceObservation]:
        height, width = frame.shape[:2]
        result = []
        for box in self._boxes(frame):
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

    def _boxes(self, frame) -> list[_Box]:
        self.face.set_tensor(self.face_in["index"], self._image(frame, self.face_in))
        self.face.invoke()
        output = [self.face.get_tensor(x["index"]) for x in self.face.get_output_details()]
        raw_boxes = next((x[0] for x in output if x.shape[-1] >= 16), None)
        raw_scores = next((x[0].reshape(-1) for x in output if x.shape[-1] == 1), None)
        if raw_boxes is None or raw_scores is None or len(raw_boxes) != 896:
            raise RuntimeError("Expected the checked-in 896-anchor BlazeFace model.")
        ih, iw = self.face_in["shape"][1:3]
        anchors = self._anchors(iw, ih)
        h, w = frame.shape[:2]
        boxes = []
        for raw, score, anchor in zip(raw_boxes, raw_scores, anchors):
            score = float(score)
            if score < 0 or score > 1:
                score = 1 / (1 + self.np.exp(-score))
            if score < .65:
                continue
            cx = raw[0] / iw + anchor[0]
            cy = raw[1] / ih + anchor[1]
            bw, bh = raw[2] / iw, raw[3] / ih
            boxes.append(_Box(max(0, (cx-bw/2)*w), max(0, (cy-bh/2)*h),
                              min(w, (cx+bw/2)*w), min(h, (cy+bh/2)*h), score))
        return self._nms(boxes)

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

    @staticmethod
    def _anchors(width, height):
        anchors = []
        for stride, repeats in ((8, 2), (16, 6)):
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    anchors.extend([((x+stride/2)/width, (y+stride/2)/height)] * repeats)
        return anchors

    @staticmethod
    def _nms(boxes, threshold=.3):
        kept = []
        for box in sorted(boxes, key=lambda x: x.score, reverse=True):
            if all(TFLiteFaceGazeDetector._iou(box, old) < threshold for old in kept):
                kept.append(box)
        return kept

    @staticmethod
    def _iou(a, b):
        l, t, r, d = max(a.left,b.left), max(a.top,b.top), min(a.right,b.right), min(a.bottom,b.bottom)
        overlap = max(0,r-l)*max(0,d-t)
        return overlap / max(1, (a.right-a.left)*(a.bottom-a.top)+(b.right-b.left)*(b.bottom-b.top)-overlap)

