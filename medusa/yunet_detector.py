"""OpenCV YuNet face detector (qnx/face_detection_yunet_2023mar.onnx), pure OpenCV + NumPy.

YuNet is trained on WIDER FACE and finds faces down to ~10 px, so one pass over the frame
at half resolution finds a 23 px face (10 m from a 1080p laptop camera). Works with
cv2.FaceDetectorYN in OpenCV >= 4.8 (the Mac's 5.x and the Pi's 4.12), nothing else.
``detect`` matches TFLiteFaceDetector's: FaceBoxes in the given image's pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tflite_detector import FaceBox

DEFAULT_SCORE = 0.6
DEFAULT_NMS = 0.3
DEFAULT_SCALE = 0.5  # run on the frame at this scale; boxes come back in full-size pixels


@dataclass(frozen=True)
class YuNetFace:
    box: FaceBox
    # right eye, left eye, nose tip, right and left mouth corners (pixels), as YuNet orders them
    points: tuple[tuple[float, float], ...]


class YuNetFaceDetector:
    def __init__(self, model_path: str, *, score_threshold: float = DEFAULT_SCORE,
                 nms_threshold: float = DEFAULT_NMS, scale: float = DEFAULT_SCALE) -> None:
        import cv2

        self.cv2, self.scale = cv2, scale
        self.model = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320),
                                               score_threshold, nms_threshold, 5000)
        self._size = None

    def faces(self, frame) -> list[YuNetFace]:
        cv2, scale = self.cv2, self.scale
        image = frame
        if scale != 1.0:
            image = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
        if self._size != (width, height):
            self.model.setInputSize((width, height))
            self._size = (width, height)
        _, rows = self.model.detect(image)
        found = []
        for row in [] if rows is None else rows:
            x, y, w, h = (float(v) / scale for v in row[:4])
            points = tuple((float(row[4 + 2 * i]) / scale, float(row[5 + 2 * i]) / scale)
                           for i in range(5))
            found.append(YuNetFace(FaceBox(x, y, x + w, y + h, float(row[14]), points[2]), points))
        return found

    def detect(self, frame) -> list[FaceBox]:
        return [face.box for face in self.faces(frame)]
