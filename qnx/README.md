# TFLite models

These are the models used by the QNX service:

- `blaze_face_short_range.tflite` detects faces.
- `face_landmark_detector.tflite` estimates 468 face landmarks.
- The service derives a conservative screen-facing/head-direction signal from
  eye and nose landmark geometry.

The QNX Pi runs them locally through the QNX OSS TensorFlow Lite runtime. No
model is downloaded during a demo. Before presenting, verify that the exact
QNX image can load both models and their required operators.

