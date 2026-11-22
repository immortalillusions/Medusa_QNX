# QNX camera bridge and models

`camera_capture.c` is a small Sensor Framework bridge that `medusa/qnx_camera.py`
loads as `libmedusa_camera.so`. Build it on the Pi and check it with
`python3 capture_face.py` from the repo root. The
[snapshot setup](../docs/qnx-deployment.md#take-a-face-detection-snapshot)
lists the dependencies and build commands.

## Models

Run on the Pi through the QNX OSS TensorFlow Lite runtime:

- `face_detector.tflite`: BlazeFace "back" (256x256), the turret's face detector.
- `blaze_face_short_range.tflite`: BlazeFace short-range (128x128), for
  `capture_face.py` and the older Pi-only mode.
- `face_landmark_detector.tflite`: 468 face landmarks.
- `cascades/haarcascade_profileface.xml`, `cascades/haarcascade_upperbody.xml`: OpenCV's
  own Haar cascades (copied from `cv2/data`; BSD-style Intel and ETH Zurich licences, text in each file's
  header), for `medusa/body_detector.py`'s head-from-body fallback. The Pi's cv2 has no
  `cv2.data`, so they are checked in here.

Run on the MacBook through MediaPipe:

- `face_landmarker.task`: face landmarks with iris points, used by
  `laptop_detector.py` for gaze.
- `blaze_face_short_range.tflite`: used by `laptop_detector.py --detector blazeface`
  and `face_webcam.py`.

`face_webcam.py` is the Windows webcam variant (see
[docs/windows-desktop-switching.md](../docs/windows-desktop-switching.md)).
`laptop_detector.py` reuses its `detect_faces` helper. All models are checked
in; nothing is downloaded at run time.
