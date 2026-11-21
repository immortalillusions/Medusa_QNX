# QNX deployment and judging checklist

## Required demo architecture

```text
C270 fixed camera -> QNX capture -> TensorFlow Lite face + landmark models
                    -> sustained snooper selector -> Arduino safety controller
                    -> local event -> laptop opens the blank document
```

The Pi is the AI host. Frames stay on the Pi; the laptop receives only an
`active`/`inactive` privacy event.

## Required QNX OSS AI module

Use **TensorFlow Lite runtime** from QNX OSS on the Pi. The project uses the
checked-in `qnx/blaze_face_short_range.tflite` and
`qnx/face_landmark_detector.tflite` files. The first finds faces; the second
provides landmarks used to estimate whether a face is directed at the display.

## Run on the QNX Pi

1. Confirm QNX sees the C270 and can acquire frames with the intended index.
2. Install the TensorFlow Lite runtime/Python binding matching the exact QNX
   AArch64 image.
3. Copy this repository, including the `qnx/` model files, to the Pi.
4. Start the Arduino safety controller, then the laptop event listener.
5. Run:

   ```sh
   python3 qnx_ai_service.py --event-host 192.168.1.20
   ```

If a camera frame, runtime, model, or inference result is unavailable, the
service emits an inactive event. The Arduino must park the servo and turn the
LED off after a missed QNX heartbeat.

## Pre-demo proof points

- Both TFLite models load on the QNX Pi.
- The service processes a C270 frame and emits a local event.
- Disconnecting the camera emits an inactive event.
- Stopping QNX causes the Arduino to park the servo and turn off the LED.
- The application never downloads a model or sends video to the cloud.

QNX OSS has published an AArch64 TensorFlow Lite runtime package; verify the
matching package against the installed QNX image before deployment.

