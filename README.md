# Medusa

Medusa uses a USB camera connected to a Raspberry Pi 5 running QNX. The snapshot command takes one photo, detects faces locally with the bundled BlazeFace model, and saves the photo with a green box around each detected face.

## Take a photo and detect faces

Plug the webcam into the Pi and follow the one-time [QNX camera setup](docs/qnx-deployment.md#take-a-face-detection-snapshot). Then run from the repo directory **on the Pi**.

```sh
python3 capture_face.py
```

The command uses Sensor Framework camera unit 5, the USB camera unit in the
standard Pi 5 configuration. It prints the number of faces and writes
`captures/face.jpg`, even when the count is zero. Use `--camera-unit` for a
different Sensor Framework unit or `--output captures/second.jpg` to keep
another photo.

## Privacy demo

Medusa was built for Hack Western 13. The continuous demo uses the fixed
camera to detect a second person looking at the laptop screen and emit a
privacy event.

## Architecture

```text
C270 fixed camera -> QNX Pi -> TensorFlow Lite face + landmark AI
                              -> stable snooper decision
                              -> Arduino safety controller -> servo + LED
                              -> local privacy event -> laptop -> docs.new
```

The QNX Pi is the AI host and decision maker. Frames remain on the Pi; the laptop receives only a small `active`/`inactive` event. The camera does not rotate. Only the LED mount moves.

## QNX OSS AI requirement

Medusa uses **TensorFlow Lite runtime** from QNX OSS on the Raspberry Pi. The checked-in BlazeFace and face-landmark models are stored locally and execute with no cloud service.

For the Python service, install `python3-tflite-runtime` (the QNX 8.0.3
AArch64 repository lists `python3-tflite-runtime-2.21.0-r1.apk`). Its runtime
dependency is `tflite-runtime-2.21.0-r1.apk`. Confirm the Pi's QNX version
before installation; do not mix packages from different QNX releases.

## AI models

- **MediaPipe BlazeFace short-range** (`qnx/blaze_face_short_range.tflite`):
  detects faces in the fixed camera frame.
- **MediaPipe face landmark detector** (`qnx/face_landmark_detector.tflite`):
  returns facial landmarks.
- **Medusa head-direction rule:** compares nose and eye landmark geometry to
  conservatively decide whether a face is directed toward the screen.

These are MediaPipe-derived TFLite model files, but the Pi runs them through
the QNX OSS TensorFlow Lite runtime—not the full MediaPipe runtime.

## Run the QNX demo

1. Complete the [QNX camera setup](docs/qnx-deployment.md#take-a-face-detection-snapshot) and take a test photo.
2. Start the laptop listener.

   ```powershell
   python laptop_event_listener.py
   ```

3. Start the QNX AI service.

   ```sh
   python3 qnx_ai_service.py --qnx-camera-unit 5 --event-host <laptop-ip> --arduino-port /dev/ser1
   ```

Before this command, upload `arduino/medusa_controller/medusa_controller.ino`
to the UNO Q. It accepts `M,yaw,led,sequence` commands at 115200 baud and
parks the servo plus turns off the LED if the heartbeat stops for 300 ms.
Use `--servo-reversed` if left/right movement is backwards.

## Local preview dashboard

The QNX service exposes a LAN-only camera preview and status on port 8080.
The live video draws boxes around detected faces. With at least two faces,
it labels the smallest bounding box by area as **SNOOPER** and draws a line
from the preview center to that person's nose. The dashboard shows the
center-to-nose distance and horizontal and vertical offsets in camera
pixels. It uses the middle of the camera frame as the screen center.
The privacy alert uses the existing screen-facing and stability criteria.

On the laptop, start the dashboard:

```powershell
cd dashboard
npm install
npm run dev
```

Open `http://localhost:3000`, enter the Pi address (for example
`http://192.168.1.40:8080`), and click **Connect**. The preview is for setup
and debugging; it is never uploaded or sent to the cloud.

See [QNX deployment](docs/qnx-deployment.md) for model requirements, safety proof points, and the judging checklist. See [system plan](docs/system-plan.md) and [wiring diagram](docs/wiring-diagram.md) for the hardware design.
