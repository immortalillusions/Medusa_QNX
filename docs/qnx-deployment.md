# QNX deployment and judging checklist

## Take a face detection snapshot

Use a Raspberry Pi 5 running a QNX image with USB camera support, such as the
QNX 8.0.5 Quick Start Target Image. Plug in the webcam and copy this checkout
to the Pi, including the files in `qnx/`. The standard Pi 5 image assigns the
USB camera to Sensor Framework unit 5. The QNX workshop verified USB capture
with the Logitech C270 and C920. [QNX camera setup](https://qnx.com/developers/docs/qnxeverywhere/com.qnx.doc.target_images/topic/qsti/next-steps.html#camera) · [QNX AI workshop](https://github.com/qnx/QNX-AI-workshop#camera)

To copy the Pi files from Windows, run these commands in PowerShell from the
local repo directory and replace `PI_IP` with the Pi's address. QNX documents
the `hmac-sha2-256` setting for Windows transfers. [QNX file transfer instructions](https://qnx.com/developers/docs/qnxeverywhere/com.qnx.doc.target_images/topic/qsti/next-steps.html#adding-binaries)

```powershell
ssh -o "MACs=hmac-sha2-256" qnxuser@PI_IP "mkdir -p Medusa_QNX"
scp -o "MACs=hmac-sha2-256" -r capture_face.py qnx_ai_service.py medusa qnx "qnxuser@PI_IP:Medusa_QNX/"
```

On the Pi, install the Python runtime packages and development headers from
the QNX APK repositories configured for the image. The camera headers
include `screen/screen.h`, provided by `qnx-screen-dev`. This is a build
dependency for headless setups too. The build uses a native C compiler such
as Clang, CMake, and Make.

```sh
sudo apk update
sudo apk add cmake make qnx-sf-base-dev qnx-screen-dev python3-numpy python3-opencv python3-tflite-runtime
```

From the repo directory on the Pi, build the camera library once, verify the
Python imports, and take a photo.

```sh
CC=clang cmake -S qnx -B build/qnx
cmake --build build/qnx
cp build/qnx/libmedusa_camera.so qnx/
python3 -c 'import cv2, numpy; from tflite_runtime.interpreter import Interpreter'
python3 capture_face.py --camera-unit 5 --output captures/face.jpg
```

The script prints the face count and saves the annotated JPEG on the Pi.
Each run replaces the selected output path. For the first check, stand
near the camera in good light, run the command, and open the saved JPEG.
Run it again with a clear view of an empty area to check the zero-face case.

If the camera open fails, run `usb` to check that QNX sees the webcam and
`ls /dev/sensor` to check that the Sensor Framework exposes a camera unit.
Use the configured unit number, close any app already using that camera, and
check your account's access to the camera device. The bridge needs an
uncompressed viewfinder format such as YCbYCr (YUYV). Check the sensor
configuration if the webcam is set to MJPEG. [QNX USB camera diagnostics](https://qnx.github.io/codelabs/configure-multiple-camera-sources-on-RPI5/index.html)

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

1. Complete the snapshot setup above to confirm the camera and BlazeFace work.
2. Start the Arduino safety controller, then the laptop event listener.
3. Start the continuous service, substituting the laptop's IP address.

   ```sh
   python3 qnx_ai_service.py --qnx-camera-unit 5 --event-host 192.168.1.20
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

