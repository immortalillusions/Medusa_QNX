# QNX deployment

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
scp -o "MACs=hmac-sha2-256" -r capture_face.py qnx_ai_service.py medusa_control.py medusa qnx "qnxuser@PI_IP:Medusa_QNX/"
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

## AI on the Pi

The Pi uses the **TensorFlow Lite runtime** from QNX OSS, with checked-in
models and no cloud service:

- `qnx/face_detector.tflite`: BlazeFace "back" (256x256), which finds smaller,
  farther faces. It is the default in laptop-trigger mode.
- `qnx/blaze_face_short_range.tflite`: BlazeFace short-range (128x128), used by
  `capture_face.py` and the older Pi-only mode.
- `qnx/face_landmark_detector.tflite`: 468-point face landmarks.

These are MediaPipe-derived model files run through `tflite_runtime`, not the
full MediaPipe runtime. QNX OSS publishes an AArch64 TFLite runtime package;
match it to the installed QNX release.

## Run Medusa on the Pi

1. Flash the UNO Q and wire it to the Pi header UART (`/dev/ser1`), following
   [the wiring guide](wiring-diagram.md).
2. Start the control process at priority 21, then the AI process at the
   default priority. The commands are in the
   [README](../README.md#2-start-the-pi-qnx-8).
3. Start the laptop detector on the Mac. The Pi accepts its heartbeats on
   UDP 8767 from the address given in `--trigger-from`.

On the Pi:

- `:8080/status.json` and `:8080/stream.mjpg`: the AI decision and the turret
  camera view.
- `:8082/control.json`: loop timing, Arduino link, arm and gate state,
  `arduino.needs_button` (the firmware's fault latch), `safe_reason`, and a
  `timing` block (see below).
- `POST :8082/stress?seconds=N` (0 stops): CPU hog workers at the AI's priority,
  for the real-time demo. Returns 202. The control process only sends a UDP
  datagram to `medusa_sysmon.py` (127.0.0.1:8770), which spawns the workers with
  `on -p 10` and writes `/tmp/medusa_stress.json`; control.json's `stress` block
  reads that file. Returns 503 if `medusa_sysmon.py` is not running.
- `POST :8082/arm` and `POST :8082/disarm`: the dashboard's arm / switch-off
  button. Returns 202; the loop sends one `A` line on its next tick and the
  firmware decides (kill gate, heartbeat and the fault latch still apply).

### Process layout

| Process | Priority | Job |
| --- | --- | --- |
| `medusa_control.py` | 21 | 50 Hz loop, sole UART writer, HTTP :8082. Never forks or execs. |
| `medusa_sysmon.py` | 10 | Pi usage samples (`/tmp/medusa_system.json`) and the stress workers (`/tmp/medusa_stress.json`). |
| `qnx_ai_service.py` | 10 | Camera, face models, targets to the control process over UDP :8768. |

### Fault latch

The firmware sets `needsButton` at boot, while the heartbeat is stale (no N/M
line for 300 ms, including the park that link loss causes) and while the kill
gate is open. Only a D2 press clears it, and loop() sets it again at once if the
fault still holds. While it is set, `A,1` is ignored: the dashboard disables Arm
with "press the turret's button (safety reset after a fault)". A dashboard or
button switch-off is not a fault and leaves it clear. A restart of
`medusa_control.py` stops the heartbeat, so it needs a button press afterwards.

### Reading the `timing` block

Per 10 s window: `work_ms_p99` / `work_ms_max` (one tick's own work),
`serial_ms_p99` (reading S lines, writing N/M/A lines), `udp_ms_p99` (draining AI
targets, sending medusa-actual), `late_ms_p99` (how long after its deadline a tick
started: sleep overshoot plus waiting for the GIL), `late_ticks` (ticks more than
5 ms late) and `late_ticks_http` (those of them that overlapped an HTTP request,
meaning the HTTP thread held the GIL the loop needed). Jitter comes from late
starts, not from tick work, unless work approaches 20 ms. If `late_ticks_http` is
a small part of `late_ticks`, the cause is outside this process: try
`slay` on `medusa_sysmon.py` for a minute (its `pidin` queries touch every
process) and compare.

`--http-priority N` puts only the HTTP thread at priority N (pthread_setschedprio
through ctypes; verified on the Pi). It is off by default: thread priority does
not decide who gets Python's GIL, so a lower HTTP priority does not make the loop
get the GIL sooner, and a preempted HTTP thread holding the GIL would delay it.

## Pre-demo checks

- Both processes start, and `/control.json` shows 50 Hz with p99 jitter of a
  few ms.
- With the kill plug out, the turret will not arm. With it in, the D2 button
  arms it and the red LED lights.
- Stopping `medusa_control.py` makes the turret glide home and go limp, and
  the dashboard's Arm then says to press the turret's button until D2 is pressed.
- Unplugging the turret camera sends the turret home.
- A stress run drops the AI frame rate, but the control loop period stays at
  20 ms.
- No model is downloaded and no video leaves the local network.
