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
to the UNO Q. It accepts `M,yaw,pitch,led,sequence` commands at 115200 baud and
disarms if the heartbeat stops for 300 ms. Disarmed (boot, button, heartbeat
loss, or the kill gate open on A2) means the servos are detached: no pulses, no
motion, LED off. The arm button re-attaches them where they stopped, and only
while the kill gate is closed and the heartbeat is live.
Use `--servo-reversed` if left/right movement is backwards.

When the camera is clipped on the turret's pan stage, add `--camera-on-turret`
(laptop-trigger mode only). The Pi models the firmware's eased motion to know
where the turret pointed when each frame was captured (`--camera-latency`, 0.07 s),
turns the face's image x into a world bearing with `--camera-hfov`, and filters
bearing and velocity. The command leads the face by `--pan-lead` (0.15 s from
capture) and holds perfectly still for a still face (1.0 deg to start moving,
0.4 deg to stop). After the target is lost the turret coasts for `--pan-coast`
(0.3 s), waits until `--pan-hold` (default: no extra wait), then glides home at
`--pan-return-speed` (60 deg/s). `--pan-gain`, `--pan-deadband` and
`--pan-max-step` are accepted but ignored. The firmware's `MAX_SPEED` and
`MAX_ACCEL` must match `FIRMWARE_*` in `medusa/trigger.py`. Tilt still maps
face y through the tilt range.

While the laptop reports a peeper but the turret camera sees no face, the
turret first points where the handoff map says (`--handoff-map`, default
`~/Medusa_QNX-bench/handoff.json`), then after `--handoff-timeout` (0.6 s), or
straight away without a map, sweeps the pan range at `--search-speed`
(60 deg/s), starting toward the laptop's side. The light stays off until the
turret camera itself sees a face. `--no-search` restores coast-then-home. Edit
the map with `python3 -m medusa.handoff record <laptop_x> <pan_deg>`, `show`,
`clear` (writes a timestamped backup beside the file first), or `record-now`
(stand in place, let the turret centre on you, run it). The service learns only
from exactly one turret face matched as the peeper, on an armed turret with the
gate closed, uses a monotonic fit of the bins (`status.json`: `map_direction`,
`map_rejected`), and discards a file whose setup fingerprint (pan home/min/max,
`--servo-reversed`, `--camera-hfov`) does not match, including files without one.

### Real-time control process

`medusa_control.py` owns `/dev/ser1` and runs a fixed 50 Hz loop on absolute
deadlines: it forwards the AI's latest target while it is fresh (< 0.3 s) and
otherwise sends a park line, so the Arduino heartbeat never depends on the AI
loop. It reads the firmware's `S,<armed>,<pan_x10>,<tilt_x10>,<seq>,<gate>` status
(every 100 ms on UNO Q D1 -> Pi RX pin 10; gate 1 = kill gate closed; the old
5-field line is still accepted), sends only the park heartbeat while the gate is
open (`control.json`: `arduino.gate_closed`, `safe_reason` "kill gate open"),
forwards the status to the AI process (UDP 8769) so the pan tracker uses the real
angle and resets while disarmed or gated,
and serves `GET /control.json` and `POST /stress?seconds=N` on port 8082.
Start it at priority 21 (its threads inherit it), then the AI process at the
default priority 10 with `--control-port`; the AI process then never opens the
serial port. In `start_medusa.sh`:

```sh
on -p 21 python3 -u medusa_control.py --port /dev/ser1 --tilt-min 90 --tilt-max 90 --tilt-home 90 &
python3 -u qnx_ai_service.py --qnx-camera-unit 5 --control-port 8768 --trigger-udp-port 8767 --trigger-from 172.20.10.4 --camera-on-turret --servo-reversed --tilt-min 90 --tilt-max 90 --tilt-home 90
```

## Local preview dashboard

The QNX service exposes a LAN-only camera preview and status on port 8080.
The live video shows the largest detected face box in green and labels every
other face **SNOOPER**. It uses box area as a proxy for closeness. The
second-largest face is labeled **SNOOPER (CLOSEST)** and gets a line from the
preview center to its nose. The dashboard shows the snooper count and the
closest snooper's center-to-nose distance and offsets in camera pixels. It
uses the middle of the camera frame as the screen center.
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
