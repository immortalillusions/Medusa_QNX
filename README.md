# Medusa

Medusa is a privacy turret for your laptop. When someone behind you looks at
your screen, Medusa swaps your work for a Google Doc and turns a camera turret
to face them. It was built at Hack Western 13 for the QNX track.

- **The MacBook decides whether someone is peeping.** Its webcam runs MediaPipe
  face landmarks and gaze estimation (head pose plus iris), so a side-eye
  counts even with the head turned away. A peeper is a second face looking at
  the screen. The largest face is the user and never counts.
- **The Raspberry Pi 5 running QNX 8 decides where the peeper is.** A Logitech
  C270 on a pan turret runs a TFLite BlazeFace model and keeps the peeper
  centred.
- **An Arduino UNO Q moves the servos** and holds the safety rules: an arm
  button, a kill gate, a heartbeat watchdog, and a glide home on any fault.
- **A hard real-time control loop** runs at 50 Hz and keeps its timing while
  the AI process is starved of CPU.

## Architecture

```text
 MacBook                               Raspberry Pi 5, QNX 8
 ───────                               ─────────────────────
 webcam → laptop_detector.py           qnx_ai_service.py   (priority 10)
   MediaPipe landmarks + gaze            C270 on the pan turret
   peeper tracker                        TFLite BlazeFace "back" model + OpenCV
   Google Docs switch ─────┐             peeper matching, pan tracking,
        │                  │             laptop-guided fallback, learned pan map
        │ UDP 8767         │                    │ UDP 8768 targets  ▲ UDP 8769 actual angles
        │ medusa-peeper    │                    ▼                   │
        └──────────────────┼──────────▶ medusa_control.py   (on -p 21, 50 Hz)
                           │             owns /dev/ser1, heartbeat, /control.json
     armed? (GET :8082) ◀──┘                    │ UART 115200: N/M lines out, S status in
                                                ▼
                                       Arduino UNO Q: medusa_controller.ino
                                         eased pan/tilt servos, arm button,
                                         A2 kill gate, 300 ms watchdog

 dashboard/ (Next.js, :3000) reads laptop :8081, Pi AI :8080, Pi control :8082
```

| Piece | File | Job |
| --- | --- | --- |
| Laptop detector | `laptop_detector.py`, `medusa/gaze.py`, `medusa/peeper_tracker.py` | Tracks faces across frames, smooths gaze per face, and decides when a peeper is looking. Sends a UDP heartbeat to the Pi every frame. Switches the Mac to a Google Docs tab, but only while the turret is armed. |
| Pi AI service | `qnx_ai_service.py`, `medusa/trigger.py`, `medusa/handoff.py`, `medusa/tflite_detector.py` | Finds faces in the turret camera and matches the laptop's peeper against them, so the turret aims at the peeper and not the user. Tracks with proportional pan. If the turret loses the face, it aims from the laptop's view through a pan map it learns as it tracks, then sweeps. It aims at the most recent peeper and resists feints. |
| Pi control process | `medusa_control.py`, `medusa/control.py` | The only owner of the serial port. Runs a fixed 50 Hz loop on absolute deadlines. It forwards fresh AI targets (under 0.3 s old) and otherwise sends a park line, so the Arduino heartbeat never depends on the AI. |
| UNO Q firmware | `arduino/medusa_controller/medusa_controller.ino` | Eases the servos (speed and acceleration limits) and enforces angle limits. Arming, disarming, the watchdog and the kill gate all live here. |
| Dashboard | `dashboard/` | Both camera views, the decision, and a safety panel: link lights, loop rate and jitter, arm and gate state, and a CPU stress button. |

Video never leaves the local network. All models are checked in under `qnx/`
and run on the device.

## Real time on QNX

The control process runs at priority 21 (`on -p 21`) and the AI process at the
default 10. When the dashboard's stress test (`POST :8082/stress?seconds=N`)
saturates the CPU with busy loops at priority 10, the AI drops from 13.7 to
0.5 fps. The control loop is unaffected:

| Measurement | Result |
| --- | --- |
| Control loop period | 20.00 ms (50 Hz) |
| p99 jitter, idle and under the CPU hog | 1–2 ms |
| AI frame rate, idle → under the CPU hog | 13.7 → 0.5 fps |

The loop runs on absolute deadlines, so a late tick does not push the next one
later. Its threads inherit priority 21, and the GIL switch interval is set to
1 ms. `/control.json` reports the live period, jitter and missed deadlines.
The two processes talk over localhost UDP. QNX message passing would be the
production choice, but stock Python has no binding for it.

## Safety

- **Off means limp.** At boot, or when disarmed by the button, the servos are
  detached and get no pulses at all.
- **Arm button (D2).** Arming needs the kill gate closed and a live heartbeat.
  The servos then creep to their last position over 1.5 s instead of jumping.
  Pressing the button again glides the turret home, then it goes limp.
- **Kill gate (A2).** The kill plug feeds the servo 5 V rail, and A2 senses that
  rail through a divider. Pulling the plug cuts servo power and disarms at once,
  even mid-glide.
- **Heartbeat watchdog.** If no valid command arrives for 300 ms while armed,
  the firmware glides home and then goes limp (2 s cap). Re-arming takes a
  button press.
- **Pi faults go home.** A failed camera frame or a face-model error commands
  exactly home. A stale AI target makes the control loop send park lines.
- **Home is pan 0, tilt 90.** Pan rests facing forward and tracks 0 to 70
  degrees. Tilt is held at 90. Out-of-range commands are rejected.
- **Docs only while armed.** The laptop switches to Google Docs only when
  `/control.json` reports the turret armed with the gate closed.
  `--docs-always` overrides this for bench testing.

## Run it

### 1. Flash the UNO Q

Upload `arduino/medusa_controller/medusa_controller.ino` with the
`arduino:zephyr` core (FQBN `arduino:zephyr:unoq`) and the Servo library.

```sh
arduino-cli core install arduino:zephyr
arduino-cli lib install Servo
arduino-cli compile --fqbn arduino:zephyr:unoq arduino/medusa_controller
arduino-cli upload -p <port> --fqbn arduino:zephyr:unoq arduino/medusa_controller
```

It always boots disarmed. Press the D2 button to arm; the red LED is lit while
armed. Pins and power are in [docs/wiring-diagram.md](docs/wiring-diagram.md).

### 2. Start the Pi (QNX 8)

Set up the Pi once with [docs/qnx-deployment.md](docs/qnx-deployment.md). That
covers the TFLite runtime, OpenCV and the camera bridge. Then start the header
UART (`/dev/ser1`) and run both processes, control first:

```sh
on -p 21 python3 -u medusa_control.py --port /dev/ser1 \
    --servo-min 0 --servo-max 70 --servo-home 0 --tilt-min 90 --tilt-max 90 --tilt-home 90 &
python3 -u qnx_ai_service.py --qnx-camera-unit 5 --control-port 8768 \
    --trigger-udp-port 8767 --trigger-from <mac-ip> --camera-on-turret --servo-reversed \
    --servo-min 0 --servo-max 70 --servo-home 0 --tilt-min 90 --tilt-max 90 --tilt-home 90
```

The pan and tilt ranges must match `PAN_*` and `TILT_*` in the firmware, or the
firmware rejects the commands. On the demo Pi, a `start_medusa.sh` script runs
these at boot; it lives on the Pi, not in this repo.

The learned pan map is saved to `~/Medusa_QNX-bench/handoff.json`. Use
`python3 -m medusa.handoff show|record|record-now|clear` to inspect or edit it.
Run `python3 qnx_ai_service.py --help` to see every tracking option.

### 3. Start the laptop detector (macOS)

```sh
python3 -m venv .venv && .venv/bin/pip install mediapipe opencv-python
open -a Terminal run_laptop_detector.command     # MEDUSA_PI=<pi-ip> to override 172.20.10.2
```

Launch it through Terminal.app so macOS grants camera and browser-automation
access. `calibrate_laptop_detector.command` is the bench mode: one face is
enough to trigger, Docs switching is off, and gaze numbers go to
`captures/gaze.log`. Its preview runs on `http://localhost:8081`.

### 4. Open the dashboard

```sh
cd dashboard && npm install && npm run dev
```

Open `http://localhost:3000`. The turret address defaults to
`http://172.20.10.2:8080` and the laptop to `http://localhost:8081`. The
control URL (port 8082) is derived from the turret address.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests
```

The tests cover gaze, peeper tracking and matching, pan tracking, the handoff
map, the control loop and its safety states. They also check that the
firmware's constants and park logic agree with the Python side. No hardware
is needed.

## Repo layout

```text
laptop_detector.py         MacBook peeper detector (run via run_laptop_detector.command)
qnx_ai_service.py          Pi AI process: turret camera, matching, tracking
medusa_control.py          Pi real-time control process (50 Hz, priority 21)
medusa/                    shared library: gaze, tracker, trigger, control, handoff, camera, preview
arduino/medusa_controller/ UNO Q firmware
dashboard/                 Next.js dashboard
qnx/                       TFLite/MediaPipe models and the QNX Sensor Framework camera bridge (C)
capture_face.py            one-shot Pi camera + BlazeFace check, used during setup
cad/                       3D-printed turret parts (manifold3d + trimesh script and STLs)
docs/                      Pi setup, wiring, parts
tests/                     unit tests
```

Before the laptop decided, the Pi itself picked out a second face looking at
the screen and sent `medusa-event` packets. That mode still works:
`qnx_ai_service.py --event-host <laptop-ip>` sends the packets, and
`laptop_event_listener.py` receives them. A Windows webcam variant also
remains (`qnx/face_webcam.py`, see
[docs/windows-desktop-switching.md](docs/windows-desktop-switching.md)).
Neither is part of the current demo.
