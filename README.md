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
- **A real-time control loop on QNX** runs at 50 Hz and keeps its timing while
  the AI process is starved of CPU.

## Who it helps

Shoulder-surfing is one of the simplest ways private information leaks: patient records at a
clinic front desk or nursing station, account numbers at a bank counter, an open-plan office.
Staff can't watch the screen and the room at the same time. Medusa covers the record with a
blank document the moment a second person looks, so the staff member also sees that someone
was looking. It identifies, records and uploads no one: detection runs on the laptop and the
turret, and laptop images never leave the laptop. A clinic could run screen-hiding alone, with
no turret or light. It needs no change to the clinic's software and uses a laptop's own webcam.

## Architecture

```text
 MacBook                                  Raspberry Pi 5, QNX 8
 ───────                                  ─────────────────────
 laptop_detector.py ─── UDP 8767 ───────▶ qnx_ai_service.py        (priority 10)
   webcam, MediaPipe     medusa-peeper      C270 on the pan turret, BlazeFace "back"
   landmarks + gaze                         peeper matching, pan tracking,
   peeper tracker                           laptop-guided fallback, learned pan map
   Google Docs switch                           │ UDP 8768 target   ▲ UDP 8769 actual angle
        ▲                                       ▼                   │
        └──── armed? GET :8082 ──────────  medusa_control.py        (on -p 21, 50 Hz)
                                            owns /dev/ser1, heartbeat, /control.json
                                                │ UART 115200: N/M lines out, S status in
                                                ▼
                                           Arduino UNO Q: medusa_controller.ino
                                            eased servos, arm button, A2 kill gate,
                                            300 ms watchdog

 dashboard/ (Next.js, :3000) reads laptop :8081, Pi AI :8080, Pi control :8082
```

| Piece | File | Job |
| --- | --- | --- |
| Laptop detector | `laptop_detector.py`, `medusa/gaze.py`, `medusa/peeper_tracker.py`, `medusa/zoom.py` | Finds small, distant and masked faces with OpenCV YuNet, then reads zoomed crops with MediaPipe (faces found out to ~10 m on downscaled clips). Tracks faces across frames, smooths gaze per face, and decides when a peeper is looking. With several peepers it picks the most recent steady look and ignores glances and feints. Sends a UDP heartbeat to the Pi every frame. Switches the Mac to a Google Docs tab, but only while the turret is armed. |
| Pi AI service | `qnx_ai_service.py`, `medusa/trigger.py`, `medusa/handoff.py`, `medusa/tflite_detector.py` | Finds faces in the turret camera with TensorFlow Lite (BlazeFace) and matches the laptop's peeper against them, so the turret aims at the peeper and not the user. Tracks with proportional pan. If the turret loses the face, it keeps aiming from the laptop's view through a pan map it learns as it tracks: it glides there without a jump, leads a moving peeper, searches around the map's guess when the map is wrong, and holds 1 s before heading home. Opt-in extras: `--face-model back+yunet` (OpenCV YuNet as a fallback behind TFLite for masked and small faces), `--head-fallback` (head from profile, upper-body and HOG person detectors), `--fusion` (one Kalman bearing from both cameras, map per distance band). |
| Pi control process | `medusa_control.py`, `medusa/control.py` | The only owner of the serial port. Runs a fixed 50 Hz loop on absolute deadlines. It forwards fresh AI targets (under 0.3 s old) and otherwise sends a park line, so the Arduino heartbeat never depends on the AI. |
| UNO Q firmware | `arduino/medusa_controller/medusa_controller.ino` | Eases the servos (speed and acceleration limits) and enforces angle limits. Arming, disarming, the watchdog and the kill gate all live here. |
| Dashboard | `dashboard/` | Both camera views, the decision, and a safety panel: link lights, loop rate and jitter, arm and gate state, and a CPU stress button. |

Video never leaves the local network. All models are checked in under `qnx/`
and run on the device.

## Real time on QNX

The control process runs at priority 21 (`on -p 21`) and the AI process at the
default 10. When the dashboard's stress test (`POST :8082/stress?seconds=N`)
saturates the CPU with four busy loops at priority 10, the AI drops from about
13 fps to 1 fps. The control loop is unaffected. Measured on the Pi on 22 Nov
(the latest run had the Arduino unplugged; see `docs/qnx-criteria.md`):

| Measurement | Idle | Under the CPU hog (20 s) |
| --- | --- | --- |
| Control loop rate | 50.0 Hz | 50.0 Hz |
| p99 period jitter | 0.06–0.6 ms | 2.0 ms |
| Worst period | 20.7 ms | 22.0 ms |
| Missed deadlines | 0 | 0 |
| AI frame rate | ~13 fps | ~1 fps |

The loop runs on absolute deadlines, so a late tick does not push the next one
later. Its one HTTP thread inherits priority 21, and the GIL switch interval is set to
1 ms. The control process never forks or execs: the stress workers are spawned by
`medusa_sysmon.py` at priority 10 (POST /stress only sends it a localhost UDP
datagram), because spawning them from the priority-21 process stalled the loop for
140–165 ms at every stress start. `/control.json` reports the live period, jitter
and missed deadlines, plus a `timing` block (per-tick work, serial and UDP time,
and how late ticks start) that shows where the loop's time goes.
The two processes talk over localhost UDP. QNX message passing would be the
production choice, but stock Python has no binding for it.

## Next hardware version

`docs/medusa-shield.md` is the PCB we'd fabricate next: the breadboard's safety circuit
(servo power with a two-pole kill switch, kill sense, arm button, Pi UART) on one UNO-format
shield, with pin map, BOM and v2 changes.

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
- **Fault latch: after any fault, a human presses the button.** The firmware
  latches `needsButton` at boot, whenever the heartbeat is stale (link loss, and
  the park it causes), and whenever the kill gate is open. Only a press of the D2
  button clears it, and a press during the fault does not count. While it is set
  the dashboard's Arm (`A,1`) is refused and its button says to press the
  turret's button. Switching off from the dashboard or the button is not a fault,
  so the dashboard can re-arm after its own switch-off. The S status line reports
  the latch as its 8th field, and `/control.json` as `arduino.needs_button`.
- **Pi faults go home.** A failed camera frame or a face-model error commands
  exactly home. A stale AI target makes the control loop send park lines.
- **Home is pan 90, tilt 0.** Pan rests facing forward and tracks 15 to 165
  degrees, 75 each side. Tilt is held at 0, where the light is level. Out-of-range commands are rejected.
- **Docs only while armed.** The laptop switches to Google Docs only when
  `/control.json` reports the turret armed with the gate closed.
  `--docs-always` overrides this for bench testing.

## Run it

For the demo itself, follow `docs/demo-day.md`: what to bring, setup order, checks, and fixes (including a one-command rollback).

### 1. Flash the UNO Q

Upload `arduino/medusa_controller/medusa_controller.ino` with the
`arduino:zephyr` core (FQBN `arduino:zephyr:unoq`) and the Servo library.

```sh
arduino-cli core install arduino:zephyr
arduino-cli lib install Servo
arduino-cli compile --fqbn arduino:zephyr:unoq arduino/medusa_controller
arduino-cli upload -p <port> --fqbn arduino:zephyr:unoq arduino/medusa_controller
```

It always boots disarmed and latched: the first arm after boot (or after any
fault) takes the D2 button, not the dashboard. The red LED is lit while armed. Pins and power are in [docs/wiring-diagram.md](docs/wiring-diagram.md).

### 2. Start the Pi (QNX 8)

Set up the Pi once with [docs/qnx-deployment.md](docs/qnx-deployment.md). That
covers the TFLite runtime, OpenCV and the camera bridge. Then start the header
UART (`/dev/ser1`) and run both processes, control first:

```sh
on -p 21 python3 -u medusa_control.py --port /dev/ser1 \
    --servo-min 15 --servo-max 165 --servo-home 90 --tilt-min 0 --tilt-max 0 --tilt-home 0 &
python3 -u qnx_ai_service.py --qnx-camera-unit auto --control-port 8768 \
    --trigger-udp-port 8767 --trigger-from <mac-ip> --camera-on-turret --servo-reversed \
    --servo-min 15 --servo-max 165 --servo-home 90 --tilt-min 0 --tilt-max 0 --tilt-home 0
```

The pan and tilt ranges must match `PAN_*` and `TILT_*` in the firmware, or the
firmware rejects the commands. On the demo Pi, `pi/start_medusa.sh` runs these (plus the low-priority
`medusa_sysmon.py` stats process) from the repo checkout; `pi/start_uart.sh` brings up the
header UART first. `pi/boot-autostart.snippet` is the block appended to
`/data/var/etc/settings/network`, which QNX's boot script sources as root, so Medusa
starts about 30 s after boot.

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
