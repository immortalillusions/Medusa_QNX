# Medusa against the QNX track criteria

Every claim here is something the build on main does, with the measurement or the file that
shows it. Numbers were measured on the bench on 22 Nov 2026.

| Criterion | Evidence | Where to see it |
| --- | --- | --- |
| **Runs on QNX** | Raspberry Pi 5 on QNX 8. Turret frames come from the C270 through the QNX camera API via our own C library (`qnx/camera_capture.c` → `libmedusa_camera.so`, built on the Pi). The AI process, the 50 Hz control process and a stats process all run on QNX and start by themselves about 30 s after boot (`pi/boot-autostart.snippet`). | `pidin` on the Pi; the dashboard's Pi panel |
| **QNX open-source AI module** | TensorFlow Lite runtime from QNX's open-source packages runs the BlazeFace face model on the turret camera at ~13 fps, with OpenCV for image work. TFLite stays the primary detector; OpenCV's YuNet is only an opt-in fallback behind it (`--face-model back+yunet`). | Turret camera panel: boxes and fps |
| **Real time** | The control process runs at priority 21 (`on -p 21`) on absolute deadlines and is the only writer to the Arduino UART. Under a 20 s CPU hog (4 busy loops at priority 10) the AI falls from ~13 to ~1 fps while the loop holds **50.0 Hz, p99 jitter 2.0 ms, worst period 22 ms, 0 missed deadlines**; idle it holds p99 0.06–0.6 ms, worst 20.7 ms. The control process never forks: stress workers and CPU/memory/temperature sampling run in a separate priority-10 process (`medusa_sysmon.py`), because doing either inside the control process cost 140–165 ms stalls and dropped the loop to 48 Hz. `/control.json` has a per-tick `timing` breakdown. | Dashboard: stress button, period chart, timing tiles |
| **Cannot-fail design** | AI advises, safety code decides. The UNO Q boots disarmed and limp, enforces angle, speed and acceleration limits, and on a 300 ms heartbeat loss glides home then goes limp. A kill plug cuts servo power in hardware and the firmware senses it on A2. After **any** fault (boot, link loss, kill plug) a firmware latch refuses remote arming until a human presses the turret's physical button. Pi-side faults (camera, model) command home. Docs switching only happens while the turret reports armed. | Failure demo: kill the AI, kill the control process, unplug the camera, pull the UART wire, pull the kill plug, turn off Wi-Fi |
| **Creative use of AI** | Two cameras, two AI stacks. The laptop judges gaze from head pose plus iris (side-eye counts), finds small, distant and masked faces with YuNet plus zoomed MediaPipe crops, picks the most recent peeper and ignores feints. The turret matches that peeper among its own faces, never the user, and keeps aiming from the laptop's view through a pan map it learns by itself when its own camera loses the face. | Live demo |
| **On the device, not the cloud** | All inference runs locally: MediaPipe and YuNet on the Mac, TFLite and OpenCV on the Pi. Only a small UDP heartbeat crosses the network. The dashboard loads no outside fonts or scripts. | Wi-Fi off: the turret falls safe instead of hanging |

## Honest limits

- The latest real-time run had the Arduino unplugged; re-measure with it connected before quoting.
- Gaze at 7–10 m comes from head pose only; iris reading works to about 5 m.
- Head-from-body detection (`--head-fallback`) and two-camera fusion (`--fusion`) are built and
  tested but off by default until they have been tried live on the turret.
