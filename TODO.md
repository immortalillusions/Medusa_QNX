# TODO

## QNX AI

- [ ] Boot the Pi into QNX.
- [ ] Confirm the C270 appears on QNX and captures frames.
- [ ] Install the QNX TensorFlow Lite runtime and Python binding.
- [ ] Run a one-line import test: `python3 -c "from tflite_runtime.interpreter import Interpreter"`.
- [ ] Copy the repository, including `qnx/*.tflite`, to the Pi.
- [ ] Run `python3 qnx_ai_service.py --event-host <laptop-ip>`.
- [ ] Confirm it loads both models and emits an inactive event when no faces are present.
- [ ] Test two people facing the camera; confirm a sustained event is emitted.
- [ ] Start `dashboard/` with `npm install` then `npm run dev`; confirm its preview and status update.

## Laptop

- [ ] Run `python laptop_event_listener.py`.
- [ ] Confirm an active QNX event opens `docs.new`.

## Arduino and safety

- [ ] Add a QNX UART sender to `qnx_ai_service.py` for yaw/LED/heartbeat commands.
- [ ] Write/upload Arduino firmware: parse yaw/LED/heartbeat commands.
- [ ] Enforce servo angle limits.
- [ ] On heartbeat timeout: LED off and servo centred.
- [ ] Build the wiring in [docs/wiring-diagram.md](docs/wiring-diagram.md).
- [ ] Test the kill switch cuts servo and LED power.

## Demo

- [ ] Keep the camera fixed above the display.
- [ ] Aim the LED at torso height, never eyes.
- [ ] Demonstrate: normal state, snooper detection, camera disconnect, QNX/heartbeat loss.

