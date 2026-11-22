# TODO

## Pi camera snapshot

- [ ] Build the camera bridge and run `python3 capture_face.py` on the QNX Pi.
- [ ] Confirm the JPEG shows a green box around a face and the command prints the count.
- [ ] Repeat with an empty frame and confirm the command saves a photo with zero faces.

## QNX AI

- [ ] Boot the Pi into QNX.
- [ ] Confirm the C270 appears on QNX and captures frames.
- [ ] Install the QNX TensorFlow Lite runtime and Python binding.
- [ ] Run a one-line import test: `python3 -c "from tflite_runtime.interpreter import Interpreter"`.
- [ ] Copy the repository, including `qnx/*.tflite`, to the Pi.
- [ ] Run `python3 qnx_ai_service.py --qnx-camera-unit 5 --event-host <laptop-ip> --arduino-port /dev/ser1` (replace with the verified QNX serial device).
- [ ] Confirm it loads both models and emits an inactive event when no faces are present.
- [ ] Test two people facing the camera; confirm a sustained event is emitted.
- [ ] Start `dashboard/` with `npm install` then `npm run dev`; confirm its preview and status update.

## Laptop

- [ ] Run `python laptop_event_listener.py`.
- [ ] Confirm an active QNX event opens `docs.new`.

## Arduino and safety

- [ ] Upload `arduino/medusa_controller/medusa_controller.ino` to the UNO Q.
- [ ] Wire the arm/disarm push button: Arduino D2 to one side, GND to the other.
- [ ] Confirm boot is disarmed; press once to arm (built-in LED on), press again to disarm.
- [ ] Enforce servo angle limits.
- [ ] On heartbeat timeout: LED off and servo centred.
- [ ] Build the wiring in [docs/wiring-diagram.md](docs/wiring-diagram.md).
- [ ] Test the kill switch cuts servo and LED power.

## Demo

- [ ] Keep the camera fixed above the display.
- [ ] Aim the LED at torso height, never eyes.
- [ ] Demonstrate: normal state, snooper detection, camera disconnect, QNX/heartbeat loss.

