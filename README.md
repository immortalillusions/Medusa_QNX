# Medusa

Medusa is a QNX embedded privacy device built for Hack Western 13. A fixed camera on a Raspberry Pi 5 running QNX detects when a second person is looking at the laptop screen. The Pi runs the AI locally, opens a blank document on the laptop, and directs a low-power LED toward the snooper's torso.

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

## Run the QNX demo

1. Install the matching QNX OSS TensorFlow Lite runtime package and its Python binding on the Raspberry Pi 5.
2. Confirm the fixed C270 works on QNX.
3. Copy the project and its `qnx/*.tflite` model files to the Pi.
4. Start the laptop listener:

   ```powershell
   python laptop_event_listener.py
   ```

5. Start the QNX AI service:

   ```sh
   python3 qnx_ai_service.py --event-host <laptop-ip>
   ```

## Local preview dashboard

The QNX service exposes a LAN-only camera preview and status on port 8080.
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
