# Windows desktop switching

This is an older Windows variant, kept working but outside the current demo.
The current build runs on a MacBook with `laptop_detector.py`; see the
[README](../README.md).

The Windows webcam demo and the Pi event listener can activate the most
recently used window of a pinned app when Medusa confirms a snooper. Use
`--switch-desktop` to opt in and
`--taskbar-slot` to pick the app. Slot 1 is the default. Slots 1 through 9 use
their matching number keys, and slot 10 uses the 0 key. [Windows taskbar shortcuts](https://support.microsoft.com/en-au/accessibility/windows/keyboard-shortcuts-in-windows)

## Pin Excel and verify the slot

1. Open the workbook you want to show. Right-click the Excel taskbar icon and
   select **Pin to taskbar**. [Microsoft taskbar instructions](https://support.microsoft.com/en-us/windows/experience/personalization/customize-the-taskbar-in-windows)
2. Drag Excel to the first pinned app position for slot 1, or count to its
   position from left to right among the app icons. Start and Search are
   taskbar controls, outside the app positions.
3. If you have multiple workbooks open, click the workbook you want to make it
   the most recently used Excel window. From another app, press
   **Win+Ctrl+1** to verify your slot. Use the number for your Excel position.

## Test before turning on the camera

From the repo root in Windows PowerShell, run this with the workbook open.
The 15-second countdown gives you time to bring your game forward. Use
**Borderless Windowed** in the game for the first test.

```powershell
python .\qnx\face_webcam.py --test-switch --taskbar-slot 1 --countdown 15
```

The test feeds one simulated confirmation through the same switch trigger the
webcam uses. It requires only Python's standard library. A log such as
`Win+Ctrl+1 sent. Foreground focus was not verified.` means Windows accepted
the key events. Check that the intended workbook appeared, then return to the
game manually when you're ready.

## Enable switching for live detections

Use the Python environment you normally use for the webcam. To set one up,
install 64-bit Python 3.13 with the Windows launcher, then run these commands
from the repo root. [Python for Windows](https://www.python.org/downloads/windows/) · [MediaPipe Python setup](https://developers.google.com/edge/mediapipe/solutions/setup_python)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install mediapipe opencv-contrib-python
.\.venv\Scripts\python.exe .\qnx\face_webcam.py --camera 0 --switch-desktop --taskbar-slot 1
```

With the dependencies already installed for `python`, the webcam-only command
is `python qnx/face_webcam.py --camera 0`. Append `--switch-desktop` to enable
switching. Try `--camera 1` if Windows assigns the webcam a different index.

The webcam passes BlazeFace landmark observations to the same
`SnooperSelector` used by the Pi service. It looks for two screen-facing faces
with at least 0.65 confidence and confirms after 15 frames with the selected
face in the same horizontal region. The first confirmed frame queues one
shortcut on a background thread. Another attempt requires 30 consecutive
webcam frames with fewer than two detected faces. Frames with two faces and an
uncertain confirmation interrupt that clear streak. The attempt counts for the
encounter even when Windows rejects the shortcut.

## Enable switching with the Pi camera

Open your workbook and keep the USB camera plugged into the Pi. Copy the
updated `qnx_ai_service.py` and `medusa/` files to the Pi, then run the listener
from the repo root in Windows PowerShell. Substitute your Pi's IP address
for `172.20.10.2` if it changed.

```powershell
python .\laptop_event_listener.py --switch-desktop --pi-ip 172.20.10.2 --taskbar-slot 1
```

Leave the listener open. Restart the service in your Pi SSH terminal and
substitute the Windows laptop's LAN IP address for `172.20.10.3`. The Windows
command `ipconfig` shows the laptop's address on the network shared with the Pi.

```sh
cd ~/Medusa_QNX
python3 qnx_ai_service.py --qnx-camera-unit 5 --event-host 172.20.10.3
```

The Pi sends confirmations to the listener on UDP port 8766. The dashboard's
**Privacy alert active** status indicates a confirmed snooper. The listener
re-arms when the Pi reports 30 consecutive frames with fewer than two
detected faces.

## If Excel stays behind the game

- Verify the physical **Win+Ctrl+number** shortcut from an ordinary desktop
  app first. Check that Excel is pinned in the specified slot and has the
  intended workbook open.
- Repeat the countdown in **Borderless Windowed** and release Shift, Alt, and
  the slot key before it expires. Those keys can change the shortcut.
- If the physical shortcut works and Windows rejects the generated shortcut,
  try the game and Medusa in the same regular Windows desktop session with
  both launched normally. Windows can reject input sent from a process at a
  lower integrity level. [Windows SendInput behavior](https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-sendinput)
- If the simulated test works, wait for the webcam preview to start, bring the
  game forward, and have two people face the camera. Add `--verbose` to the
  webcam command to see why a frame was skipped. Use **Q** or **Esc** in the
  webcam preview to quit.

Run the automated tests with `python -m unittest discover -s tests -v`.
