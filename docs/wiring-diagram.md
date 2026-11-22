# Hardware setup

## 1. Mount

1. Put the pan servo in the printed yoke on the base plate.
2. Clip the C270 on the pan stage so the camera turns with the turret. Plug it
   into a Pi USB port.
3. Seat the servo horn so pan 0 degrees faces forward. The turret tracks from
   0 to 70 degrees, which is one side of forward.

## 2. Power

1. Power the Pi from its USB-C wall charger.
2. Power the UNO Q from USB.
3. Power the servos from a separate regulated 5 V supply. Do **not** use the Pi
   or UNO Q 5 V pins.
4. Join **servo-supply GND**, **UNO Q GND** and **Pi GND**.
5. Put the 470 uF capacitor across the servo supply, near the servos.
6. Route the servo +5 V rail through the kill plug, so pulling it cuts servo power.

## 3. UNO Q connections

| UNO Q pin | Connect to |
| --- | --- |
| D9 | Pan servo signal |
| D10 | Tilt servo signal (optional; tilt is held at 90) |
| D2 | One side of the arm/disarm push button; other side to GND |
| A2 | Kill-gate sense: three 10 kOhm resistors in series from the servo +5 V rail (after the kill plug) to GND, A2 tapped one resistor above GND |
| D0 (RX) | Pi UART TX (header pin 8) |
| D1 (TX) | Pi UART RX (header pin 10) |
| GND | Servo-supply GND and Pi GND |

With the plug in, A2 reads about 1.67 V. With it out, A2 reads 0 V, and the
firmware disarms and refuses to arm. D5 drove the light, which was dropped;
leave it unconnected. The protocol's `led` field is a leftover from it.

The button uses `INPUT_PULLUP`, so do not connect it to 5 V. The red LED on
the board is lit while armed.

## 4. Pi UART

Use the Pi header UART, `/dev/ser1` on QNX, at 115200 baud, through a
**verified level-safe interface**. Do not join the Pi and UNO Q 3.3 V or 5 V
pins.

- Pi → UNO Q: the control process sends `N,<yaw_x10>,<pitch_x10>,<led>,<seq>`
  (tenths of a degree) every 20 ms, or `M,<yaw>,<pitch>,0,<seq>` to park.
- UNO Q → Pi, every 100 ms: `S,<armed>,<pan_x10>,<tilt_x10>,<seq>,<gate>,<parking>`.
  The Pi ignores the Zephyr boot banner that also appears on D1 at reset.

## 5. Safe first test

1. With the kill plug out, confirm the button will not arm.
2. Put the plug in, start `medusa_control.py`, and arm. Over about 1.5 s the
   servos should creep to their last position, not jump.
3. Stop `medusa_control.py`. The turret should glide home (pan 0,
   tilt 90), then go limp.
4. Arm again and pull the kill plug. The servos must lose power at once.
