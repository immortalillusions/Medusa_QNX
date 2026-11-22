# Parts

## Compute

- Raspberry Pi 5 with a QNX 8 microSD image and the Active Cooler, powered from
  its wall charger. A laptop USB-C port can brown it out.
- MacBook with a built-in camera, on the same network as the Pi.
- Arduino UNO Q.

## Turret

- Logitech C270 USB webcam, clipped on the pan stage so it turns with the turret.
- SG90 micro servo for pan. A second SG90 for tilt is optional; the demo holds
  tilt at 90.
- 3D-printed base plate and pan yoke (`cad/`).

## Power and safety

- A dedicated regulated 5 V supply for the servos, never the Pi or UNO Q 5 V pin.
- A 470 uF or larger capacitor across the servo supply, near the servos.
- A kill plug or pull-tab that cuts the servo 5 V rail.
- Three 10 kOhm resistors: a divider from the servo rail to A2, so the firmware
  can sense the kill gate.
- A push button for arm/disarm (D2 to GND).
- A level-safe connection for the Pi to UNO Q UART, plus breadboard and jumper wires.
- A multimeter.
