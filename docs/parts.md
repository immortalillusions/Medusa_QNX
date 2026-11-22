# Parts

## MLH & QNX Provided

- 1 Raspberry Pi 5 (8 GB) with QNX microSD card
- 1 Raspberry Pi Active Cooler
- 1 Logitech C270 webcam
- 2 Arduino UNO Q (the second is the servos' 5 V supply; it runs no code)
- 1 micro-HDMI to HDMI cable
- Jumper wires (bundle)
- 1 MacBook with built-in webcam, also the power source for everything (ours)
- 1 Apple USB-C charger, backup power for the Pi (ours)
- 1 realistic dummy head (ours)

## Joanna Provided

- 2 SG90 micro servos, with their horns (and their small screws, if they have them)
- 1 2N2222A transistor
- 3 resistors around 1 kΩ (anything from 470 Ω to 2.2 kΩ)
- 6 resistors around 10 kΩ (anything from 4.7 kΩ to 47 kΩ)
- 1 breadboard
- 1 MB102 power module
- 1 push button
- 1 multimeter
- 1 capacitor, 470 µF or bigger (required: steadies the servos' power)
- About 20 male-to-male and 3 female-to-male jumper wires

## Externally Provided

- 1 i-Zoom Rota8 puck light (Dollarama), base removed, 2 × CR2032 cells
- 3D printed parts: base, yoke, light clamp, fit coupon (see [printing.md](printing.md))
- 10 small zip ties plus 3 spares (2.5–3.6 mm wide; at least one 150 mm long, for the coins), plus 1 zip tie 300 mm long (or 2 × 150 mm joined) for the light
- About 11 toonies for the counterweight
- Hot glue, low-temperature gun if possible (the makerspace)
- 4 small rubber feet and a 20 × 20 mm patch of foam tape or mounting putty
- About 1 m of solid-core wire, for the horn wire loops and the light's two leads
- 2 USB-C to USB-C cables, one for each UNO Q (the Pi uses the Apple charger's cable)
- 1 paper marker (ArUco ID 7, 4 cm square) on matte paper
- Kitchen foil, thin card and tape (for the light's on/off switch), and black card for a 35 mm tube over the light
- Recommended: 2 spare SG90 servos

## Power

Everything runs off the laptop's three USB-C ports:

1. Pi
2. UNO Q
3. UNO Q #2, which feeds the servos' 5 V through the kill plug

The light uses its own batteries. With all three USB-C ports in use, the laptop charges by MagSafe or runs on a full battery. Stop it from sleeping, or it may cut USB power mid-demo.

## Bench notes (22 Nov, demo build)

- Power the Pi from its wall charger: on a laptop USB-C port it can brown out and drop.
- The servo 5 V rail runs through the kill plug, with the 470 µF capacitor near the
  servos and a 10 kΩ divider from that rail to A2 so the firmware senses the kill gate.
- The arm button is D2 to GND.
- Pan rests forward at 90° and tracks 15–165°; tilt is held at 0°, where the light sits
  level.
- The ArUco marker and the foil on/off switch for the light were dropped; the light is
  left on.
