# Hardware setup

## 1. Mount

1. Fix the C270 above the laptop screen, pointing at the people in front of it.
2. Mount the LED on the left/right servo arm.
3. Keep the camera fixed. Only the LED mount moves.

## 2. Power

1. Power the Pi from its USB-C charger.
2. Power the Arduino from USB.
3. Power the servo from a separate regulated 5 V supply.
4. Connect **servo-supply GND** to **Arduino GND**.
5. Do **not** power the servo from the Pi or Arduino 5 V pins.
6. Put the 470 uF capacitor across the servo supply, near the servo.
7. Add a kill switch that cuts servo and LED power.

## 3. Arduino connections

| Arduino pin | Connect to |
| --- | --- |
| D9 | Left/right servo signal |
| D10 | Optional tilt-servo signal |
| D5 | 1 kOhm resistor, then 2N2222A base |
| D2 | One side of arm/disarm push button; other side to GND |
| GND | Servo-supply GND and 2N2222A emitter |

Connect the LED/light positive side to its approved power source. Connect its
negative side to the 2N2222A collector. Check the transistor's pinout before
powering it.

The button uses Arduino `INPUT_PULLUP`: do not connect it to 5 V. The built-in
Arduino LED is on while armed and off while disarmed. At boot or after a button
press to disarm, Medusa is disarmed: the servos stop where they are and go limp,
the LED turns off, and QNX commands are ignored until the button is pressed
again. After a 300 ms heartbeat timeout while armed, the servos first glide home
(at most 2 s), then go limp the same way.

## 4. Pi connections

1. Plug the C270 into a Pi USB port.
2. Connect Pi GND to Arduino GND.
3. Connect Pi UART TX/RX to Arduino RX/TX through a **verified level-safe
   interface**. Do not join the Pi and Arduino 3.3 V or 5 V pins.
4. Verify the exact QNX UART device and UNO Q voltage levels before connecting
   serial data.

The Pi sends `M,yaw,pitch,led,sequence` at 115200 baud (old `M,yaw,led,sequence` still accepted). Example:
`M,92,80,1,1842`. The Arduino parks both servos and turns the LED off after 300 ms
without a valid command.

## 5. Safe first test

1. Keep the LED disconnected.
2. Test the Arduino moves the servo only within safe angles.
3. Stop its heartbeat input: the servo must centre and the LED control must go off.
4. Connect the LED and confirm the kill switch removes its power.

