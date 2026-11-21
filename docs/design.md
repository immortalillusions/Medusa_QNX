# Medusa design

## Demo architecture

```text
                    local Wi-Fi
QNX Pi event --------------------> laptop event listener -> blank Google Doc
    |
    | USB camera + local TensorFlow Lite inference
    v
snooper decision -> Arduino heartbeat/target -> servo-mounted low-power LED
```

The Raspberry Pi 5 runs QNX and is the authoritative embedded computer. It uses the QNX OSS TensorFlow Lite runtime to execute local face-detection and face-landmark models. Frames remain on the Pi; the laptop receives only a small decision event.

## Safety design

The Arduino UNO Q is an independent actuator safety layer. It accepts only safe angle limits, defaults the LED to off, and parks the servo when the QNX heartbeat expires. A physical kill switch removes servo and LED power.

## Detection rule

The QNX model identifies faces and estimates whether each face is directed toward the screen. When at least two high-confidence screen-facing faces remain visible for the hold period, Medusa treats the smaller/farther face as the likely snooper. This is a distance heuristic, not face identification.

The camera is fixed above the display. The moving component is the LED mount, not the camera.
