# Medusa system plan

## What Medusa does

Medusa is a QNX embedded privacy device. A fixed USB camera connected to the Raspberry Pi 5 observes the viewing area. The Pi runs a local AI model to identify faces looking toward the screen. When two or more faces persist, it selects the farther/smaller face as the likely snooper, covers the laptop screen, and points a low-power LED at that person's torso.

The judged configuration does not send video to the cloud or to the laptop.

## Component roles

| Part | Job |
| --- | --- |
| **Raspberry Pi 5 running QNX** | Owns camera capture, TensorFlow Lite inference, snooper selection, events, and actuator targets. |
| **C270 fixed camera** | Looks at the whole viewing area from above the display. It never rotates, so the coordinate system is stable. |
| **TensorFlow Lite runtime from QNX OSS** | Runs the face detector and landmark model locally on the Pi. This is the project's required QNX OSS AI module. |
| **Arduino UNO Q** | Moves the servo, switches the LED, enforces safe angle limits, and fails off when the Pi heartbeat stops. |
| **Servo and LED** | The LED mount moves left/right. The LED is low power and aimed at the torso, never the eyes. |
| **Laptop** | Receives only an `active`/`inactive` local event from QNX and opens a blank Google Doc. It is not the AI host. |

## Decision flow

```text
fixed C270 -> QNX camera capture -> TFLite face + landmark models
          -> stable two-or-more-face rule -> snooper target
          -> Arduino target + heartbeat -> servo/LED

QNX privacy event -> laptop event listener -> docs.new
```

The model provides faces, a confidence that each face is looking at the display, and relative face size. Smaller face size is the first distance heuristic. The selected result must persist for approximately 0.5--1 second before Medusa responds. This prevents a passer-by from triggering it.

## Why the camera stays fixed

A rotating detection camera loses people at the edge of the frame and changes the meaning of every pixel position while it moves. The C270 therefore stays fixed above the display. Only the LED mount moves. A one-time calibration maps the selected face's horizontal image position to a safe servo angle.

## Reliability and fail-safe behaviour

- QNX has one bounded local inference loop; it does not depend on cloud access.
- Any missing frame, invalid model output, low confidence, or lost tracking results in an inactive decision.
- The Arduino receives a frequent heartbeat. If it expires, it turns the LED off and parks the servo at its neutral angle.
- The LED defaults off at boot/reset.
- A physical kill switch removes power from the LED and servos regardless of software state.
- The laptop can be absent without making the physical response unsafe.

See [QNX deployment](qnx-deployment.md) for the installed AI runtime, model contract, and demo proof checklist. See [wiring diagram](wiring-diagram.md) for physical connections.

