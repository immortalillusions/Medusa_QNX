# Medusa Shield: what we'd fabricate next

Today the turret's safety electronics live on a breadboard. The Medusa Shield puts the same
circuit on one UNO-format board (about 68.6 × 53.3 mm) that plugs straight onto the Arduino
UNO Q, so nothing on the turret depends on a loose jumper. The current firmware runs on it
unchanged: same pins, same behaviour.

## Pin map (from the bench netlist)

| UNO Q pin | Net | Goes to |
| --- | --- | --- |
| D9 | PAN_SIG | Pan servo header J2, 10 kΩ pull-down |
| D10 | TILT_SIG | Tilt servo header J3, 10 kΩ pull-down |
| D5 | LIGHT | 1 kΩ to the light switch transistor Q1, 10 kΩ pull-down |
| D2 | ARM | Arm button SW2 to GND (INPUT_PULLUP) |
| A2 | KILL_SENSE | 10k/10k/10k divider from the servo 5 V rail: ≈1.67 V with the kill switch closed, 0 V open. The firmware reads it to disarm at once and to latch a fault |
| D0 (RX) | PI_TX | 1 kΩ from Pi header pin 8 |
| D1 (TX) | PI_RX | 1 kΩ to Pi header pin 10 |
| GND | GND | Pi pin 6, servo supply ground |

## Blocks

- **Servo power in, kill pole A, bulk cap.** J1 screw terminal for a dedicated 5 V servo supply
  (today: a second UNO Q's 5 V pin), polarity diode, kill switch pole A, 470 µF next to the
  servo headers. Servo current never runs through the main board.
- **Servo headers ×2.** Pull-downs hold both servos still while the UNO Q boots or resets. The
  firmware clamps pan to 15–165° and tilt to 0–70° and eases every move.
- **Light switch, kill pole B.** Q1 switches the puck light from D5; kill pole B breaks the
  light circuit in hardware. (On the demo build the light is simply left on.)
- **Arm button and Pi UART header.** SW2 arms and disarms; after any fault the firmware only
  re-arms on this physical button. The Pi shares only ground and the two data lines.

Layout rule: the high-current servo path (J1 → diode → kill pole → C1 → J2/J3) stays short
along one edge, away from the UART and sense lines; the kill switch sits on the board edge
within reach.

## Bill of materials (v1, rough small-batch USD, not quotes)

| Ref | Part | Qty | Cost |
| --- | --- | --- | --- |
| PCB | 2-layer UNO-format shield | 1 | ≈ $1–2 |
| — | Male pin headers for the UNO rows | 1 set | ≈ $1.00 |
| J1 | 2-pin 5.08 mm screw terminal, servo 5 V in | 1 | ≈ $0.40 |
| D1 | SS34 Schottky, polarity protection | 1 | ≈ $0.15 |
| SW1 | Two-pole kill switch (DPST toggle or 2×2 header + double shunt) | 1 | ≤ $1.50 |
| C1 | 470 µF, ≥ 10 V | 1 | ≈ $0.30 |
| J2, J3 | 1×3 servo headers | 2 | ≈ $0.20 |
| Q1 | 2N2222A | 1 | ≈ $0.10 |
| R1, R9, R10 | 1 kΩ | 3 | ≈ $0.05 |
| R2, R4–R8 | 10 kΩ | 6 | ≈ $0.10 |
| J4 | 2-pin JST-PH, light leads | 1 | ≈ $0.30 |
| SW2 | 6 mm tactile button, ARM | 1 | ≈ $0.10 |
| J5 | 1×3 header to the Pi (pins 8, 10, 6) | 1 | ≈ $0.20 |
| | **Per board** | | **≈ $4–6** |

## v2 changes

- Logic-level N-MOSFET (e.g. AO3400A) in place of the 2N2222A, gate resistor and 100 kΩ pull-down.
- Onboard 5 V: USB-C PD input and a buck regulator for the servos, so the second UNO Q is no
  longer a power source; sized after measuring both servos at stall.
- Resettable polyfuse on the servo input; ideal-diode (P-MOSFET) polarity protection.
- Status LEDs: servo 5 V present (after the kill switch), armed, light on.

The full illustrated version is the "Medusa Shield" page our team built during the hackathon.
