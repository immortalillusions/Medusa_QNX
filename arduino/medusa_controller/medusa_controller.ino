#include <Servo.h>

// Serial protocol from QNX: N,<yaw_x10>,<pitch_x10>,<led 0|1>,<sequence>\n
// with angles in tenths of a degree (925 = 92.5), each within its axis range (PAN_*/TILT_*).
// Whole-degree M,<yaw>,<pitch>,<led>,<sequence>\n and the old M,<yaw>,<led>,<sequence>\n
// (pitch = tilt home) are still accepted.
// Off means off: while disarmed both servos are DETACHED (no pulses at all), so
// they cannot move or twitch. Disarming (button, a 300 ms gap in valid commands,
// the kill gate opening, boot) stops the axes where they are, detaches them and
// turns the LED off; the position estimate is kept. Arming (button press, only
// while the kill gate is closed) re-attaches both servos at that estimate, so
// nothing jumps; only then do commands move them.
// Kill gate: KILL 1 of the kill plug feeds the servo +5 V rail, and a 3 x 10k
// divider from that rail feeds A2 (about 1.67 V with the tab in, 0 V with it out).
// An open gate forces a disarm and refuses to arm.
// Status back to QNX about every 100 ms on the same UART (UNO Q D1 -> Pi RX):
// S,<armed 0|1>,<pan_x10>,<tilt_x10>,<sequence>,<gate 1 closed|0 open>\n with the
// eased ACTUAL axis positions in tenths of a degree. The Zephyr boot banner also
// appears on D1 at reset; the Pi ignores any line that does not match.

const byte SERVO_PIN = 9;
const byte TILT_PIN = 10;
const byte LED_PIN = 5;
const byte ARM_BUTTON_PIN = 2;
const byte ARM_STATUS_PIN = LED_BUILTIN;
const byte KILL_SENSE_PIN = A2;
// Pan rests at 0 deg, which faces forward on this turret (set on the bench 22 Nov),
// so it can only track to one side of forward (0..70). Tilt rests at 90.
const int PAN_MIN = 0, PAN_MAX = 70, PAN_HOME = 0;
const int TILT_MIN = 55, TILT_MAX = 125, TILT_HOME = 90;
const unsigned long HEARTBEAT_TIMEOUT_MS = 300;
const unsigned long BUTTON_DEBOUNCE_MS = 40;
// analogRead on the arduino:zephyr core is 10-bit (0..1023) unless
// analogReadResolution() is called, over the MCU's 3.3 V: 1.67 V reads ~517.
// Closed needs > ~1.0 V (310) for 200 ms; open needs < ~0.8 V (248) for 20 ms.
// Readings in between keep the current state (hysteresis); a failed read (< 0)
// counts as open.
const int GATE_CLOSED_ABOVE = 310;
const int GATE_OPEN_BELOW = 248;
const unsigned long GATE_CLOSE_MS = 200;
const unsigned long GATE_OPEN_MS = 20;
const unsigned long GATE_SAMPLE_MS = 5;
// Pan and tilt moves are eased: a speed cap plus an acceleration limit stop
// the turret overshooting and bouncing on sharp aims. Pulses use microseconds.
// The Pi models this motion to know where the camera points: keep MAX_SPEED,
// MAX_ACCEL and MOTION_STEP_MS equal to FIRMWARE_* in medusa/trigger.py.
const float MAX_SPEED = 180.0;   // degrees per second
const float MAX_ACCEL = 900.0;   // degrees per second squared
const unsigned long MOTION_STEP_MS = 5;
const unsigned long STATUS_PERIOD_MS = 100;

struct Axis {
  Servo servo;
  float position;  // estimate: kept while detached (an unpowered servo has no feedback)
  float velocity;
  float target;
  int minimum, maximum, home;
  byte pin;
};

Axis pan = {Servo(), PAN_HOME, 0, PAN_HOME, PAN_MIN, PAN_MAX, PAN_HOME, SERVO_PIN};
Axis tilt = {Servo(), TILT_HOME, 0, TILT_HOME, TILT_MIN, TILT_MAX, TILT_HOME, TILT_PIN};
char line[40];
byte lineLength = 0;
unsigned long lastCommandMs = 0;
unsigned long buttonChangedMs = 0;
bool armed = false;
bool lastButtonReading = HIGH;
bool stableButtonReading = HIGH;
unsigned long lastMotionMs = 0;
unsigned long lastStatusMs = 0;
unsigned long statusSequence = 0;
bool gateClosed = false;  // open until A2 proves otherwise
unsigned long gateDisagreeSinceMs = 0;
bool gateDisagreeing = false;
unsigned long lastGateSampleMs = 0;

int degreesToMicroseconds(float degrees) {
  return MIN_PULSE_WIDTH + (int)(degrees * (MAX_PULSE_WIDTH - MIN_PULSE_WIDTH) / 180.0 + 0.5);
}

void stepAxis(Axis& axis, float dt) {
  // Detached: no pulses, no motion. (The zephyr Servo library also must not be
  // written while detached: it would index its servo table out of bounds.)
  if (!axis.servo.attached()) return;
  float error = axis.target - axis.position;
  // Fastest speed from which the axis can still stop at the target.
  float wanted = min(MAX_SPEED, sqrtf(2 * MAX_ACCEL * fabsf(error)));
  if (error < 0) wanted = -wanted;
  float limit = MAX_ACCEL * dt;
  axis.velocity += constrain(wanted - axis.velocity, -limit, limit);
  axis.position = constrain(axis.position + axis.velocity * dt, (float)axis.minimum, (float)axis.maximum);
  if (fabsf(error) < 0.05 && fabsf(axis.velocity) < 2) {
    axis.position = axis.target;
    axis.velocity = 0;
  }
  axis.servo.writeMicroseconds(degreesToMicroseconds(axis.position));
}

void detachAxis(Axis& axis) {
  // Stop where it is: no easing anywhere, the estimate stays at the last position.
  axis.target = axis.position;
  axis.velocity = 0;
  if (axis.servo.attached()) axis.servo.detach();
  digitalWrite(axis.pin, LOW);  // A detach mid-pulse could leave the pin high.
}

void attachAxis(Axis& axis) {
  axis.target = axis.position;  // Hold here until a command says otherwise.
  axis.velocity = 0;
  if (!axis.servo.attached()) axis.servo.attach(axis.pin);
  axis.servo.writeMicroseconds(degreesToMicroseconds(axis.position));
}

void setArmed(bool shouldArm) {
  if (!gateClosed) shouldArm = false;  // No servo power: refuse to arm.
  // No live QNX heartbeat: refuse too, rather than attach for one loop and drop out.
  if (millis() - lastCommandMs > HEARTBEAT_TIMEOUT_MS) shouldArm = false;
  armed = shouldArm;
  // The UNO Q's red LED is active-low: LOW lights it. Lit means armed.
  digitalWrite(ARM_STATUS_PIN, armed ? LOW : HIGH);
  if (armed) {
    attachAxis(pan);
    attachAxis(tilt);
  } else {
    detachAxis(pan);
    detachAxis(tilt);
    digitalWrite(LED_PIN, LOW);
  }
}

void updateKillGate() {
  if (millis() - lastGateSampleMs < GATE_SAMPLE_MS) return;
  lastGateSampleMs = millis();
  int reading = analogRead(KILL_SENSE_PIN);
  bool readsClosed = reading > GATE_CLOSED_ABOVE;
  bool readsOpen = reading < GATE_OPEN_BELOW;  // includes a failed read (< 0)
  bool disagrees = gateClosed ? readsOpen : readsClosed;
  if (!disagrees) {
    gateDisagreeing = false;  // Back on the current side, or in the hysteresis band.
    return;
  }
  if (!gateDisagreeing) {
    gateDisagreeing = true;
    gateDisagreeSinceMs = millis();
  }
  if (millis() - gateDisagreeSinceMs >= (gateClosed ? GATE_OPEN_MS : GATE_CLOSE_MS)) {
    gateClosed = !gateClosed;
    gateDisagreeing = false;
    if (!gateClosed && armed) setArmed(false);  // Closing never arms: that takes the button.
  }
}

void updateArmButton() {
  bool reading = digitalRead(ARM_BUTTON_PIN);
  if (reading != lastButtonReading) buttonChangedMs = millis();
  if (millis() - buttonChangedMs >= BUTTON_DEBOUNCE_MS && reading != stableButtonReading) {
    stableButtonReading = reading;
    // INPUT_PULLUP means a pressed button reads LOW.
    if (stableButtonReading == LOW) setArmed(!armed);
  }
  lastButtonReading = reading;
}

bool inRange(const Axis& axis, int value, int scale) {
  return value >= axis.minimum * scale && value <= axis.maximum * scale;
}

void applyCommand(char* command) {
  int yaw, pitch = TILT_HOME, led;
  int scale = 1;  // 10 for N lines (tenths of a degree)
  unsigned long sequence;
  // Count commas to pick the format (4 = 5-field N or M line, 3 = old 4-field M line),
  // so the parse never relies on %n, whose support varies between C libraries.
  int commas = 0;
  for (char* c = command; *c; c++) if (*c == ',') commas++;
  if (commas == 4 && command[0] == 'N') {
    if (sscanf(command, "N,%d,%d,%d,%lu", &yaw, &pitch, &led, &sequence) != 4) return;
    scale = 10;
  } else if (commas == 4) {
    if (sscanf(command, "M,%d,%d,%d,%lu", &yaw, &pitch, &led, &sequence) != 4) return;
  } else if (commas == 3) {
    if (sscanf(command, "M,%d,%d,%lu", &yaw, &led, &sequence) != 3) return;
  } else {
    return;
  }
  if ((led != 0 && led != 1) || !inRange(pan, yaw, scale) || !inRange(tilt, pitch, scale)) return;

  // Keep receiving heartbeats while disarmed, but never move or light up.
  if (armed) {
    pan.target = (float)yaw / scale;
    tilt.target = (float)pitch / scale;
    digitalWrite(LED_PIN, led ? HIGH : LOW);
  }
  lastCommandMs = millis();
}

long tenths(float degrees) {
  return (long)(degrees * 10 + (degrees < 0 ? -0.5 : 0.5));
}

void sendStatus() {
  char status[40];
  snprintf(status, sizeof(status), "S,%d,%ld,%ld,%lu,%d\n", armed ? 1 : 0,
           tenths(pan.position), tenths(tilt.position), ++statusSequence, gateClosed ? 1 : 0);
  Serial1.print(status);
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(ARM_BUTTON_PIN, INPUT_PULLUP);
  pinMode(ARM_STATUS_PIN, OUTPUT);
  pinMode(SERVO_PIN, OUTPUT);  // Detached servo pins are held low.
  pinMode(TILT_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  setArmed(false);  // Boot disarmed and detached: the servos get no pulses at all.
  Serial1.begin(115200);
}

void loop() {
  updateKillGate();
  updateArmButton();
  while (Serial1.available()) {
    char incoming = (char)Serial1.read();
    if (incoming == '\n') {
      line[lineLength] = '\0';
      applyCommand(line);
      lineLength = 0;
    } else if (incoming != '\r' && lineLength < sizeof(line) - 1) {
      line[lineLength++] = incoming;
    } else {
      lineLength = 0;
    }
  }

  if (armed && millis() - lastCommandMs > HEARTBEAT_TIMEOUT_MS) {
    // A lost QNX heartbeat requires a physical button press before movement
    // can resume; this prevents a surprise re-arm after a restart.
    setArmed(false);
  }

  if (millis() - lastMotionMs >= MOTION_STEP_MS) {
    lastMotionMs = millis();
    stepAxis(pan, MOTION_STEP_MS / 1000.0);
    stepAxis(tilt, MOTION_STEP_MS / 1000.0);
  }

  if (millis() - lastStatusMs >= STATUS_PERIOD_MS) {
    lastStatusMs = millis();
    sendStatus();
  }
}

