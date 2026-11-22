#include <Servo.h>

// Serial protocol from QNX: N,<yaw_x10>,<pitch_x10>,<led 0|1>,<sequence>\n
// with angles in tenths of a degree (925 = 92.5), within SERVO_MIN*10..SERVO_MAX*10.
// Whole-degree M,<yaw>,<pitch>,<led>,<sequence>\n and the old M,<yaw>,<led>,<sequence>\n
// (pitch = home) are still accepted, within SERVO_MIN..SERVO_MAX.
// If valid commands stop for 300 ms, disarm, centre both servos, LED off.
// Status back to QNX about every 100 ms on the same UART (UNO Q D1 -> Pi RX):
// S,<armed 0|1>,<pan_x10>,<tilt_x10>,<sequence>\n with the eased ACTUAL axis
// positions in tenths of a degree. The Zephyr boot banner also appears on D1
// at reset; the Pi ignores any line that does not match.

const byte SERVO_PIN = 9;
const byte TILT_PIN = 10;
const byte LED_PIN = 5;
const byte ARM_BUTTON_PIN = 2;
const byte ARM_STATUS_PIN = LED_BUILTIN;
const int SERVO_MIN = 55;
const int SERVO_MAX = 125;
const int SERVO_HOME = 90;
const unsigned long HEARTBEAT_TIMEOUT_MS = 300;
const unsigned long BUTTON_DEBOUNCE_MS = 40;
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
  float position;
  float velocity;
  float target;
};

Axis pan = {Servo(), SERVO_HOME, 0, SERVO_HOME};
Axis tilt = {Servo(), SERVO_HOME, 0, SERVO_HOME};
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

int degreesToMicroseconds(float degrees) {
  return MIN_PULSE_WIDTH + (int)(degrees * (MAX_PULSE_WIDTH - MIN_PULSE_WIDTH) / 180.0 + 0.5);
}

void stepAxis(Axis& axis, float dt) {
  float error = axis.target - axis.position;
  // Fastest speed from which the axis can still stop at the target.
  float wanted = min(MAX_SPEED, sqrtf(2 * MAX_ACCEL * fabsf(error)));
  if (error < 0) wanted = -wanted;
  float limit = MAX_ACCEL * dt;
  axis.velocity += constrain(wanted - axis.velocity, -limit, limit);
  axis.position = constrain(axis.position + axis.velocity * dt, (float)SERVO_MIN, (float)SERVO_MAX);
  if (fabsf(error) < 0.05 && fabsf(axis.velocity) < 2) {
    axis.position = axis.target;
    axis.velocity = 0;
  }
  axis.servo.writeMicroseconds(degreesToMicroseconds(axis.position));
}

void parkSafely() {
  pan.target = SERVO_HOME;  // both eased back to centre by stepAxis()
  tilt.target = SERVO_HOME;
  digitalWrite(LED_PIN, LOW);
}

void setArmed(bool shouldArm) {
  armed = shouldArm;
  // The UNO Q's red LED is active-low: LOW lights it. Lit means armed.
  digitalWrite(ARM_STATUS_PIN, armed ? LOW : HIGH);
  if (!armed) parkSafely();
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

bool inRange(int value, int scale) {
  return value >= SERVO_MIN * scale && value <= SERVO_MAX * scale;
}

void applyCommand(char* command) {
  int yaw, pitch = SERVO_HOME, led;
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
  if ((led != 0 && led != 1) || !inRange(yaw, scale) || !inRange(pitch, scale)) return;

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
  snprintf(status, sizeof(status), "S,%d,%ld,%ld,%lu\n", armed ? 1 : 0,
           tenths(pan.position), tenths(tilt.position), ++statusSequence);
  Serial1.print(status);
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(ARM_BUTTON_PIN, INPUT_PULLUP);
  pinMode(ARM_STATUS_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  pan.servo.attach(SERVO_PIN);
  tilt.servo.attach(TILT_PIN);
  pan.servo.writeMicroseconds(degreesToMicroseconds(SERVO_HOME));
  tilt.servo.writeMicroseconds(degreesToMicroseconds(SERVO_HOME));
  setArmed(false);
  Serial1.begin(115200);
}

void loop() {
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

