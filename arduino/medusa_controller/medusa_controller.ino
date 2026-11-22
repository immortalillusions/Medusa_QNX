#include <Servo.h>

// Serial protocol from QNX: M,<yaw 0..180>,<led 0|1>,<sequence>\n
// If valid commands stop for 300 ms, centre the servo and turn off the LED.

const byte SERVO_PIN = 9;
const byte LED_PIN = 5;
const byte ARM_BUTTON_PIN = 2;
const byte ARM_STATUS_PIN = LED_BUILTIN;
const int SERVO_MIN = 35;
const int SERVO_MAX = 145;
const int SERVO_HOME = 90;
const unsigned long HEARTBEAT_TIMEOUT_MS = 300;
const unsigned long BUTTON_DEBOUNCE_MS = 40;

Servo panServo;
char line[40];
byte lineLength = 0;
unsigned long lastCommandMs = 0;
unsigned long buttonChangedMs = 0;
bool armed = false;
bool lastButtonReading = HIGH;
bool stableButtonReading = HIGH;

void parkSafely() {
  panServo.write(SERVO_HOME);
  digitalWrite(LED_PIN, LOW);
}

void setArmed(bool shouldArm) {
  armed = shouldArm;
  digitalWrite(ARM_STATUS_PIN, armed ? HIGH : LOW);
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

void applyCommand(char* command) {
  int yaw, led;
  unsigned long sequence;
  if (sscanf(command, "M,%d,%d,%lu", &yaw, &led, &sequence) != 3) return;
  if ((led != 0 && led != 1) || yaw < SERVO_MIN || yaw > SERVO_MAX) return;

  // Keep receiving heartbeats while disarmed, but never move or light up.
  if (armed) {
    panServo.write(yaw);
    digitalWrite(LED_PIN, led ? HIGH : LOW);
  }
  lastCommandMs = millis();
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(ARM_BUTTON_PIN, INPUT_PULLUP);
  pinMode(ARM_STATUS_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  panServo.attach(SERVO_PIN);
  setArmed(false);
  Serial.begin(115200);
}

void loop() {
  updateArmButton();
  while (Serial.available()) {
    char incoming = (char)Serial.read();
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
}

