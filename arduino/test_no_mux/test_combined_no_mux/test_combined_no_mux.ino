// =============================================================
// Test: All Sensors Combined — Direct ADC (No MUX)
// Reads flex, FSR, IMU, and touch without a multiplexer.
// Each analog sensor gets its own ADC pin.
// Outputs JSON matching backend format at 50 Hz.
//
// Flex sensors: ~25.5kΩ–26.6kΩ range (narrow, 1.1kΩ change).
// Measured: 1.100V straight, 1.038V at 90° bend (10kΩ divider).
// Voltage drops with bend. 64x oversampling + auto-calibration.
// Flex output remapped to 800–3800 for ML pipeline.
//
// Pin assignments:
//   Flex:  GPIO 36, 39, 34, 35, 32  (ADC1) + [10kΩ → GND]
//   FSR:   GPIO 33, 25, 26          (ADC1 + ADC2) + [47kΩ → GND]
//   IMU:   GPIO 21 (SDA), 22 (SCL)  (I2C)
//   Touch: GPIO 4, 0, 2, 15, 13     (T0-T4)
//
// NOTE: GPIO25/26 are ADC2 — won't work with WiFi active.
//       For serial-only testing this is fine.
// =============================================================

#include <Wire.h>
#include <math.h>

// ---- Configuration ----
const int HAND_ID = 1;  // 1 = right, 2 = left
const uint8_t MPU6050_ADDR = 0x68;  // 0x68 right, 0x69 left

// ---- Direct ADC Pins (No MUX) ----
const int FLEX_PINS[] = {36, 39, 34, 35, 32};
const int FSR_PINS[]  = {33, 25, 26};

// ---- Flex oversampling & calibration ----
const int FLEX_OVERSAMPLE = 64;          // noise ÷ √64 ≈ ±2.5 counts
const int ML_FLEX_MIN = 800;
const int ML_FLEX_MAX = 3800;
long flexCalStraight[5] = {0};  // ADC readings when fingers straight
long flexCalCurl[5] = {0};      // ADC readings when fingers bent

// ---- Touch Pins ----
const int TOUCH_PINS[] = {4, 0, 2, 15, 13};
const int TOUCH_THRESHOLD = 40;

// ---- MPU6050 Registers ----
const uint8_t REG_PWR_MGMT_1   = 0x6B;
const uint8_t REG_ACCEL_CONFIG  = 0x1C;
const uint8_t REG_GYRO_CONFIG   = 0x1B;
const uint8_t REG_ACCEL_XOUT_H  = 0x3B;

// ---- Scale Factors ----
const float ACCEL_SCALE = 9.80665 / 8192.0;  // +/-4g to m/s^2
const float GYRO_SCALE  = 1.0 / 65.5;         // +/-500 deg/s

// ---- Timing ----
const unsigned long SAMPLE_INTERVAL_MS = 20;  // 50 Hz
unsigned long lastSampleTime = 0;

// ---- Sensor Data ----
int flexValues[5];
int fsrValues[3];
uint8_t contactBitmask;
float accel[3];
float gyro[3];
float quaternion[4];

// Complementary filter state
float roll = 0, pitch = 0, yaw = 0;
const float ALPHA = 0.98;

// ---- Helper Functions ----

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

// Read a single flex pin with 64x oversampling
long readFlexOversampled(int pin) {
  long sum = 0;
  for (int i = 0; i < FLEX_OVERSAMPLE; i++) {
    sum += analogRead(pin);
  }
  return sum / FLEX_OVERSAMPLE;
}

// Collect N oversampled readings for one pin and return average
long collectFlexCalSamples(int pin, int numSamples) {
  long total = 0;
  for (int i = 0; i < numSamples; i++) {
    total += readFlexOversampled(pin);
    delay(20);
  }
  return total / numSamples;
}

void readFlexSensors() {
  for (int i = 0; i < 5; i++) {
    long raw = readFlexOversampled(FLEX_PINS[i]);
    // Map directly: calStraight→800, calCurl→3800
    long mapped = map(raw, flexCalStraight[i], flexCalCurl[i], ML_FLEX_MIN, ML_FLEX_MAX);
    flexValues[i] = constrain(mapped, ML_FLEX_MIN, ML_FLEX_MAX);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Flex sensor pins
  for (int i = 0; i < 5; i++) {
    pinMode(FLEX_PINS[i], INPUT);
  }

  // FSR sensor pins
  for (int i = 0; i < 3; i++) {
    pinMode(FSR_PINS[i], INPUT);
  }

  // I2C for MPU6050
  Wire.begin(21, 22);
  Wire.setClock(400000);

  // Initialize MPU6050
  Wire.beginTransmission(MPU6050_ADDR);
  uint8_t error = Wire.endTransmission();
  if (error != 0) {
    Serial.println("{\"error\": \"MPU6050 not found — check wiring\"}");
    while (1) delay(1000);
  }

  writeRegister(REG_PWR_MGMT_1, 0x00);   // Wake up
  delay(100);
  writeRegister(REG_ACCEL_CONFIG, 0x08);  // +/-4g
  writeRegister(REG_GYRO_CONFIG, 0x08);   // +/-500 deg/s

  Serial.println("{\"status\": \"initialized_no_mux\", \"hand_id\": " + String(HAND_ID) + "}");
  Serial.println();
  Serial.println("Pin map: Flex=[36,39,34,35,32]+10k FSR=[33,25,26]+47k");
  Serial.println("IMU=I2C(21,22) Touch=[4,0,2,15,13]");
  Serial.println("Flex: 64x oversampling, per-finger calibration");
  Serial.println();

  // ---- ADC Warm-up ----
  // ESP32 ADC drifts ~50-60 counts in first 10s after boot.
  // Discard readings until it stabilises.
  Serial.println(">>> Warming up ADC (10 seconds)...");
  for (int w = 0; w < 500; w++) {
    readFlexOversampled(FLEX_PINS[0]);  // discard — just warming up the ADC
    delay(20);
    if (w % 50 == 0) Serial.print(".");
  }
  Serial.println(" Done.");
  Serial.println();

  // ---- Flex Auto-Calibration ----
  const char* fingerNames[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};

  Serial.println(">>> FLEX CALIBRATION 1/2: Hold ALL fingers STRAIGHT.");
  Serial.println(">>> Collecting in 3 seconds...");
  delay(3000);
  Serial.print(">>> Sampling... ");
  for (int i = 0; i < 5; i++) {
    flexCalStraight[i] = collectFlexCalSamples(FLEX_PINS[i], 50);
  }
  Serial.println("Done.");
  for (int i = 0; i < 5; i++) {
    Serial.print(">>>   ");
    Serial.print(fingerNames[i]);
    Serial.print(" straight = ");
    Serial.println(flexCalStraight[i]);
  }

  Serial.println(">>> FLEX CALIBRATION 2/2: BEND ALL fingers to 90 degrees.");
  Serial.println(">>> Collecting in 3 seconds...");
  delay(3000);
  Serial.print(">>> Sampling... ");
  for (int i = 0; i < 5; i++) {
    flexCalCurl[i] = collectFlexCalSamples(FLEX_PINS[i], 50);
  }
  Serial.println("Done.");
  for (int i = 0; i < 5; i++) {
    Serial.print(">>>   ");
    Serial.print(fingerNames[i]);
    Serial.print(" bent = ");
    Serial.println(flexCalCurl[i]);
  }

  // Validate per-finger calibration (no swapping)
  for (int i = 0; i < 5; i++) {
    long span = abs(flexCalStraight[i] - flexCalCurl[i]);
    if (span < 10) {
      Serial.print(">>> WARNING: ");
      Serial.print(fingerNames[i]);
      Serial.println(" span < 10 counts!");
    }
  }
  Serial.println(">>> Flex calibration complete. Output mapped to 800-3800.");
  Serial.println();
}

void readFsrSensors() {
  for (int i = 0; i < 3; i++) {
    fsrValues[i] = analogRead(FSR_PINS[i]);
  }
}

void readTouchSensors() {
  contactBitmask = 0;
  for (int i = 0; i < 5; i++) {
    int rawValue = touchRead(TOUCH_PINS[i]);
    if (rawValue < TOUCH_THRESHOLD) {
      contactBitmask |= (1 << i);
    }
  }
}

void readIMU() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, (uint8_t)14);

  int16_t ax_raw = (Wire.read() << 8) | Wire.read();
  int16_t ay_raw = (Wire.read() << 8) | Wire.read();
  int16_t az_raw = (Wire.read() << 8) | Wire.read();
  int16_t temp_raw = (Wire.read() << 8) | Wire.read();  // skip temp
  int16_t gx_raw = (Wire.read() << 8) | Wire.read();
  int16_t gy_raw = (Wire.read() << 8) | Wire.read();
  int16_t gz_raw = (Wire.read() << 8) | Wire.read();

  accel[0] = ax_raw * ACCEL_SCALE;
  accel[1] = ay_raw * ACCEL_SCALE;
  accel[2] = az_raw * ACCEL_SCALE;
  gyro[0]  = gx_raw * GYRO_SCALE;
  gyro[1]  = gy_raw * GYRO_SCALE;
  gyro[2]  = gz_raw * GYRO_SCALE;

  // Complementary filter for orientation
  float dt = SAMPLE_INTERVAL_MS / 1000.0;
  float accelRoll  = atan2(accel[1], accel[2]) * 180.0 / M_PI;
  float accelPitch = atan2(-accel[0], sqrt(accel[1]*accel[1] + accel[2]*accel[2])) * 180.0 / M_PI;

  roll  = ALPHA * (roll  + gyro[0] * dt) + (1.0 - ALPHA) * accelRoll;
  pitch = ALPHA * (pitch + gyro[1] * dt) + (1.0 - ALPHA) * accelPitch;
  yaw   = yaw + gyro[2] * dt;

  if (pitch > 85.0) pitch = 85.0;
  if (pitch < -85.0) pitch = -85.0;

  // Euler to quaternion
  float cr = cos(roll  * M_PI / 360.0);
  float sr = sin(roll  * M_PI / 360.0);
  float cp = cos(pitch * M_PI / 360.0);
  float sp = sin(pitch * M_PI / 360.0);
  float cy = cos(yaw   * M_PI / 360.0);
  float sy = sin(yaw   * M_PI / 360.0);

  quaternion[0] = cr * cp * cy + sr * sp * sy;  // w
  quaternion[1] = sr * cp * cy - cr * sp * sy;  // x
  quaternion[2] = cr * sp * cy + sr * cp * sy;  // y
  quaternion[3] = cr * cp * sy - sr * sp * cy;  // z
}

void sendJSON() {
  String json = "{";
  json += "\"timestamp_ms\":" + String(millis()) + ",";
  json += "\"hand_id\":" + String(HAND_ID) + ",";

  json += "\"flex\":[";
  for (int i = 0; i < 5; i++) {
    json += String(flexValues[i]);
    if (i < 4) json += ",";
  }
  json += "],";

  json += "\"fsr\":[";
  for (int i = 0; i < 3; i++) {
    json += String(fsrValues[i]);
    if (i < 2) json += ",";
  }
  json += "],";

  json += "\"contact_bitmask\":" + String(contactBitmask) + ",";

  json += "\"accel\":[";
  for (int i = 0; i < 3; i++) {
    json += String(accel[i], 2);
    if (i < 2) json += ",";
  }
  json += "],";

  json += "\"gyro\":[";
  for (int i = 0; i < 3; i++) {
    json += String(gyro[i], 1);
    if (i < 2) json += ",";
  }
  json += "],";

  json += "\"quaternion\":[";
  for (int i = 0; i < 4; i++) {
    json += String(quaternion[i], 4);
    if (i < 3) json += ",";
  }
  json += "]";

  json += "}";
  Serial.println(json);
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = now;

    readFlexSensors();
    readFsrSensors();
    readTouchSensors();
    readIMU();
    sendJSON();
  }
}
