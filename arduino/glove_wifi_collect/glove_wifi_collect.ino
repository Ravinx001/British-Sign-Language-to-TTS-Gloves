// ================================================================
// BSL Sign Language Gloves — Production WiFi Collection Firmware
//
// Flash this sketch to BOTH ESP32 boards.
// Change HAND_ID and MPU_ADDR before flashing each one:
//   Right glove:  HAND_ID = 1, MPU_ADDR = 0x68  (AD0 → GND)
//   Left  glove:  HAND_ID = 2, MPU_ADDR = 0x69  (AD0 → 3.3V)
//
// Required libraries (install via Arduino Library Manager):
//   • arduinoWebSockets by Markus Sattler  (links2004/arduinoWebSockets)
//   • ArduinoJson       by Benoit Blanchon (v6 — tested 6.21+)
//   • ESP32 board package by Espressif (≥2.0.0) — adds Wire, WiFi
//
// Pin wiring matches docs/new-docs/guide-hardware-build.md:
//   MUX SIG  → GPIO 36 (ADC1 only — safe with WiFi)
//   MUX S0-S3→ GPIO 16, 17, 18, 19
//   MUX EN   → GPIO 5  (pull LOW to enable)
//   IMU SDA  → GPIO 21, SCL → GPIO 22 (I²C)
// ================================================================

#include <WiFi.h>
#include <Wire.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <time.h>

// ================================================================
// USER CONFIGURATION — edit these before flashing
// ================================================================
#define HAND_ID     2        // 1 = right glove,  2 = left glove
#define MPU_ADDR    0x69     // 0x68 = right (AD0 low), 0x69 = left (AD0 to 3.3V)

const char* WIFI_SSID  = "A14";
const char* WIFI_PASS  = "458173*/";
const char* WS_HOST    = "10.66.209.201";  // << laptop LAN IP (run ipconfig)
const uint16_t WS_PORT = 8000;
const char* WS_PATH    = "/ws/collect";

// ================================================================
// Pin map — all ADC1 channels (ADC2 doesn't work with WiFi active)
// ================================================================
const int MUX_SIG = 36;   // ADC1 — sole analog input
const int MUX_S0  = 16;   // MUX select bit 0
const int MUX_S1  = 17;   // MUX select bit 1
const int MUX_S2  = 18;   // MUX select bit 2
const int MUX_S3  = 19;   // MUX select bit 3
const int MUX_EN  = 5;    // MUX enable (active LOW)

const int I2C_SDA = 21;
const int I2C_SCL = 22;

// MUX channel assignments (matches guide-hardware-build.md)
//   0–4 : Flex sensors (thumb → pinky)
//   5–7 : FSR sensors  (thumb tip, index tip, pinky tip)
const int CH_FLEX[5] = {0, 1, 2, 3, 4};
const int CH_FSR[3]  = {5, 6, 7};

// ================================================================
// Timing
// ================================================================
const unsigned long SAMPLE_MS = 20;   // 50 Hz

// ================================================================
// MPU6050 register constants
// ================================================================
const uint8_t MPU_REG_PWR_MGMT_1   = 0x6B;
const uint8_t MPU_REG_ACCEL_CONFIG  = 0x1C;
const uint8_t MPU_REG_GYRO_CONFIG   = 0x1B;
const uint8_t MPU_REG_ACCEL_XOUT_H  = 0x3B;

// Scale factors: ±4g and ±500°/s
const float ACCEL_SCALE = 9.80665f / 8192.0f;
const float GYRO_SCALE  = 1.0f / 65.5f;

// Complementary filter weight (higher = trust gyro more)
const float CF_ALPHA = 0.98f;

// ================================================================
// Sensor data buffers
// ================================================================
int   flexADC[5];       // raw ADC (0–4095)
int   fsrADC[3];        // raw ADC (0–4095)
float accel[3];         // m/s²    [x, y, z]
float gyro[3];          // °/s     [x, y, z]
float quat[4];          // unit quaternion [w, x, y, z]

// Complementary filter state
float rollDeg  = 0.0f;
float pitchDeg = 0.0f;
float yawDeg   = 0.0f;
unsigned long lastIMUus = 0;

// ================================================================
// WebSocket
// ================================================================
WebSocketsClient wsClient;
volatile bool wsConnected = false;
unsigned long lastSampleMs = 0;

// ================================================================
// Helpers — MUX
// ================================================================
void muxSelect(int ch) {
  digitalWrite(MUX_S0, (ch >> 0) & 1);
  digitalWrite(MUX_S1, (ch >> 1) & 1);
  digitalWrite(MUX_S2, (ch >> 2) & 1);
  digitalWrite(MUX_S3, (ch >> 3) & 1);
  delayMicroseconds(80);  // MUX settling time
}

// Average of 4 readings on the selected MUX channel
int readMux(int ch) {
  muxSelect(ch);
  analogRead(MUX_SIG);  // discard first sample after channel switch
  long sum = 0;
  for (int i = 0; i < 4; i++) sum += analogRead(MUX_SIG);
  return (int)(sum / 4);
}

// ================================================================
// Helpers — MPU6050
// ================================================================
void mpuWriteReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void mpuInit() {
  Wire.begin(I2C_SDA, I2C_SCL, 400000UL);
  mpuWriteReg(MPU_REG_PWR_MGMT_1,  0x00);  // wake up, disable sleep
  delay(100);
  mpuWriteReg(MPU_REG_ACCEL_CONFIG, 0x08);  // ±4g full scale
  mpuWriteReg(MPU_REG_GYRO_CONFIG,  0x08);  // ±500°/s full scale
  lastIMUus = micros();
}

void readMPU() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(MPU_REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (size_t)14);
  if (Wire.available() < 14) return;

  int16_t ax = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t ay = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t az = ((int16_t)Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();                                   // skip temperature
  int16_t gx = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t gy = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t gz = ((int16_t)Wire.read() << 8) | Wire.read();

  accel[0] = ax * ACCEL_SCALE;
  accel[1] = ay * ACCEL_SCALE;
  accel[2] = az * ACCEL_SCALE;
  gyro[0]  = gx * GYRO_SCALE;
  gyro[1]  = gy * GYRO_SCALE;
  gyro[2]  = gz * GYRO_SCALE;

  // Complementary filter
  unsigned long nowUs = micros();
  float dt = (nowUs - lastIMUus) / 1e6f;
  lastIMUus = nowUs;

  float accelRoll  = atan2f(accel[1], accel[2]) * 180.0f / M_PI;
  float accelPitch = atan2f(-accel[0],
                    sqrtf(accel[1]*accel[1] + accel[2]*accel[2])) * 180.0f / M_PI;

  rollDeg  = CF_ALPHA * (rollDeg  + gyro[0] * dt) + (1.0f - CF_ALPHA) * accelRoll;
  pitchDeg = CF_ALPHA * (pitchDeg + gyro[1] * dt) + (1.0f - CF_ALPHA) * accelPitch;
  yawDeg  += gyro[2] * dt;   // gyro-only integration (no magnetometer)

  // Euler (roll, pitch, yaw) → unit quaternion [w, x, y, z]
  float r = rollDeg  * M_PI / 180.0f;
  float p = pitchDeg * M_PI / 180.0f;
  float y = yawDeg   * M_PI / 180.0f;
  float cr = cosf(r * 0.5f), sr = sinf(r * 0.5f);
  float cp = cosf(p * 0.5f), sp = sinf(p * 0.5f);
  float cy = cosf(y * 0.5f), sy = sinf(y * 0.5f);
  quat[0] = cr*cp*cy + sr*sp*sy;  // w
  quat[1] = sr*cp*cy - cr*sp*sy;  // x
  quat[2] = cr*sp*cy + sr*cp*sy;  // y
  quat[3] = cr*cp*sy - sr*sp*cy;  // z
}

// ================================================================
// NTP timestamp — gives both boards the same epoch-based clock
// so paired frames share the same timestamp_ms value.
// ================================================================
uint32_t getTimestampMs() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  // 32-bit truncation is fine within a session (wraps every ~49 days)
  return (uint32_t)((uint64_t)tv.tv_sec * 1000ULL + tv.tv_usec / 1000ULL);
}

// ================================================================
// Read all sensors
// ================================================================
void readSensors() {
  for (int i = 0; i < 5; i++) flexADC[i] = readMux(CH_FLEX[i]);
  for (int i = 0; i < 3; i++) fsrADC[i]  = readMux(CH_FSR[i]);
  readMPU();
}

// ================================================================
// Serialize and send one JSON packet
// ================================================================
void sendPacket() {
  // ArduinoJson v6: StaticJsonDocument with enough capacity.
  // Packet is ~210 chars; 512 bytes gives comfortable headroom.
  StaticJsonDocument<512> doc;

  doc["timestamp_ms"]    = getTimestampMs();
  doc["hand_id"]         = (int)HAND_ID;
  doc["contact_bitmask"] = 0;  // Touch pins removed; kept for backend compatibility

  JsonArray flex = doc.createNestedArray("flex");
  for (int i = 0; i < 5; i++) flex.add(flexADC[i]);

  JsonArray fsr = doc.createNestedArray("fsr");
  for (int i = 0; i < 3; i++) fsr.add(fsrADC[i]);

  JsonArray accelArr = doc.createNestedArray("accel");
  accelArr.add(serialized(String(accel[0], 4)));
  accelArr.add(serialized(String(accel[1], 4)));
  accelArr.add(serialized(String(accel[2], 4)));

  JsonArray gyroArr = doc.createNestedArray("gyro");
  gyroArr.add(serialized(String(gyro[0], 3)));
  gyroArr.add(serialized(String(gyro[1], 3)));
  gyroArr.add(serialized(String(gyro[2], 3)));

  JsonArray quatArr = doc.createNestedArray("quaternion");
  quatArr.add(serialized(String(quat[0], 5)));
  quatArr.add(serialized(String(quat[1], 5)));
  quatArr.add(serialized(String(quat[2], 5)));
  quatArr.add(serialized(String(quat[3], 5)));

  char buf[512];
  size_t n = serializeJson(doc, buf, sizeof(buf));
  wsClient.sendTXT(buf, n);
}

// ================================================================
// WebSocket event handler
// ================================================================
void wsEventHandler(WStype_t type, uint8_t* payload, size_t len) {
  switch (type) {
    case WStype_CONNECTED:
      wsConnected = true;
      Serial.printf("[WS] Connected  ws://%s:%u%s  hand_id=%d\n",
                    WS_HOST, WS_PORT, WS_PATH, HAND_ID);
      break;

    case WStype_DISCONNECTED:
      wsConnected = false;
      Serial.println("[WS] Disconnected — retrying in 2s…");
      break;

    case WStype_TEXT:
      // Optional: parse server ack {"type":"ack","paired":true,...}
      // Uncomment to debug:
      // Serial.printf("[ACK] %.*s\n", (int)len, payload);
      break;

    case WStype_ERROR:
      Serial.println("[WS] Error");
      break;

    default:
      break;
  }
}

// ================================================================
// setup
// ================================================================
void setup() {
  Serial.begin(115200);
  Serial.printf("\n[BOOT] hand_id=%d  mpu_addr=0x%02X\n", HAND_ID, MPU_ADDR);

  // MUX
  pinMode(MUX_S0, OUTPUT); pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT); pinMode(MUX_S3, OUTPUT);
  pinMode(MUX_EN, OUTPUT);
  digitalWrite(MUX_EN, LOW);   // Active-LOW enable; pull LOW to select all channels
  analogSetAttenuation(ADC_11db);  // 0–3.3V ADC range

  // IMU
  mpuInit();
  Serial.println("[IMU] MPU6050 initialized");

  // WiFi
  Serial.printf("[WiFi] Connecting to \"%s\"", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500); Serial.print("."); attempts++;
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[WiFi] FAILED — check SSID/password. Rebooting in 5s.");
    delay(5000);
    ESP.restart();
  }
  Serial.printf("\n[WiFi] Connected  IP: %s\n", WiFi.localIP().toString().c_str());

  // NTP sync — both boards must share the same clock for frame pairing
  Serial.print("[NTP]  Syncing time");
  configTime(0, 0, "pool.ntp.org", "time.google.com");
  int ntpAttempts = 0;
  while (time(nullptr) < 1000000UL && ntpAttempts < 30) {
    delay(500); Serial.print("."); ntpAttempts++;
  }
  if (time(nullptr) > 1000000UL)
    Serial.printf("\n[NTP]  Synced  epoch=%lu\n", (unsigned long)time(nullptr));
  else
    Serial.println("\n[NTP]  WARN: sync failed — timestamps may not pair correctly");

  // WebSocket
  wsClient.begin(WS_HOST, WS_PORT, WS_PATH);
  wsClient.onEvent(wsEventHandler);
  wsClient.setReconnectInterval(2000);
  Serial.printf("[WS]  Connecting to ws://%s:%u%s\n", WS_HOST, WS_PORT, WS_PATH);
}

// ================================================================
// loop — maintains WebSocket and fires sensor read at 50 Hz
// ================================================================
void loop() {
  // Must call every iteration for WebSocket keep-alive & reconnect
  wsClient.loop();

  unsigned long now = millis();
  if (now - lastSampleMs >= SAMPLE_MS) {
    lastSampleMs = now;
    readSensors();

    if (wsConnected) {
      sendPacket();
    } else {
      // Serial debug while not connected
      Serial.printf("[DBG] flex=[%d %d %d %d %d]  fsr=[%d %d %d]  "
                    "roll=%.1f  pitch=%.1f  yaw=%.1f\n",
        flexADC[0], flexADC[1], flexADC[2], flexADC[3], flexADC[4],
        fsrADC[0],  fsrADC[1],  fsrADC[2],
        rollDeg, pitchDeg, yawDeg);
    }
  }

  // Passive WiFi watchdog — reconnect if association lost
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Connection lost — reconnecting…");
    WiFi.reconnect();
    delay(1000);
  }
}
