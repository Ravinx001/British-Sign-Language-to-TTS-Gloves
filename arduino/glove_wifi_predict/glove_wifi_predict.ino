// ================================================================
// BSL Sign Language Gloves — Real-Time Prediction Firmware
//
// Flash this sketch to BOTH ESP32 boards.
// Only change the two lines marked "CHANGE THIS":
//
//   RIGHT glove:
//     #define HAND_ID   1
//     #define MPU_ADDR  0x68    (AD0 pin → GND)
//
//   LEFT glove:
//     #define HAND_ID   2
//     #define MPU_ADDR  0x69    (AD0 pin → 3.3V)
//
// This sketch connects to /ws/predict (not /ws/collect).
// The backend pairs frames from both hands by timestamp_ms,
// runs the XGBoost static classifier or CNN-LSTM (for J/Z),
// and sends back:
//   {"sign":"A","confidence":0.97,"route":"static","motion_energy":0.12}
//
// Required libraries (Arduino Library Manager):
//   • arduinoWebSockets  by Markus Sattler  (links2004/arduinoWebSockets)
//   • ArduinoJson        by Benoit Blanchon (v6 — tested 6.21+)
//   • ESP32 board package by Espressif (>=2.0.0)
//
// Pin wiring — identical to glove_wifi_collect:
//   MUX SIG  → GPIO 36  (ADC1 — only safe ADC pin with WiFi active)
//   MUX S0   → GPIO 16
//   MUX S1   → GPIO 17
//   MUX S2   → GPIO 18
//   MUX S3   → GPIO 19
//   MUX EN   → GPIO 5   (active LOW — pull to GND to enable)
//   IMU SDA  → GPIO 21
//   IMU SCL  → GPIO 22
//
// MUX channel map (matches data collection wiring):
//   CH 0-4 : Flex sensors  (thumb, index, middle, ring, pinky)
//   CH 5-7 : FSR sensors   (thumb tip, index tip, pinky tip)
// ================================================================

#include <WiFi.h>
#include <Wire.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <time.h>

// ================================================================
// USER CONFIGURATION — change these two lines before each flash
// ================================================================
#define HAND_ID    1         // CHANGE THIS: 1 = right glove, 2 = left glove
#define MPU_ADDR   0x68      // CHANGE THIS: 0x68 = right (AD0→GND), 0x69 = left (AD0→3.3V)

const char* WIFI_SSID = "A14";          // Your WiFi network name
const char* WIFI_PASS = "458173*/";     // Your WiFi password
const char* WS_HOST   = "10.66.209.201"; // << Run `ipconfig` on your laptop; use the IPv4 LAN address
const uint16_t WS_PORT = 8000;
const char* WS_PATH   = "/ws/predict"; // Prediction endpoint (NOT /ws/collect)

// ================================================================
// Pin definitions — ADC1 only (ADC2 is disabled when WiFi is on)
// ================================================================
const int MUX_SIG = 36;
const int MUX_S0  = 16;
const int MUX_S1  = 17;
const int MUX_S2  = 18;
const int MUX_S3  = 19;
const int MUX_EN  = 5;

const int I2C_SDA = 21;
const int I2C_SCL = 22;

// MUX channel assignments
const int CH_FLEX[5] = {0, 1, 2, 3, 4};  // thumb → pinky
const int CH_FSR[3]  = {5, 6, 7};         // thumb tip, index tip, pinky tip

// ================================================================
// Sampling rate
// ================================================================
const unsigned long SAMPLE_MS = 20;   // 50 Hz

// ================================================================
// MPU6050 registers
// ================================================================
const uint8_t MPU_PWR_MGMT_1  = 0x6B;
const uint8_t MPU_ACCEL_CFG   = 0x1C;
const uint8_t MPU_GYRO_CFG    = 0x1B;
const uint8_t MPU_ACCEL_XOUT  = 0x3B;

// ±4g → 8192 LSB/g;  ±500°/s → 65.5 LSB/(°/s)
const float ACCEL_SCALE = 9.80665f / 8192.0f;
const float GYRO_SCALE  = 1.0f / 65.5f;

// Complementary filter alpha (gyro weight)
const float CF_ALPHA = 0.98f;

// ================================================================
// Sensor buffers
// ================================================================
int   flexADC[5];
int   fsrADC[3];
float accel[3];
float gyro[3];
float quat[4];   // [w, x, y, z]

// Complementary filter state
float rollDeg  = 0.0f;
float pitchDeg = 0.0f;
float yawDeg   = 0.0f;
unsigned long lastIMUus = 0;

// ================================================================
// WebSocket client
// ================================================================
WebSocketsClient ws;
volatile bool wsConnected = false;
unsigned long lastSampleMs = 0;

// ================================================================
// Prediction result display state
// ================================================================
String lastSign       = "";
float  lastConfidence = 0.0f;
String lastRoute      = "";
unsigned long lastPredMs = 0;

// How long to keep showing a prediction on Serial before clearing
const unsigned long PRED_DISPLAY_MS = 1500;

// ================================================================
// MUX helpers
// ================================================================
void muxSelect(int ch) {
  digitalWrite(MUX_S0, (ch >> 0) & 1);
  digitalWrite(MUX_S1, (ch >> 1) & 1);
  digitalWrite(MUX_S2, (ch >> 2) & 1);
  digitalWrite(MUX_S3, (ch >> 3) & 1);
  delayMicroseconds(80);   // settling time
}

int readMux(int ch) {
  muxSelect(ch);
  analogRead(MUX_SIG);     // discard first sample after channel switch
  long sum = 0;
  for (int i = 0; i < 4; i++) sum += analogRead(MUX_SIG);
  return (int)(sum / 4);
}

// ================================================================
// MPU6050 helpers
// ================================================================
void mpuWriteReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void mpuInit() {
  Wire.begin(I2C_SDA, I2C_SCL, 400000UL);
  mpuWriteReg(MPU_PWR_MGMT_1, 0x00);  // wake, disable sleep
  delay(100);
  mpuWriteReg(MPU_ACCEL_CFG,  0x08);  // ±4g
  mpuWriteReg(MPU_GYRO_CFG,   0x08);  // ±500°/s
  lastIMUus = micros();
}

void readMPU() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(MPU_ACCEL_XOUT);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (size_t)14);
  if (Wire.available() < 14) return;

  int16_t ax = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t ay = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t az = ((int16_t)Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();   // skip temperature bytes
  int16_t gx = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t gy = ((int16_t)Wire.read() << 8) | Wire.read();
  int16_t gz = ((int16_t)Wire.read() << 8) | Wire.read();

  accel[0] = ax * ACCEL_SCALE;
  accel[1] = ay * ACCEL_SCALE;
  accel[2] = az * ACCEL_SCALE;
  gyro[0]  = gx * GYRO_SCALE;
  gyro[1]  = gy * GYRO_SCALE;
  gyro[2]  = gz * GYRO_SCALE;

  // Complementary filter — fuses accel (orientation) + gyro (rate)
  unsigned long nowUs = micros();
  float dt = (nowUs - lastIMUus) / 1e6f;
  lastIMUus = nowUs;

  float aRoll  = atan2f(accel[1], accel[2]) * 180.0f / M_PI;
  float aPitch = atan2f(-accel[0],
                 sqrtf(accel[1]*accel[1] + accel[2]*accel[2])) * 180.0f / M_PI;

  rollDeg  = CF_ALPHA * (rollDeg  + gyro[0] * dt) + (1.0f - CF_ALPHA) * aRoll;
  pitchDeg = CF_ALPHA * (pitchDeg + gyro[1] * dt) + (1.0f - CF_ALPHA) * aPitch;
  yawDeg  += gyro[2] * dt;   // gyro-only yaw (no magnetometer)

  // Euler → unit quaternion [w, x, y, z]
  float r = rollDeg  * M_PI / 180.0f;
  float p = pitchDeg * M_PI / 180.0f;
  float y = yawDeg   * M_PI / 180.0f;
  float cr = cosf(r*0.5f), sr = sinf(r*0.5f);
  float cp = cosf(p*0.5f), sp = sinf(p*0.5f);
  float cy = cosf(y*0.5f), sy = sinf(y*0.5f);
  quat[0] = cr*cp*cy + sr*sp*sy;  // w
  quat[1] = sr*cp*cy - cr*sp*sy;  // x
  quat[2] = cr*sp*cy + sr*cp*sy;  // y
  quat[3] = cr*cp*sy - sr*sp*cy;  // z
}

// ================================================================
// NTP-based timestamp (shared clock = frame pairing works)
// ================================================================
uint32_t getTimestampMs() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
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
// Build and send one JSON frame to /ws/predict
//
// Packet format (matches backend/routers/websocket.py expectation):
// {
//   "timestamp_ms": <uint32>,
//   "hand_id":      1 or 2,
//   "flex":         [thumb, index, middle, ring, pinky],  // raw ADC 0-4095
//   "fsr":          [thumb_tip, index_tip, pinky_tip],    // raw ADC 0-4095
//   "accel":        [x, y, z],                            // m/s²
//   "gyro":         [x, y, z],                            // deg/s
//   "quaternion":   [w, x, y, z],                         // unit quaternion
//   "contact_bitmask": 0                                   // always 0 (touch removed)
// }
// ================================================================
void sendFrame() {
  StaticJsonDocument<512> doc;

  doc["timestamp_ms"]    = getTimestampMs();
  doc["hand_id"]         = (int)HAND_ID;
  doc["contact_bitmask"] = 0;   // Touch sensors removed; kept for schema compat

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
  ws.sendTXT(buf, n);
}

// ================================================================
// Parse and display a prediction result from the server.
//
// Server response format:
// {
//   "sign":          "A",        // predicted BSL letter (or "Unknown")
//   "confidence":    0.97,       // 0.0-1.0
//   "route":         "static",   // "static" | "dynamic" | "unknown"
//   "motion_energy": 0.12        // gravity-compensated accel magnitude
// }
// ================================================================
void handleServerMessage(uint8_t* payload, size_t len) {
  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, payload, len);
  if (err) {
    Serial.printf("[WS] JSON parse error: %s\n", err.c_str());
    return;
  }

  const char* sign = doc["sign"] | "?";
  float conf       = doc["confidence"] | 0.0f;
  const char* route = doc["route"] | "?";

  // Only update display for non-Unknown predictions
  if (strcmp(sign, "Unknown") != 0) {
    lastSign       = String(sign);
    lastConfidence = conf;
    lastRoute      = String(route);
    lastPredMs     = millis();

    // Big clear visual for the predicted sign
    Serial.println("\n========================================");
    Serial.printf("  SIGN: %s  (%.0f%%)  [%s]\n", sign, conf * 100.0f, route);
    Serial.println("========================================\n");
  }
}

// ================================================================
// WebSocket event handler
// ================================================================
void wsEvent(WStype_t type, uint8_t* payload, size_t len) {
  switch (type) {

    case WStype_CONNECTED:
      wsConnected = true;
      Serial.printf("[WS] Connected  ws://%s:%u%s  hand_id=%d\n",
                    WS_HOST, WS_PORT, WS_PATH, HAND_ID);
      Serial.println("[WS] Streaming sensor data to prediction server...");
      break;

    case WStype_DISCONNECTED:
      wsConnected = false;
      Serial.println("[WS] Disconnected — retrying in 2s...");
      break;

    case WStype_TEXT:
      handleServerMessage(payload, len);
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
  delay(300);
  Serial.println("\n\n================================================");
  Serial.printf("  BSL Prediction Glove  |  hand_id=%d  addr=0x%02X\n",
                HAND_ID, MPU_ADDR);
  Serial.println("================================================");

  // --- MUX ---
  pinMode(MUX_S0, OUTPUT); pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT); pinMode(MUX_S3, OUTPUT);
  pinMode(MUX_EN, OUTPUT);
  digitalWrite(MUX_EN, LOW);         // Active-LOW: pull LOW to enable MUX
  analogSetAttenuation(ADC_11db);    // Full 0–3.3V range on ADC
  Serial.println("[MUX] Ready");

  // --- IMU ---
  mpuInit();
  Serial.println("[IMU] MPU6050 initialized  (±4g / ±500 dps)");

  // --- WiFi ---
  Serial.printf("[WiFi] Connecting to \"%s\"", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int att = 0;
  while (WiFi.status() != WL_CONNECTED && att < 40) {
    delay(500);
    Serial.print(".");
    att++;
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[WiFi] FAILED — check SSID/password. Rebooting...");
    delay(5000);
    ESP.restart();
  }
  Serial.printf("\n[WiFi] Connected  IP: %s\n", WiFi.localIP().toString().c_str());

  // --- NTP ---
  // Both gloves MUST share the same NTP-synced clock so the server can
  // pair right-hand and left-hand frames by their timestamp_ms value.
  Serial.print("[NTP]  Syncing time");
  configTime(0, 0, "pool.ntp.org", "time.google.com");
  int ntpAtt = 0;
  while (time(nullptr) < 1000000UL && ntpAtt < 30) {
    delay(500);
    Serial.print(".");
    ntpAtt++;
  }
  if (time(nullptr) > 1000000UL) {
    Serial.printf("\n[NTP]  Synced  epoch=%lu\n", (unsigned long)time(nullptr));
  } else {
    Serial.println("\n[NTP]  WARN: sync failed — frame pairing may not work!");
  }

  // --- WebSocket ---
  ws.begin(WS_HOST, WS_PORT, WS_PATH);
  ws.onEvent(wsEvent);
  ws.setReconnectInterval(2000);
  Serial.printf("[WS]   Connecting to ws://%s:%u%s\n", WS_HOST, WS_PORT, WS_PATH);
  Serial.println("\nWaiting for WebSocket connection...");
}

// ================================================================
// loop — 50 Hz sensor read + WebSocket keep-alive
// ================================================================
void loop() {
  ws.loop();   // Must be called every iteration

  unsigned long now = millis();

  // --- 50 Hz sensor read + send ---
  if (now - lastSampleMs >= SAMPLE_MS) {
    lastSampleMs = now;
    readSensors();

    if (wsConnected) {
      sendFrame();

      // Periodic sensor debug line (every ~2s = every 100 frames)
      static uint16_t frameCount = 0;
      if (++frameCount >= 100) {
        frameCount = 0;
        Serial.printf("[LIVE] flex=[%4d %4d %4d %4d %4d]  "
                      "fsr=[%4d %4d %4d]  "
                      "roll=%5.1f  pitch=%5.1f\n",
          flexADC[0], flexADC[1], flexADC[2], flexADC[3], flexADC[4],
          fsrADC[0],  fsrADC[1],  fsrADC[2],
          rollDeg, pitchDeg);
      }
    } else {
      // Offline debug — full sensor dump at 50 Hz
      Serial.printf("[DBG] flex=[%4d %4d %4d %4d %4d]  "
                    "fsr=[%4d %4d %4d]  "
                    "roll=%5.1f  pitch=%5.1f\n",
        flexADC[0], flexADC[1], flexADC[2], flexADC[3], flexADC[4],
        fsrADC[0],  fsrADC[1],  fsrADC[2],
        rollDeg, pitchDeg);
    }
  }

  // --- WiFi watchdog ---
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Lost connection — reconnecting...");
    WiFi.reconnect();
    delay(1000);
  }
}
