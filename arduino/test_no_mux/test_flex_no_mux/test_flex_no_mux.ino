// =============================================================
// Test: Flex Sensors — Direct ADC (No MUX)
// Each flex sensor wired directly to an ESP32 ADC pin
// with a 10kΩ pull-down voltage divider.
//
// Sensor: ~25.5kΩ (straight) to ~26.6kΩ (90° bend)
//         Only 1.1kΩ change — requires 64x oversampling
//         and per-finger auto-calibration.
//
// Measured voltages (10kΩ divider, 3.3V supply):
//   Straight: ~1.100V  |  90° bend: ~1.038V
//   Voltage drops with bend (inverted polarity)
//
// Wiring per sensor:
//   3.3V → [Flex Sensor] → ADC Pin + [10kΩ → GND]
//
// Calibration on boot maps narrow ADC range to 800–3800
// for ML pipeline compatibility.
// =============================================================

// Direct ADC pins for each flex sensor (all on ADC1 — WiFi safe)
const int FLEX_PINS[] = {36, 39, 34, 35, 32};
const char* FINGER_NAMES[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
const int NUM_FLEX = 5;

// ---- Oversampling ----
const int OVERSAMPLE_COUNT = 64;  // noise ÷ √64 ≈ ±2.5 counts

// ---- Per-finger calibration state ----
// Stored as-read (no swapping) — map() handles inverted ranges correctly
long calStraight[5] = {0};  // ADC readings when fingers straight
long calCurl[5] = {0};       // ADC readings when fingers bent 90°

// ---- ML pipeline output range ----
const int ML_FLEX_MIN = 800;   // maps to straight
const int ML_FLEX_MAX = 3800;  // maps to fully bent

// Read a single flex pin with 64x oversampling
long readFlexOversampled(int pin) {
  long sum = 0;
  for (int i = 0; i < OVERSAMPLE_COUNT; i++) {
    sum += analogRead(pin);
  }
  return sum / OVERSAMPLE_COUNT;
}

// Collect N oversampled readings for one pin and return average
long collectCalibrationSamples(int pin, int numSamples) {
  long total = 0;
  for (int i = 0; i < numSamples; i++) {
    total += readFlexOversampled(pin);
    delay(20);
  }
  return total / numSamples;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  for (int i = 0; i < NUM_FLEX; i++) {
    pinMode(FLEX_PINS[i], INPUT);
  }

  Serial.println("=== Flex Sensor Test (Direct ADC — No MUX) ===");
  Serial.println();
  Serial.println("Wiring (per sensor):");
  Serial.println("  3.3V -> [Flex Sensor] -> GPIO pin + [10k -> GND]");
  Serial.println();
  Serial.println("Pin assignments:");
  for (int i = 0; i < NUM_FLEX; i++) {
    Serial.print("  ");
    Serial.print(FINGER_NAMES[i]);
    Serial.print(" -> GPIO ");
    Serial.println(FLEX_PINS[i]);
  }
  Serial.println();
  Serial.println("Expected: ~1.100V straight, ~1.038V at 90 deg bend");
  Serial.println("64x oversampling + per-finger calibration enabled.");
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

  // ---- Auto-Calibration ----
  Serial.println(">>> CALIBRATION STEP 1/2: Hold ALL fingers STRAIGHT (open hand).");
  Serial.println(">>> Collecting in 3 seconds...");
  delay(3000);
  Serial.print(">>> Sampling (50 readings per finger)... ");
  for (int i = 0; i < NUM_FLEX; i++) {
    calStraight[i] = collectCalibrationSamples(FLEX_PINS[i], 50);
  }
  Serial.println("Done.");
  for (int i = 0; i < NUM_FLEX; i++) {
    Serial.print(">>>   ");
    Serial.print(FINGER_NAMES[i]);
    Serial.print(" straight ADC = ");
    Serial.println(calStraight[i]);
  }
  Serial.println();

  Serial.println(">>> CALIBRATION STEP 2/2: BEND ALL fingers to 90 degrees.");
  Serial.println(">>> Collecting in 3 seconds...");
  delay(3000);
  Serial.print(">>> Sampling (50 readings per finger)... ");
  for (int i = 0; i < NUM_FLEX; i++) {
    calCurl[i] = collectCalibrationSamples(FLEX_PINS[i], 50);
  }
  Serial.println("Done.");
  for (int i = 0; i < NUM_FLEX; i++) {
    Serial.print(">>>   ");
    Serial.print(FINGER_NAMES[i]);
    Serial.print(" bent ADC = ");
    Serial.println(calCurl[i]);
  }
  Serial.println();

  // Validate per-finger calibration (no swapping)
  for (int i = 0; i < NUM_FLEX; i++) {
    long span = abs(calStraight[i] - calCurl[i]);
    if (span < 10) {
      Serial.print(">>> WARNING: ");
      Serial.print(FINGER_NAMES[i]);
      Serial.print(" span only ");
      Serial.print(span);
      Serial.println(" counts — check wiring!");
    }
  }

  Serial.println();
  Serial.println(">>> Calibration complete. Streaming mapped values (800-3800)...");
  Serial.println();
}

void loop() {
  Serial.print("[FLEX] ");
  for (int i = 0; i < NUM_FLEX; i++) {
    long raw = readFlexOversampled(FLEX_PINS[i]);

    // Map directly: calStraight→800, calCurl→3800
    long mapped = map(raw, calStraight[i], calCurl[i], ML_FLEX_MIN, ML_FLEX_MAX);
    mapped = constrain(mapped, ML_FLEX_MIN, ML_FLEX_MAX);

    Serial.print(FINGER_NAMES[i]);
    Serial.print("(");
    Serial.print(FLEX_PINS[i]);
    Serial.print("): ");
    Serial.print(mapped);

    if (i < NUM_FLEX - 1) Serial.print("  |  ");
  }
  Serial.println();

  delay(100);  // 10 Hz
}
