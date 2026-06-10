// =============================================================
// Test: Single Flex Sensor — Direct ADC (No MUX)
// Minimal test for ONE flex sensor on GPIO36
// Use this to verify your very first flex sensor works
// before wiring all 5.
//
// Sensor: ~25.5kΩ (straight) to ~26.6kΩ (90° bend)
//         Only 1.1kΩ change — requires 64x oversampling
//         and auto-calibration for reliable readings.
//
// Measured voltages (10kΩ divider, 3.3V supply):
//   Straight: ~1.100V  →  ADC ~1365
//   90° bend: ~1.038V  →  ADC ~1288
//   Span: ~77 ADC counts (inverted: voltage drops with bend)
//
// Wiring:
//   3.3V → [Flex Sensor] → GPIO36 + [10kΩ → GND]
//
// Calibration on boot maps narrow ADC range to 800–3800
// for ML pipeline compatibility.
// =============================================================

const int FLEX_PIN = 36;  // ADC1_CH0 — input only, most reliable

// ---- Oversampling ----
const int OVERSAMPLE_COUNT = 64;  // noise ÷ √64 ≈ ±2.5 counts

// ---- Calibration state ----
// Stored as-read (no swapping) — map() handles inverted ranges correctly
long calStraight = 0;  // ADC reading when straight (higher voltage)
long calCurl = 0;       // ADC reading when 90° bent  (lower voltage)

// ---- ML pipeline output range ----
const int ML_FLEX_MIN = 800;   // maps to straight
const int ML_FLEX_MAX = 3800;  // maps to fully bent

// Read flex sensor with 64x oversampling for noise reduction
long readFlexOversampled() {
  long sum = 0;
  for (int i = 0; i < OVERSAMPLE_COUNT; i++) {
    sum += analogRead(FLEX_PIN);
  }
  return sum / OVERSAMPLE_COUNT;
}

// Collect N oversampled readings and return their average
long collectCalibrationSamples(int numSamples) {
  long total = 0;
  for (int i = 0; i < numSamples; i++) {
    total += readFlexOversampled();
    delay(20);  // 50 Hz pace
  }
  return total / numSamples;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(FLEX_PIN, INPUT);

  Serial.println("=== Single Flex Sensor Test (GPIO36, No MUX) ===");
  Serial.println();
  Serial.println("Wiring:");
  Serial.println("  3.3V --[Flex Sensor]--+---> GPIO36");
  Serial.println("                        |");
  Serial.println("                      [10kΩ]");
  Serial.println("                        |");
  Serial.println("                       GND");
  Serial.println();
  Serial.println("Expected: ~1.100V straight, ~1.038V at 90 deg bend");
  Serial.println("64x oversampling + auto-calibration enabled.");
  Serial.println();

  // ---- ADC Warm-up ----
  // ESP32 ADC drifts ~50-60 counts in first 10s after boot.
  // Discard readings until it stabilises.
  Serial.println(">>> Warming up ADC (10 seconds)...");
  for (int w = 0; w < 500; w++) {
    readFlexOversampled();  // discard — just warming up the ADC
    delay(20);
    if (w % 50 == 0) Serial.print(".");
  }
  Serial.println(" Done.");
  Serial.println();

  // ---- Auto-Calibration ----
  Serial.println(">>> CALIBRATION STEP 1/2: Hold finger STRAIGHT (open hand).");
  Serial.println(">>> Collecting in 3 seconds...");
  delay(3000);
  Serial.print(">>> Sampling (50 readings)... ");
  calStraight = collectCalibrationSamples(50);
  Serial.print("Done. Straight ADC = ");
  Serial.println(calStraight);
  Serial.println();

  Serial.println(">>> CALIBRATION STEP 2/2: BEND the finger to 90 degrees.");
  Serial.println(">>> Collecting in 3 seconds...");
  delay(3000);
  Serial.print(">>> Sampling (50 readings)... ");
  calCurl = collectCalibrationSamples(50);
  Serial.print("Done. Bent ADC = ");
  Serial.println(calCurl);
  Serial.println();

  // Report calibration — no swapping, map() handles either direction
  long calSpan = abs(calStraight - calCurl);
  Serial.print(">>> Straight ADC: ");
  Serial.print(calStraight);
  Serial.print("  |  Bent ADC: ");
  Serial.println(calCurl);
  Serial.print(">>> Span: ");
  Serial.print(calSpan);
  Serial.println(" ADC counts");

  if (calSpan < 10) {
    Serial.println(">>> WARNING: Calibrated span < 10 counts!");
    Serial.println(">>> Check wiring and sensor. Readings may be unreliable.");
  } else if (calSpan < 30) {
    Serial.println(">>> Note: Narrow span — oversampling is critical.");
  }

  if (calStraight > calCurl) {
    Serial.println(">>> Polarity: voltage DROPS with bend (normal for this sensor).");
  } else {
    Serial.println(">>> Polarity: voltage RISES with bend.");
  }

  Serial.println();
  Serial.println(">>> Calibration complete. Streaming mapped values...");
  Serial.println(">>> Output: 800 = straight, 3800 = fully bent");
  Serial.println();
}

void loop() {
  long raw = readFlexOversampled();

  // Map directly from calStraight→800, calCurl→3800
  // Arduino map() handles inverted input ranges correctly
  long mapped = map(raw, calStraight, calCurl, ML_FLEX_MIN, ML_FLEX_MAX);
  mapped = constrain(mapped, ML_FLEX_MIN, ML_FLEX_MAX);

  // Visual bar (0-50 chars mapped from 800–3800)
  int barLen = map(mapped, ML_FLEX_MIN, ML_FLEX_MAX, 0, 50);
  barLen = constrain(barLen, 0, 50);

  Serial.print("Raw: ");
  Serial.print(raw);
  Serial.print("  Mapped: ");
  Serial.print(mapped);
  Serial.print("  ");

  // Classify bend amount using mapped values
  if (mapped < 1550) {
    Serial.print("[STRAIGHT]  ");
  } else if (mapped < 2300) {
    Serial.print("[SLIGHT]    ");
  } else if (mapped < 3050) {
    Serial.print("[HALF BEND] ");
  } else {
    Serial.print("[FULL CURL] ");
  }

  // ASCII bar graph
  Serial.print("|");
  for (int i = 0; i < barLen; i++) Serial.print("█");
  for (int i = barLen; i < 50; i++) Serial.print("░");
  Serial.println("|");

  delay(100);
}
