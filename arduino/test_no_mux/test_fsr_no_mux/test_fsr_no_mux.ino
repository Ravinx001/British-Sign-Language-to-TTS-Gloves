// =============================================================
// Test: FSR Sensors — Direct ADC (No MUX)
// Each FSR wired directly to an ESP32 ADC pin
// with a 47kΩ pull-down voltage divider
//
// Wiring per sensor:
//   3.3V → [FSR 402] → ADC Pin + [47kΩ → GND]
//
// Expected: ~0-100 (no pressure) to ~2500-4095 (firm press)
// =============================================================

// Direct ADC pins for each FSR
// GPIO33 is on ADC1 (WiFi safe), GPIO25/26 are on ADC2
const int FSR_PINS[] = {33, 25, 26};
const char* FSR_NAMES[] = {"Thumb Tip", "Index Tip", "Palm Pad"};
const int NUM_FSR = 3;

void setup() {
  Serial.begin(115200);
  delay(1000);

  for (int i = 0; i < NUM_FSR; i++) {
    pinMode(FSR_PINS[i], INPUT);
  }

  Serial.println("=== FSR Pressure Sensor Test (Direct ADC — No MUX) ===");
  Serial.println();
  Serial.println("Wiring (per sensor):");
  Serial.println("  3.3V -> [FSR 402] -> GPIO pin + [47k -> GND]");
  Serial.println();
  Serial.println("Pin assignments:");
  for (int i = 0; i < NUM_FSR; i++) {
    Serial.print("  ");
    Serial.print(FSR_NAMES[i]);
    Serial.print(" -> GPIO ");
    Serial.println(FSR_PINS[i]);
  }
  Serial.println();
  Serial.println("NOTE: GPIO25 and GPIO26 are on ADC2 — they won't work if WiFi is active.");
  Serial.println("      For testing without WiFi, this is fine.");
  Serial.println();
  Serial.println("Press each FSR pad — values should rise from ~0 to ~4095");
  Serial.println();
}

void loop() {
  Serial.print("[FSR] ");
  for (int i = 0; i < NUM_FSR; i++) {
    int rawValue = analogRead(FSR_PINS[i]);
    float pressurePercent = (rawValue / 4095.0) * 100.0;

    Serial.print(FSR_NAMES[i]);
    Serial.print("(");
    Serial.print(FSR_PINS[i]);
    Serial.print("): ");
    Serial.print(rawValue);
    Serial.print(" (");
    Serial.print(pressurePercent, 1);
    Serial.print("%)");

    if (i < NUM_FSR - 1) Serial.print("  |  ");
  }
  Serial.println();

  delay(100);  // 10 Hz
}
