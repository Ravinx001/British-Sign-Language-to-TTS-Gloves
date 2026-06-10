// =============================================================
// Test: Single FSR Sensor — Direct ADC (No MUX)
// Minimal test for ONE FSR on GPIO33
// Use this to verify your first FSR works before wiring all 3.
//
// Wiring:
//   3.3V → [FSR 402] → GPIO33 + [47kΩ → GND]
//
// Expected: ~0-100 (no pressure) to ~2500-4095 (firm press)
// =============================================================

const int FSR_PIN = 33;  // ADC1_CH5

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(FSR_PIN, INPUT);

  Serial.println("=== Single FSR Sensor Test (GPIO33, No MUX) ===");
  Serial.println();
  Serial.println("Wiring:");
  Serial.println("  3.3V --[FSR 402]--+---> GPIO33");
  Serial.println("                    |");
  Serial.println("                  [47kΩ]");
  Serial.println("                    |");
  Serial.println("                   GND");
  Serial.println();
  Serial.println("Press the FSR pad with varying force.");
  Serial.println("No press: ~0-100  |  Firm press: ~2500-4095");
  Serial.println();
}

void loop() {
  int raw = analogRead(FSR_PIN);
  float percent = (raw / 4095.0) * 100.0;

  // Visual bar
  int barLen = map(raw, 0, 4095, 0, 50);
  barLen = constrain(barLen, 0, 50);

  Serial.print("FSR: ");
  Serial.print(raw);
  Serial.print(" (");
  Serial.print(percent, 1);
  Serial.print("%)  ");

  // Classify pressure
  if (raw < 100) {
    Serial.print("[NONE]   ");
  } else if (raw < 1000) {
    Serial.print("[LIGHT]  ");
  } else if (raw < 2500) {
    Serial.print("[MEDIUM] ");
  } else {
    Serial.print("[FIRM]   ");
  }

  // ASCII bar graph
  Serial.print("|");
  for (int i = 0; i < barLen; i++) Serial.print("█");
  for (int i = barLen; i < 50; i++) Serial.print("░");
  Serial.println("|");

  delay(100);
}
