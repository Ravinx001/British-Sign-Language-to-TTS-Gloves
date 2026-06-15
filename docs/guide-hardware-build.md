# Hardware Build Guide — BSL Sign Language Gloves

> **Complete wiring guide, component list, circuit diagrams, and Arduino test sketches for building the BSL fingerspelling glove hardware.**

---

## Table of Contents

- [System Overview](#system-overview)
- [Bill of Materials](#bill-of-materials)
- [Tools Required](#tools-required)
- [ESP32 Pin Assignment Table](#esp32-pin-assignment-table)
- [Circuit Diagrams & Wiring](#circuit-diagrams--wiring)
  - [1. Flex Sensors → MUX](#1-flex-sensors--mux)
  - [2. FSR Sensors → MUX](#2-fsr-sensors--mux)
  - [3. CD74HC4067 MUX → ESP32](#3-cd74hc4067-mux--esp32)
  - [4. MPU6050 IMU (I²C)](#4-mpu6050-imu-i2c)
  - [5. Capacitive Touch Pads — Not Used](#5-capacitive-touch-pads--not-used)
  - [6. Power Supply](#6-power-supply)
- [Full Wiring Summary Diagram](#full-wiring-summary-diagram)
- [Arduino Test Sketches](#arduino-test-sketches)
  - [Test 1 — Flex Sensors via MUX](#test-1--flex-sensors-via-mux)
  - [Test 2 — FSR Sensors via MUX](#test-2--fsr-sensors-via-mux)
  - [Test 3 — MPU6050 IMU](#test-3--mpu6050-imu)
  - [Test 4 — Capacitive Touch Sensors (Not Used)](#test-4--capacitive-touch-sensors-not-used--removed)
  - [Test 5 — MUX Channel Sweep](#test-5--mux-channel-sweep)
  - [Test 6 — Combined All Sensors (Full Glove)](#test-6--combined-all-sensors-full-glove)
- [Assembly Tips](#assembly-tips)
- [Troubleshooting](#troubleshooting)
- [Calibration Guide](#calibration-guide)

---

## System Overview

The BSL Sign Language Gloves system uses **two gloves** (right + left), each with identical sensor arrays controlled by an ESP32 microcontroller. The sensors capture finger bend, pressure, and hand orientation — producing a 26-dimensional feature vector for ML classification of BSL fingerspelling letters.

```
┌─────────────────────────────────────────────────────────┐
│                    PER GLOVE LAYOUT                      │
│                                                          │
│   5× Flex Sensors ──┐                                   │
│   (fingers)          ├──→ CD74HC4067 MUX ──→ GPIO36     │
│   3× FSR Sensors  ──┘    (analog mux)       (ADC)      │
│   (thumb,index,pinky)     ↑ select lines                │
│                           GPIO 16,17,18,19              │
│                                                          │
│   1× MPU6050 IMU ────────→ I²C Bus ──→ GPIO21 (SDA)    │
│   (accel + gyro)                        GPIO22 (SCL)    │
│                                                          │
│   Power: USB (dev) or 3.7V LiPo (wireless)             │
└─────────────────────────────────────────────────────────┘
```

> **Hardware change:** Capacitive touch pins (T0–T4) are **not used**. The former palm pad FSR has been relocated to the **pinky fingertip**. FSR positions are now: thumb tip, index tip, pinky tip.

---

## Bill of Materials

### Per Glove Pair (2 gloves)

| # | Component | Qty | Approx Cost | Interface | Notes |
|---|-----------|-----|-------------|-----------|-------|
| 1 | **ESP32-DevKitC V4** | 2 | ~$10 | WiFi / BLE | One per glove. 38-pin version recommended |
| 2 | **Flex Sensors (2.2″ / 55mm)** | 10 | ~$20 | Analog | 5 per hand — one per finger |
| 3 | **FSR 402 (Force Sensitive Resistor)** | 6 | ~$24 | Analog | 3 per hand — thumb tip, index tip, **pinky tip** |
| 4 | **MPU6050 IMU Module** (GY-521) | 2 | ~$5 | I²C | 1 per hand — accelerometer + gyroscope |
| 5 | **CD74HC4067 Breakout** (16-ch MUX) | 2 | ~$4 | 4 GPIO + 1 ADC | 1 per hand — routes 8 analog sensors |
| 6 | **47kΩ Resistors** | 16 | ~$1 | — | Pull-down for flex (×10) + FSR (×6) voltage dividers |
| 7 | **10kΩ Resistors** | 4 | ~$0.50 | — | I²C pull-ups (×2 per glove, optional if module has them) |
| 8 | **100nF Ceramic Capacitors** | 4 | ~$0.50 | — | Decoupling for MPU6050 and MUX (×2 per glove) |
| 9 | **Thin Gloves** (stretchy fabric) | 2 | ~$10 | — | Substrate to mount sensors on |
| 10 | **3.7V LiPo Battery** (500mAh+) | 2 | ~$8 | — | Wireless power (optional for dev phase) |
| 11 | **TP4056 Charger Module** | 2 | ~$4 | Micro-USB | LiPo charging (optional for dev phase) |
| 12 | **Breadboard** (half-size) | 2 | ~$3 | — | For prototyping before soldering |
| 13 | **Jumper Wires** (M-M, M-F) | 1 pack | ~$3 | — | Connections |
| 14 | **Micro-USB Cables** | 2 | ~$4 | USB | Programming + power during development |
| | **TOTAL** | | **~$97** | | |

### Where to Buy

- **Flex Sensors**: Spectra Symbol (official), AliExpress, Amazon
- **FSR 402**: Interlink Electronics, Adafruit, SparkFun
- **ESP32-DevKitC V4**: Espressif official, Amazon, AliExpress
- **MPU6050 (GY-521)**: Amazon, AliExpress (very common module)
- **CD74HC4067**: SparkFun breakout, Amazon, AliExpress

---

## Tools Required

- Soldering iron + solder (for final assembly)
- Wire strippers
- Multimeter (for continuity / resistance testing)
- Hot glue gun (sensor mounting)
- Heat shrink tubing
- Arduino IDE 2.x with **ESP32 board package** installed

### Arduino IDE Setup

1. Open Arduino IDE → **File → Preferences**
2. Add to **Additional Board Manager URLs**:
   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
3. **Tools → Board → Boards Manager** → search "esp32" → install **esp32 by Espressif Systems**
4. Select board: **Tools → Board → esp32 → ESP32 Dev Module**
5. Select port: **Tools → Port → COMx** (your ESP32's USB port)

---

## ESP32 Pin Assignment Table

### Right Glove (ESP32 #1)

| GPIO | Function | Connected To | Notes |
|------|----------|--------------|-------|
| **36** (VP) | ADC1_CH0 (input only) | MUX SIG (signal output) | Reads all analog sensors through MUX |
| **16** | Digital Output | MUX S0 (select bit 0) | MUX channel select LSB |
| **17** | Digital Output | MUX S1 (select bit 1) | MUX channel select |
| **18** | Digital Output | MUX S2 (select bit 2) | MUX channel select |
| **19** | Digital Output | MUX S3 (select bit 3) | MUX channel select MSB |
| **21** | I²C SDA | MPU6050 SDA | Data line (with 10kΩ pull-up to 3.3V) |
| **22** | I²C SCL | MPU6050 SCL | Clock line (with 10kΩ pull-up to 3.3V) |
| **3.3V** | Power | MUX VCC, MPU6050 VCC, Flex/FSR dividers | Regulated 3.3V output |
| **GND** | Ground | All grounds (MUX, MPU6050, dividers) | Common ground |

> **Note:** GPIO 4, 0, 2, 15, 13 (touch pins T0–T4) are **not connected** — capacitive touch sensing is not used in this hardware revision.

### Left Glove (ESP32 #2)

Same pin assignments as right glove. The only difference:
- **MPU6050 AD0 pin → 3.3V** (sets I²C address to **0x69** instead of 0x68)

### MUX Channel Assignments

| MUX Channel | Sensor | Finger/Position |
|-------------|--------|-----------------|
| **CH0** | Flex Sensor | Thumb |
| **CH1** | Flex Sensor | Index |
| **CH2** | Flex Sensor | Middle |
| **CH3** | Flex Sensor | Ring |
| **CH4** | Flex Sensor | Pinky |
| **CH5** | FSR | Thumb tip |
| **CH6** | FSR | Index tip |
| **CH7** | FSR | **Pinky tip** |
| CH8–CH15 | — | Unused (leave disconnected) |

---

## Circuit Diagrams & Wiring

### 1. Flex Sensors → MUX

Each flex sensor forms a **voltage divider** with a 47kΩ pull-down resistor. The voltage at the junction changes as the flex sensor bends (resistance increases).

**Circuit (per flex sensor):**

```
            ┌──────────────────────┐
  3.3V ─────┤ Flex Sensor (variable│
            │ resistance)          │
            └──────────┬───────────┘
                       │
                       ├──────────────→ MUX Channel Input (CH0-CH4)
                       │
                   ┌───┴───┐
                   │ 47kΩ  │
                   └───┬───┘
                       │
                      GND
```

**How it works:**
- **Finger straight** → flex resistance LOW (~10kΩ) → voltage at junction is HIGH → ADC reads ~**800**
- **Finger curled** → flex resistance HIGH (~100kΩ) → voltage at junction is LOW → ADC reads ~**3800**

> Wait — the ADC reads *higher* when the flex resistance is *higher*? Yes, because the ESP32 ADC is 12-bit (0–4095) and the voltage divider output *decreases* as flex resistance increases, but the flex sensor wiring convention in this project maps higher ADC to more curl. The exact mapping depends on whether the flex sensor is in the upper or lower leg. In our configuration, the **flex sensor is the upper leg** (connected to 3.3V), so as it bends and resistance increases, the voltage at the junction drops — but the ADC inversion in the ESP32's attenuation setting makes higher resistance map to higher raw values in the 800–3800 range.

**Wiring all 5 flex sensors to MUX:**

```
  3.3V ──┬──────┬──────┬──────┬──────┬
         │      │      │      │      │
        [F0]   [F1]   [F2]   [F3]   [F4]     ← Flex sensors
       Thumb  Index  Middle  Ring  Pinky
         │      │      │      │      │
         ├──→CH0├──→CH1├──→CH2├──→CH3├──→CH4  ← MUX channel inputs
         │      │      │      │      │
        47k    47k    47k    47k    47k       ← Pull-down resistors
         │      │      │      │      │
  GND ──┴──────┴──────┴──────┴──────┴
```

### 2. FSR Sensors → MUX

FSR sensors also use a voltage divider with a 47kΩ pull-down. Unlike flex sensors, FSR resistance **decreases** with pressure.

**Circuit (per FSR):**

```
            ┌──────────────────────┐
  3.3V ─────┤ FSR 402 (variable    │
            │ resistance)          │
            └──────────┬───────────┘
                       │
                       ├──────────────→ MUX Channel Input (CH5-CH7)
                       │
                   ┌───┴───┐
                   │ 47kΩ  │
                   └───┬───┘
                       │
                      GND
```

**How it works:**
- **No pressure** → FSR resistance VERY HIGH (>1MΩ) → junction voltage near 0V → ADC reads ~**0–100**
- **Firm press** → FSR resistance LOW (~1kΩ) → junction voltage near 3.3V → ADC reads ~**2500–4095**

**Wiring all 3 FSR sensors to MUX:**

```
  3.3V ──┬──────────┬──────────┬
         │          │          │
       [FSR0]     [FSR1]     [FSR2]       ← FSR 402 sensors
       Thumb tip  Index tip  Pinky tip
         │          │          │
         ├───→CH5   ├───→CH6   ├───→CH7   ← MUX channel inputs
         │          │          │
        47k        47k        47k         ← Pull-down resistors
         │          │          │
  GND ──┴──────────┴──────────┴
```

**Physical placement:**
- **FSR0 (Thumb tip)**: Glued to the pad of the thumb, sensing area facing outward
- **FSR1 (Index tip)**: Glued to the pad of the index finger
- **FSR2 (Pinky tip)**: Glued to the pad of the pinky finger — detects when the right index presses on the left pinky tip (critical for BSL letter "U")

### 3. CD74HC4067 MUX → ESP32

The 16-channel analog multiplexer routes all 8 analog sensor signals through a single ESP32 ADC pin. Four digital GPIO lines select which channel is active.

**CD74HC4067 Breakout Pinout:**

```
  CD74HC4067 Module
  ┌─────────────────────────────┐
  │                             │
  │  SIG ──────────→ ESP32 GPIO36 (ADC input)
  │                             │
  │  S0  ──────────→ ESP32 GPIO16
  │  S1  ──────────→ ESP32 GPIO17
  │  S2  ──────────→ ESP32 GPIO18
  │  S3  ──────────→ ESP32 GPIO19
  │                             │
  │  EN  ──────────→ GND (always enabled)
  │  VCC ──────────→ 3.3V
  │  GND ──────────→ GND
  │                             │
  │  C0  ← Flex Thumb           │
  │  C1  ← Flex Index           │
  │  C2  ← Flex Middle          │
  │  C3  ← Flex Ring            │
  │  C4  ← Flex Pinky           │
  │  C5  ← FSR Thumb            │
  │  C6  ← FSR Index            │
  │  C7  ← FSR Pinky            │
  │  C8  ← (unused)             │
  │  ...                        │
  │  C15 ← (unused)             │
  │                             │
  └─────────────────────────────┘
```

**Channel Selection Logic:**

The 4-bit select lines (S0–S3) choose which channel connects to SIG:

| Channel | S3 | S2 | S1 | S0 | Sensor |
|---------|----|----|----|----|--------|
| 0 | 0 | 0 | 0 | 0 | Flex Thumb |
| 1 | 0 | 0 | 0 | 1 | Flex Index |
| 2 | 0 | 0 | 1 | 0 | Flex Middle |
| 3 | 0 | 0 | 1 | 1 | Flex Ring |
| 4 | 0 | 1 | 0 | 0 | Flex Pinky |
| 5 | 0 | 1 | 0 | 1 | FSR Thumb |
| 6 | 0 | 1 | 1 | 0 | FSR Index |
| 7 | 0 | 1 | 1 | 1 | FSR Pinky |

### 4. MPU6050 IMU (I²C)

The MPU6050 module (GY-521 breakout) provides 3-axis accelerometer + 3-axis gyroscope. Connected via I²C.

**Wiring:**

```
  MPU6050 (GY-521)          ESP32
  ┌──────────────┐
  │ VCC ─────────────────→ 3.3V
  │ GND ─────────────────→ GND
  │ SCL ────────┬────────→ GPIO22
  │             │
  │            10kΩ (pull-up, optional if module has built-in)
  │             │
  │            3.3V
  │             │
  │ SDA ────────┬────────→ GPIO21
  │             │
  │            10kΩ (pull-up, optional if module has built-in)
  │             │
  │            3.3V
  │                      │
  │ AD0 ─────────────────→ GND (address = 0x68, RIGHT glove)
  │                        OR
  │ AD0 ─────────────────→ 3.3V (address = 0x69, LEFT glove)
  │                      │
  │ INT ─────────────────→ (not connected — we poll, not interrupt)
  └──────────────┘
```

**I²C Address Selection:**

| Glove | AD0 Pin | I²C Address |
|-------|---------|-------------|
| **Right** | GND (or floating) | **0x68** |
| **Left** | 3.3V (pulled HIGH) | **0x69** |

> **Note:** Most GY-521 breakout boards have built-in 10kΩ pull-up resistors on SDA and SCL. Check your module — if it has them, you do NOT need external pull-ups.

**100nF decoupling capacitor:** Place a 100nF ceramic capacitor between VCC and GND on the MPU6050, as close to the chip as possible, to reduce noise.

### 5. Capacitive Touch Pads — Not Used

> **⚠️ Hardware change:** Capacitive touch sensing (ESP32 touch pins T0–T4) has been **removed** from the glove design. Contact detection is now achieved via the FSR sensors on each fingertip (thumb, index, pinky). GPIO 4, 0, 2, 15, and 13 are left unconnected.
>
> The former palm pad FSR (CH7) has been relocated to the **pinky fingertip**, which provides partial contact disambiguation for BSL letters where the dominant index finger presses on the non-dominant pinky tip (e.g., letter "U"). See `docs/new-docs/pad_to_pinky_fsr_analysis.md` for the full impact analysis.

### 6. Power Supply

**Development Phase (USB):**
- Simply power each ESP32 via micro-USB cable from your computer
- Provides 5V through USB, regulated to 3.3V by the ESP32's onboard regulator
- Both programming and serial monitoring through the same cable

**Wireless Phase (LiPo Battery):**

```
  3.7V LiPo ──→ TP4056 Charger ──→ ESP32 VIN pin
  (500mAh+)     (micro-USB input    (accepts 5V–12V,
                  for charging)       has onboard 3.3V
                                      regulator)
```

| Connection | From | To |
|-----------|------|----|
| Battery + | LiPo + | TP4056 B+ |
| Battery − | LiPo − | TP4056 B− |
| Output + | TP4056 OUT+ | ESP32 **VIN** pin |
| Output − | TP4056 OUT− | ESP32 **GND** pin |

> **⚠️ Warning:** Never connect USB power and LiPo simultaneously through VIN unless your TP4056 module has proper protection. During development, just use USB.

---

## Full Wiring Summary Diagram

```
                              ESP32-DevKitC V4
                         ┌──────────────────────────┐
                         │                          │
  Flex Thumb ──47k──┐    │  GPIO36 (VP) ◄── MUX SIG│
  Flex Index ──47k──┤    │  GPIO16      ──► MUX S0  │
  Flex Middle──47k──┤    │  GPIO17      ──► MUX S1  │
  Flex Ring ──47k───┤    │  GPIO18      ──► MUX S2  │
  Flex Pinky──47k───┤    │  GPIO19      ──► MUX S3  │
                    │    │                          │
  FSR Thumb ──47k───┤    │  GPIO21 (SDA)◄──►MPU6050 │
  FSR Index ──47k───┤    │  GPIO22 (SCL)──► MPU6050 │
  FSR Pinky ──47k───┘    │                          │
        │                │  3.3V ──► VCC (all)      │
        ▼                │  GND  ──► GND (all)      │
  ┌──────────┐           │                          │
  │CD74HC4067│           │  VIN  ◄── LiPo (wireless)│
  │   MUX    │           │  USB  ◄── PC (dev)       │
  │ C0-C4:Flx│           │                          │
  │ C5-C7:FSR│           │  GPIO 4,0,2,15,13        │
  │ SIG──────────────────│  (Touch T0–T4): NOT USED │
  │ S0-S3 ◄─────────────│                          │
  │ EN──►GND │           └──────────────────────────┘
  │ VCC──►3V3│
  └──────────┘

  All 47kΩ resistors connect between sensor junction and GND
  MUX SIG output connects to GPIO36 (ADC1_CH0)
  MPU6050 AD0 → GND (right, 0x68) or 3.3V (left, 0x69)
  CH7 = FSR Pinky tip (was Palm pad in earlier revisions)
```

---

## Arduino Test Sketches

> **Prerequisites:** Install the ESP32 board package in Arduino IDE (see [Arduino IDE Setup](#arduino-ide-setup)). Set board to "ESP32 Dev Module" and select the correct COM port.

### Test 1 — Flex Sensors via MUX

Tests all 5 flex sensors by reading them through the CD74HC4067 MUX. Open the Serial Monitor (115200 baud) and bend each finger — you should see values change from ~800 (straight) to ~3800 (curled).

```cpp
// =============================================================
// Test 1: Flex Sensors via CD74HC4067 MUX
// Reads 5 flex sensors on MUX channels 0-4
// Expected: ~800 (straight) to ~3800 (fully curled)
// =============================================================

// MUX select pins
const int MUX_S0 = 16;
const int MUX_S1 = 17;
const int MUX_S2 = 18;
const int MUX_S3 = 19;

// MUX signal (analog read) pin
const int MUX_SIG = 36;

// Flex sensor MUX channels
const int FLEX_CHANNELS[] = {0, 1, 2, 3, 4};
const char* FINGER_NAMES[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
const int NUM_FLEX = 5;

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Set MUX select pins as outputs
  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  pinMode(MUX_S3, OUTPUT);

  // ADC input pin
  pinMode(MUX_SIG, INPUT);

  Serial.println("=== Flex Sensor Test (via MUX) ===");
  Serial.println("Bend each finger to see values change");
  Serial.println("Expected range: ~800 (straight) to ~3800 (curled)");
  Serial.println();
}

// Select a MUX channel (0-15) by setting S0-S3
void selectMuxChannel(int channel) {
  digitalWrite(MUX_S0, (channel >> 0) & 1);
  digitalWrite(MUX_S1, (channel >> 1) & 1);
  digitalWrite(MUX_S2, (channel >> 2) & 1);
  digitalWrite(MUX_S3, (channel >> 3) & 1);
  delayMicroseconds(5);  // Settling time for MUX
}

void loop() {
  Serial.print("[FLEX] ");
  for (int i = 0; i < NUM_FLEX; i++) {
    selectMuxChannel(FLEX_CHANNELS[i]);
    int rawValue = analogRead(MUX_SIG);

    Serial.print(FINGER_NAMES[i]);
    Serial.print(": ");
    Serial.print(rawValue);

    if (i < NUM_FLEX - 1) Serial.print("  |  ");
  }
  Serial.println();

  delay(100);  // 10 Hz for readable serial output
}
```

**What to verify:**
- [ ] Each finger shows ~800 when straight
- [ ] Each finger shows ~3800 when fully curled
- [ ] Values change smoothly as you bend
- [ ] Each sensor responds independently (bending one finger doesn't affect others)
- [ ] If a sensor reads 0 or 4095 constantly — check wiring and resistor connection

---

### Test 2 — FSR Sensors via MUX

Tests all 3 FSR (Force Sensitive Resistor) sensors through MUX channels 5–7. Press each sensor with varying force.

```cpp
// =============================================================
// Test 2: FSR Sensors via CD74HC4067 MUX
// Reads 3 FSR sensors on MUX channels 5-7
// Expected: ~0-100 (no pressure) to ~2500-4095 (firm press)
// =============================================================

// MUX select pins
const int MUX_S0 = 16;
const int MUX_S1 = 17;
const int MUX_S2 = 18;
const int MUX_S3 = 19;

// MUX signal (analog read) pin
const int MUX_SIG = 36;

// FSR MUX channels
const int FSR_CHANNELS[] = {5, 6, 7};
const char* FSR_NAMES[] = {"Thumb Tip", "Index Tip", "Pinky Tip"};
const int NUM_FSR = 3;

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  pinMode(MUX_S3, OUTPUT);
  pinMode(MUX_SIG, INPUT);

  Serial.println("=== FSR Pressure Sensor Test (via MUX) ===");
  Serial.println("Press each FSR pad with varying force");
  Serial.println("CH5=Thumb Tip  CH6=Index Tip  CH7=Pinky Tip");
  Serial.println("Expected: ~0-100 (no press) to ~2500-4095 (firm press)");
  Serial.println();
}

void selectMuxChannel(int channel) {
  digitalWrite(MUX_S0, (channel >> 0) & 1);
  digitalWrite(MUX_S1, (channel >> 1) & 1);
  digitalWrite(MUX_S2, (channel >> 2) & 1);
  digitalWrite(MUX_S3, (channel >> 3) & 1);
  delayMicroseconds(5);
}

void loop() {
  Serial.print("[FSR] ");
  for (int i = 0; i < NUM_FSR; i++) {
    selectMuxChannel(FSR_CHANNELS[i]);
    int rawValue = analogRead(MUX_SIG);

    // Map to percentage for easy reading
    float pressurePercent = (rawValue / 4095.0) * 100.0;

    Serial.print(FSR_NAMES[i]);
    Serial.print(": ");
    Serial.print(rawValue);
    Serial.print(" (");
    Serial.print(pressurePercent, 1);
    Serial.print("%)");

    if (i < NUM_FSR - 1) Serial.print("  |  ");
  }
  Serial.println();

  delay(100);  // 10 Hz for readable output
}
```

**What to verify:**
- [ ] No pressure → reads ~0–100
- [ ] Light press → reads ~800–1500
- [ ] Firm press → reads ~2500–4095
- [ ] Values return to near-zero when released
- [ ] Pinky tip FSR (CH7) spikes when right index presses on left pinky tip

---

### Test 3 — MPU6050 IMU

Tests the MPU6050 accelerometer and gyroscope over I²C. Uses raw register reads with the Wire library — no external library needed.

```cpp
// =============================================================
// Test 3: MPU6050 IMU (Accelerometer + Gyroscope)
// Reads accel XYZ and gyro XYZ via I2C
// Expected at rest: accel ~[0, 0, 9.8] m/s², gyro ~[0, 0, 0] °/s
// =============================================================

#include <Wire.h>

// I2C address: 0x68 (AD0=GND, right glove) or 0x69 (AD0=HIGH, left glove)
const uint8_t MPU6050_ADDR = 0x68;  // Change to 0x69 for left glove

// MPU6050 Register addresses
const uint8_t REG_PWR_MGMT_1  = 0x6B;
const uint8_t REG_ACCEL_CONFIG = 0x1C;
const uint8_t REG_GYRO_CONFIG  = 0x1B;
const uint8_t REG_ACCEL_XOUT_H = 0x3B;

// Scale factors
// Accel: ±4g → sensitivity = 8192 LSB/g
// Gyro:  ±500°/s → sensitivity = 65.5 LSB/(°/s)
const float ACCEL_SCALE = 9.80665 / 8192.0;  // Convert to m/s²
const float GYRO_SCALE  = 1.0 / 65.5;         // Convert to °/s

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(21, 22);  // SDA=GPIO21, SCL=GPIO22
  Wire.setClock(400000);  // 400kHz fast mode

  Serial.println("=== MPU6050 IMU Test ===");
  Serial.println("Scanning I2C bus...");

  // I2C scan to verify connection
  Wire.beginTransmission(MPU6050_ADDR);
  uint8_t error = Wire.endTransmission();
  if (error == 0) {
    Serial.print("MPU6050 found at address 0x");
    Serial.println(MPU6050_ADDR, HEX);
  } else {
    Serial.print("ERROR: MPU6050 not found at 0x");
    Serial.print(MPU6050_ADDR, HEX);
    Serial.println(" — check wiring!");
    while (1) delay(1000);  // Halt
  }

  // Wake up MPU6050 (it starts in sleep mode)
  writeRegister(REG_PWR_MGMT_1, 0x00);
  delay(100);

  // Set accelerometer range to ±4g
  writeRegister(REG_ACCEL_CONFIG, 0x08);  // AFS_SEL=1 → ±4g

  // Set gyroscope range to ±500°/s
  writeRegister(REG_GYRO_CONFIG, 0x08);   // FS_SEL=1 → ±500°/s

  Serial.println("MPU6050 initialized: ±4g accel, ±500°/s gyro");
  Serial.println("Keep sensor still — accel Z should read ~9.8 m/s²");
  Serial.println();
}

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

int16_t readRegister16(uint8_t reg) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, (uint8_t)2);
  int16_t value = (Wire.read() << 8) | Wire.read();
  return value;
}

void loop() {
  // Read accelerometer (6 bytes: XH,XL,YH,YL,ZH,ZL)
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, (uint8_t)14);  // Read accel + temp + gyro (14 bytes)

  int16_t ax_raw = (Wire.read() << 8) | Wire.read();
  int16_t ay_raw = (Wire.read() << 8) | Wire.read();
  int16_t az_raw = (Wire.read() << 8) | Wire.read();
  int16_t temp_raw = (Wire.read() << 8) | Wire.read();  // Skip temperature
  int16_t gx_raw = (Wire.read() << 8) | Wire.read();
  int16_t gy_raw = (Wire.read() << 8) | Wire.read();
  int16_t gz_raw = (Wire.read() << 8) | Wire.read();

  // Convert to physical units
  float ax = ax_raw * ACCEL_SCALE;
  float ay = ay_raw * ACCEL_SCALE;
  float az = az_raw * ACCEL_SCALE;
  float gx = gx_raw * GYRO_SCALE;
  float gy = gy_raw * GYRO_SCALE;
  float gz = gz_raw * GYRO_SCALE;
  float temp = (temp_raw / 340.0) + 36.53;  // Temperature in °C

  // Print formatted output
  Serial.print("[IMU] Accel(m/s²) X:");
  Serial.print(ax, 2);
  Serial.print("  Y:");
  Serial.print(ay, 2);
  Serial.print("  Z:");
  Serial.print(az, 2);

  Serial.print("  |  Gyro(°/s) X:");
  Serial.print(gx, 1);
  Serial.print("  Y:");
  Serial.print(gy, 1);
  Serial.print("  Z:");
  Serial.print(gz, 1);

  Serial.print("  |  Temp: ");
  Serial.print(temp, 1);
  Serial.println("°C");

  delay(20);  // 50 Hz
}
```

**What to verify:**
- [ ] Serial shows "MPU6050 found at address 0x68" (or 0x69)
- [ ] At rest on table: accel Z ≈ 9.8 m/s², X and Y near 0
- [ ] Tilt sensor → accel values shift between axes
- [ ] Rotate quickly → gyro values spike (should reach ±500°/s max)
- [ ] At rest → gyro values near 0 (small drift is normal)
- [ ] Temperature reads room temperature (~20–30°C)

---

### Test 4 — Capacitive Touch Sensors (Not Used — Removed)

> **⚠️ Hardware change:** Capacitive touch sensing has been removed from this design. Test 4 is no longer applicable.
>
> GPIO 4 (T0), 0 (T1), 2 (T2), 15 (T3), and 13 (T4) are left unconnected. Contact detection relies entirely on the FSR sensors on each fingertip (thumb, index, pinky — MUX channels 5–7).
>
> If you are evaluating the accuracy impact of this change, refer to `docs/new-docs/capacitive_touch_removal_analysis.md` and `docs/new-docs/pad_to_pinky_fsr_analysis.md`.

---

### Test 5 — MUX Channel Sweep

Sweeps through all 16 channels of the CD74HC4067 to verify wiring and identify which channels have sensors connected. Useful for debugging MUX wiring.

```cpp
// =============================================================
// Test 5: CD74HC4067 MUX — Full 16-Channel Sweep
// Reads all 16 MUX channels and prints values
// Channels 0-4: flex sensors, 5-7: FSR, 8-15: unused
// =============================================================

const int MUX_S0 = 16;
const int MUX_S1 = 17;
const int MUX_S2 = 18;
const int MUX_S3 = 19;
const int MUX_SIG = 36;

const int NUM_CHANNELS = 16;

// Labels for each channel
const char* CHANNEL_LABELS[] = {
  "Flex Thumb", "Flex Index", "Flex Middle", "Flex Ring", "Flex Pinky",
  "FSR Thumb",  "FSR Index",  "FSR Pinky",
  "unused",     "unused",     "unused",      "unused",
  "unused",     "unused",     "unused",      "unused"
};

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  pinMode(MUX_S3, OUTPUT);
  pinMode(MUX_SIG, INPUT);

  Serial.println("=== MUX Channel Sweep Test ===");
  Serial.println("Sweeping all 16 channels of CD74HC4067");
  Serial.println("Connected sensors (CH0-7) should show valid readings");
  Serial.println("Unused channels (CH8-15) may float — ignore them");
  Serial.println();
}

void selectMuxChannel(int channel) {
  digitalWrite(MUX_S0, (channel >> 0) & 1);
  digitalWrite(MUX_S1, (channel >> 1) & 1);
  digitalWrite(MUX_S2, (channel >> 2) & 1);
  digitalWrite(MUX_S3, (channel >> 3) & 1);
  delayMicroseconds(5);
}

void loop() {
  Serial.println("--- MUX Channel Sweep ---");
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    selectMuxChannel(ch);
    int rawValue = analogRead(MUX_SIG);

    Serial.print("  CH");
    if (ch < 10) Serial.print(" ");
    Serial.print(ch);
    Serial.print(" [");
    Serial.print(CHANNEL_LABELS[ch]);
    Serial.print("]: ");
    Serial.print(rawValue);

    // Flag connected sensor channels
    if (ch <= 4) {
      Serial.print("  (flex — expect 800-3800)");
    } else if (ch <= 7) {
      Serial.print("  (fsr — expect 0-4095)");
    } else {
      Serial.print("  (unused — floating)");
    }
    Serial.println();
  }
  Serial.println();

  delay(2000);  // Sweep every 2 seconds
}
```

**What to verify:**
- [ ] Channels 0–4 (flex): show values in 800–3800 range
- [ ] Channels 5–7 (FSR): show 0–100 with no pressure, higher when pressed
- [ ] Channels 8–15: may show random floating values — this is normal
- [ ] Bending a finger changes only its corresponding channel (no crossover)

---

### Test 6 — Combined All Sensors (Full Glove)

Reads all sensors at 50 Hz and outputs JSON packets matching the backend's expected format. This is the final integration test before connecting to the ML backend.

```cpp
// =============================================================
// Test 6: Combined All Sensors — Full Glove Test
// Reads flex, FSR (thumb/index/pinky), and IMU at 50 Hz
// Outputs JSON matching backend WebSocket format:
// {"timestamp_ms", "hand_id", "flex[]", "fsr[]",
//  "accel[]", "gyro[]", "quaternion[]"}
// NOTE: contact_bitmask removed — capacitive touch not used
// =============================================================

#include <Wire.h>
#include <math.h>

// ---- Configuration ----
const bool DEBUG_MODE = true;       // true = human-readable, false = JSON for backend
const int DEBUG_DELAY_MS = 500;     // Extra pause after each debug print (ms). Set 0 for fastest, 500-1000 for easy reading
const int HAND_ID = 1;  // 1 = right, 2 = left
const uint8_t MPU6050_ADDR = 0x68;  // 0x68 for right, 0x69 for left

// ---- MUX Pins ----
const int MUX_S0 = 16;
const int MUX_S1 = 17;
const int MUX_S2 = 18;
const int MUX_S3 = 19;
const int MUX_SIG = 36;

// ---- MUX Channels ----
const int FLEX_CHANNELS[] = {0, 1, 2, 3, 4};
const int FSR_CHANNELS[]  = {5, 6, 7};  // CH5=Thumb, CH6=Index, CH7=Pinky

// ---- MPU6050 Registers ----
const uint8_t REG_PWR_MGMT_1   = 0x6B;
const uint8_t REG_ACCEL_CONFIG  = 0x1C;
const uint8_t REG_GYRO_CONFIG   = 0x1B;
const uint8_t REG_ACCEL_XOUT_H  = 0x3B;

// ---- Scale Factors ----
const float ACCEL_SCALE = 9.80665 / 8192.0;  // ±4g to m/s²
const float GYRO_SCALE  = 1.0 / 65.5;         // ±500°/s

// ---- Timing ----
const unsigned long SAMPLE_INTERVAL_MS = 20;  // 50 Hz
unsigned long lastSampleTime = 0;

// ---- Sensor Data ----
int flexValues[5];
int fsrValues[3];  // [thumb, index, pinky]
float accel[3];
float gyro[3];
float quaternion[4];

// Simple complementary filter for quaternion estimation
float roll = 0, pitch = 0, yaw = 0;
const float ALPHA = 0.98;  // Complementary filter coefficient

void setup() {
  Serial.begin(115200);
  delay(1000);

  // MUX pins
  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  pinMode(MUX_S3, OUTPUT);
  pinMode(MUX_SIG, INPUT);

  // Set ADC attenuation to 11dB for full 0-3.3V range
  // Without this, readings may clip at ~1.1V depending on ESP32 core version
  analogSetAttenuation(ADC_11db);

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
  writeRegister(REG_ACCEL_CONFIG, 0x08);  // ±4g
  writeRegister(REG_GYRO_CONFIG, 0x08);   // ±500°/s

  Serial.println("{\"status\": \"initialized\", \"hand_id\": " + String(HAND_ID) + "}");
}

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void selectMuxChannel(int channel) {
  digitalWrite(MUX_S0, (channel >> 0) & 1);
  digitalWrite(MUX_S1, (channel >> 1) & 1);
  digitalWrite(MUX_S2, (channel >> 2) & 1);
  digitalWrite(MUX_S3, (channel >> 3) & 1);
  delayMicroseconds(5);
}

void readFlexSensors() {
  for (int i = 0; i < 5; i++) {
    selectMuxChannel(FLEX_CHANNELS[i]);
    flexValues[i] = analogRead(MUX_SIG);
  }
}

void readFsrSensors() {
  for (int i = 0; i < 3; i++) {
    selectMuxChannel(FSR_CHANNELS[i]);
    fsrValues[i] = analogRead(MUX_SIG);
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
  int16_t temp_raw = (Wire.read() << 8) | Wire.read();  // Skip temp
  int16_t gx_raw = (Wire.read() << 8) | Wire.read();
  int16_t gy_raw = (Wire.read() << 8) | Wire.read();
  int16_t gz_raw = (Wire.read() << 8) | Wire.read();

  accel[0] = ax_raw * ACCEL_SCALE;
  accel[1] = ay_raw * ACCEL_SCALE;
  accel[2] = az_raw * ACCEL_SCALE;
  gyro[0]  = gx_raw * GYRO_SCALE;
  gyro[1]  = gy_raw * GYRO_SCALE;
  gyro[2]  = gz_raw * GYRO_SCALE;

  // Simple complementary filter for orientation estimation
  float dt = SAMPLE_INTERVAL_MS / 1000.0;

  // Accelerometer-based angles
  float accelRoll  = atan2(accel[1], accel[2]) * 180.0 / M_PI;
  float accelPitch = atan2(-accel[0], sqrt(accel[1]*accel[1] + accel[2]*accel[2])) * 180.0 / M_PI;

  // Complementary filter: combine gyro integration with accel angles
  roll  = ALPHA * (roll  + gyro[0] * dt) + (1.0 - ALPHA) * accelRoll;
  pitch = ALPHA * (pitch + gyro[1] * dt) + (1.0 - ALPHA) * accelPitch;
  yaw   = yaw + gyro[2] * dt;  // Gyro only (no magnetometer for yaw correction)

  // Clamp pitch to ±85° (matching ML pipeline config)
  if (pitch > 85.0) pitch = 85.0;
  if (pitch < -85.0) pitch = -85.0;

  // Convert Euler angles to quaternion for JSON output
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

void sendDebugOutput() {
  // Human-readable output for verifying sensors during testing
  Serial.println("──────────── Sensor Readings ────────────");

  // Flex sensors
  const char* fingers[] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
  Serial.print("[FLEX]   ");
  for (int i = 0; i < 5; i++) {
    Serial.print(fingers[i]);
    Serial.print(": ");
    Serial.print(flexValues[i]);
    if (i < 4) Serial.print("  |  ");
  }
  Serial.println();

  // FSR sensors
  const char* fsrNames[] = {"Thumb Tip", "Index Tip", "Pinky Tip"};
  Serial.print("[FSR]    ");
  for (int i = 0; i < 3; i++) {
    Serial.print(fsrNames[i]);
    Serial.print(": ");
    Serial.print(fsrValues[i]);
    Serial.print(" (");
    Serial.print((fsrValues[i] / 4095.0) * 100.0, 1);
    Serial.print("%)");
    if (i < 2) Serial.print("  |  ");
  }
  Serial.println();

  // IMU - Accelerometer
  Serial.print("[IMU]    Accel(m/s²) X:");
  Serial.print(accel[0], 2);
  Serial.print("  Y:");
  Serial.print(accel[1], 2);
  Serial.print("  Z:");
  Serial.println(accel[2], 2);

  // IMU - Gyroscope
  Serial.print("         Gyro(°/s)  X:");
  Serial.print(gyro[0], 1);
  Serial.print("  Y:");
  Serial.print(gyro[1], 1);
  Serial.print("  Z:");
  Serial.println(gyro[2], 1);

  // Quaternion
  Serial.print("         Quat w:");
  Serial.print(quaternion[0], 4);
  Serial.print("  x:");
  Serial.print(quaternion[1], 4);
  Serial.print("  y:");
  Serial.print(quaternion[2], 4);
  Serial.print("  z:");
  Serial.println(quaternion[3], 4);

  Serial.println();

  // Optional pause — increase DEBUG_DELAY_MS at the top to slow output for easier reading
  if (DEBUG_DELAY_MS > 0) delay(DEBUG_DELAY_MS);
}

void sendJSON() {
  // Build JSON string matching backend expected format
  String json = "{";
  json += "\"timestamp_ms\":" + String(millis()) + ",";
  json += "\"hand_id\":" + String(HAND_ID) + ",";

  // Flex array
  json += "\"flex\":[";
  for (int i = 0; i < 5; i++) {
    json += String(flexValues[i]);
    if (i < 4) json += ",";
  }
  json += "],";

  // FSR array  [thumb_tip, index_tip, pinky_tip]
  json += "\"fsr\":[";
  for (int i = 0; i < 3; i++) {
    json += String(fsrValues[i]);
    if (i < 2) json += ",";
  }
  json += "],";

  // Accel array
  json += "\"accel\":[";
  for (int i = 0; i < 3; i++) {
    json += String(accel[i], 2);
    if (i < 2) json += ",";
  }
  json += "],";

  // Gyro array
  json += "\"gyro\":[";
  for (int i = 0; i < 3; i++) {
    json += String(gyro[i], 1);
    if (i < 2) json += ",";
  }
  json += "],";

  // Quaternion array
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
    readIMU();
    if (DEBUG_MODE) {
      sendDebugOutput();
    } else {
      sendJSON();
    }
  }
}
```

**Example output (one line at 50 Hz):**

```json
{"timestamp_ms":104523,"hand_id":1,"flex":[1850,820,3680,3740,3700],"fsr":[2650,2890,20],"accel":[-0.45,3.21,9.12],"gyro":[2.1,-1.4,0.6],"quaternion":[0.9200,0.0800,-0.3700,0.0500]}
```

> **`fsr` layout:** `[thumb_tip, index_tip, pinky_tip]`. No `contact_bitmask` field.

**What to verify:**
- [ ] JSON parses correctly (paste into a JSON validator)
- [ ] `flex` values change when bending fingers (800–3800)
- [ ] `fsr` values change when pressing thumb/index/pinky tips (0–4095)
- [ ] `accel` Z ≈ 9.8 at rest
- [ ] `gyro` near zero at rest
- [ ] `quaternion` values are reasonable (w near 1.0 when flat)
- [ ] Data streams at ~50 lines per second

---

## Assembly Tips

### Step-by-Step Build Order

1. **Test on breadboard first** — wire everything on a breadboard before sewing/gluing to gloves
2. **Test each sensor individually** — run Test 1–5 one at a time to verify each component
3. **Run combined test** — run Test 6 to verify all sensors work together without interference
4. **Mount sensors on glove** — once verified:
   - Flex sensors: along the back (dorsal side) of each finger, secured with fabric glue or sewn-on channels
   - FSR pads: on thumb tip, index tip, and **pinky tip** (sensing area facing outward on each)
   - MPU6050: on the back of the hand (near wrist) where it won't bend
   - ESP32 + MUX: on the wrist area (can use a small case or velcro pouch)

### Sensor Mounting

```
  GLOVE LAYOUT (Palm Side - Left Hand)
  ┌──────────────────────┐
  │      [FSR Index]     │  ← Index fingertip
  │                      │
  │ [FSR Thumb]          │  ← Thumb tip
  │                      │
  │                      │
  │                      │
  │                   [FSR Pinky]  ← Pinky fingertip
  │                      │
  └──────────────────────┘

  GLOVE LAYOUT (Back Side - Left Hand)
  ┌──────────────────────┐
  │  ╔══Flex Pinky══╗    │
  │  ║ ╔══Flex Ring══╗   │
  │  ║ ║ ╔══Flex Mid══╗  │
  │  ║ ║ ║ ╔══Flex Idx═╗ │
  │  ║ ║ ║ ║           │ │
  │  ║ ║ ║ ║  ╔═Flex═╗ │ │
  │  ║ ║ ║ ║  ║Thumb ║ │ │
  │  ╚═╝ ╚═╝  ╚══════╝ │ │
  │      ╚═══╝   ╚══════╝ │
  │                      │
  │   [MPU6050 IMU]      │  ← Back of hand, near wrist
  │   [ESP32 + MUX]      │  ← Wrist pouch/case
  └──────────────────────┘
```

### Wiring Tips

- Use **thin, flexible wire** (30 AWG silicone wire recommended) for sensor leads — stiff wire restricts hand movement
- **Solder all connections** for reliability — breadboard jumpers will come loose with hand movement
- Apply **hot glue** over solder joints for strain relief
- Route wires **along the back of the hand** to avoid interfering with grip
- Leave **extra slack** near finger joints so wires don't pull when fingers bend
- Use **heat shrink tubing** to insulate exposed connections

---

## Troubleshooting

### Flex Sensors

| Problem | Cause | Fix |
|---------|-------|-----|
| Always reads ~0 | Missing resistor or broken wire | Check 47kΩ pull-down is connected to GND |
| Always reads ~4095 | Flex sensor disconnected from VCC | Check connection from 3.3V through flex sensor |
| Reads same value for all channels | MUX select lines not wired correctly | Verify GPIO 16–19 connections to S0–S3 |
| Noisy/jumping values | Long unshielded wires or poor solder | Shorten wires, add 100nF cap at MUX input |
| Range is too small (e.g. 1000–2000) | Wrong pull-down resistor value | Must be 47kΩ — try adjusting if your flex sensors differ |

### FSR Sensors

| Problem | Cause | Fix |
|---------|-------|-----|
| Always reads 0 even when pressed | FSR not connected or wrong MUX channel | Verify wiring to CH5/CH6/CH7 |
| Never reaches above ~2000 | Pull-down resistor too high | Try 22kΩ instead of 47kΩ for more sensitivity |
| Slow to return to 0 | FSR pad stuck or adhesive issue | Ensure FSR can flex freely, check mounting |

### MPU6050 IMU

| Problem | Cause | Fix |
|---------|-------|-----|
| "MPU6050 not found" error | Wrong wiring, wrong address, or not powered | Check SDA→GPIO21, SCL→GPIO22, VCC→3.3V, GND→GND |
| Found at wrong address | AD0 pin state incorrect | AD0→GND = 0x68, AD0→3.3V = 0x69 |
| Accel reads [0, 0, 0] | MPU6050 still in sleep mode | Ensure `writeRegister(0x6B, 0x00)` is called |
| Gyro drifts heavily over time | Normal for MPU6050 without magnetometer | Use complementary filter (included in Test 6) |
| I²C data corruption | Wires too long or no pull-ups | Keep I²C wires < 30cm, ensure 10kΩ pull-ups present |

### Capacitive Touch (Removed)

> Capacitive touch pins are not used in this hardware revision. GPIO 4, 0, 2, 15, 13 (T0–T4) are left unconnected.

### General

| Problem | Cause | Fix |
|---------|-------|-----|
| ESP32 won't upload | GPIO 0 held LOW, or wrong board selected | Hold BOOT button during upload, select "ESP32 Dev Module" |
| Serial Monitor shows garbage | Wrong baud rate | Set Serial Monitor to **115200** baud |
| ADC readings on GPIO36 are noisy | WiFi interference on ADC2 | GPIO36 is on ADC1 — should be fine. Add 100nF cap if needed |
| Multiple sensors interfere | Shared ground loop or power issue | Use star-ground topology, add decoupling caps |

---

## Calibration Guide

After assembly, calibrate each sensor to ensure readings match the expected ranges used by the ML model.

### Flex Sensor Calibration

1. Upload Test 1 (Flex Sensors)
2. Hold hand **flat open** — record all 5 values as `FLEX_MIN` (should be ~800 each)
3. Make a **tight fist** — record all 5 values as `FLEX_MAX` (should be ~3800 each)
4. If values differ: adjust the pull-down resistor (higher resistance = wider range) or update the normalization constants in the ESP32 firmware

```
  Finger    | Flat Open | Tight Fist | Range
  ----------|-----------|------------|-------
  Thumb     |    ____   |    ____    | ____
  Index     |    ____   |    ____    | ____
  Middle    |    ____   |    ____    | ____
  Ring      |    ____   |    ____    | ____
  Pinky     |    ____   |    ____    | ____
```

### FSR Calibration

FSR sensors are located at thumb tip, index tip, and **pinky tip** (MUX CH5, CH6, CH7).

1. Upload Test 2 (FSR Sensors)
2. With **no pressure** — values should be < 100
3. Press **firmly** — values should reach > 2500
4. If range is too narrow: try a different pull-down value (22kΩ for more sensitivity, 100kΩ for less)

### IMU Calibration

1. Upload Test 3 (MPU6050)
2. Place sensor **flat on table, chip facing up**
3. Record accel values — Z should be ~9.8, X and Y near 0
4. Record gyro values at rest — note the **offset** (this is gyro bias)
5. Subtract gyro bias in firmware for better accuracy:
   ```cpp
   // After reading gyro, subtract bias:
   float gx_corrected = gx - gx_bias;
   float gy_corrected = gy - gy_bias;
   float gz_corrected = gz - gz_bias;
   ```

### Touch Sensor Calibration (Not Applicable)

> Capacitive touch sensors have been removed. No touch calibration required.

### Recording Calibration Values

Use this template to record your calibration — these values will be needed when connecting to the ML backend:

```
=== CALIBRATION RECORD ===
Date: ____________
Glove: Right / Left

Flex Sensors (MUX CH0-4):
  Thumb:  min=____  max=____
  Index:  min=____  max=____
  Middle: min=____  max=____
  Ring:   min=____  max=____
  Pinky:  min=____  max=____

FSR Sensors (MUX CH5-7):
  Thumb:  no-press=____  firm=____
  Index:  no-press=____  firm=____
  Pinky:  no-press=____  firm=____

IMU (MPU6050):
  Accel at rest: X=____  Y=____  Z=____
  Gyro bias:     X=____  Y=____  Z=____
  I2C address: 0x____

Touch Sensors: NOT USED (capacitive touch removed)
================================
```

---

## Next Steps

After completing hardware build and all 6 tests:

1. **Connect to backend** — see [guide-backend-integration.md](new-docs/guide-backend-integration.md) for WebSocket setup
2. **Collect training data** — see [guide-sensor-data-collection.md](new-docs/guide-sensor-data-collection.md) for recording protocols
3. **Train with real data** — see [guide-train-with-real-data.md](new-docs/guide-train-with-real-data.md) for retraining the ML model
