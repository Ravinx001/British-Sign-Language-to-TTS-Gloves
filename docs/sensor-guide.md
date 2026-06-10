# Sensor Guide — BSL Sign Language Gloves

## 1. Flex Sensors (×5 per hand)

**Purpose:** Measure how much each finger is bent/curled. A resistive strip whose resistance increases as it bends. Mounted along the back of each finger.

- **Straight finger** → low resistance → low ADC value (~800)
- **Fully curled finger** → high resistance → high ADC value (~3800)

```
[FLEX] Thumb:  820  Index: 3650  Middle: 3720  Ring: 3690  Pinky: 3710   // Fist
[FLEX] Thumb:  810  Index:  830  Middle:  825  Ring:  815  Pinky:  840   // Flat open hand
```

---

## 2. IMU — MPU6050 (×1 per hand)

**Purpose:** Tracks palm orientation (which way the hand faces) and hand movement (acceleration during dynamic signs). Outputs 3-axis accelerometer + 3-axis gyroscope. Critical for distinguishing signs that have the same finger shape but different palm orientation or involve motion (e.g., waving for "Hello").

- **Accelerometer** — gravitational + linear acceleration (m/s²). At rest, one axis reads ~9.8.
- **Gyroscope** — rotational velocity (°/s). Near zero when still.

```
[IMU] Accel X: -0.32  Y: 0.18  Z: 9.74  | Gyro X: 1.2  Y: -0.8  Z: 0.3    // Hand at rest, palm down
[IMU] Accel X:  2.15  Y: 8.40  Z: 4.12  | Gyro X: 45.3  Y: -12.1  Z: 8.7   // Hand mid-wave (dynamic sign)
```

---

## 3. FSR — Force Sensitive Resistors (×3 per hand)

**Purpose:** Measure fingertip pressure. Placed on the thumb fingertip, index fingertip, and pinky fingertip of each hand. In BSL fingerspelling, one hand often presses against the other — the FSR detects *how hard* and *whether* contact is being made. The pinky FSR is particularly useful for discriminating BSL letters where the index finger presses on the left pinky tip (e.g., BSL "U"). This distinguishes "lightly touching" from "pressing firmly," which differs between certain BSL letters.

- **No pressure** → high resistance → low ADC (~0–100)
- **Firm press** → low resistance → high ADC (~2500–4095)

```
[FSR] R_Thumb: 2780  R_Index: 3100  R_Pinky:  20  L_Thumb:  45  L_Index:  30  L_Pinky: 1200  // Right pressing on left pinky
[FSR] R_Thumb:   20  R_Index:   15  R_Pinky:  10  L_Thumb:  10  L_Index:  12  L_Pinky:   15  // No contact
```

---

## 4. Analog MUX — CD74HC4067 (×1 per hand)

**Purpose:** Not a sensor itself — it's a channel expander. The ESP32 has limited ADC pins, but each glove needs 8 analog readings (5 flex + 3 FSR). The MUX routes 16 analog inputs through a single ADC pin, selected by 4 digital GPIO lines. It cycles through channels at each sample tick.

No direct serial output — it's transparent to the data. The flex and FSR values above are read *through* the MUX.

> **Note:** The capacitive touch pins (T0–T4) have been removed from the hardware design. Contact detection is now handled solely by the three FSR sensors (thumb, index, pinky) on each hand.

---

## Full Combined Serial Output (one sample at 50 Hz)

This is what a complete frame from **one glove** looks like at a single timestamp:

```
--- [RIGHT HAND] t=104523ms ---
[FLEX]  Thumb: 1850  Index:  820  Middle: 3680  Ring: 3740  Pinky: 3700
[FSR]   Thumb: 2650  Index: 2890  Pinky:  20
[IMU]   Ax: -0.45  Ay: 3.21  Az: 9.12  |  Gx:  2.1  Gy: -1.4  Gz:  0.6
[QUAT]  W: 0.92  X: 0.08  Y: -0.37  Z: 0.05

--- [LEFT HAND] t=104525ms ---
[FLEX]  Thumb:  830  Index:  810  Middle:  825  Ring:  840  Pinky:  820
[FSR]   Thumb:  350  Index:  410  Pinky: 1200
[IMU]   Ax:  0.12  Ay: -0.08  Az: 9.78  |  Gx:  0.3  Gy:  0.5  Gz: -0.2
[QUAT]  W: 0.99  X: 0.01  Y: -0.02  Z: 0.03
```

**Reading the example above:** Right hand has index finger extended (flex ~820), other fingers curled (~3700), pressing thumb and index onto the left hand (FSR ~2650/2890). Left hand is flat and open (all flex ~820), palm facing up (Az ≈ 9.8). This pattern — right index pointing into left open palm — corresponds to **BSL letter "A"**.

---

## JSON Packet Sent to Backend (WebSocket)

```json
{
  "timestamp_ms": 104523,
  "hand_id": 1,
  "flex": [1850, 820, 3680, 3740, 3700],
  "fsr": [2650, 2890, 20],
  "accel": [-0.45, 3.21, 9.12],
  "gyro": [2.1, -1.4, 0.6],
  "quaternion": [0.92, 0.08, -0.37, 0.05]
}
```

> **`fsr` array layout:** `[thumb_tip, index_tip, pinky_tip]`. The capacitive `contact_bitmask` field has been removed.

---

## Summary Table

| Sensor | What it measures | Range | Key insight |
|---|---|---|---|
| **Flex** | Finger bend angle | 800 (straight) – 3800 (curled) | Primary handshape feature |
| **IMU Accel** | Hand orientation + movement | ±4g (~±39 m/s²) | Palm facing direction, motion detection |
| **IMU Gyro** | Rotation speed | ±500 °/s | Detects waving, twisting, nodding motions |
| **FSR** | Fingertip pressure (thumb, index, pinky) | 0 (none) – 4095 (max press) | Inter-hand contact force; pinky FSR helps distinguish BSL letters U, E |
