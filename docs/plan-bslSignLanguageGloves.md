## Plan: ESP32 BSL Sign Language Recognition Gloves with ML

**TL;DR** — Build a 4-tier system: (1) Two ESP32 sensor gloves (one per hand) each reading 5 flex sensors, 1 IMU, and 3 FSR pressure sensors (thumb tip, index tip, pinky tip), (2) Python FastAPI backend with SQLite for data collection and model serving, (3) Two-stage ML pipeline — scikit-learn Random Forest/XGBoost for static BSL fingerspelling poses + TensorFlow CNN-LSTM for dynamic BSL signs, (4) React dashboard displaying recognized signs as live text. BSL uses a **two-handed** manual alphabet (unlike ASL), so both gloves are mandatory. Start with Phase 1 (backend + ML + simulator, no hardware needed), then integrate hardware incrementally. Total BOM ~$85–130. Initial target: 26 BSL fingerspelling letters + 20 common signs.

---

### Sensor Hardware (~$96 per pair of gloves)

| Sensor | Module | Qty | Interface | Est. Cost | Purpose |
|---|---|---|---|---|---|
| **Flex sensors** (finger bend) | Generic 2.2″ resistive flex | 10 (5/hand) | Analog → MUX → ADC1 | $20 | Finger curl angle — primary handshape feature |
| **IMU** (orientation + motion) | MPU6050 GY-521 breakout | 2 (1/hand) | I2C (0x68 left, 0x69 right) | $5 | Palm orientation, hand movement trajectory |
| **Force/pressure** (fingertip) | FSR 402 (Interlink) | 6 (thumb+index+pinky, both hands) | Analog → MUX → ADC1 | $24 | Fingertip pressure — inter-hand contact force; pinky FSR helps distinguish BSL letters E/U |
| **Analog MUX** | CD74HC4067 16-ch | 2 (1/hand) | 4 digital GPIO + 1 ADC | $4 | Expands ESP32's 8 ADC1 channels to handle all analog sensors |
| **MCU** | ESP32-DevKitC V4 | 2 (1/hand) | — | $10 | One per glove — avoids long inter-hand wiring |
| **Gloves** | Thin lycra/cycling gloves | 2 | — | $10 | Sensor mounting substrate |
| **Power** | 3.7V 1000mAh LiPo + TP4056 | 2 | — | $12 | Wireless operation (optional — USB for dev) |
| **Misc** | Breadboards, 47kΩ resistors (×8), wires, connectors | 1 lot | — | $10 | Voltage dividers for flex sensors + wiring |
| **Total** | | | | **~$87** | Budget: ~$50 (skip FSRs, generic flex) / Premium: ~$190 (BNO055 IMU, SparkFun flex) |

**Wiring per glove:** 5 flex sensors + 3 FSRs → CD74HC4067 MUX channels 0–7 → single ADC1 pin (GPIO 36). MUX select lines on GPIO 16–19. MPU6050 on I2C (GPIO 21 SDA / GPIO 22 SCL). 47kΩ pull-down resistors for each flex sensor voltage divider. Capacitive touch pins are not used.

**Sampling rate:** 50 Hz per sensor — captures gesture dynamics (BSL signs last 200ms–2s). Data packet: ~55 bytes/hand/sample → 5.5 KB/s total for both gloves (smaller than original estimate due to removal of contact_bitmask field).

---

### Steps

**Phase 1 — Backend + Dashboard + Simulator + ML Pipeline (no hardware)**

1. Scaffold project in `sign-language-gloves/` with subdirs: `firmware/`, `backend/`, `frontend/`, `ml/`, `scripts/`, `docs/`, `data/`

2. Create FastAPI backend in `backend/app/main.py` with SQLAlchemy ORM against SQLite. Tables: `users` (calibration profiles), `sensor_readings` (timestamped raw frames from both gloves), `signs` (BSL sign dictionary — name, type static/dynamic, description), `training_samples` (labeled sensor sequences + sign_id), `predictions` (timestamp, predicted_sign, confidence, model_version), `sessions` (recording sessions for grouping data). Define Pydantic schemas for the dual-glove JSON data packet: `{timestamp_ms, hand_id, flex[5], fsr[3], accel[3], gyro[3], quaternion[4]}` — note: `fsr[3]` is `[thumb_tip, index_tip, pinky_tip]`; capacitive touch (`contact_bitmask`) is not present

3. Implement API endpoints:
   - `POST /api/v1/readings` — ingest real-time sensor frames from both ESP32s
   - `POST /api/v1/calibrate/{user_id}` — start/store per-user sensor calibration (fist → flat hand → spread)
   - `POST /api/v1/training/record` — start a labeled recording session (sign_id + raw frames)
   - `GET /api/v1/training/samples` — list collected training data per sign
   - `POST /api/v1/ml/train` — trigger model training, return accuracy metrics
   - `GET /api/v1/predict/latest` — get current sign prediction + confidence
   - `WS /api/v1/predict/stream` — WebSocket for real-time prediction streaming
   - `CRUD /api/v1/signs` — manage BSL sign dictionary
   - `GET /api/v1/signs/dictionary` — return all target signs with descriptions

4. Build `scripts/simulate_gloves.py` — sends synthetic dual-glove sensor readings at 50 Hz to the API, cycling through BSL fingerspelling letters and common signs. Simulate realistic flex sensor curves: fist (all flex high ~3800), flat hand (all flex low ~800), pointing (index low, rest high), "V" sign (index+middle low, rest high). Add Gaussian noise (σ=3% of range) and IMU orientation variation

5. Build `ml/bsl_sign_definitions.py` — define the target sign set with expected sensor signatures:
   - **Phase 1 signs (26 BSL fingerspelling letters)**: All static poses, both hands. Example: BSL "A" = right index finger extended pointing at left open palm; "B" = right hand flat against left palm; "C" = both hands curved
   - **Phase 2 signs (20 common BSL words)**: Hello (wave), Thank you, Yes (nod fist), No (finger wag), Please, Sorry, Help, Good, Bad, Name, What, Where, When, I/Me, You, Drink, Eat, More, Stop, Understand

6. Build `ml/synthetic_data.py` — generate 5,000+ labeled training samples per sign based on the BSL sign definitions. For each sign: define the "ideal" sensor vector → generate N samples with Gaussian noise, slight orientation variation, and timing jitter. Apply data augmentation: amplitude scaling (±8%), additive noise (σ=5%), time warping (±15% for dynamic signs), sensor dropout (randomly zero 1 sensor with 5% probability)

7. Build the **two-stage ML training pipeline** in `ml/train.py`:
   - **Stage 1 — Static Pose Classifier** (fingerspelling + static signs): scikit-learn `Pipeline` → `ColumnTransformer` (StandardScaler) → `GradientBoostingClassifier` (XGBoost). Features per frame: 10 normalized flex values + 6 FSR values (3 per hand: thumb, index, pinky) + 6 orientation angles (roll/pitch/yaw per hand) + 4 derived features (hand openness L/R = mean flex, inter-hand orientation delta roll/pitch) = **26 features**. Train/test split 80/20, stratified
   - **Stage 2 — Dynamic Gesture Classifier** (moving signs): TensorFlow/Keras CNN-LSTM. Input: 100 timesteps × 26 features (2-second window at 50 Hz). Architecture: Conv1D(64, k=5) → BatchNorm → MaxPool → Conv1D(128, k=3) → BatchNorm → MaxPool → LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → Dense(64, relu) → Dense(n_classes, softmax). Loss: categorical crossentropy, optimizer: Adam(lr=0.001)
   - **Sign Segmentation**: Detect sign boundaries using a motion energy threshold — when IMU acceleration magnitude exceeds threshold, start recording; when it drops below for 500ms, end recording. Feed segment to appropriate classifier
   - **Anomaly/Unknown Detector**: If max softmax confidence < 0.6, classify as "Unknown" — prevents false positives

8. Build prediction service in `backend/app/services/prediction_service.py`:
   - Maintain a rolling buffer of the last 100 frames (2s) for both hands
   - On each incoming frame: update buffer → compute static features from latest frame → run Stage 1 classifier → if confidence > 0.85, emit static prediction
   - Simultaneously: detect motion segments → when segment complete, extract feature sequence → run Stage 2 CNN-LSTM → if confidence > 0.6, emit dynamic prediction
   - Apply **debouncing**: same sign must be predicted for 3+ consecutive frames (60ms) before emitting, to avoid flicker
   - Build recognized text as a running string: fingerspelled letters accumulate into words, word signs append directly, 2s pause = space

9. Create React frontend (Vite + TypeScript + Tailwind CSS + Recharts + TanStack Query):
   - **Live Recognition** page: Large text display area showing recognized signs/words in real-time (WebSocket), current sign prediction with confidence bar, "sentence so far" accumulator with clear/backspace buttons
   - **Sensor Monitor** page: 10 flex sensor gauges (5 per hand), IMU orientation 3D hand visualization (Three.js or simple pitch/roll/yaw indicators), contact point indicators, FSR pressure bars — useful for debugging sensor issues
   - **Training** page: Sign selector dropdown, "Record" button (3-2-1 countdown → record 3 seconds → label → save), sample counter per sign, progress bar toward training data goal (30 samples/sign minimum)
   - **Dictionary** page: BSL sign reference with descriptions and expected hand shapes for each target sign
   - **Model** page: Training trigger button, accuracy metrics display (per-sign confusion matrix, overall accuracy, F1 scores), model version history

**Phase 2 — ESP32 Firmware + Single Glove (Dominant Hand)**

10. Set up PlatformIO project in `firmware/` targeting `esp32dev`. Install libraries: `Wire`, `MPU6050_light` or `Adafruit_MPU6050`, `ArduinoJson`, `WiFi`, `WebSocketsClient`. Note: capacitive touch library not required — touch pins are not used.

11. Implement sensor drivers as modular files:
    - `sensors/flex_mux.cpp` — CD74HC4067 multiplexer driver: set S0–S3 GPIO pins → read ADC → cycle through 8 channels (5 flex + 3 FSR) at each sample tick
    - `sensors/imu.cpp` — MPU6050 initialization (gyro ±500°/s, accel ±4g, DLPF 42Hz), read accel[3] + gyro[3], optionally compute quaternion via Madgwick filter
    - `sensors/calibration.cpp` — On-boot calibration routine: prompt user (via Serial) to make fist → record max flex values, then spread hand → record min flex values. Store in NVS (non-volatile storage)

12. Implement `network/wifi_stream.cpp` — connect to WiFi, open WebSocket to backend, JSON-serialize sensor packet, send at 50 Hz. Fallback: buffer to SPIFFS if WiFi drops, flush when reconnected

13. Main loop: read all sensors → normalize using calibration → build JSON packet → send via WebSocket → 20ms delay (50 Hz). Config stored in `config.h`: WiFi credentials, backend URL, sample rate, calibration values

**Phase 3 — Second Glove + Inter-Hand Features**

14. Replicate firmware to second ESP32 for non-dominant (left) hand. Assign `hand_id=0`. Both gloves stream independently to the backend, which aligns frames by timestamp (±10ms tolerance)

15. Add inter-hand feature computation in the backend: relative orientation difference (Δroll, Δpitch, Δyaw), simultaneous contact detection (correlate touch events across hands within same time window), combined handshape encoding. These features are critical for BSL fingerspelling where one hand acts on the other

16. Inter-hand contact is detected through the FSR sensors on each hand's fingertips (thumb, index, pinky). When the dominant hand's fingertips press onto the non-dominant hand's fingers, the FSR values spike — the backend correlates these pressure readings into "inter-hand contact" features

**Phase 4 — Real Data Collection + Model Refinement**

17. Use the Training page to record 30–50 real samples per sign, per user. Start with the 26 BSL fingerspelling letters (static poses — easiest to record). Have the signer perform each letter naturally with slight variation. Store raw sensor sequences tagged with sign labels

18. Retrain Stage 1 classifier with real data blended with synthetic data (50/50 ratio initially, then shift to 80% real / 20% synthetic as data grows). Evaluate accuracy per letter — expect >90% on fingerspelling with 30+ samples/letter. Identify confused pairs (e.g., BSL "M" vs "N") and record additional samples for those

19. Record dynamic signs (Phase 2 sign set). Each recording: rest → sign → rest (3s window). 50+ samples per sign. Train Stage 2 CNN-LSTM. Apply augmentation (time warp, noise injection) to multiply dataset. Tune hyperparameters: `n_epochs`, `learning_rate`, `dropout_rate`, `lstm_units`

20. Add per-user calibration profiles — different hand sizes and signing styles produce different sensor baselines. The calibration routine (fist/flat/spread) normalizes each user's range. Store calibration in `users` table, apply before feature extraction

**Phase 5 — Refinement + Word/Sentence Assembly**

21. Implement sentence assembly logic: fingerspelled letters accumulate into a word buffer → 1.5s pause between letters triggers word completion → word added to sentence. Dynamic sign predictions insert directly as words. Add common BSL word shortcuts (if a whole word is signed, prefer that over letter-by-letter spelling)

22. Add a confidence-based correction UI: when prediction confidence is 0.6–0.85, show top-3 candidates and let the user tap to correct. Corrected labels are saved back as additional training data (active learning loop)

23. Upgrade IMU from MPU6050 to BNO055 if orientation accuracy is a bottleneck — BNO055 outputs quaternions directly via onboard sensor fusion, eliminating Madgwick/Kalman filter drift issues. Drop-in I2C replacement, only firmware driver change needed

---

### Verification

- **Backend**: Run `pytest tests/` — verify sensor ingestion, training sample storage, prediction endpoints, WebSocket streaming. Run `scripts/simulate_gloves.py` and confirm data flows through to predictions
- **ML Stage 1**: After training on synthetic data, expect >92% accuracy on 26 BSL fingerspelling letters (test set). Inspect confusion matrix for commonly confused letter pairs. After real data collection (30+ samples/letter), target >95%
- **ML Stage 2**: After training CNN-LSTM on 20 dynamic signs with 50+ samples each, target >88% accuracy. Verify motion segmentation correctly isolates sign boundaries
- **Firmware**: Serial monitor output shows plausible flex sensor values (800–3800 range), IMU data (accel ~9.8 m/s² at rest), touch detects contact reliably. Calibration routine produces consistent min/max per user
- **End-to-end**: Wear both gloves → fingerspell "HELLO" → dashboard displays H-E-L-L-O accumulating in real-time within <200ms latency per letter. Sign BSL "Thank you" → dashboard displays "Thank you" with confidence >0.7

---

### Decisions

- **Two ESP32s** (one per glove) over single ESP32 with long inter-hand wires — more reliable, simpler wiring, each glove is a self-contained unit
- **MPU6050** over BNO055 for initial POC — $2 vs $25, adequate with Madgwick filter; upgrade path to BNO055 in Phase 5
- **50 Hz sample rate** — captures BSL gesture dynamics (signs last 200ms–2s) without excessive data (6.2 KB/s)
- **Two-stage ML pipeline** (static classifier + dynamic CNN-LSTM) over single end-to-end model — static poses (fingerspelling) are recognized instantly without waiting for gesture completion; dynamic signs use temporal context
- **Synthetic data first** — entire ML pipeline can be developed and evaluated before any hardware arrives
- **Phase 1 is software-only** — backend + dashboard + ML can be built and tested with `simulate_gloves.py` while hardware BOM (~$96) ships
- **WebSocket** over HTTP polling for sensor streaming — 50 Hz data rate needs persistent connection
- **BSL fingerspelling first** (26 static letters) before dynamic signs — highest accuracy with least training data, validates the full pipeline end-to-end
- **No body-location tracking** in POC — BSL signs at specific body locations (forehead, chin, chest) would require additional sensors (chest-mounted IMU or camera); out of scope for glove-only POC, acknowledged as a limitation
```
