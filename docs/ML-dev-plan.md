# ML Development Plan — BSL Sign Language Gloves

> **Project:** Real-Time British Sign Language Recognition System Using Dual Sensor Gloves and Machine Learning  
> **Module:** IE3044 – Design Project | SLIIT  
> **Document Version:** 1.1 (Revised)
> **Date:** March 2026

---

## Executive Summary

This document defines the finalized, phase-by-phase ML development plan for a wearable dual-glove BSL-to-text recognition system. The system uses 5 flex sensors, 1 IMU (MPU6050), and 3 FSR pressure sensors (thumb tip, index tip, pinky tip) per hand — producing ~28 raw values per frame at 50 Hz, engineered down to **26 ML features**. Capacitive touch pins have been removed from the hardware. A two-stage pipeline (XGBoost for 24 static fingerspelling letters + CNN-LSTM for 2 dynamic letters and 20 common signs) targets ≥89% static accuracy and ≥88% dynamic accuracy. Development follows an incremental five-phase approach, starting fully software-based with synthetic data and progressively integrating hardware, real data, and refinement.

No public BSL sensor-glove dataset exists. All training begins with synthetic data generated from BSL SignBank definitions.

---

## Table of Contents

1. [Sensor Stack & Feature Space](#1-sensor-stack--feature-space)
2. [36 Engineered ML Features](#2-36-engineered-ml-features)
3. [ML Architecture — Two-Stage Pipeline](#3-ml-architecture--two-stage-pipeline)
4. [Synthetic Data Strategy](#4-synthetic-data-strategy)
5. [Phase 1 — ML Pipeline (Software-Only)](#5-phase-1--ml-pipeline-software-only)
6. [Phase 2 — Single Glove Firmware & Integration](#6-phase-2--single-glove-firmware--integration)
7. [Phase 3 — Dual Glove Integration & Inter-Hand Features](#7-phase-3--dual-glove-integration--inter-hand-features)
8. [Phase 4 — Real Data Collection & Model Refinement](#8-phase-4--real-data-collection--model-refinement)
9. [Phase 5 — End-to-End Refinement & Deployment](#9-phase-5--end-to-end-refinement--deployment)
10. [Success Metrics & KPIs](#10-success-metrics--kpis)
11. [Risk Register & Mitigation](#11-risk-register--mitigation)
12. [Project Directory Structure](#12-project-directory-structure)
13. [Technology Stack](#13-technology-stack)
14. [Timeline & Milestones](#14-timeline--milestones)

---

## 1. Sensor Stack & Feature Space

Each glove produces raw sensor data at **50 Hz** (20 ms sample interval). A dual-glove pair yields ~38 raw values per frame:

| Sensor | Per Hand | Total (L+R) | Raw Range | ML Role |
|---|---|---|---|---|
| Flex ×5 | 5 analog values | 10 | 800–3800 (ADC) | Primary handshape — finger curl angle |
| IMU Accel ×3 | 3-axis acceleration | 6 | ±39 m/s² | Palm orientation, motion detection |
| IMU Gyro ×3 | 3-axis rotation rate | 6 | ±500 °/s | Waving, twisting, trajectory |
| FSR ×3 (thumb/index/pinky) | 3 pressure values | 6 | 0–4095 (ADC) | Inter-hand contact force; pinky FSR helps distinguish U/E |
| Quaternion | 4 values | 8 | Unit quaternion (±1.0) | Precise 3D orientation |

> **Hardware note:** Capacitive touch pins (T0–T4) are removed. FSR positions: thumb tip, index tip, pinky tip. No `contact_bitmask` in the data packet.

**Data throughput:** ~55 bytes/hand/sample → 5.5 KB/s total (both gloves) → manageable over WiFi WebSocket.

---

## 2. 26 Engineered ML Features

Raw sensor values are transformed into a fixed 26-dimensional feature vector per frame. Capacitive touch (10 features) has been removed; the palm pad FSR has been relocated to the pinky fingertip:

| # | Feature | Source | Transform |
|---|---|---|---|
| 1–5 | `R_flex_thumb..pinky` | Right flex[5] | Min-max normalize: `(raw - 800) / 3000` → [0, 1] |
| 6–10 | `L_flex_thumb..pinky` | Left flex[5] | Same normalization |
| 11–13 | `R_fsr_thumb, R_fsr_index, R_fsr_pinky` | Right FSR[0:3] | `raw / 4095` → [0, 1] |
| 14–16 | `L_fsr_thumb, L_fsr_index, L_fsr_pinky` | Left FSR[0:3] | Same |
| 17–19 | `R_roll, R_pitch, R_yaw` | Right quaternion[4] | Quaternion → Euler angles (degrees) |
| 20–22 | `L_roll, L_pitch, L_yaw` | Left quaternion[4] | Same |
| 23 | `R_hand_openness` | mean(R_flex_norm) | Derived: 0 = fully open, 1 = fist |
| 24 | `L_hand_openness` | mean(L_flex_norm) | Same |
| 25 | `inter_hand_delta_roll` | \|R_roll − L_roll\| | Relative palm orientation |
| 26 | `inter_hand_delta_pitch` | \|R_pitch − L_pitch\| | Same (pitch axis) |

> **Note — Quaternion → Euler gimbal lock:** At ±90° pitch, Euler representation loses a degree of freedom. Mitigation: clamp pitch to ±85° in feature engineering; monitor signs requiring extreme wrist angles (P, Q) for degraded orientation data. If problematic, switch to rotation matrix (adds 3 features) in Phase 4.

### Feature Discrimination Rationale

- **Flex (10 features):** Separates ~60% of BSL alphabet alone — dominant discriminator.
- **FSR (6 features):** All 3 per hand. Pinky FSR helps separate BSL "U" (R index on L pinky) and "E" (R index on L index) from the confusable I/O group. Thumb/index FSR detect broader contact pressure. Without touch pins, FSR is now the primary contact sensor.
- **Orientation (6 features):** Distinguishes letters with identical flex but different palm facing (e.g., C vs O, P vs Q).
- **Derived (4 features):** Higher-level aggregates capturing overall hand state and relative positioning.

> **Accuracy note:** Removing capacitive touch drops baseline accuracy from ~97.4% to ~89–91% (ablation study). The primary remaining confusion is between BSL letters I and O (no FSR on L middle/ring). The pinky FSR recovers E/U discrimination. See `docs/new-docs/pad_to_pinky_fsr_analysis.md` for full impact analysis.

---

## 3. ML Architecture — Two-Stage Pipeline

### 3.1 Pipeline Flow

```
Sensor Frame (50 Hz, 36 features)
  │
  ├─ Compute Motion Energy: ME = sqrt((ax - gx)² + (ay - gy)² + (az - gz)²)
  │   where g = estimated gravity vector from low-pass filtered accel
  │
  ├─ ME ≤ 2.5 m/s² ──────────────────► STAGE 1: XGBoost (Static Classifier)
  │                                        │
  │                                        ├─ Confidence ≥ 0.85 → Debounce (3 frames / 60ms) → Emit sign
  │                                        └─ Confidence < 0.85 → Hold / "Unknown"
  │
  └─ ME > 2.5 for 3+ frames ────────────► Buffer 100 frames (2s window)
                                           │
                                           ► STAGE 2: CNN-LSTM (Dynamic Classifier)
                                              │
                                              ├─ Confidence ≥ 0.60 → Debounce → Emit sign
                                              └─ Confidence < 0.60 → "Unknown"
```

### 3.2 Stage 1 — XGBoost Static Pose Classifier

Handles 24 static BSL fingerspelling letters (all except J, Z which involve motion).

| Aspect | Specification |
|---|---|
| Algorithm | XGBoost (Gradient Boosted Trees) |
| Input | Single frame × **26 features** |
| Output | 24-class probability (+ expandable to static common signs) |
| Inference | < 1 ms per frame |
| Confidence threshold | ≥ 0.85 to emit prediction |

**Hyperparameter search space (5-fold GridSearchCV):**

| Parameter | Range |
|---|---|
| `n_estimators` | 200, 300, 500 |
| `max_depth` | 4, 6, 8 |
| `learning_rate` | 0.05, 0.1, 0.15 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 3 |
| `reg_alpha` (L1) | 0.1 |
| `reg_lambda` (L2) | 1.0 |

**Baselines trained in parallel:** Random Forest, SVM (RBF kernel). Best model selected on validation F1.

### 3.3 Stage 2 — CNN-LSTM Dynamic Gesture Classifier

Handles 2 dynamic fingerspelling letters (J, Z) plus 20 common BSL signs in later phases.

| Aspect | Specification |
|---|---|
| Architecture | Conv1D → BatchNorm → MaxPool → Conv1D → BatchNorm → MaxPool → LSTM → LSTM → Dense |
| Input shape | **(100, 26)** — 100 timesteps × 26 features (2s at 50 Hz) |
| Output | n-class softmax (initially 2, expanding to 22) |
| Inference | < 10 ms per segment |
| Confidence threshold | ≥ 0.60 to emit prediction |

**Layer specification:**

```
Input(100, 26)
  → Conv1D(64, kernel=5, relu, padding=same)
  → BatchNormalization
  → MaxPooling1D(pool=2)            # → 50 steps
  → Conv1D(128, kernel=3, relu, padding=same)
  → BatchNormalization
  → MaxPooling1D(pool=2)            # → 25 steps
  → LSTM(128, return_sequences=True)
  → Dropout(0.3)
  → LSTM(64)
  → Dropout(0.3)
  → Dense(64, relu)
  → Dropout(0.2)
  → Dense(n_classes, softmax)
```

**Training config:** Adam(lr=0.001), `categorical_crossentropy`, EarlyStopping(patience=10), ReduceLROnPlateau(factor=0.5, patience=5), batch_size=32, max 100 epochs.

> **Phase 1 note (2-class risk):** Training CNN-LSTM on only J and Z (2 classes) risks severe overfitting despite regularization. Mitigation: in Phase 1, also generate synthetic dynamic sequences for 4–6 "distractor" gesture classes (random hand waves, scratching, adjusting gloves) to teach the model meaningful temporal discrimination. This grows the dynamic class set to ~8 without requiring real BSL sign definitions, and the distractor classes are dropped when real dynamic signs are added in Phase 4.

### 3.4 Motion Segmentation

Sign boundaries are detected via **gravity-compensated** motion energy thresholding:

- **Gravity removal:** Apply a low-pass filter (α=0.1) on raw accelerometer to estimate gravity vector. Subtract gravity from raw accel to get linear acceleration. This prevents false motion detection from hand tilting (raw accel magnitude is always ~9.8 m/s² even at rest).
- **Motion energy:** `ME = sqrt(linear_ax² + linear_ay² + linear_az²)` — near 0 at rest, spikes during actual hand movement.
- **Start recording:** `ME > 2.5 m/s²` sustained for 3+ consecutive frames.
- **End recording:** `ME < 2.5 m/s²` sustained for 25 frames (500 ms cooldown).
- **Segment extraction:** Pad/truncate captured window to exactly 100 frames → feed to CNN-LSTM.

> **Why the threshold changed from earlier drafts:** Previous drafts used `ME = sqrt(ax² + ay² + az²) > 12.0`, but this formula **includes gravity** (~9.8 m/s²), so the threshold barely exceeded the resting value — making detection unreliable and orientation-dependent. Gravity-compensated ME starts near 0 at rest, making a 2.5 m/s² threshold robust and orientation-independent.

### 3.5 Unknown Rejection & Debouncing

- If `max(softmax_output) < threshold` → classify as **"Unknown"** — prevents false positive emissions.
- **Debouncing:** Same sign must be predicted for 3+ consecutive frames (60 ms) before emitting output — eliminates prediction flicker.

---

## 4. Synthetic Data Strategy

### 4.1 Justification

| Dataset Evaluated | Type | Rejection Reason |
|---|---|---|
| BOBSL / BSL-1K | Video-based BSL | Vision data — no flex/FSR/IMU readings |
| BSL Corpus | Video-based BSL | Linguistic annotated video — different modality |
| UCI Sign Language | Glove sensor (ASL) | ASL one-handed; CyberGlove 22 sensors — different config |
| Kaggle ASL Alphabet | Image-based ASL | Camera images — not sensor data |
| SignFi | WiFi CSI-based | Completely different modality |
| UniMiB SHAR / UCI HAR | Accelerometer only | No finger flex, no touch, no FSR |

**Conclusion:** Zero public datasets exist for BSL two-handed fingerspelling with this sensor configuration. Synthetic generation from BSL SignBank definitions is the only viable path.

### 4.2 Generation Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Samples per sign | 5,000 | Sufficient for tree + NN training at 26 classes |
| Total static samples | 130,000 (5,000 × 26) | Balanced across all letters |
| Gaussian noise | σ = 3% of sensor range | Simulates real ADC noise |
| Amplitude scaling | ±8% random factor | Simulates different hand sizes |
| Baseline shift | N(0, 0.03) additive | Simulates calibration drift |
| Orientation jitter | ±15° per axis | Natural hand position variation |
| Sensor dropout | 5% per-sensor probability | Robustness to occasional sensor failure |
| Time warping (dynamic) | ±15% stretch/compress | Signing speed variation between users |
| Virtual user profiles | 10, each with ±10% baseline offset | Prevents memorizing single calibration |
| **Correlated finger noise** | Covariance matrix (ρ=0.3 adjacent fingers) | Adjacent flex sensors move together when one finger bends — models real cross-talk |
| **Transition frames** | 5–10 interpolated frames prepended/appended per sample | Simulates approach-to-pose and release-from-pose; prevents model from only recognizing "perfect" static holds |

### 4.3 Data Split

- **70% train / 15% validation / 15% test** — stratified by sign label AND virtual user.
- **No user leakage:** Same virtual user never appears in both train and test sets.
- Validation set used for hyperparameter tuning; test set held untouched until final evaluation.

### 4.4 Output File Structure

```
data/
├── raw/synthetic/
│   ├── static_dataset.npz        # 130K samples × 26 features + labels
│   └── dynamic_dataset.npz       # J, Z + distractor temporal sequences (100 frames × 26 features)
├── processed/
│   ├── train.npz                 # 70% stratified
│   ├── val.npz                   # 15% stratified
│   ├── test.npz                  # 15% stratified
│   └── scaler.pkl                # Fitted StandardScaler
└── models/
    ├── xgboost_static_v1.pkl     # Best static classifier
    ├── cnn_lstm_dynamic_v1.h5    # Dynamic classifier (Keras)
    ├── training_report.json      # All metrics, per-sign F1, confusion matrix
    └── feature_importance.json   # SHAP values for interpretability
```

---

## 5. Phase 1 — ML Pipeline (Software-Only)

> **Goal:** Prove that the 36-feature sensor vector distinguishes 26 BSL fingerspelling letters at ≥92% accuracy — before building any backend, frontend, or hardware.  
> **Duration:** ~4 weeks  
> **Hardware required:** None

### Step 1: Project Scaffolding & Configuration

| Task | Output |
|---|---|
| Create directory structure (`ml/`, `data/`, `scripts/`, `docs/`) | Clean project skeleton |
| Define `requirements.txt` | numpy, pandas, scikit-learn, xgboost, tensorflow, matplotlib, seaborn, scipy, shap, mlflow, onnx, tf2onnx |
| Create `ml/config.py` | Sensor ranges, 36 feature names, noise params, hyperparameter grids, class labels, confidence thresholds |
| Set up **MLflow** experiment tracking | `mlflow server` for local tracking; all training runs log params, metrics, artifacts, and model versions automatically |

### Step 2: BSL Sign Definitions

| Task | Output |
|---|---|
| Create `ml/bsl_sign_definitions.py` | Normalized ideal sensor vectors for all 26 letters + static/dynamic flag |
| Cross-reference every letter against BSL SignBank | Verified finger positions, palm orientations, contact points |
| Document key discriminating signals per letter | Feature-level justification for each sign template |

**BSL Fingerspelling Sensor Signature Summary (26 letters):**

| Letter | Right Hand | Left Hand | Key Signals |
|---|---|---|---|
| A | Index extended, others curled | Flat open, palm up | R_index low; L_all low; FSR high |
| B | Flat, pressed against L palm | Flat open | All flex low both; FSR high (contact) |
| C | Curved fingers (C-shape) | Passive/mirroring | R_flex mid (~0.5); L relaxed |
| D | Index curved, thumb at L base | Flat, fingers extended | R_index mid; touch R_thumb+L_index |
| E | Index touches L index tip | Flat open, palm up | R_index low; touch R_index→L_index |
| F | Thumb+index circle, others out | Flat open | R_thumb+R_index mid; others low |
| G | Index points at L palm side | Fist, thumb up | R_index low, others high; L_all high |
| H | Index+middle extended | Flat open | R_index+mid low; others high |
| I | Index touches L middle tip | Flat open, palm up | R_index low; touch at L_middle |
| J | Like I + downward motion | Same as I | Same as I + gyro_Y spike **(dynamic)** |
| K | Index bent, touches L index | Flat, fingers extended | R_index mid; touch R→L_index |
| L | Thumb+index form L | Passive | R_thumb+R_index low; others high |
| M | 3 fingers over L fist | Fist, thumb tucked | R_index/mid/ring low on L; L high |
| N | 2 fingers over L fist | Fist, thumb tucked | R_index/mid low on L; L high |
| O | Index touches L ring tip | Flat open, palm up | R_index low; touch at L_ring |
| P | Index points down at L palm | Flat open, palm up | R_index low; IMU R_pitch inverted |
| Q | Like P but twisted | Flat open | Similar to P + R_roll shifted |
| R | Index+middle crossed | Flat open | R_index low, R_middle slightly higher |
| S | Fist pressed on L palm | Flat open, palm up | R_all high; L_all low; FSR high |
| T | Thumb between index+middle | Flat, thumb visible | Touch from thumb position |
| U | Index touches L pinky tip | Flat open, palm up | R_index low; touch at L_pinky |
| V | Index+middle extended (V) | Passive | R_index+mid low; others high |
| W | Index+mid+ring extended | Passive | R_index/mid/ring low; others high |
| X | Hooked index finger | Flat open | R_index mid-range, bent shape |
| Y | Thumb+pinky extended | Passive | R_thumb+R_pinky low; others high |
| Z | Index traces Z in air | Passive | R_index low; gyro zigzag **(dynamic)** |

> **24 static** + **2 dynamic** (J, Z) letters.

### Step 3: Feature Engineering Module

| Task | Output |
|---|---|
| Build `ml/feature_engineering.py` | `extract_features(raw_frame) → np.array[32]` |
| Implement: flex normalization, FSR normalization, bitmask→binary, quaternion→Euler, derived features | Tested with sample data |
| Unit tests for each transform | Verified edge cases (min/max sensor values, zero quaternion) |

### Step 4: Synthetic Data Generation

| Task | Output |
|---|---|
| Build `ml/synthetic_data.py` | `generate_static_dataset()`, `generate_dynamic_dataset()` |
| Generate 130,000 static samples + J/Z dynamic sequences | Saved as `.npz` |
| Visualize feature distributions per class (histograms, t-SNE) | Verify inter-class separation; adjust definitions if overlap |
| Apply 70/15/15 split with virtual user stratification | `train.npz`, `val.npz`, `test.npz` |

### Step 5: Train Static Classifier (XGBoost)

| Task | Output |
|---|---|
| Build `ml/train_static.py` | XGBoost + RF + SVM pipelines with GridSearchCV; all runs logged to MLflow |
| Run 5-fold cross-validation across hyperparameter grid | Best model saved as `.pkl` + exported to ONNX for portability |
| Evaluate on held-out test set | Accuracy, per-letter P/R/F1 |
| Generate 26×26 confusion matrix | Identify confused letter pairs |
| Compute SHAP feature importance | Understand which sensors drive which predictions |
| **Iterate if accuracy < 92%:** analyze confused pairs → adjust sign definitions → regenerate → retrain | Repeat until target met |

### Step 6: Train Dynamic Classifier (CNN-LSTM)

| Task | Output |
|---|---|
| Build `ml/train_dynamic.py` | CNN-LSTM model with EarlyStopping + LR scheduling; logged to MLflow |
| Train on J/Z + 4–6 distractor gesture classes | Model saved as `.h5` + `.tflite` (quantized for potential edge deployment) |
| Evaluate accuracy and per-class metrics | ≥88% target on J/Z (distractor recall ≥80%) |

### Step 7: Motion Segmentation Module

| Task | Output |
|---|---|
| Build `ml/segmentation.py` | `detect_motion()`, `extract_segment()` functions |
| Test on synthetic motion sequences | Verify correct sign boundary detection (IoU > 0.75) |

### Step 8: Unified Prediction Service

| Task | Output |
|---|---|
| Build `ml/predict.py` | `predict(frame) → (sign, confidence)` |
| Implement: rolling buffer, motion routing, static/dynamic classification, confidence thresholding, debouncing | Full inference pipeline |
| **Ensure single scaler instance** — scaler fitted once during data generation (`scaler.pkl`), loaded by prediction service. Do NOT add a second StandardScaler inside the XGBoost pipeline — this causes double-scaling and degrades accuracy. | Scaler consistency verified |
| Build `scripts/simulate_and_predict.py` | End-to-end test: simulate "HELLO" + "BSL" → verify correct recognition |

### Step 9: Ablation & Robustness Testing

| Task | Output |
|---|---|
| Build `ml/ablation.py` | Feature group removal experiments |
| Remove each sensor group independently → retrain → measure accuracy drop | Degradation curve per sensor type |
| Test at 2× noise level (σ = 6%) | Noise robustness report |
| Document all results | `docs/ml_evaluation_report.md` |

### Phase 1 Deliverables

- [ ] Trained XGBoost static classifier (≥92% accuracy, 26 letters)
- [ ] Trained CNN-LSTM dynamic classifier (≥88% accuracy, J/Z)
- [ ] 130,000+ synthetic training samples with full augmentation
- [ ] Feature engineering pipeline with unit tests
- [ ] Motion segmentation module
- [ ] Unified prediction service passing end-to-end test
- [ ] Ablation study and noise robustness report
- [ ] SHAP-based feature importance analysis
- [ ] MLflow experiment dashboard with all training runs tracked
- [ ] ONNX export of best static model + TFLite export of dynamic model

---

## 6. Phase 2 — Single Glove Firmware & Integration

> **Goal:** Develop ESP32 firmware for the dominant-hand glove. Validate real sensor readings against synthetic assumptions. Connect glove to backend.  
> **Duration:** ~3 weeks  
> **Hardware required:** 1× ESP32, 5× flex sensors, 1× MPU6050, 3× FSR, 1× CD74HC4067 MUX

### Step 1: Firmware Development (PlatformIO)

| Task | Output |
|---|---|
| Set up PlatformIO project targeting `esp32dev` | `firmware/` directory with `platformio.ini` |
| Implement `sensors/flex_mux.cpp` | CD74HC4067 MUX driver — cycle S0–S3 → read 8 ADC channels (5 flex + 3 FSR) |
| Implement `sensors/imu.cpp` | MPU6050 init (gyro ±500°/s, accel ±4g, DLPF 42Hz), accel[3] + gyro[3] + quaternion (Madgwick filter) |
| Implement `sensors/calibration.cpp` | On-boot routine: fist → record max flex, spread → record min flex, store in NVS |

> **Note:** `sensors/touch.cpp` is **not needed** — capacitive touch pins are removed. FSR positions: CH5=thumb tip, CH6=index tip, CH7=pinky tip.

### Step 2: Wireless Streaming

| Task | Output |
|---|---|
| Implement `network/wifi_stream.cpp` | WiFi connect → WebSocket to backend → JSON packets at 50 Hz |
| Implement SPIFFS buffer fallback | Buffer to flash if WiFi drops, flush on reconnect |
| Define `config.h` | WiFi credentials, backend URL, sample rate, calibration defaults |

### Step 3: Backend Integration

| Task | Output |
|---|---|
| Build FastAPI backend (`backend/app/`) | SQLAlchemy + SQLite, Pydantic schemas |
| Implement `POST /api/v1/readings` | Ingest real-time sensor frames |
| Implement `WS /api/v1/predict/stream` | WebSocket for real-time prediction streaming |
| Integrate prediction service from Phase 1 | Live inference on incoming sensor data |

### Step 4: Sensor Validation

| Task | Output |
|---|---|
| Compare real sensor output to synthetic assumptions | Verify flex ranges (800–3800), FSR response, IMU at-rest values |
| Adjust `ml/config.py` sensor ranges if needed | Updated normalization parameters |
| Record 50+ samples of known signs → compare feature vectors to synthetic | Domain gap analysis report |

### Phase 2 Deliverables

- [ ] Working single-glove streaming real sensor data at 50 Hz over WebSocket
- [ ] Calibration routine functional (fist/flat/spread)
- [ ] Backend ingesting and serving predictions on live data
- [ ] Sensor validation report (real vs synthetic gap analysis)

---

## 7. Phase 3 — Dual Glove Integration & Inter-Hand Features

> **Goal:** Add second glove. Implement inter-hand feature computation — critical for BSL's two-handed fingerspelling.  
> **Duration:** ~2 weeks  
> **Hardware required:** 2nd ESP32 + sensor set, conductive fabric patches

### Step 1: Second Glove Deployment

| Task | Output |
|---|---|
| Replicate firmware to second ESP32 (`hand_id = 0`) | Both gloves streaming independently |
| Backend frame alignment by timestamp (±10 ms tolerance) | Synchronized dual-glove data |

### Step 2: Inter-Hand Feature Computation

| Task | Output |
|---|---|
| Compute relative orientation: Δroll, Δpitch, Δyaw between hands | Features 25–26 from real data (renumbered from 36-feature vector) |
| Implement cross-hand FSR correlation | Correlate FSR pressure events across hands at same timestamp |

### Step 3: Model Update

| Task | Output |
|---|---|
| Verify dual-glove feature vector matches the **26-feature** spec | Feature parity check |
| Run existing models on dual-glove data | Baseline accuracy on real two-handed input |

### Phase 3 Deliverables

- [ ] Both gloves streaming and synchronized in backend
- [ ] Inter-hand FSR correlation working across gloves
- [ ] Dual-glove feature vector validated against 26-feature specification
- [ ] Initial real-data accuracy benchmarks

---

## 8. Phase 4 — Real Data Collection & Model Refinement

> **Goal:** Collect real BSL sign samples. Blend with synthetic data and retrain. Achieve production-level accuracy.  
> **Duration:** ~4 weeks  
> **Hardware required:** Dual gloves operational

### Step 1: Training Data Collection

| Task | Output |
|---|---|
| Use web training interface: select sign → countdown → record 3s → label → save | Structured recording workflow |
| Record 30–50 real samples per letter (26 letters) | 780–1,300 real static samples |
| Record 50+ real samples per dynamic sign (20 common signs) | 1,000+ real dynamic samples |
| Multiple users record where possible | Captures inter-user variation |
| **Version all datasets with DVC** (Data Version Control) | Reproducible data lineage — every model links to exact dataset version |

### Step 2: Data Blending & Retraining

| Task | Output |
|---|---|
| Blend real + synthetic data (initial 50/50 ratio) | Combined training set |
| Retrain XGBoost with blended data | Updated `xgboost_static_v2.pkl` |
| Shift to 80% real / 20% synthetic as real data grows | Progressive synthetic phase-out |
| Retrain CNN-LSTM on 20 dynamic signs with augmentation (10× via time warp + noise) | Updated `cnn_lstm_dynamic_v2.h5` |

### Step 3: Confusion Analysis & Targeted Collection

| Task | Output |
|---|---|
| Generate updated confusion matrix | Identify remaining confused pairs (e.g., M/N, D/K) |
| Record 50+ additional samples for each confused pair | Targeted data augmentation |
| Retrain and evaluate | Iterate until ≤ 2 pairs with >10% confusion |

> **Small-dataset evaluation:** With only 30–50 real samples per sign, a single train/test split is unreliable. Use **stratified 5-fold cross-validation** for all Phase 4 evaluations. Report mean ± std accuracy across folds. Only switch to a held-out test set once real samples exceed 100 per sign.

### Step 4: Per-User Calibration

| Task | Output |
|---|---|
| Implement per-user calibration profiles in `users` table | Different hand sizes → different baselines |
| Calibration routine: fist → flat → spread → store min/max per user | Stored in DB, applied before feature extraction |
| Evaluate accuracy across calibration profiles | Per-user accuracy report |

### Phase 4 Deliverables

- [ ] ≥95% accuracy on static fingerspelling with real data
- [ ] ≥90% accuracy on 20 dynamic signs with real data
- [ ] Confusion matrix showing ≤ 2 pairs with >10% confusion
- [ ] Per-user calibration system working
- [ ] Model version v2 artifacts saved

---

## 9. Phase 5 — End-to-End Refinement & Deployment

> **Goal:** Sentence assembly, active learning, UX refinement, IMU upgrade path. Prepare for demonstration.  
> **Duration:** ~3 weeks

### Step 1: Sentence Assembly Logic

| Task | Output |
|---|---|
| Fingerspelled letters accumulate into word buffer | Letter-by-letter → word formation |
| 1.5s pause between letters → word completion | Words joined into sentences |
| Dynamic sign predictions insert directly as words | Whole-word recognition alongside fingerspelling |
| Common BSL word shortcuts (prefer whole-word over letter-by-letter) | Faster recognition for frequent signs |

### Step 2: Active Learning & Correction UI

| Task | Output |
|---|---|
| When confidence is 0.60–0.85: show top-3 candidates in UI | User selects correct sign |
| Corrected labels saved as additional training data | Growing dataset through usage |
| Periodic model retraining with accumulated corrections | Continuously improving accuracy |

### Step 3: Model Optimization & Export

| Task | Output |
|---|---|
| Export final XGBoost model to **ONNX** format | Portable model for any serving framework |
| Convert final CNN-LSTM to **TensorFlow Lite** (int8 quantized) | Enables potential on-device inference on ESP32-S3 in future |
| Benchmark inference latency: Python `.pkl` vs ONNX Runtime vs TFLite | Latency comparison report |
| Archive all model artifacts + training configs in MLflow Model Registry | Versioned, reproducible model lineage |

### Step 4: IMU Upgrade Assessment

| Task | Output |
|---|---|
| Evaluate MPU6050 drift over extended sessions | Quantify orientation drift |
| If drift is problematic: upgrade to BNO055 (onboard sensor fusion) | Drop-in I2C replacement, firmware driver change only |
| Validate accuracy improvement with upgraded IMU | A/B comparison |

### Step 5: End-to-End Testing & Demonstration

| Task | Output |
|---|---|
| Full pipeline test: wear gloves → fingerspell "HELLO" → dashboard shows H-E-L-L-O | < 200ms latency per letter |
| Sign "Thank you" → dashboard shows "Thank you" | Dynamic sign with confidence > 0.7 |
| Multi-user testing (different hand sizes) | Calibration robustness validation |
| **Battery life profiling** — measure ESP32 power draw during 50 Hz streaming | Expected runtime per charge (target: ≥2 hours continuous) |
| Record demonstration video | Project submission artifact |

### Phase 5 Deliverables

- [ ] Sentence assembly producing readable text from mixed fingerspelling + signs
- [ ] Active learning loop operational
- [ ] IMU upgrade assessment complete
- [ ] End-to-end demonstration passing all acceptance criteria
- [ ] ONNX + TFLite optimized model artifacts
- [ ] Battery life / power profiling report
- [ ] Final documentation and presentation

---

## 10. Success Metrics & KPIs

### ML Model Performance

| Metric | Phase 1 Target (Synthetic) | Phase 4 Target (Real Data) |
|---|---|---|
| Static fingerspelling accuracy | **≥ 89%** (26-feature vector, no touch) | ≥ 92% |
| Per-letter precision | ≥ 75% every letter | ≥ 80% every letter |
| Confused letter pairs | ≤ 5 pairs with >10% confusion | ≤ 3 pairs |
| Dynamic sign accuracy (J, Z) | ≥ 88% | ≥ 90% |
| Dynamic sign accuracy (20 signs) | N/A | ≥ 88% |
| Motion segmentation IoU | > 0.75 | > 0.80 |

> **Accuracy note:** Touch removal drops baseline from ~97.4% to ~89–91%. The primary remaining confusion is I vs O (no FSR on L middle/ring fingertip). Accept this as a known limitation or apply contextual spell-correction.

### System Performance

| Metric | Target |
|---|---|
| Static inference latency | < 1 ms per frame |
| Dynamic inference latency | < 10 ms per segment |
| End-to-end recognition latency | < 200 ms per letter |
| Sensor sampling rate | 50 Hz sustained |
| WebSocket throughput | 6.2 KB/s (both gloves) |

### Robustness

| Metric | Target |
|---|---|
| Noise robustness (2× training noise) | ≥ 85% accuracy |
| Single sensor group removal | ≥ 70% accuracy maintained |
| Unknown rejection rate | < 5% false positive emission |

---

## 11. Risk Register & Mitigation

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Synthetic data doesn't match real sensors | High | High | BSL SignBank-sourced definitions; domain gap analysis in Phase 2; fine-tuning in Phase 4 |
| 2 | M/N, D/K letter confusion | Medium | Medium | FSR + touch as discriminating features; targeted extra samples; iterate confusion matrix |
| 3 | CNN-LSTM overfits on 2 dynamic signs | Medium | Low | Dropout 0.3, augmentation, early stopping; expand class count in Phase 4 |
| 4 | Wrong sign template definitions | Medium | High | Cross-reference BSL SignBank for every letter; flag uncertain definitions; validate in Phase 2 |
| 5 | Hand size variation across users | High | Medium | 10 virtual user profiles in synthetic data; per-user calibration routine in Phase 4 |
| 6 | Flex sensor inconsistency (non-linear, unit-to-unit) | Medium | Medium | Per-sensor calibration normalizes readings; ML model tolerates variance |
| 7 | IMU gyroscope drift (MPU6050) | Low | Low | Madgwick filter; signs are < 2s so drift is minimal; BNO055 upgrade path in Phase 5 |
| 8 | WiFi latency/packet loss | Low | Medium | WebSocket persistent connection; SPIFFS buffer fallback on ESP32; debouncing tolerates drops |
| 9 | Sensor cross-talk (adjacent finger flex correlation) | Medium | Low | **Now modeled:** Correlated noise (ρ=0.3) for adjacent fingers in synthetic data generation; validated via ablation |
| 10 | No body-location tracking | Low | Medium | Documented as POC limitation; future work (chest IMU or camera) |
| 11 | Quaternion → Euler gimbal lock at ±90° pitch | Low | Medium | Clamp pitch to ±85° in feature engineering; monitor P/Q letter accuracy; fallback to rotation matrix representation |
| 12 | Double-scaling (scaler in data pipeline + model pipeline) | Medium | High | Single scaler fitted once during data generation; loaded by prediction service; XGBoost pipeline uses passthrough scaler |
| 13 | CNN-LSTM overfitting on 2-class dynamic set | High | Medium | Add 4–6 distractor gesture classes in Phase 1; expand to full 22 classes in Phase 4 |

---

## 12. Project Directory Structure

```
sign_gloves/
├── firmware/                        # ESP32 PlatformIO project (Phase 2+)
│   ├── src/
│   │   ├── main.cpp
│   │   ├── sensors/
│   │   │   ├── flex_mux.cpp         # CD74HC4067 MUX driver
│   │   │   ├── imu.cpp              # MPU6050 driver + Madgwick filter
│   │   │   └── calibration.cpp      # On-boot calibration + NVS storage
│   │   └── network/
│   │       └── wifi_stream.cpp      # WebSocket streaming
│   ├── include/config.h
│   └── platformio.ini
│
├── backend/                         # FastAPI server (Phase 1+)
│   ├── app/
│   │   ├── main.py
│   │   ├── models.py                # SQLAlchemy ORM
│   │   ├── schemas.py               # Pydantic validation
│   │   ├── database.py              # SQLite setup
│   │   ├── routers/
│   │   │   ├── readings.py          # Sensor ingestion
│   │   │   ├── training.py          # Training data management
│   │   │   ├── prediction.py        # Prediction endpoints + WS
│   │   │   ├── signs.py             # BSL sign dictionary
│   │   │   └── calibration.py       # User calibration
│   │   └── services/
│   │       ├── prediction_service.py
│   │       └── feature_engineering.py
│   ├── tests/
│   └── requirements.txt
│
├── ml/                              # ML pipeline (Phase 1 — core)
│   ├── config.py                    # Sensor specs, hyperparams, thresholds
│   ├── bsl_sign_definitions.py      # 26 letter sensor templates
│   ├── feature_engineering.py       # Raw → 36 features
│   ├── synthetic_data.py            # Data generation + augmentation
│   ├── train_static.py              # XGBoost + RF + SVM pipeline
│   ├── train_dynamic.py             # CNN-LSTM pipeline
│   ├── segmentation.py              # Motion energy sign detection
│   ├── predict.py                   # Unified two-stage prediction
│   ├── evaluate.py                  # Metrics, confusion matrices, SHAP
│   ├── ablation.py                  # Feature group removal tests
│   └── models/                      # Saved model artifacts
│
├── scripts/
│   ├── simulate_gloves.py           # Synthetic sensor data at 50 Hz
│   └── simulate_and_predict.py      # End-to-end simulation test
│
├── frontend/                        # React dashboard (Phase 1+)
│   ├── src/pages/
│   │   ├── LiveRecognition.tsx
│   │   ├── SensorMonitor.tsx
│   │   ├── Training.tsx
│   │   ├── Dictionary.tsx
│   │   └── Model.tsx
│   ├── package.json
│   └── vite.config.ts
│
├── data/                            # Datasets
│   ├── raw/synthetic/
│   ├── processed/
│   └── models/
│
├── docs/                            # Documentation
└── general-docs/                    # Project charter, presentations
```

---

## 13. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Firmware** | C++ / PlatformIO / Arduino framework | ESP32 sensor drivers + WiFi streaming |
| **Backend** | Python 3.11+ / FastAPI / SQLAlchemy / SQLite | API, data ingestion, model serving |
| **ML — Static** | scikit-learn / XGBoost / SHAP | Static pose classification + interpretability |
| **ML — Dynamic** | TensorFlow / Keras | CNN-LSTM temporal classification |
| **ML — Data** | NumPy / Pandas / SciPy | Data generation, feature engineering |
| **ML — Viz** | Matplotlib / Seaborn | Confusion matrices, distributions, SHAP plots |
| **ML — Tracking** | MLflow | Experiment tracking, model registry, artifact versioning |
| **ML — Export** | ONNX / TFLite | Portable model formats for serving + potential edge deployment |
| **Data — Versioning** | DVC (Data Version Control) | Dataset lineage tracking, reproducible experiments |
| **Frontend** | React / Vite / TypeScript / Tailwind CSS | Dashboard UI |
| **Frontend — Data** | TanStack Query / Recharts | Data fetching + sensor visualization |
| **Communication** | WebSocket (50 Hz) | Real-time sensor streaming + predictions |
| **Hardware** | ESP32-DevKitC V4, MPU6050, CD74HC4067, flex sensors, FSR 402 | Sensor glove pair (~$96 BOM) |

---

## 14. Timeline & Milestones

| Phase | Milestone | Target Date | Key Deliverable |
|---|---|---|---|
| **Phase 1** | ML Pipeline Complete | April 2026 | Trained models ≥92% static, ≥88% dynamic on synthetic data |
| **Phase 2** | Single Glove Streaming | May 2026 | ESP32 firmware + backend ingestion + live inference |
| **Phase 3** | Dual Gloves Integrated | June 2026 | Both gloves synchronized, inter-hand features computed |
| **Phase 4** | Real Data Models | August 2026 | ≥95% static, ≥88% dynamic with real data blend |
| **Phase 5** | Final System | October 2026 | Sentence assembly, active learning, demonstration-ready |

### Key Decision Gates

| Gate | Criteria | Action if NOT met |
|---|---|---|
| Phase 1 → 2 | Static accuracy ≥ 92% on synthetic test set | Iterate sign definitions and retrain before proceeding |
| Phase 2 → 3 | Real sensor readings within 15% of synthetic assumptions | Adjust normalization params and regenerate synthetic data |
| Phase 3 → 4 | Dual-glove feature vector validated against 36-feature spec | Debug inter-hand alignment before collecting training data |
| Phase 4 → 5 | Real-data accuracy ≥ 90% on fingerspelling (5-fold CV mean) | Collect more samples for confused pairs; do not proceed to refinement |

---

---

## Appendix A — Changelog

| Version | Date | Changes |
|---|---|---|
| 1.0 | March 2026 | Initial plan |
| 1.1 | March 2026 | Fixed motion energy gravity bug (was including 9.8 m/s² gravity in ME); expanded feature set from 32→36 (added FSR pad + pinky touch per hand); added correlated finger noise model + transition frames to synthetic data; added MLflow experiment tracking + ONNX/TFLite model export; added DVC for dataset versioning; fixed CNN-LSTM 2-class overfit risk with distractor classes; added gimbal lock mitigation note; added double-scaling prevention; added 5-fold CV requirement for small real datasets (Phase 4); added battery profiling to Phase 5; updated risk register with 3 new risks |
| 1.2 | May 2026 | Hardware change: removed capacitive touch pins (T0–T4); relocated palm pad FSR to pinky fingertip. Feature vector reduced from 36 → 26 features (touch features 16–25 removed; FSR feature names updated pad→pinky; orientation/derived features renumbered). Static accuracy target revised to ≥89% synthetic (vs 97.4% baseline). Updated firmware structure (touch.cpp removed). I/O confusion identified as primary remaining limitation. |

> **Document prepared by:** ML Engineering Team  
> **Next review:** End of Phase 1 — post model evaluation checkpoint
