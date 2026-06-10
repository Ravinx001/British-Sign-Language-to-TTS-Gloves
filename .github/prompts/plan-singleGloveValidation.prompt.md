# Plan: Single-Glove (Right Hand) Hardware + ML Validation

**TL;DR** — Build the **right-hand glove first**. SHAP analysis on the trained XGBoost model ([docs/ml_evaluation_report.md](docs/ml_evaluation_report.md#L185)) shows the right hand contributes 5 of the top 10 most-important features (R_flex_middle is #1, R_pitch #2, R_fsr_index #3), and BSL fingerspelling is dominant-hand–led. Since BSL is inherently two-handed, single-glove validation cannot test full inter-hand contact letters, so we validate via three complementary tracks: (1) **hardware/data-pipeline integrity**, (2) **right-hand-only static pose subset** using a retrained 18-feature model, and (3) **dynamic J/Z** which is purely right-handed.

## Why Right Hand

- Top-3 SHAP features are all R-hand; 5 of top 10 are R-flex.
- BSL fingerspelling is led by the dominant (typically right) hand; left is the passive base.
- Dynamic letters J and Z are produced entirely by the right hand → CNN-LSTM can be fully validated.
- MPU6050 default I²C address `0x68` is for the right glove; left will need `AD0→3.3V` later.

## Phases

1. **Hardware bring-up** — build per [docs/guide-hardware-build.md](docs/guide-hardware-build.md), flash incrementally from [arduino/test_no_mux/](arduino/test_no_mux), verify 50 Hz WebSocket ingest at [backend/routers/websocket.py](backend/routers/websocket.py), record JSONL per [guide-sensor-data-collection.md](docs/new-docs/guide-sensor-data-collection.md). *Blocking.*
2. **Calibration & sanity** — capture per-finger flex min/max, FSR baseline, IMU rest quaternion; run KS-test on real vs. synthetic R-hand feature distributions (`data/processed/train.npz` cols 0–4, 10–12, 16–20, 26–28, 32). *Depends on 1.*
3. **Pose subset selection** — from [ml/bsl_sign_definitions.py](ml/bsl_sign_definitions.py), pick ~10 letters whose R-hand-only signature is unique (candidates: A, C, F, G, L, V, W, Y + dynamic J, Z); exclude inter-hand-contact letters (B, M, N, R, S, T, U, X). *Parallel with 1–2.*
4. **Real data collection** — ≥ 50 samples × 12 signs × 1 signer, then expand to 3 signers; convert to 36-feature NPZ via the template in [guide-train-with-real-data.md](docs/new-docs/guide-train-with-real-data.md) Step 2. Use **mock-left mode** (zero flex/FSR/touch, neutral euler) for the missing hand. *Depends on 1–3.*
5. **Validation — two parallel tracks:**
   - **E1**: feed real-R + mock-L through existing 36-feature [ml/predict.py](ml/predict.py) using `data/models/cnn_lstm_dynamic_v1.pt` and the static XGBoost; report confusion matrix, latency, synthetic→real gap.
   - **E2**: build `ml/train_static_right_only.py` and `ml/train_dynamic_right_only.py` with 18 R-hand features (5 flex + 3 FSR + 5 touch + 3 euler + R-openness + motion-energy); retrain from synthetic, validate on real.
6. **Reporting** — `docs/single_glove_validation_report.md` with hardware metrics, calibration profile, KS scores, per-letter confusion matrices for E1 vs. E2, latency, go/no-go for second glove.

## Relevant Files

- [docs/guide-hardware-build.md](docs/guide-hardware-build.md), [docs/sensor-guide.md](docs/sensor-guide.md), [docs/new-docs/guide-sensor-data-collection.md](docs/new-docs/guide-sensor-data-collection.md), [docs/new-docs/guide-train-with-real-data.md](docs/new-docs/guide-train-with-real-data.md)
- [ml/bsl_sign_definitions.py](ml/bsl_sign_definitions.py), [ml/feature_engineering.py](ml/feature_engineering.py), [ml/config.py](ml/config.py), [ml/train_static.py](ml/train_static.py), [ml/train_dynamic.py](ml/train_dynamic.py), [ml/predict.py](ml/predict.py)
- [backend/routers/websocket.py](backend/routers/websocket.py), [backend/routers/predict.py](backend/routers/predict.py)
- New: `ml/train_static_right_only.py`, `ml/train_dynamic_right_only.py`, `tests/test_real_data_quality.py`, `docs/new-docs/plan-singleGloveValidation.md`, `docs/single_glove_validation_report.md`

## Verification

1. Live stream sustains 50 ± 1 Hz for ≥ 10 min; quality-gate test passes (flex range ≥ 1500 ADC, FSR pad ≥ 2500 firm press, IMU drift < 2°/min).
2. `python -m ml.predict --source ws --mock-left` runs with < 50 ms median latency.
3. **E1** ≥ 75% accuracy on R-hand subset (mock-left penalty expected).
4. **E2** ≥ 88% static and ≥ 80% dynamic accuracy on real test split.
5. Per-feature KS statistic < 0.25 for ≥ 80% of R-hand features after calibration.
6. Live-spell a word using only R-hand-distinguishable letters (e.g., "FLAG", "WAVY").

## Decisions

- **Hand**: Right (SHAP + BSL convention).
- **Excluded**: full BSL fingerspelling, two-handed common signs (Hello, Thank-you), the second glove BOM.
- **Mock-left**: zero-filled neutral pose — simpler than synthesizing a plausible left hand; the resulting accuracy drop *is* the metric of interest for E1.

## Further Considerations

1. **Signer count for collection** — A) 1 signer only / **B) 1 → 3 staged (recommended)** / C) jump straight to 5.
2. **Run both validation tracks?** — A) E1 only (faster) / **B) both E1 + E2 (recommended, cleaner signal)** / C) E2 only.
3. **Calibration tooling** — A) full Three.js UI from [plan-3dHandCalibrationSimulation.md](docs/plan-3dHandCalibrationSimulation.md) / **B) minimal CLI capture now, UI later (recommended)** / C) skip calibration.
