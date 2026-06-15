# ML Recognition and Recovery Deep Analysis

Last updated: 2026-06-09

This document analyzes the current ML recognition subsystem in depth: feature
engineering, static/dynamic routing, real-data training, deployed artifacts,
calibration, weak-letter recovery, validation, and the live risks that remain.

## Current Recognition Contract

The active recognizer consumes a 26-feature dual-glove frame and returns a
thresholded, debounced, display-safe prediction object. It supports two input
styles:

- Streaming raw packets from two ESP32 gloves through `/ws/predict`.
- One-shot REST calls through `/api/predict/raw` or `/api/predict/features`.

The streaming path is the real product path. It uses motion buffers, dynamic
segment detection, debounce history, dynamic latching, and dashboard
diagnostics. The REST one-shot path bypasses debounce and motion routing so a
manual form or external client can get a single static prediction without
waiting for several frames.

## Feature Vector

`ml/config.py` defines the current feature order:

| Index | Name | Notes |
| --- | --- | --- |
| 0 | `R_flex_thumb` | Active, currently carries real thumb data after rewiring. |
| 1 | `R_flex_index` | Active. |
| 2 | `R_flex_middle` | Active and upweighted for weak-letter recovery. |
| 3 | `R_flex_ring` | Active and upweighted for weak-letter recovery. |
| 4 | `R_flex_pinky` | Masked dead channel. |
| 5 | `L_flex_thumb` | Masked unreliable channel. |
| 6 | `L_flex_index` | Active. |
| 7 | `L_flex_middle` | Active. |
| 8 | `L_flex_ring` | Active. |
| 9 | `L_flex_pinky` | Active. |
| 10 | `R_fsr_thumb` | Active. |
| 11 | `R_fsr_index` | Active and strongly weighted. |
| 12 | `R_fsr_pinky` | Active. |
| 13 | `L_fsr_thumb` | Active. |
| 14 | `L_fsr_index` | Active and strongly weighted after left pinky pad moved here. |
| 15 | `L_fsr_pinky` | Masked unavailable channel. |
| 16-18 | `R_roll`, `R_pitch`, `R_yaw` | Quaternion-derived Euler angles. |
| 19-21 | `L_roll`, `L_pitch`, `L_yaw` | Quaternion-derived Euler angles. |
| 22 | `R_hand_openness` | Mean of working right flex channels. |
| 23 | `L_hand_openness` | Mean of working left flex channels. |
| 24 | `inter_hand_delta_roll` | Absolute right/left roll delta. |
| 25 | `inter_hand_delta_pitch` | Absolute right/left pitch delta. |

Normalization is simple and deterministic:

- Flex ADC values are normalized from the configured 800-3800 range to 0-1.
- FSR ADC values are normalized from 0-4095 to 0-1.
- Quaternions are converted from `[w, x, y, z]` to Euler roll/pitch/yaw using
  scipy, with pitch clamped to avoid gimbal-lock instability.
- Derived features are recomputed whenever calibration or augmentation changes
  their source channels.
- Masked channels are zeroed at the end of feature extraction and again after
  calibration/augmentation paths that could disturb them.

Touch features are intentionally absent. Touch bitmasks remain in API models
only as compatibility fields.

## Sensor Mask and Reliability Weights

Current masked indices are:

```text
MASKED_FEATURE_INDICES = (4, 5, 15)
```

Current feature weights are not baked into the raw feature vector by default.
They are used by training/inference components that explicitly support them:

- XGBoost receives feature weights through its model construction path.
- The MLP candidate pre-multiplies inputs and marks the model payload as
  weighted so inference can apply the same weighting.
- Augmentation suppresses noise on zero-weight channels so masked channels
  remain exactly zero.

The recovery emphasis is intentional:

- E/I/O/U rely heavily on the active index/pinky FSR channels.
- I/O/M/N/W rely on right middle/ring flex channels now that right pinky flex
  and left thumb flex are unavailable.
- G/W/Z/J rely on orientation and inter-hand deltas enough that these channels
  must survive model feature sampling and live drift.

Tests assert that masked channels are both masked and zero-weighted, and that
the active discriminative channels keep minimum useful weights.

## Label Topology

The current BSL labels are split by route:

| Route | Labels |
| --- | --- |
| Static | A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, REST_1, REST_2, REST_3 |
| Dynamic | H, J, OTHER |

H and J are dynamic. Z is static in this dataset. OTHER is not a displayed sign;
it is a dynamic rejection class trained from static/rest distractor sequences so
normal movement does not have to be forced into H or J.

The static label map uses original global letter indices plus REST indices. The
static model payload stores an inverse label map and label names so deployed
models can train on contiguous labels while returning global display labels.

## Static Recognition

The deployed static model is resolved at startup by reading
`data/models/training_report_real.json`. If the report names a valid best model,
the predictor loads the matching canonical artifact. If not, it falls back to
the random forest artifact because RF was historically the best real-data
fallback.

Current latest report:

| Model | Val accuracy | Test accuracy | Weighted | Notes |
| --- | --- | --- | --- | --- |
| XGBoost | 1.0 | 1.0 | false | Current best model. |
| Random Forest | 1.0 | 1.0 | false | Strong fallback. |
| SVM | 0.9937 | 0.9977 | false | Slightly weaker. |
| MLP | 0.9897 | 0.9994 | true | Uses weighted input payload. |

The static runtime path is:

1. Calibration-adjusted 26-feature vector arrives.
2. `StandardScaler` transforms it.
3. Weighted models apply `FEATURE_WEIGHTS` if required.
4. The model emits class probabilities.
5. Top-k diagnostics are recorded.
6. Confidence below `STATIC_CONFIDENCE_THRESHOLD = 0.55` becomes Unknown.
7. Streaming predictions enter the rolling debounce filter.

The static model includes REST classes. REST predictions are represented with
`display_label`, `is_rest`, and `rest_pose`; `sign` remains null because REST is
not a letter.

## Dynamic Recognition

The dynamic classifier is a PyTorch CNN-LSTM loaded from
`data/models/cnn_lstm_dynamic_v2.pt`. It receives 125-frame windows with 26
features per frame. Latest report:

| Property | Value |
| --- | --- |
| Classes | H, J, OTHER |
| Parameters | 219587 |
| Epochs trained | 43 |
| Train/val/test samples | 2492 / 37 / 37 |
| Test accuracy | 1.0 |
| H recall | 1.0 |
| J recall | 1.0 |
| OTHER recall | 1.0 |

The dynamic runtime path is more guarded than the static path:

1. Feature-delta and accel-delta scores update per frame.
2. Movement starts when feature delta, accel delta, or K-shaped dynamic-start
   context crosses configured thresholds for enough frames.
3. Movement ends after enough below-threshold frames, or times out after a
   full dynamic window.
4. The extracted segment is resampled to 125 frames.
5. The CNN-LSTM emits H/J/OTHER probabilities.
6. OTHER is rejected as Unknown.
7. Low H/J confidence below `DYNAMIC_CONFIDENCE_THRESHOLD = 0.60` is rejected.
8. H/J passes a dynamic context gate unless confidence is strong enough to
   bypass context.
9. Accepted dynamic predictions are force-emitted and latched.

Dynamic diagnostics include:

- Feature delta and accel delta.
- Whether feature-motion buffering is active.
- Segment source: `feature_settle`, `feature_timeout`, `gravity_settle`, or
  `dynamic_start_probe`.
- Segment frame count and final 125-frame window count.
- Dynamic top-k.
- Dynamic gate payload with class, confidence, static context label,
  context confidence, allowed labels, and reason.

## Live H/J Safeguards

The newest routing tests document the current safeguards:

- Dynamic threshold must remain the configured value, not be clamped back to
  0.90.
- Strong dynamic confidence can bypass static context mismatch.
- Dynamic diagnostics are cleared per frame to prevent stale debug state.
- OTHER remains Unknown and is never latched.
- Long movement without settling still produces a fallback segment.
- K-shaped starts do not immediately emit static K if J is plausible.
- A strong early J probe can preempt static K.
- If a dynamic probe says OTHER, static K recovers.
- Unknown dynamic segments do not create an Unknown latch.

These safeguards are the difference between an offline dynamic model and a
usable live stream. H/J errors are usually routing or temporal-state problems,
not just CNN-LSTM classification problems.

## Real-Data Build Pipeline

The current full validation pipeline is `scripts/run_full_user_validation.py`.
It builds from:

```text
data/raw/real/ravindu-8dd5c02e
```

The pipeline:

1. Analyzes raw JSONL sessions for missing fields, malformed records, bad
   dimensions, sensor ranges, quaternion norms, frame counts, and label
   coverage.
2. Splits sessions 80/10/10 by label.
3. Uses settled windows for static labels.
4. Uses full motion sequences for H and J.
5. Builds OTHER dynamic examples from static/rest recordings.
6. Applies real-data augmentation, including label-specific weak-letter
   policies.
7. Fits a real-data StandardScaler on static training frames.
8. Scales static and dynamic splits with the same scaler.
9. Writes `scaler_stats.json` for live calibration.
10. Trains static models from scratch.
11. Trains the dynamic CNN-LSTM from scratch.
12. Runs acceptance gates.
13. Copies artifacts to canonical live paths when the decision is `GO`.

Latest build values:

| Field | Value |
| --- | --- |
| User id | `8dd5c02e` |
| Total sessions | 1069 |
| Total paired frames | 207669 |
| Static split | 24 static letters plus 3 REST labels |
| Dynamic split | H, J, OTHER |
| Static train/val/test samples | 107676 / 1746 / 1746 |
| Dynamic train/val/test samples | 2492 / 37 / 37 |
| Static augmentation multiplier | 5 |
| Dynamic augmentation multiplier | 16 |
| OTHER dynamic augmentation multiplier | 8 |
| Weak validation letters | E, G, I, J, M, N, O, U, W, Z |

The training data is not a raw frame dump. It is filtered, windowed,
augmented, scaled, labeled, and split by session. That distinction matters
when interpreting model metrics.

## Weak-Letter Recovery

`scripts/live_accuracy_recovery.py` defines the current recovery workflow for
letters that are weak in live use:

```text
E, G, I, J, M, N, O, U, W, Z
```

Correction-session targets:

| Letter | Target sessions |
| --- | --- |
| E | 50 |
| G | 50 |
| I | 45 |
| J | 60 |
| M | 45 |
| N | 60 |
| O | 60 |
| U | 60 |
| W | 45 |
| Z | 45 |

The normal training command refuses to retrain until the correction targets are
met. `train --existing-data` bypasses that blocker explicitly for a
known-existing-data recovery pass, while still printing the missing-session
status.

The weak validation profile runs interactive live validation for the weak
letters:

```text
python -m scripts.live_accuracy_recovery validate weak
```

Equivalent raw validator behavior:

- Letters: E,G,I,J,M,N,O,U,W,Z
- Reps per letter: 5
- Hold seconds: 3
- Live debug: enabled
- Target top-1: 0.80
- Target top-3: 0.91
- Minimum per-letter top-1: 4 out of 5
- Output: `data/figures/live_validation_weak_letters_after_retrain.json`

The all-letter profile runs:

```text
python -m scripts.live_accuracy_recovery validate all
```

It covers A-Z with 3 reps per letter and writes
`data/figures/live_validation_all_letters_after_retrain.json`.

## Label-Specific Augmentation

Generic augmentation is not enough for the current hardware. The code now uses
label-specific policies:

- E/I/O/U keep FSR dropout low and FSR amplitude variation high because these
  labels depend on subtle target pressure.
- L/M/N reduce flex noise because these boundaries depend on which right-hand
  fingers are extended or curled.
- G/W/Z increase orientation jitter and yaw bias because live errors clustered
  around orientation drift.
- J broadens sequence timing and orientation while preserving FSR signal.
- T/V get additional orientation jitter because they are H-adjacent static
  shapes.

Full-user validation also oversamples weak static letters:

| Letter group | Current behavior |
| --- | --- |
| E, M, N, O, U | Static multiplier at least 9. |
| G, I, Z | Static multiplier at least 8. |
| W | Static multiplier 10. |
| J | Dynamic multiplier 20. |

This is not just "more data"; it is targeted distribution shaping around the
known weak separations.

## Calibration Analysis

Runtime calibration closes train/serve drift without retraining:

1. The user captures neutral frames.
2. The service compares the mean neutral vector to the training neutral
   baseline in `scaler_stats.json`.
3. It applies offsets only to configured offset channels:
   flex, Euler angles, and openness.
4. FSRs are not offset because zero pressure is meaningful signal.
5. Optional A/B/C reference captures estimate flex amplitude scale.
6. Calibration then recomputes derived features and reapplies the sensor mask.

The calibration order is important:

```text
raw frame
-> extract 26 features
-> apply calibration offset/scale
-> recompute derived features
-> reapply broken-sensor mask
-> StandardScaler
-> static or dynamic model
```

Applying calibration after scaling would be wrong because the offsets and
amplitude scales are defined in the unscaled feature domain.

## Validation Evidence

The latest reports in the workspace show:

| Evidence | Result |
| --- | --- |
| `training_report_real.json` best model | XGBoost |
| Static test accuracy | 1.0 |
| Static macro F1 | 1.0 |
| Static min class recall | 1.0 |
| REST rejection accuracy | 1.0 |
| Dynamic test accuracy | 1.0 |
| Dynamic H/J min recall | 1.0 |
| Full validation decision | GO |
| Deployment report | Artifacts copied to canonical live paths |

This is strong evidence for the current captured-user validation set. It is not
the same as evidence that every new wearer will pass. The live validation
harness and calibration profiles are still required for new users.

## Failure Modes to Watch

The main current failure modes are:

- Train/serve feature mismatch: any path that assumes 36 features or includes
  touch is stale.
- Missing hand pairing: the live model expects both gloves. Do not substitute
  a neutral hand unless a model is explicitly trained for that mode.
- Dynamic false rejection: H/J may be rejected by context gates if end-pose
  labels drift and confidence is not strong enough.
- Dynamic false static: K-shaped J starts can look static before the motion
  resolves.
- Weak pressure letters: E/I/O/U are sensitive to FSR target placement and
  dropout.
- Finger-count letters: M/N/L/W depend on working flex channels after the
  right pinky and left thumb masks.
- Orientation drift: G/W/Z/J need robust IMU-derived features and calibration.
- Single-user overconfidence: held-out session scores can be perfect while
  new-user live performance still shifts.
- Legacy batch endpoint: `/api/predict/batch` still validates 36 columns and
  should not be treated as representative of the active 26-feature predictor.

## What To Preserve When Changing ML Code

Any future ML changes should preserve these invariants unless the whole system
is intentionally migrated:

- `NUM_FEATURES` remains 26 across feature engineering, scaler, static models,
  dynamic models, dashboards, and validators.
- Masked channels stay zero in both training and live inference.
- Derived features are recomputed after calibration and augmentation.
- H and J remain the only dynamic BSL output letters unless data, dynamic
  labels, route gates, validation, and dashboards are all updated.
- OTHER remains a dynamic rejection class and must not emit as a displayed
  sign.
- The live WebSocket path must serialize access to the stateful predictor.
- Calibration must run before scaling.
- New validation reports should be read before changing deployed model paths.
- Weak-letter live validation should be run after retraining, not only offline
  test-set evaluation.

## Source of Truth Files

For recognition behavior, use these files before trusting any older note:

- `ml/config.py`
- `ml/feature_engineering.py`
- `ml/predict.py`
- `backend/services/predictor.py`
- `scripts/run_full_user_validation.py`
- `scripts/live_accuracy_recovery.py`
- `scripts/validate_live_recognition.py`
- `data/models/training_report_real.json`
- `data/models/dynamic_training_report_real.json`
- `data/models/full_user_validation_report.json`
- `tests/test_dynamic_routing.py`
- `tests/test_weak_letter_recovery_config.py`
- `tests/test_live_accuracy_recovery.py`

These files are the current recognition contract. Historical plans and guide
documents were removed because they described obsolete architecture, draft
intent, or pre-recovery assumptions.
