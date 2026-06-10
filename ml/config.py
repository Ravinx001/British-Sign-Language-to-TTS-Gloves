"""
Central configuration for the BSL Sign Language Gloves ML pipeline.

All sensor specifications, feature definitions, hyperparameter search spaces,
augmentation parameters, and thresholds are defined here. Values are sourced
from docs/ML-dev-plan.md v1.1 and docs/sensor-guide.md.
"""

# =============================================================================
# Reproducibility
# =============================================================================
SEED = 42

# =============================================================================
# Sampling
# =============================================================================
SAMPLE_RATE_HZ = 50          # Sensor sampling frequency
DYNAMIC_WINDOW_FRAMES = 125  # 2.5 seconds at 50 Hz (raised from 100; real dynamic runs are often >100 frames)

# =============================================================================
# Sensor Ranges (raw ADC / physical units)
# =============================================================================
FLEX_MIN = 800       # ADC — straight finger
FLEX_MAX = 3800      # ADC — fully curled
FLEX_RANGE = FLEX_MAX - FLEX_MIN  # 3000

FSR_MIN = 0          # ADC — no pressure
FSR_MAX = 4095       # ADC — maximum pressure

ACCEL_RANGE = 39.0   # m/s² (±4g)
GYRO_RANGE = 500.0   # °/s

# Touch sensor: inverted logic (sensor-guide.md)
# Raw high (~60-80) = no contact; raw low (~5-20) = contact
TOUCH_THRESHOLD_RAW = 40  # Below this → contact detected (binary 1)

# =============================================================================
# 26 Engineered ML Features (ordered, 0-indexed)
# Capacitive touch removed (hardware change, ML-dev-plan v1.2).
# FSR third sensor renamed pad → pinky (pinky fingertip).
# =============================================================================
FEATURE_NAMES = [
    # 0–4: Right flex (normalized 0=straight, 1=curled)
    'R_flex_thumb', 'R_flex_index', 'R_flex_middle', 'R_flex_ring', 'R_flex_pinky',
    # 5–9: Left flex
    'L_flex_thumb', 'L_flex_index', 'L_flex_middle', 'L_flex_ring', 'L_flex_pinky',
    # 10–12: Right FSR — thumb tip, index tip, pinky tip (normalized 0–1)
    'R_fsr_thumb', 'R_fsr_index', 'R_fsr_pinky',
    # 13–15: Left FSR
    'L_fsr_thumb', 'L_fsr_index', 'L_fsr_pinky',
    # 16–18: Right Euler angles (degrees, from quaternion)
    'R_roll', 'R_pitch', 'R_yaw',
    # 19–21: Left Euler angles
    'L_roll', 'L_pitch', 'L_yaw',
    # 22–23: Derived hand openness (0=open, 1=fist)
    'R_hand_openness', 'L_hand_openness',
    # 24–25: Inter-hand orientation deltas (degrees)
    'inter_hand_delta_roll', 'inter_hand_delta_pitch',
]

NUM_FEATURES = len(FEATURE_NAMES)  # 26
assert NUM_FEATURES == 26, f"Expected 26 features, got {NUM_FEATURES}"

# =============================================================================
# Broken-sensor masking
# =============================================================================
# Indices of feature channels to zero out everywhere — both at training
# materialisation and at live inference. Used to disable hardware-broken
# sensors so the model stops relying on them.
#
# Current setup: R_flex_thumb (idx 0) was broken on the deployed glove,
# so the WORKING R_flex_pinky sensor was physically rewired into the thumb
# position. Feature index 0 now carries real thumb data; feature index 4
# (R_flex_pinky) is now the dead channel and is masked here.
#
# When a right-hand flex index (0..4) is masked, R_hand_openness (idx 22)
# is recomputed from the remaining working right-hand channels inside
# ml/feature_engineering.py. Symmetric logic applies to left-hand flex
# (5..9) and L_hand_openness (idx 23).
#
# If a sensor is repaired, set this to () and rebuild the dataset.
#
# v3 update: left thumb flex (idx 5) is also unreliable on the deployed glove
# (intermittent ADC dropouts) — masked here. With no spare working flex sensor
# available (right thumb destroyed, right pinky relocated), the left thumb is
# masked rather than replaced. L_hand_openness (idx 23) is recomputed from the
# remaining 4 working left-hand channels inside ml/feature_engineering.py.
MASKED_FEATURE_INDICES: tuple[int, ...] = (4, 5)

# =============================================================================
# Per-feature reliability weights (length 26)
# =============================================================================
# Higher weight = sensor is clean and discriminative; lower weight = noisy,
# narrow dynamic range, or drift-prone. Consumed by:
#   - XGBoost via the `feature_weights` argument (sampling bias during tree
#     construction).
#   - The MLP candidate by pre-multiplying X *before* fitting (and during
#     inference). The model payload carries `weighted=True` so predict.py
#     knows to multiply at inference.
#   - real_augmentation: noise on weight=0 channels is suppressed (so masked
#     channels stay exactly 0 after augmentation).
#
# Justification per channel:
#   0  R_flex_thumb  1.00  physically rewired to the relocated pinky sensor; clean
#   1-3 R_flex idx/mid/ring  1.00  healthy
#   4  R_flex_pinky  0.00  masked (floating ADC, sensor moved away)
#   5  L_flex_thumb  0.00  masked (unreliable)
#   6-9 L_flex idx/mid/ring/pinky  1.00  healthy
#  10  R_fsr_thumb   0.80  healthy
#  11  R_fsr_index   0.80  healthy
#  12  R_fsr_pinky   0.25  narrow dynamic range (0-200 vs 0-4095 elsewhere)
#  13-15 L_fsr_*     0.80  healthy
#  16,17 R_roll,pitch  1.00  healthy
#  18  R_yaw         0.90  yaw drifts with gyro bias
#  19,20 L_roll,pitch  1.00  healthy
#  21  L_yaw         0.90  yaw drifts
#  22,23 hand_openness  0.70  derived from flex; informative but redundant
#  24,25 inter-hand deltas  0.60  derived from euler; redundant
FEATURE_WEIGHTS: tuple[float, ...] = (
    1.00, 1.00, 1.00, 1.00, 0.00,   # 0-4  R_flex (idx 4 masked)
    0.00, 1.00, 1.00, 1.00, 1.00,   # 5-9  L_flex (idx 5 masked)
    0.80, 0.80, 0.25,               # 10-12 R_fsr (R_pinky FSR low-range)
    0.80, 0.80, 0.80,               # 13-15 L_fsr
    1.00, 1.00, 0.90,               # 16-18 R_roll/pitch/yaw
    1.00, 1.00, 0.90,               # 19-21 L_roll/pitch/yaw
    0.70, 0.70,                     # 22-23 hand_openness (derived)
    0.60, 0.60,                     # 24-25 inter-hand deltas
)
assert len(FEATURE_WEIGHTS) == 26, f"FEATURE_WEIGHTS length {len(FEATURE_WEIGHTS)} != 26"

# When True, extract_features multiplies the output by FEATURE_WEIGHTS at the
# point of feature extraction. When False (default), models apply weights at
# their own discretion (e.g., XGB `feature_weights`, MLP pre-multiply). False
# is the safer default because it keeps the StandardScaler fit on the
# unweighted distribution.
BAKE_WEIGHTS_INTO_FEATURES = False

# =============================================================================
# Feature Group Indices (for ablation studies)
# =============================================================================
FEATURE_GROUPS = {
    'flex':        list(range(0, 10)),    # 0–9
    'fsr':         list(range(10, 16)),   # 10–15
    'orientation': list(range(16, 22)),   # 16–21
    'derived':     list(range(22, 26)),   # 22–25
}

# =============================================================================
# Class Labels — 26 BSL Fingerspelling Letters
# =============================================================================
LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
NUM_CLASSES = len(LETTERS)  # 26
REST_LABELS = ['REST_1', 'REST_2', 'REST_3']
REST_LABEL_TO_IDX = {
    label: NUM_CLASSES + idx for idx, label in enumerate(REST_LABELS)
}
REST_IDX_TO_LABEL = {idx: label for label, idx in REST_LABEL_TO_IDX.items()}
REST_POSES = [
    {
        'label': 'REST_1',
        'name': 'Open table rest',
        'instruction': 'Both hands open and flat on the table.',
    },
    {
        'label': 'REST_2',
        'name': 'Relaxed hover rest',
        'instruction': 'Hands relaxed in a natural hover or lap pose.',
    },
    {
        'label': 'REST_3',
        'name': 'Hands apart idle',
        'instruction': 'Hands apart in a natural idle position.',
    },
]

# Static (24) vs dynamic (2) classification.
# In the current BSL glove dataset, H and J are motion gestures. Z is static.
DYNAMIC_LETTERS = {'H', 'J'}
STATIC_LETTERS = sorted(set(LETTERS) - DYNAMIC_LETTERS)
NUM_STATIC_CLASSES = len(STATIC_LETTERS)   # 24
NUM_DYNAMIC_CLASSES = len(DYNAMIC_LETTERS)  # 2

# Letter-to-index mapping
LETTER_TO_IDX = {letter: idx for idx, letter in enumerate(LETTERS)}
IDX_TO_LETTER = {idx: letter for letter, idx in LETTER_TO_IDX.items()}
STATIC_LABEL_TO_IDX = {**LETTER_TO_IDX, **REST_LABEL_TO_IDX}
STATIC_IDX_TO_LABEL = {idx: label for label, idx in STATIC_LABEL_TO_IDX.items()}

# Static-only index mapping (for XGBoost classifier)
STATIC_LETTER_TO_IDX = {letter: idx for idx, letter in enumerate(STATIC_LETTERS)}
STATIC_IDX_TO_LETTER = {idx: letter for letter, idx in STATIC_LETTER_TO_IDX.items()}

# Known confused pairs — require extra attention during data gen & evaluation
CONFUSED_PAIRS = [
    ('M', 'N'),  # Differ only in number of fingers over fist (3 vs 2)
    ('N', 'L'),  # Live hardware frequently collapses two-finger drape vs L shape
    ('I', 'O'),  # Same right-index action; left-hand contact/pressure differs
    ('U', 'O'),  # Same right-index action; left-hand contact/pressure differs
    ('I', 'U'),  # Adjacent left-hand fingertip targets can swap live
    ('D', 'K'),  # Both involve index touching L index area
    ('P', 'Q'),  # Same shape, differ in wrist rotation
    ('D', 'E'),  # Both touch L_index area, differ in R_index curl
    ('H', 'V'),  # Both have index+middle extended, differ in orientation
    ('H', 'T'),  # H false positives often happen during T-like hand movement
]

# =============================================================================
# Per-new-user runtime calibration
# =============================================================================
# Captured at the dashboard before live signing. The neutral pose maps the
# wearer's per-flex baseline and IMU mount angle into the training
# distribution; the optional reference letters refine the per-flex
# amplitude scale. The offset_channel_mask gates WHICH features get an
# additive correction — flex and Euler-angle channels yes; FSR and
# inter-hand deltas no (they encode actual signal, not baseline).
CALIBRATION = {
    'neutral_frames': 150,                # 3 s @ 50 Hz, both hands resting flat
    'reference_letters': ['A', 'B', 'C'],
    'frames_per_reference': 100,          # 2 s per reference letter
    # length-26 list of booleans; True = channel is offset-corrected, False = left alone.
    'offset_channel_mask': (
        # flex 0-9: all eligible (mask filters out broken ones at runtime)
        True, True, True, True, True,  True, True, True, True, True,
        # fsr 10-15: NEVER offset (zero-press = real zero, not a baseline)
        False, False, False, False, False, False,
        # euler 16-21: ALL eligible (mount-angle baseline correction)
        True, True, True, True, True, True,
        # derived 22-23 openness: eligible (mirrors flex correction)
        True, True,
        # derived 24-25 inter-hand deltas: NOT offset (recomputed anyway)
        False, False,
    ),
    # Threshold (mean of R/L hand_openness) below which a training frame is
    # considered "neutral pose" and contributes to train_neutral_baseline.
    'neutral_openness_threshold': 0.15,
    # Clip range for per-flex amplitude scale derived from reference letters.
    'amplitude_scale_clip': (0.7, 1.4),
}
assert len(CALIBRATION['offset_channel_mask']) == 26

# =============================================================================
# Real-data augmentation parameters (used by ml/real_augmentation.py)
# =============================================================================
# Per-group ranges so the augmentation distribution matches the realistic
# variability a NEW user would introduce, with a safety buffer.
REAL_AUG = {
    # Global Gaussian noise (after per-group amplitude scale).
    'noise_sigma': 0.03,

    # Per-group amplitude scale: multiplicative range applied to that group.
    'amplitude_scale': {
        'flex': 0.10,        # ±10 % (avoid blowing up M/N, D/K boundaries)
        'fsr':  0.20,        # ±20 %
        'euler': 0.05,       # ±5 % (orientation jitter is the bigger lever — see below)
        'derived': 0.0,      # never scaled directly; recomputed
    },

    # Per-group additive baseline shift (post-normalize units).
    'baseline_shift_sigma': {
        'flex': 0.04,
        'fsr':  0.05,
        'euler': 0.0,        # baseline shift expressed via euler_jitter_deg instead
        'derived': 0.0,
    },

    # Orientation jitter applied to Euler channels (degrees), plus a small
    # probability of a constant yaw bias across an entire frame.
    'euler_jitter_deg': 8.0,
    'yaw_bias_prob': 0.05,
    'yaw_bias_deg': 15.0,

    # Per-channel dropout (zero a single channel) probability.
    'sensor_dropout_prob': 0.04,
    # Additional dropout that specifically targets a single FSR channel.
    'fsr_dropout_prob': 0.06,

    # Sequence augmentation (dynamic only).
    'time_warp_range': 0.20,
    'temporal_shift_max': 8,
}

# Label-specific real-data augmentation overrides. These keep the generic
# augmentation broad while preserving the fine boundaries that matter for the
# live-confused letters. In particular:
#   - I/O/U depend on subtle left-hand target pressure, so FSR dropout is lower.
#   - M/N/L depend on which right fingers are extended/curled, so flex noise is
#     gentler and per-label oversampling in the build script supplies breadth.
#   - H-adjacent static shapes (V/T/M/N) get moderate orientation jitter so the
#     dynamic gate sees realistic end-pose variation.
REAL_AUG_LABEL_POLICIES = {
    'I': {
        'noise_sigma': 0.02,
        'amplitude_scale': {'flex': 0.07, 'fsr': 0.24, 'euler': 0.04, 'derived': 0.0},
        'baseline_shift_sigma': {'flex': 0.025, 'fsr': 0.08, 'euler': 0.0, 'derived': 0.0},
        'euler_jitter_deg': 5.0,
        'sensor_dropout_prob': 0.02,
        'fsr_dropout_prob': 0.025,
    },
    'O': {
        'noise_sigma': 0.02,
        'amplitude_scale': {'flex': 0.07, 'fsr': 0.24, 'euler': 0.04, 'derived': 0.0},
        'baseline_shift_sigma': {'flex': 0.025, 'fsr': 0.08, 'euler': 0.0, 'derived': 0.0},
        'euler_jitter_deg': 5.0,
        'sensor_dropout_prob': 0.02,
        'fsr_dropout_prob': 0.025,
    },
    'U': {
        'noise_sigma': 0.02,
        'amplitude_scale': {'flex': 0.07, 'fsr': 0.24, 'euler': 0.04, 'derived': 0.0},
        'baseline_shift_sigma': {'flex': 0.025, 'fsr': 0.08, 'euler': 0.0, 'derived': 0.0},
        'euler_jitter_deg': 5.0,
        'sensor_dropout_prob': 0.02,
        'fsr_dropout_prob': 0.025,
    },
    'L': {
        'noise_sigma': 0.018,
        'amplitude_scale': {'flex': 0.055, 'fsr': 0.12, 'euler': 0.04, 'derived': 0.0},
        'baseline_shift_sigma': {'flex': 0.02, 'fsr': 0.04, 'euler': 0.0, 'derived': 0.0},
        'euler_jitter_deg': 5.0,
        'sensor_dropout_prob': 0.015,
        'fsr_dropout_prob': 0.03,
    },
    'M': {
        'noise_sigma': 0.018,
        'amplitude_scale': {'flex': 0.055, 'fsr': 0.12, 'euler': 0.04, 'derived': 0.0},
        'baseline_shift_sigma': {'flex': 0.02, 'fsr': 0.04, 'euler': 0.0, 'derived': 0.0},
        'euler_jitter_deg': 5.0,
        'sensor_dropout_prob': 0.015,
        'fsr_dropout_prob': 0.03,
    },
    'N': {
        'noise_sigma': 0.018,
        'amplitude_scale': {'flex': 0.055, 'fsr': 0.12, 'euler': 0.04, 'derived': 0.0},
        'baseline_shift_sigma': {'flex': 0.02, 'fsr': 0.04, 'euler': 0.0, 'derived': 0.0},
        'euler_jitter_deg': 5.0,
        'sensor_dropout_prob': 0.015,
        'fsr_dropout_prob': 0.03,
    },
    'T': {
        'euler_jitter_deg': 10.0,
        'yaw_bias_prob': 0.08,
        'yaw_bias_deg': 18.0,
        'sensor_dropout_prob': 0.025,
    },
    'V': {
        'euler_jitter_deg': 10.0,
        'yaw_bias_prob': 0.08,
        'yaw_bias_deg': 18.0,
        'sensor_dropout_prob': 0.025,
    },
}

# =============================================================================
# Synthetic Data Generation Parameters
# =============================================================================
DATA_GEN = {
    'n_per_sign': 5000,                # Samples per letter → 130K total static
    'n_virtual_users': 10,             # Virtual user profiles
    'virtual_user_offset': 0.10,       # ±10% baseline calibration offset

    # Augmentation
    'noise_sigma': 0.03,              # Gaussian noise σ = 3% of sensor range
    'amplitude_scale': 0.08,          # ±8% amplitude scaling
    'baseline_shift_sigma': 0.03,     # N(0, 0.03) additive baseline shift
    'orientation_jitter_deg': 15.0,   # ±15° orientation jitter
    'sensor_dropout_prob': 0.05,      # 5% probability per sensor per sample
    'time_warp_range': 0.15,          # ±15% time warping (dynamic only)
    'correlated_finger_rho': 0.3,     # Adjacent finger noise correlation
    'transition_frames': (5, 10),     # Min/max interpolated frames between poses

    # Distractor classes for dynamic classifier
    'n_distractor_classes': 5,        # Random waves, scratching, adjusting, etc.
    'n_distractor_samples': 500,      # Samples per distractor class

    # Data split
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}

# =============================================================================
# Motion Energy / Segmentation
# =============================================================================
MOTION_ENERGY_THRESHOLD = 2.5     # m/s² (gravity-compensated linear accel)
MOTION_ONSET_FRAMES = 3           # Consecutive frames above threshold to start
MOTION_OFFSET_FRAMES = 25         # Consecutive frames below threshold to stop (500ms)
GRAVITY_LPF_ALPHA = 0.1           # Low-pass filter coefficient for gravity estimation

# Settle-window threshold used by ml/real_data.find_settled_window during
# training-dataset construction. Looser than the live MOTION_ENERGY_THRESHOLD
# so we admit training frames that match the natural micro-jitter of a live
# hand. (Was 1.5; raised to 2.5 to fix train/serve distribution mismatch.)
SETTLE_THRESHOLD_TRAIN = 2.5

# =============================================================================
# Confidence Thresholds & Debouncing
# =============================================================================
# v3: lowered to 0.55 (was 0.85). With LOUO-validated, calibrated probabilities
# the magnitudes are honest but smaller. Debounce frames raised to 4 to keep
# flicker out at the lower threshold.
STATIC_CONFIDENCE_THRESHOLD = 0.55
DYNAMIC_CONFIDENCE_THRESHOLD = 0.60
DEBOUNCE_FRAMES = 4               # Consecutive identical predictions before emit (80ms)

# =============================================================================
# XGBoost Hyperparameter Search Space
# =============================================================================
XGBOOST_PARAM_GRID = {
    'n_estimators': [200, 400],
    'max_depth': [4, 6],
    'learning_rate': [0.1],
    'subsample': [0.7],
    'colsample_bytree': [0.7],
    'min_child_weight': [5],
    'reg_alpha': [0.5],
    'reg_lambda': [3.0],
}

# =============================================================================
# Random Forest Hyperparameter Search Space
# =============================================================================
RF_PARAM_GRID = {
    # Cap depth (was: allowed None=unrestricted, which made it trivial to
    # memorize per-session sensor jitter). Increase min_samples_leaf so a leaf
    # represents at least a handful of frames, not a single one.
    'n_estimators': [300, 500],
    'max_depth': [8, 12],
    'min_samples_split': [5],
    'min_samples_leaf': [3, 5],
    'max_features': ['sqrt'],
}

# =============================================================================
# SVM Hyperparameter Search Space
# =============================================================================
SVM_PARAM_GRID = {
    'C': [0.1, 1.0, 10.0],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf'],
}

# =============================================================================
# CNN-LSTM Architecture Parameters
# =============================================================================
CNN_LSTM = {
    # Conv layers
    'conv1_filters': 64,
    'conv1_kernel': 5,
    'conv2_filters': 128,
    'conv2_kernel': 3,
    'pool_size': 2,

    # LSTM layers
    'lstm1_units': 128,
    'lstm2_units': 64,

    # Dense layers
    'dense_units': 64,

    # Regularization
    'lstm_dropout': 0.3,
    'dense_dropout': 0.2,

    # Training
    'learning_rate': 0.001,
    'batch_size': 32,
    'max_epochs': 100,
    'early_stop_patience': 10,
    'lr_reduce_factor': 0.5,
    'lr_reduce_patience': 5,
}

# =============================================================================
# GridSearchCV Settings
# =============================================================================
CV_FOLDS = 5
CV_SCORING = 'f1_weighted'

# =============================================================================
# File Paths (relative to project root)
# =============================================================================
PATHS = {
    # --- Synthetic data (kept for baseline comparison) ---
    'raw_static': 'data/raw/synthetic/static_dataset.npz',
    'raw_dynamic': 'data/raw/synthetic/dynamic_dataset.npz',
    'train': 'data/processed/train.npz',
    'val': 'data/processed/val.npz',
    'test': 'data/processed/test.npz',

    # --- Real data splits (built by scripts/build_real_dataset.py) ---
    'real_train': 'data/processed/real/train.npz',
    'real_val':   'data/processed/real/val.npz',
    'real_test':  'data/processed/real/test.npz',
    'real_dynamic_train': 'data/processed/real/dynamic_train.npz',
    'real_dynamic_val':   'data/processed/real/dynamic_val.npz',
    'real_dynamic_test':  'data/processed/real/dynamic_test.npz',

    # --- Scaler: fitted on real training data ---
    'scaler': 'data/processed/real/scaler.pkl',
    # Sidecar JSON with scaler.mean_, scaler.scale_, train_neutral_baseline,
    # and letter_flex_amplitudes — used by the calibration routine without
    # unpickling sklearn.
    'scaler_stats': 'data/processed/real/scaler_stats.json',

    # --- Models: v2 trained on real data ---
    'xgboost_model': 'data/models/xgboost_static_v2.pkl',
    'rf_model': 'data/models/rf_static_v2.pkl',
    'svm_model': 'data/models/svm_static_v2.pkl',
    'mlp_model':  'data/models/mlp_static_v2.pkl',
    'cnn_lstm_model': 'data/models/cnn_lstm_dynamic_v2.pt',

    # --- LOUO outputs (Leave-One-User-Out) ---
    'louo_root': 'data/processed/real/louo',          # parent dir for louo_<user>/
    'louo_models_root': 'data/models/louo',           # parent dir for per-user models
    'louo_summary': 'data/models/louo_summary.json',
    'louo_report': 'data/models/louo_report.json',

    # --- Per-user calibration profiles ---
    'calibrations_dir': 'data/calibrations',

    # --- Reports ---
    'dynamic_training_report': 'data/models/dynamic_training_report.json',
    'dynamic_training_report_real': 'data/models/dynamic_training_report_real.json',
    'training_report': 'data/models/training_report.json',
    'training_report_real': 'data/models/training_report_real.json',
    'feature_importance': 'data/models/feature_importance.json',
    'confusion_matrix': 'data/models/confusion_matrix.png',
}

# Authoritative source of real captured sessions (outside the repo).
# scripts/build_real_dataset.py reads JSONL files from here.
REAL_DATA_SOURCE = r'c:\laragon\www\BSL_Gloves\data\real'

# =============================================================================
# Euler Angle Limits (gimbal lock mitigation)
# =============================================================================
PITCH_CLAMP_DEG = 85.0  # Clamp pitch to ±85° to avoid gimbal lock at ±90°

# =============================================================================
# Quaternion → Euler Convention
# =============================================================================
EULER_CONVENTION = 'xyz'  # Roll (X), Pitch (Y), Yaw (Z)
