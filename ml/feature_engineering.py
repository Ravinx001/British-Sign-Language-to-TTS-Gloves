"""
Feature engineering: raw sensor frame → 26-dimensional ML feature vector.

Transforms raw dual-glove sensor readings into the normalized feature space
defined in ml/config.py. Handles flex normalization, FSR normalization,
quaternion→Euler conversion (with gimbal-lock pitch clamping), and derived
feature computation. Capacitive touch removed (hardware change, v1.2).

References:
    docs/ML-dev-plan.md §2 — 26 feature specification
    docs/sensor-guide.md   — raw sensor ranges
"""

import numpy as np
from scipy.spatial.transform import Rotation

from ml.config import (
    BAKE_WEIGHTS_INTO_FEATURES,
    EULER_CONVENTION,
    FEATURE_WEIGHTS,
    FLEX_MIN,
    FLEX_RANGE,
    FSR_MAX,
    MASKED_FEATURE_INDICES,
    NUM_FEATURES,
    PITCH_CLAMP_DEG,
)

_FEATURE_WEIGHTS_ARR = np.asarray(FEATURE_WEIGHTS, dtype=np.float64)


def normalize_flex(raw_values):
    """Min-max normalize flex sensor readings to [0, 1].

    0 = straight/extended, 1 = fully curled.

    Args:
        raw_values: array-like of raw ADC values (800–3800).

    Returns:
        np.ndarray clipped to [0, 1].
    """
    arr = np.asarray(raw_values, dtype=np.float64)
    normalized = (arr - FLEX_MIN) / FLEX_RANGE
    return np.clip(normalized, 0.0, 1.0)


def normalize_fsr(raw_values):
    """Normalize FSR readings to [0, 1].

    0 = no pressure, 1 = max pressure.

    Args:
        raw_values: array-like of raw ADC values (0–4095).

    Returns:
        np.ndarray clipped to [0, 1].
    """
    arr = np.asarray(raw_values, dtype=np.float64)
    normalized = arr / FSR_MAX
    return np.clip(normalized, 0.0, 1.0)


def expand_touch_bitmask(bitmask):
    """Expand a 5-bit contact bitmask into 5 binary features.

    Bit order: bit0=thumb, bit1=index, bit2=middle, bit3=ring, bit4=pinky.

    Args:
        bitmask: integer (0–31).

    Returns:
        np.ndarray of shape (5,) with 0/1 values.
    """
    return np.array([(bitmask >> i) & 1 for i in range(5)], dtype=np.float64)


def touch_raw_to_binary(raw_values):
    """Convert raw capacitive touch readings to binary contact flags.

    Inverted logic per sensor-guide.md: high value = no contact,
    low value = contact. Values below TOUCH_THRESHOLD_RAW → 1 (contact).

    Args:
        raw_values: array-like of 5 raw touch readings.

    Returns:
        np.ndarray of shape (5,) with 0/1 values.
    """
    arr = np.asarray(raw_values, dtype=np.float64)
    return (arr < TOUCH_THRESHOLD_RAW).astype(np.float64)


def quaternion_to_euler(quat):
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] in degrees.

    Applies pitch clamping at ±PITCH_CLAMP_DEG to mitigate gimbal lock.

    Args:
        quat: array-like of [w, x, y, z].

    Returns:
        np.ndarray of [roll, pitch, yaw] in degrees.
    """
    arr = np.asarray(quat, dtype=np.float64)
    if arr.shape != (4,) or not np.isfinite(arr).all():
        return np.zeros(3, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float64)

    w, x, y, z = arr
    # scipy expects [x, y, z, w] (scalar-last)
    rot = Rotation.from_quat([x, y, z, w])
    euler = rot.as_euler(EULER_CONVENTION, degrees=True)  # [roll, pitch, yaw]
    # Clamp pitch to avoid gimbal lock instability
    euler[1] = np.clip(euler[1], -PITCH_CLAMP_DEG, PITCH_CLAMP_DEG)
    return euler


_MASKED_SET = frozenset(MASKED_FEATURE_INDICES)
_R_FLEX_WORKING = tuple(i for i in range(0, 5) if i not in _MASKED_SET)
_L_FLEX_WORKING = tuple(i for i in range(5, 10) if i not in _MASKED_SET)


def apply_sensor_mask(features: np.ndarray) -> np.ndarray:
    """Zero out hardware-broken feature channels in-place.

    Reads ``MASKED_FEATURE_INDICES`` from ml.config to decide which channels
    to disable. When any right-hand flex channel (0-4) is masked,
    R_hand_openness (idx 22) is recomputed from the remaining working
    right-hand channels so the derived feature doesn't carry a constant
    bias from the dead sensor. Same for the left hand (5-9) and
    L_hand_openness (idx 23).

    Called as the final step of ``extract_features`` AND from the build
    script after augmentation, so live inference and training data are
    bit-for-bit consistent on every masked channel.
    """
    if not _MASKED_SET:
        return features
    for idx in _MASKED_SET:
        features[idx] = 0.0
    # Recompute hand-openness whenever ANY flex channel on that hand is masked.
    if _R_FLEX_WORKING and any(i in _MASKED_SET for i in range(0, 5)):
        features[22] = float(np.mean([features[i] for i in _R_FLEX_WORKING]))
    if _L_FLEX_WORKING and any(i in _MASKED_SET for i in range(5, 10)):
        features[23] = float(np.mean([features[i] for i in _L_FLEX_WORKING]))
    return features


def recompute_derived(features: np.ndarray) -> np.ndarray:
    """Recompute indices 22-25 from current source channels.

    Used after calibration or augmentation has perturbed flex/euler so the
    derived features stay semantically consistent. Hand openness honours
    MASKED_FEATURE_INDICES (mean of working channels only).
    """
    if _R_FLEX_WORKING:
        features[22] = float(np.mean([features[i] for i in _R_FLEX_WORKING]))
    else:
        features[22] = 0.0
    if _L_FLEX_WORKING:
        features[23] = float(np.mean([features[i] for i in _L_FLEX_WORKING]))
    else:
        features[23] = 0.0
    features[24] = abs(features[16] - features[19])  # inter_hand_delta_roll
    features[25] = abs(features[17] - features[20])  # inter_hand_delta_pitch
    return features


def apply_feature_weights(features: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Multiply features by per-channel reliability weights.

    Returns a NEW array. Used by the MLP candidate (pre-multiply before fit
    and before inference) and any consumer that wants to bake the reliability
    signal directly into the feature vector. Tree models that accept native
    ``feature_weights`` (XGBoost) should NOT call this — they receive weights
    through their constructor.
    """
    w = _FEATURE_WEIGHTS_ARR if weights is None else np.asarray(weights, dtype=np.float64)
    return features * w


def extract_features(raw_frame):
    """Transform a single raw sensor frame into a 26-feature vector.

    Args:
        raw_frame: dict with keys:
            'R_flex': [5] raw flex ADC values (thumb..pinky)
            'L_flex': [5] raw flex ADC values
            'R_fsr':  [3] raw FSR ADC values (thumb, index, pinky)
            'L_fsr':  [3] raw FSR ADC values
            'R_quaternion': [4] (w, x, y, z)
            'L_quaternion': [4] (w, x, y, z)

    Returns:
        np.ndarray of shape (26,).
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float64)

    # --- Flex (0–9) ---
    features[0:5] = normalize_flex(raw_frame['R_flex'])
    features[5:10] = normalize_flex(raw_frame['L_flex'])

    # --- FSR (10–15) ---
    features[10:13] = normalize_fsr(raw_frame['R_fsr'])
    features[13:16] = normalize_fsr(raw_frame['L_fsr'])

    # --- Orientation (16–21) ---
    r_euler = quaternion_to_euler(raw_frame['R_quaternion'])
    l_euler = quaternion_to_euler(raw_frame['L_quaternion'])
    features[16:19] = r_euler
    features[19:22] = l_euler

    # --- Derived (22–25) ---
    features[22] = np.mean(features[0:5])    # R_hand_openness
    features[23] = np.mean(features[5:10])   # L_hand_openness
    features[24] = abs(r_euler[0] - l_euler[0])  # inter_hand_delta_roll
    features[25] = abs(r_euler[1] - l_euler[1])  # inter_hand_delta_pitch

    # Disable hardware-broken sensors (and re-derive dependent features).
    apply_sensor_mask(features)

    if BAKE_WEIGHTS_INTO_FEATURES:
        features = apply_feature_weights(features)

    return features


def extract_features_batch(raw_frames):
    """Vectorized batch extraction for multiple frames.

    Args:
        raw_frames: list of raw_frame dicts (same format as extract_features).

    Returns:
        np.ndarray of shape (N, 36).
    """
    return np.array([extract_features(f) for f in raw_frames], dtype=np.float64)


def extract_features_from_arrays(
    r_flex, l_flex, r_fsr, l_fsr,
    r_quaternion, l_quaternion,
):
    """Extract features from separate arrays (convenience for synthetic data).

    All args are array-like with shapes matching their sensor counts.
    Returns np.ndarray of shape (26,).
    """
    raw_frame = {
        'R_flex': r_flex,
        'L_flex': l_flex,
        'R_fsr': r_fsr,
        'L_fsr': l_fsr,
        'R_quaternion': r_quaternion,
        'L_quaternion': l_quaternion,
    }
    return extract_features(raw_frame)
