"""Real-data augmentation helpers.

Per-group augmentation policy designed to broaden the training distribution
toward what a NEW wearer's signal actually looks like, with a safety buffer.

Three callers:

- ``augment_features_frame`` — for static-classifier frames (each row is an
  independent 26-dim feature vector). Applies per-group Gaussian noise,
  amplitude scaling, baseline shift, Euler jitter (with rare constant yaw
  bias), channel + FSR-specific dropout.

- ``augment_sequence`` — for dynamic CNN-LSTM sequences (shape ``T x 26``).
  Adds time warping and small temporal shifts on top of the frame-level
  augmentations, preserving temporal coherence.

Both routes finish with: apply_sensor_mask → recompute_derived (so masked
channels stay exactly 0 and indices 22-25 stay consistent with their source
channels). Noise on weight-zero channels is suppressed entirely.

All augmentations apply AFTER feature extraction but BEFORE the StandardScaler
is fitted, so the broadened distribution is what the scaler captures.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ml.config import FEATURE_WEIGHTS, NUM_FEATURES, REAL_AUG, REAL_AUG_LABEL_POLICIES
from ml.feature_engineering import apply_sensor_mask, recompute_derived

# Feature group slices — mirror ml/config.py FEATURE_GROUPS but as slices
# for fast NumPy indexing.
FLEX_SLICE = slice(0, 10)          # R + L flex, normalized [0, 1]
FSR_SLICE = slice(10, 16)          # R + L FSR, normalized [0, 1]
ORIENT_SLICE = slice(16, 22)       # Roll/pitch/yaw per hand, degrees
DERIVED_SLICE = slice(22, 26)      # Hand openness + inter-hand deltas (recomputed)

# Mask channels with weight 0 → augmentation must not introduce signal there
# (they're physically broken; the consumer expects exactly 0).
_NOISE_MASK = np.asarray(FEATURE_WEIGHTS, dtype=np.float64) > 0.0


def _merged_policy(label: str | None = None, override: dict | None = None) -> dict:
    """Return REAL_AUG with optional label-specific and call-site overrides."""
    out = deepcopy(REAL_AUG)
    label_key = str(label).upper() if label else None
    for policy in (REAL_AUG_LABEL_POLICIES.get(label_key, {}), override or {}):
        for key, value in policy.items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = {**out[key], **value}
            else:
                out[key] = value
    return out


def _per_channel_amp_scale(rng: np.random.Generator, n: int, amp: dict) -> np.ndarray:
    """Build an (n, 26) multiplicative-scale matrix from per-group amplitudes."""
    scale = np.ones((n, NUM_FEATURES), dtype=np.float64)
    if amp.get('flex', 0) > 0:
        a = amp['flex']
        scale[:, FLEX_SLICE] = 1.0 + rng.uniform(-a, a, size=(n, 10))
    if amp.get('fsr', 0) > 0:
        a = amp['fsr']
        scale[:, FSR_SLICE] = 1.0 + rng.uniform(-a, a, size=(n, 6))
    if amp.get('euler', 0) > 0:
        a = amp['euler']
        scale[:, ORIENT_SLICE] = 1.0 + rng.uniform(-a, a, size=(n, 6))
    return scale


def _per_channel_baseline_shift(rng: np.random.Generator, n: int, sig: dict) -> np.ndarray:
    """Build an (n, 26) additive-shift matrix from per-group sigmas."""
    shift = np.zeros((n, NUM_FEATURES), dtype=np.float64)
    if sig.get('flex', 0) > 0:
        shift[:, FLEX_SLICE] = rng.normal(0.0, sig['flex'], size=(n, 10))
    if sig.get('fsr', 0) > 0:
        shift[:, FSR_SLICE] = rng.normal(0.0, sig['fsr'], size=(n, 6))
    if sig.get('euler', 0) > 0:
        shift[:, ORIENT_SLICE] = rng.normal(0.0, sig['euler'], size=(n, 6))
    return shift


def augment_features_frame(
    X: np.ndarray,
    rng: np.random.Generator,
    policy: dict | None = None,
    label: str | None = None,
) -> np.ndarray:
    """Apply per-frame augmentation to a feature matrix using a group policy.

    Args:
        X: shape ``(N, NUM_FEATURES)``; each row is an independent frame.
        rng: numpy Generator for deterministic augmentation.
        policy: dict overriding ml.config.REAL_AUG entries. The policy schema
            mirrors REAL_AUG (per-group amplitude and baseline_shift dicts;
            scalar noise/dropout params).
        label: optional class label. When supplied, label-specific policies
            from REAL_AUG_LABEL_POLICIES are merged before the explicit policy.

    Returns:
        A NEW array of the same shape; X is not modified.
    """
    if X.size == 0:
        return X.copy()

    pol = _merged_policy(label=label, override=policy)

    X = X.astype(np.float64, copy=True)
    n, d = X.shape
    assert d == NUM_FEATURES, f"expected {NUM_FEATURES}-dim features, got {d}"

    # 1) Per-channel amplitude scale (multiplicative, per-group).
    amp = pol.get('amplitude_scale') or {}
    if any(v > 0 for v in amp.values()):
        X *= _per_channel_amp_scale(rng, n, amp)

    # 2) Per-channel baseline shift (additive, per-group).
    bsh = pol.get('baseline_shift_sigma') or {}
    if any(v > 0 for v in bsh.values()):
        X += _per_channel_baseline_shift(rng, n, bsh)

    # 3) Global Gaussian noise — but suppress on weight=0 channels so masked
    #    features stay exactly 0 after augmentation.
    sigma = float(pol.get('noise_sigma', 0.0) or 0.0)
    if sigma > 0:
        noise = rng.normal(0.0, sigma, size=X.shape)
        noise *= _NOISE_MASK  # zero noise on masked channels
        X += noise

    # 4) Euler jitter (degrees) — added directly to the orientation channels.
    ej = float(pol.get('euler_jitter_deg', 0.0) or 0.0)
    if ej > 0:
        X[:, ORIENT_SLICE] += rng.uniform(-ej, ej, size=(n, 6))
    # 4b) Rare constant yaw bias across a whole frame batch (one value applied
    #     uniformly to each row's yaw channels).
    yaw_p = float(pol.get('yaw_bias_prob', 0.0) or 0.0)
    yaw_deg = float(pol.get('yaw_bias_deg', 0.0) or 0.0)
    if yaw_p > 0 and yaw_deg > 0:
        # Decide per-frame whether to apply a yaw bias, then sample its value.
        apply = rng.random(n) < yaw_p
        bias = rng.uniform(-yaw_deg, yaw_deg, size=n)
        X[apply, 18] += bias[apply]   # R_yaw
        X[apply, 21] += bias[apply]   # L_yaw

    # 5) Single-channel dropout — replace with column mean to mimic a flaky
    #    sensor briefly missing.
    dp = float(pol.get('sensor_dropout_prob', 0.0) or 0.0)
    if dp > 0:
        mask = rng.random((n, d)) < dp
        col_mean = X.mean(axis=0, keepdims=True)
        X = np.where(mask, np.broadcast_to(col_mean, X.shape), X)

    # 6) Extra FSR-only dropout — pick at most ONE FSR channel per row and zero it.
    fdp = float(pol.get('fsr_dropout_prob', 0.0) or 0.0)
    if fdp > 0:
        drop_row = rng.random(n) < fdp
        if drop_row.any():
            # Choose which FSR index (10..15) to drop per affected row
            fsr_idx = rng.integers(10, 16, size=drop_row.sum())
            row_ids = np.where(drop_row)[0]
            X[row_ids, fsr_idx] = 0.0

    # 7) Hard-clip channels with [0, 1] semantics.
    X[:, FLEX_SLICE] = np.clip(X[:, FLEX_SLICE], 0.0, 1.0)
    X[:, FSR_SLICE] = np.clip(X[:, FSR_SLICE], 0.0, 1.0)

    # 8) Apply hardware mask + recompute derived features so 22-25 reflect
    #    the perturbed source channels (not the originals).
    for i in range(n):
        apply_sensor_mask(X[i])
        recompute_derived(X[i])

    # Clip derived hand-openness too (defensive).
    X[:, 22:24] = np.clip(X[:, 22:24], 0.0, 1.0)

    return X


def augment_sequence(
    X: np.ndarray,
    rng: np.random.Generator,
    time_warp_range: float | None = None,
    temporal_shift_max: int | None = None,
    policy: dict | None = None,
    label: str | None = None,
) -> np.ndarray:
    """Apply per-sequence augmentation to a dynamic CNN-LSTM segment.

    Args:
        X: shape ``(T, NUM_FEATURES)``.
        rng: numpy Generator.
        time_warp_range: stretch/shrink the timeline by ±this fraction, then
            resample back to T frames. Defaults to REAL_AUG['time_warp_range'].
        temporal_shift_max: max ± frames of circular shift. Defaults to
            REAL_AUG['temporal_shift_max'].
        policy: forwarded to augment_features_frame.
        label: optional class label for label-specific frame augmentation.

    Returns:
        Augmented sequence of the same shape as X.
    """
    if X.size == 0:
        return X.copy()

    pol = _merged_policy(label=label, override=policy)
    if time_warp_range is None:
        time_warp_range = float(pol.get('time_warp_range', 0.0) or 0.0)
    if temporal_shift_max is None:
        temporal_shift_max = int(pol.get('temporal_shift_max', 0) or 0)

    X = X.astype(np.float64, copy=True)
    T = X.shape[0]

    # 1) Time warp: resample to a different length, then back to T
    if time_warp_range > 0:
        warp = 1.0 + float(rng.uniform(-time_warp_range, time_warp_range))
        src_T = max(2, int(round(T * warp)))
        src_x = np.linspace(0.0, T - 1, src_T)
        dst_x = np.linspace(0.0, src_T - 1, T)
        intermediate = np.empty((src_T, X.shape[1]), dtype=np.float64)
        for j in range(X.shape[1]):
            intermediate[:, j] = np.interp(src_x, np.arange(T), X[:, j])
        warped = np.empty_like(X)
        for j in range(X.shape[1]):
            warped[:, j] = np.interp(dst_x, np.arange(src_T), intermediate[:, j])
        X = warped

    # 2) Small circular temporal shift
    if temporal_shift_max > 0:
        shift = int(rng.integers(-temporal_shift_max, temporal_shift_max + 1))
        if shift != 0:
            X = np.roll(X, shift, axis=0)

    # 3) Frame-level augmentation (applies mask + recompute_derived per row)
    X = augment_features_frame(X, rng, policy=policy, label=label)

    return X
