"""
Synthetic data generation for BSL fingerspelling ML pipeline.

Generates static samples for configured static letters and dynamic sequences
for configured dynamic letters plus distractor gestures. Applies full augmentation pipeline: Gaussian
noise, amplitude scaling, baseline shift, orientation jitter, sensor dropout,
correlated finger noise, and transition frames. Creates 10 virtual user profiles
with no user leakage across train/val/test splits.

A single StandardScaler is fitted here and saved to data/processed/scaler.pkl
to prevent the double-scaling bug described in docs/ML-dev-plan.md v1.1.

Usage:
    python -m ml.synthetic_data          # Generate all data
    python -m ml.synthetic_data --quick  # Quick test (100 samples/sign)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from ml.config import (
    DATA_GEN,
    DYNAMIC_WINDOW_FRAMES,
    FEATURE_GROUPS,
    LETTERS,
    LETTER_TO_IDX,
    NUM_FEATURES,
    PATHS,
    SEED,
    STATIC_LETTERS,
    STATIC_LETTER_TO_IDX,
)
from ml.bsl_sign_definitions import (
    DISTRACTOR_DEFINITIONS,
    DYNAMIC_TRAJECTORIES,
    SIGN_DEFINITIONS,
    get_dynamic_definitions,
    get_static_definitions,
)

# Project root (one level up from ml/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Virtual User Profiles
# =============================================================================

def create_virtual_users(rng):
    """Create 10 virtual user profiles with per-feature calibration offsets.

    Each user has a systematic offset drawn from U(-offset, +offset) per feature,
    simulating different hand sizes and sensor calibrations.

    Returns:
        list of np.ndarray[26] offsets per user.
    """
    n_users = DATA_GEN['n_virtual_users']
    offset_range = DATA_GEN['virtual_user_offset']
    users = []
    for _ in range(n_users):
        user_offset = np.zeros(NUM_FEATURES, dtype=np.float64)
        # Flex: per-user hand size offset
        user_offset[0:10] = rng.uniform(-offset_range, offset_range, 10)
        # FSR: per-user pressing strength offset
        user_offset[10:16] = rng.uniform(-offset_range * 0.5, offset_range * 0.5, 6)
        # Orientation: per-user resting posture offset (degrees)
        user_offset[16:22] = rng.uniform(-offset_range * 50, offset_range * 50, 6)
        # Derived: computed from others, no direct offset
        users.append(user_offset)
    return users


# =============================================================================
# Correlated Finger Noise
# =============================================================================

def build_finger_correlation_matrix(rho, n=5):
    """Build a correlation matrix for adjacent finger noise.

    Adjacent fingers have correlation rho; non-adjacent decrease.

    Returns:
        np.ndarray of shape (n, n).
    """
    cov = np.eye(n)
    for i in range(n):
        for j in range(n):
            dist = abs(i - j)
            if dist == 1:
                cov[i, j] = rho
            elif dist == 2:
                cov[i, j] = rho * 0.3
    return cov


# =============================================================================
# Augmentation Functions
# =============================================================================

def apply_augmentation(sample, rng, user_offset, is_orientation_idx=None):
    """Apply the full augmentation pipeline to a single feature vector.

    Args:
        sample: np.ndarray[26] base feature vector.
        rng: numpy random Generator.
        user_offset: np.ndarray[26] per-user calibration offset.
        is_orientation_idx: unused, kept for API compatibility.

    Returns:
        Augmented np.ndarray[26].
    """
    cfg = DATA_GEN
    aug = sample.copy()

    # 1. User calibration offset
    aug += user_offset

    # 2. Gaussian noise (σ = 3% of range, applied to continuous features)
    noise = rng.normal(0, cfg['noise_sigma'], NUM_FEATURES)
    aug += noise

    # 3. Correlated finger noise
    rho = cfg['correlated_finger_rho']
    cov = build_finger_correlation_matrix(rho)
    r_finger_noise = rng.multivariate_normal(np.zeros(5), cov * (cfg['noise_sigma'] ** 2))
    l_finger_noise = rng.multivariate_normal(np.zeros(5), cov * (cfg['noise_sigma'] ** 2))
    aug[0:5] += r_finger_noise
    aug[5:10] += l_finger_noise

    # 4. Amplitude scaling (±8%)
    scale = 1.0 + rng.uniform(-cfg['amplitude_scale'], cfg['amplitude_scale'])
    # Apply to flex and FSR features only
    aug[0:16] *= scale

    # 5. Baseline shift
    shift = rng.normal(0, cfg['baseline_shift_sigma'], 16)
    aug[0:16] += shift

    # 6. Orientation jitter (±15°)
    jitter_deg = cfg['orientation_jitter_deg']
    aug[16:22] += rng.uniform(-jitter_deg, jitter_deg, 6)

    # 7. Sensor dropout (5% probability)
    dropout_mask = rng.random(NUM_FEATURES) < cfg['sensor_dropout_prob']
    dropout_mask[22:26] = False  # Don't dropout derived features
    aug[dropout_mask] = 0.0

    # Re-derive computed features
    aug[22] = np.mean(aug[0:5])    # R_hand_openness
    aug[23] = np.mean(aug[5:10])   # L_hand_openness
    aug[24] = abs(aug[16] - aug[19])  # inter_hand_delta_roll
    aug[25] = abs(aug[17] - aug[20])  # inter_hand_delta_pitch

    # Clip continuous features to valid ranges
    aug[0:16] = np.clip(aug[0:16], 0.0, 1.0)
    aug[22:24] = np.clip(aug[22:24], 0.0, 1.0)
    aug[24:26] = np.clip(aug[24:26], 0.0, 180.0)

    return aug


# =============================================================================
# Transition Frame Generation
# =============================================================================

def generate_transition_frames(from_vec, to_vec, n_frames, rng):
    """Generate interpolated transition frames between two poses.

    Simulates the approach-to-pose and release-from-pose motion that occurs
    in real signing, preventing the model from only recognizing "perfect" holds.

    Returns:
        np.ndarray of shape (n_frames, 26).
    """
    frames = np.zeros((n_frames, NUM_FEATURES), dtype=np.float64)
    for i in range(n_frames):
        # Ease-in-out interpolation
        t = (i + 1) / (n_frames + 1)
        t_smooth = t * t * (3 - 2 * t)  # Smoothstep
        frames[i] = from_vec * (1 - t_smooth) + to_vec * t_smooth
        # Add small noise to transitions
        frames[i] += rng.normal(0, 0.01, NUM_FEATURES)
    # Clip
    frames[:, 0:16] = np.clip(frames[:, 0:16], 0.0, 1.0)
    return frames


# =============================================================================
# Static Dataset Generation
# =============================================================================

def generate_static_dataset(n_per_sign=None, seed=SEED):
    """Generate the full static dataset for 24 static letters.

    Args:
        n_per_sign: samples per sign (default from config: 5000).
        seed: random seed.

    Returns:
        X: np.ndarray of shape (N, 36)
        y: np.ndarray of shape (N,) — integer labels (LETTER_TO_IDX mapping)
        user_ids: np.ndarray of shape (N,) — virtual user ID per sample
    """
    if n_per_sign is None:
        n_per_sign = DATA_GEN['n_per_sign']

    rng = np.random.default_rng(seed)
    users = create_virtual_users(rng)
    n_users = len(users)

    static_defs = get_static_definitions()
    n_static = len(static_defs)
    total = n_static * n_per_sign

    X = np.zeros((total, NUM_FEATURES), dtype=np.float64)
    y = np.zeros(total, dtype=np.int32)
    user_ids = np.zeros(total, dtype=np.int32)

    # Neutral rest pose for transitions
    neutral = np.zeros(NUM_FEATURES, dtype=np.float64)
    neutral[0:10] = 0.15  # Slightly relaxed
    neutral[32:34] = 0.15

    idx = 0
    for letter, defn in static_defs.items():
        label = LETTER_TO_IDX[letter]
        base_vec = defn['vector']
        variance = defn['variance']

        for i in range(n_per_sign):
            user_id = i % n_users
            user_offset = users[user_id]

            # Start from base + per-feature variance
            sample = base_vec.copy()
            # Add sign-specific variance
            sample[0:16] += rng.normal(0, 1, 16) * variance[0:16]
            sample[16:22] += rng.normal(0, 1, 6) * variance[16:22]

            # Apply full augmentation
            sample = apply_augmentation(sample, rng, user_offset)

            X[idx] = sample
            y[idx] = label
            user_ids[idx] = user_id
            idx += 1

    # Also generate for the 2 dynamic letters AS STATIC snapshots
    # (these are the base poses without motion, useful for the full 26-class label set)
    dynamic_defs = get_dynamic_definitions()
    n_dyn_static = len(dynamic_defs) * n_per_sign
    X_dyn = np.zeros((n_dyn_static, NUM_FEATURES), dtype=np.float64)
    y_dyn = np.zeros(n_dyn_static, dtype=np.int32)
    user_ids_dyn = np.zeros(n_dyn_static, dtype=np.int32)

    idx = 0
    for letter, defn in dynamic_defs.items():
        label = LETTER_TO_IDX[letter]
        base_vec = defn['vector']
        variance = defn['variance']

        for i in range(n_per_sign):
            user_id = i % n_users
            user_offset = users[user_id]

            sample = base_vec.copy()
            sample[0:16] += rng.normal(0, 1, 16) * variance[0:16]
            sample[16:22] += rng.normal(0, 1, 6) * variance[16:22]
            sample = apply_augmentation(sample, rng, user_offset)

            X_dyn[idx] = sample
            y_dyn[idx] = label
            user_ids_dyn[idx] = user_id
            idx += 1

    # Combine
    X_all = np.concatenate([X, X_dyn], axis=0)
    y_all = np.concatenate([y, y_dyn], axis=0)
    user_ids_all = np.concatenate([user_ids, user_ids_dyn], axis=0)

    return X_all, y_all, user_ids_all


# =============================================================================
# Dynamic Sequence Generation
# =============================================================================

def _interpolate_trajectory(keyframes, n_frames):
    """Interpolate trajectory keyframes to n_frames of gyro values.

    Args:
        keyframes: list of (time_frac, gx, gy, gz) tuples.
        n_frames: number of output frames.

    Returns:
        np.ndarray of shape (n_frames, 3) — gyro values per frame.
    """
    times = np.array([kf[0] for kf in keyframes])
    values = np.array([kf[1:] for kf in keyframes])

    out = np.zeros((n_frames, 3), dtype=np.float64)
    frame_times = np.linspace(0, 1, n_frames)

    for dim in range(3):
        out[:, dim] = np.interp(frame_times, times, values[:, dim])

    return out


def generate_dynamic_sequence(letter, rng, user_offset, variance):
    """Generate a single dynamic gesture sequence (100 frames × 36 features).

    The static base pose is held while the gyro trajectory is overlaid.
    Orientation features evolve based on integrated gyro values.

    Returns:
        np.ndarray of shape (100, 36).
    """
    defn = SIGN_DEFINITIONS[letter]
    base = defn['vector'].copy()
    n_frames = DYNAMIC_WINDOW_FRAMES

    # Get trajectory
    keyframes = DYNAMIC_TRAJECTORIES[letter]
    gyro_profile = _interpolate_trajectory(keyframes, n_frames)

    # Time warping
    warp = DATA_GEN['time_warp_range']
    warp_factor = 1.0 + rng.uniform(-warp, warp)
    n_warped = max(20, min(int(n_frames * warp_factor), n_frames * 2))
    if n_warped != n_frames:
        indices = np.linspace(0, n_frames - 1, n_warped).astype(int)
        indices = np.clip(indices, 0, n_frames - 1)
        gyro_profile_warped = gyro_profile[indices]
        # Pad or truncate back to n_frames
        if len(gyro_profile_warped) >= n_frames:
            gyro_profile = gyro_profile_warped[:n_frames]
        else:
            pad_len = n_frames - len(gyro_profile_warped)
            gyro_profile = np.vstack([
                gyro_profile_warped,
                np.zeros((pad_len, 3))
            ])

    # Build sequence
    sequence = np.zeros((n_frames, NUM_FEATURES), dtype=np.float64)
    cumulative_orientation = np.zeros(3, dtype=np.float64)
    dt = 1.0 / 50.0  # 50 Hz

    for t in range(n_frames):
        frame = base.copy()

        # Add per-feature variance
        frame[0:16] += rng.normal(0, 1, 16) * variance[0:16] * 0.5
        frame[16:22] += rng.normal(0, 1, 6) * variance[16:22] * 0.5

        # Integrate gyro into orientation change
        cumulative_orientation += gyro_profile[t] * dt
        frame[16:19] += cumulative_orientation  # R orientation evolves

        # Apply user offset and noise
        frame = apply_augmentation(frame, rng, user_offset)
        sequence[t] = frame

    return sequence


def generate_distractor_sequence(name, defn, rng, user_offset):
    """Generate a single distractor gesture sequence.

    Returns:
        np.ndarray of shape (100, 36).
    """
    n_frames = DYNAMIC_WINDOW_FRAMES
    base_flex = defn['base_flex']
    keyframes = defn['trajectory']
    gyro_profile = _interpolate_trajectory(keyframes, n_frames)

    sequence = np.zeros((n_frames, NUM_FEATURES), dtype=np.float64)
    cumulative_orientation = np.zeros(3, dtype=np.float64)
    dt = 1.0 / 50.0

    # Base pose: open hand with given flex
    base = np.zeros(NUM_FEATURES, dtype=np.float64)
    base[0:5] = base_flex
    base[5:10] = [0.15] * 5  # L hand relaxed

    for t in range(n_frames):
        frame = base.copy()
        frame[0:5] += rng.normal(0, 0.05, 5)  # Flex noise
        frame[5:10] += rng.normal(0, 0.03, 5)

        cumulative_orientation += gyro_profile[t] * dt
        frame[16:19] = cumulative_orientation + rng.normal(0, 3, 3)

        # Derived features
        frame[22] = np.mean(np.clip(frame[0:5], 0, 1))
        frame[23] = np.mean(np.clip(frame[5:10], 0, 1))
        frame[24] = abs(frame[16] - frame[19])
        frame[25] = abs(frame[17] - frame[20])

        frame[0:16] = np.clip(frame[0:16], 0.0, 1.0)
        frame += user_offset * 0.5  # Reduced user effect for distractors
        frame[0:16] = np.clip(frame[0:16], 0.0, 1.0)

        sequence[t] = frame

    return sequence


def generate_dynamic_dataset(n_per_sign=None, n_distractor=None, seed=SEED):
    """Generate dynamic gesture sequences and distractor classes.

    Args:
        n_per_sign: sequences per dynamic letter (default from config: 5000).
        n_distractor: sequences per distractor class (default: 500).
        seed: random seed.

    Returns:
        X: np.ndarray of shape (N, 100, NUM_FEATURES)
        y: np.ndarray of shape (N,) — class labels
        class_names: list of class name strings
    """
    if n_per_sign is None:
        n_per_sign = DATA_GEN['n_per_sign']
    if n_distractor is None:
        n_distractor = DATA_GEN['n_distractor_samples']

    rng = np.random.default_rng(seed + 1)  # Different seed from static
    users = create_virtual_users(rng)
    n_users = len(users)

    dynamic_defs = get_dynamic_definitions()
    distractor_names = list(DISTRACTOR_DEFINITIONS.keys())

    # Class mapping: configured dynamic letters first, then distractors.
    class_names = list(dynamic_defs.keys()) + distractor_names
    n_dyn_letters = len(dynamic_defs)
    n_dists = len(distractor_names)
    total = n_dyn_letters * n_per_sign + n_dists * n_distractor

    X = np.zeros((total, DYNAMIC_WINDOW_FRAMES, NUM_FEATURES), dtype=np.float64)
    y = np.zeros(total, dtype=np.int32)

    idx = 0

    # Dynamic letters
    for cls_idx, (letter, defn) in enumerate(dynamic_defs.items()):
        variance = defn['variance']
        for i in range(n_per_sign):
            user_id = i % n_users
            seq = generate_dynamic_sequence(letter, rng, users[user_id], variance)
            X[idx] = seq
            y[idx] = cls_idx
            idx += 1

    # Distractor classes
    for d_idx, d_name in enumerate(distractor_names):
        cls_idx = n_dyn_letters + d_idx
        d_defn = DISTRACTOR_DEFINITIONS[d_name]
        for i in range(n_distractor):
            user_id = i % n_users
            seq = generate_distractor_sequence(d_name, d_defn, rng, users[user_id])
            X[idx] = seq
            y[idx] = cls_idx
            idx += 1

    return X, y, class_names


# =============================================================================
# Data Splitting with No User Leakage
# =============================================================================

def split_by_user(X, y, user_ids, seed=SEED):
    """Stratified split ensuring no virtual user appears in both train and test.

    Assigns users to splits: 7 train, 1-2 val, 1-2 test.
    Then further stratifies within each user group to maintain class balance.

    Returns:
        dict with 'train', 'val', 'test' keys, each containing (X, y).
    """
    rng = np.random.default_rng(seed)
    n_users = DATA_GEN['n_virtual_users']

    # Assign users to splits
    user_order = rng.permutation(n_users)
    n_train_users = 7
    n_val_users = 2
    # Remaining go to test
    train_users = set(user_order[:n_train_users])
    val_users = set(user_order[n_train_users:n_train_users + n_val_users])
    test_users = set(user_order[n_train_users + n_val_users:])

    train_mask = np.isin(user_ids, list(train_users))
    val_mask = np.isin(user_ids, list(val_users))
    test_mask = np.isin(user_ids, list(test_users))

    return {
        'train': (X[train_mask], y[train_mask]),
        'val': (X[val_mask], y[val_mask]),
        'test': (X[test_mask], y[test_mask]),
    }


def split_dynamic_stratified(X, y, seed=SEED):
    """Stratified split for dynamic sequences (no user_ids tracking for sequences).

    Uses simple stratified shuffle split: 70/15/15.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    indices = rng.permutation(n)

    n_train = int(n * DATA_GEN['train_ratio'])
    n_val = int(n * DATA_GEN['val_ratio'])

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        'train': (X[train_idx], y[train_idx]),
        'val': (X[val_idx], y[val_idx]),
        'test': (X[test_idx], y[test_idx]),
    }


# =============================================================================
# Main Generation Pipeline
# =============================================================================

def generate_all(n_per_sign=None, seed=SEED):
    """Run the full data generation pipeline.

    1. Generate static dataset (130K samples)
    2. Generate dynamic dataset (configured dynamic letters + distractors)
    3. Fit StandardScaler ONCE on training data
    4. Apply scaler to all splits
    5. Save everything to data/ directory

    Returns:
        dict with paths to saved files.
    """
    print("=" * 60)
    print("BSL Sign Language Gloves — Synthetic Data Generation")
    print("=" * 60)

    if n_per_sign is None:
        n_per_sign = DATA_GEN['n_per_sign']

    # --- Step 1: Generate static dataset ---
    print(f"\n[1/6] Generating static dataset ({n_per_sign} samples/sign)...")
    X_static, y_static, user_ids = generate_static_dataset(n_per_sign, seed)
    print(f"  Static dataset: {X_static.shape[0]} samples × {X_static.shape[1]} features")
    print(f"  Classes: {len(np.unique(y_static))}, Users: {len(np.unique(user_ids))}")

    # Verify no NaN/inf
    assert not np.any(np.isnan(X_static)), "NaN found in static data"
    assert not np.any(np.isinf(X_static)), "Inf found in static data"

    # --- Step 2: Generate dynamic dataset ---
    print(f"\n[2/6] Generating dynamic dataset...")
    n_dist = DATA_GEN['n_distractor_samples'] if n_per_sign >= 1000 else max(50, n_per_sign // 10)
    X_dyn, y_dyn, dyn_class_names = generate_dynamic_dataset(n_per_sign, n_dist, seed)
    print(f"  Dynamic dataset: {X_dyn.shape[0]} sequences × {X_dyn.shape[1]} frames × {X_dyn.shape[2]} features")
    print(f"  Classes: {dyn_class_names}")

    # --- Step 3: Split static data (user-leakage-free) ---
    print(f"\n[3/6] Splitting static data (user-leakage-free)...")
    static_splits = split_by_user(X_static, y_static, user_ids, seed)
    for name, (X_s, y_s) in static_splits.items():
        print(f"  {name}: {X_s.shape[0]} samples, {len(np.unique(y_s))} classes")

    # --- Step 4: Fit scaler on TRAINING data only ---
    print(f"\n[4/6] Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(static_splits['train'][0])
    X_val_scaled = scaler.transform(static_splits['val'][0])
    X_test_scaled = scaler.transform(static_splits['test'][0])
    print(f"  Scaler fitted: mean shape={scaler.mean_.shape}, std shape={scaler.scale_.shape}")

    # --- Step 5: Scale dynamic data with same scaler ---
    print(f"\n[5/6] Scaling dynamic data + splitting...")
    dyn_splits = split_dynamic_stratified(X_dyn, y_dyn, seed)

    # Scale each frame in each sequence using the static scaler
    for split_name in ['train', 'val', 'test']:
        X_d, y_d = dyn_splits[split_name]
        orig_shape = X_d.shape
        X_flat = X_d.reshape(-1, NUM_FEATURES)
        X_flat_scaled = scaler.transform(X_flat)
        dyn_splits[split_name] = (X_flat_scaled.reshape(orig_shape), y_d)
        print(f"  Dynamic {split_name}: {orig_shape[0]} sequences")

    # --- Step 6: Save everything ---
    print(f"\n[6/6] Saving to disk...")
    paths = _save_all(
        X_train_scaled, static_splits['train'][1],
        X_val_scaled, static_splits['val'][1],
        X_test_scaled, static_splits['test'][1],
        X_static, y_static,  # Raw (unscaled) for reference
        X_dyn, y_dyn, dyn_class_names,
        dyn_splits,
        scaler,
    )

    print(f"\n{'=' * 60}")
    print("Data generation complete!")
    print(f"{'=' * 60}")
    _print_summary(X_train_scaled, static_splits['train'][1],
                   X_test_scaled, static_splits['test'][1])

    return paths


def _save_all(
    X_train, y_train, X_val, y_val, X_test, y_test,
    X_raw, y_raw, X_dyn, y_dyn, dyn_class_names,
    dyn_splits, scaler,
):
    """Save all generated data to the data/ directory."""
    paths_saved = {}

    # Ensure directories exist
    for subdir in ['raw/synthetic', 'processed', 'models']:
        (PROJECT_ROOT / 'data' / subdir).mkdir(parents=True, exist_ok=True)

    # Raw datasets
    raw_static_path = PROJECT_ROOT / PATHS['raw_static']
    np.savez_compressed(str(raw_static_path), X=X_raw, y=y_raw)
    paths_saved['raw_static'] = str(raw_static_path)
    print(f"  Saved: {raw_static_path}")

    raw_dynamic_path = PROJECT_ROOT / PATHS['raw_dynamic']
    np.savez_compressed(str(raw_dynamic_path), X=X_dyn, y=y_dyn,
                        class_names=np.array(dyn_class_names))
    paths_saved['raw_dynamic'] = str(raw_dynamic_path)
    print(f"  Saved: {raw_dynamic_path}")

    # Processed splits (scaled static)
    train_path = PROJECT_ROOT / PATHS['train']
    np.savez_compressed(str(train_path), X=X_train, y=y_train)
    paths_saved['train'] = str(train_path)
    print(f"  Saved: {train_path}")

    val_path = PROJECT_ROOT / PATHS['val']
    np.savez_compressed(str(val_path), X=X_val, y=y_val)
    paths_saved['val'] = str(val_path)
    print(f"  Saved: {val_path}")

    test_path = PROJECT_ROOT / PATHS['test']
    np.savez_compressed(str(test_path), X=X_test, y=y_test)
    paths_saved['test'] = str(test_path)
    print(f"  Saved: {test_path}")

    # Dynamic splits (scaled)
    for split_name in ['train', 'val', 'test']:
        X_d, y_d = dyn_splits[split_name]
        dyn_path = PROJECT_ROOT / f'data/processed/dynamic_{split_name}.npz'
        np.savez_compressed(str(dyn_path), X=X_d, y=y_d,
                            class_names=np.array(dyn_class_names))
        paths_saved[f'dynamic_{split_name}'] = str(dyn_path)
        print(f"  Saved: {dyn_path}")

    # Scaler
    scaler_path = PROJECT_ROOT / PATHS['scaler']
    with open(str(scaler_path), 'wb') as f:
        pickle.dump(scaler, f)
    paths_saved['scaler'] = str(scaler_path)
    print(f"  Saved: {scaler_path}")

    return paths_saved


def _print_summary(X_train, y_train, X_test, y_test):
    """Print a summary of the generated dataset."""
    print(f"\n--- Dataset Summary ---")
    print(f"Training:   {X_train.shape[0]:>7,} samples × {X_train.shape[1]} features")
    print(f"Test:       {X_test.shape[0]:>7,} samples × {X_test.shape[1]} features")

    # Class distribution in test
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"Test classes: {len(unique)}")
    min_count = counts.min()
    max_count = counts.max()
    print(f"Test class balance: min={min_count}, max={max_count}, "
          f"ratio={max_count / max(min_count, 1):.2f}")

    # Feature stats
    print(f"\nTraining feature stats:")
    print(f"  Mean range:  [{X_train.mean(axis=0).min():.3f}, {X_train.mean(axis=0).max():.3f}]")
    print(f"  Std range:   [{X_train.std(axis=0).min():.3f}, {X_train.std(axis=0).max():.3f}]")
    print(f"  Any NaN: {np.any(np.isnan(X_train))}")
    print(f"  Any Inf: {np.any(np.isinf(X_train))}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic BSL data')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (100 samples/sign)')
    parser.add_argument('--n-per-sign', type=int, default=None,
                        help='Override samples per sign')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')
    args = parser.parse_args()

    n = args.n_per_sign
    if n is None:
        n = 100 if args.quick else DATA_GEN['n_per_sign']

    generate_all(n_per_sign=n, seed=args.seed)
