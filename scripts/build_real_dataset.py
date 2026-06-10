"""
Build Real Dataset — one-shot orchestrator.

Walks data/raw/real/, loads and trims all JSONL sessions, performs a
session-level 80/10/10 stratified split by (user, letter), fits a
StandardScaler on real training frames only, and saves:

    data/processed/real/train.npz
    data/processed/real/val.npz
    data/processed/real/test.npz
    data/processed/real/dynamic_train.npz
    data/processed/real/dynamic_val.npz
    data/processed/real/dynamic_test.npz
    data/processed/real/scaler.pkl
    data/processed/real/coverage_report.json

This script is the ONLY way real data enters the ML pipeline.
All downstream training reads .npz files; never re-walks the JSONL tree.

Usage:
    python -m scripts.build_real_dataset
    python -m scripts.build_real_dataset --min-settled 0.6
    python -m scripts.build_real_dataset --dry-run   # prints inventory only
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Project root setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import (
    CALIBRATION,
    DYNAMIC_LETTERS,
    DYNAMIC_WINDOW_FRAMES,
    FEATURE_GROUPS,
    LETTER_TO_IDX,
    NUM_FEATURES,
    PATHS,
    SAMPLE_RATE_HZ,
    SEED,
)
from ml.feature_engineering import apply_sensor_mask
from ml.real_data import (
    compute_motion_energy_series,
    discover_sessions,
    extract_session_features,
    find_settled_window,
    load_users,
    pair_frames,
    SETTLE_THRESHOLD,
)
from ml.real_augmentation import augment_features_frame, augment_sequence
from ml.segmentation import extract_segment

OUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'real'
LOUO_ROOT = PROJECT_ROOT / PATHS['louo_root']


# =============================================================================
# Session inventory (load metadata without materialising features)
# =============================================================================

def build_session_inventory(
    min_settled_seconds: float = 0.6,
    settle_threshold: float = SETTLE_THRESHOLD,
    verbose: bool = True,
) -> dict:
    """Scan all user/letter JSONL sessions and record metadata.

    Returns:
        {
          'static': [
            {user_id, letter, path, settle_start, settle_end, n_settled_frames},
            ...
          ],
          'dynamic': [
            {user_id, letter, path, n_frames},
            ...
          ],
        }
    """
    users = load_users()
    static_letters = sorted(set(LETTER_TO_IDX.keys()) - DYNAMIC_LETTERS)
    min_settled_frames = int(min_settled_seconds * SAMPLE_RATE_HZ)

    static_sessions = []
    dynamic_sessions = []
    discarded = defaultdict(int)

    for user_id, user_info in users.items():
        if verbose:
            print(f"  Scanning user: {user_info['name']} ({user_id})")

        # --- Static letters ---
        for letter in static_letters:
            sessions = discover_sessions(user_id, letter)
            for sess_path in sessions:
                pairs = pair_frames(sess_path)
                if len(pairs) < min_settled_frames:
                    discarded['too_short_raw'] += 1
                    continue

                energy = compute_motion_energy_series(pairs)
                s_start, s_end = find_settled_window(energy, threshold=settle_threshold)
                n_settled = s_end - s_start + 1

                if n_settled < min_settled_frames:
                    discarded['too_short_settled'] += 1
                    continue

                static_sessions.append({
                    'user_id': user_id,
                    'letter': letter,
                    'path': str(sess_path),
                    'settle_start': s_start,
                    'settle_end': s_end,
                    'n_settled_frames': n_settled,
                })

        # --- Dynamic letters ---
        for letter in sorted(DYNAMIC_LETTERS):
            sessions = discover_sessions(user_id, letter)
            for sess_path in sessions:
                pairs = pair_frames(sess_path)
                if len(pairs) > 10:  # Minimum viable sequence
                    dynamic_sessions.append({
                        'user_id': user_id,
                        'letter': letter,
                        'path': str(sess_path),
                        'n_frames': len(pairs),
                    })

    if verbose:
        print(f"\n  Static sessions kept:   {len(static_sessions)}")
        print(f"  Static discarded (raw < min):      {discarded['too_short_raw']}")
        print(f"  Static discarded (settled < min):  {discarded['too_short_settled']}")
        print(f"  Dynamic sessions:       {len(dynamic_sessions)}")

    return {
        'static': static_sessions,
        'dynamic': dynamic_sessions,
        'discarded': dict(discarded),
    }


# =============================================================================
# Session-level split  80 / 10 / 10  stratified by (user, letter)
# =============================================================================

def split_sessions(
    sessions: list,
    seed: int = SEED,
    mode: str = 'same_user_session',
    test_user: str | None = None,
) -> dict:
    """Split sessions into train / val / test.

    Modes:
        ``same_user_session`` (default, current behaviour):
            For each (user, letter) bucket, shuffle and take 80/10/10.
            All users appear in all three splits — *inflates* test accuracy
            because the model has seen each user during training. Kept as the
            default because with only 2 wearers it's the most data-efficient
            way to produce a deployable model; pair with augmentation to
            broaden the distribution.

        ``leave_one_user_out``:
            All sessions from ``test_user`` go to test. All sessions from
            every OTHER user are split 90/10 between train and val. Honest
            cross-wearer generalization metric, but with only 2 users this
            halves the training data — use only for diagnostic evaluation,
            not for the deployed model.
    """
    rng = np.random.default_rng(seed)

    if mode == 'leave_one_user_out':
        if not test_user:
            raise ValueError("leave_one_user_out requires test_user")
        train, val, test = [], [], []
        # Test = every session by the held-out user
        held = [s for s in sessions if s['user_id'] == test_user]
        rest = [s for s in sessions if s['user_id'] != test_user]
        test.extend(held)
        # Stratify the remaining users' sessions by (user, letter) into 90/10
        buckets: dict[tuple, list] = defaultdict(list)
        for sess in rest:
            buckets[(sess['user_id'], sess['letter'])].append(sess)
        for (_, _), bucket in sorted(buckets.items()):
            shuffled = list(bucket)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_val = max(1, round(n * 0.10))
            n_train = max(0, n - n_val)
            train.extend(shuffled[:n_train])
            val.extend(shuffled[n_train:n_train + n_val])
        return {'train': train, 'val': val, 'test': test}

    # Default: same_user_session
    buckets = defaultdict(list)
    for sess in sessions:
        buckets[(sess['user_id'], sess['letter'])].append(sess)

    train, val, test = [], [], []
    for (_user_id, _letter), bucket in sorted(buckets.items()):
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, round(n * 0.10))
        n_val = max(1, round(n * 0.10))
        n_train = n - n_val - n_test
        if n_train <= 0:
            train.extend(shuffled)
            continue
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    return {'train': train, 'val': val, 'test': test}


# =============================================================================
# Feature materialisation
# =============================================================================

def _stratified_subsample(
    feats: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample frames stratified across early/middle/late thirds of the
    settled window.

    A uniform random pick can land entirely in one micro-pose within a session,
    which lets tree models memorize that pose. Splitting into thirds and
    drawing roughly n_target/3 from each third forces the per-session sample
    to span the pose's entire duration.
    """
    n = len(feats)
    if n <= n_target:
        return feats
    # Split indices into three roughly-equal contiguous thirds.
    third_edges = [0, n // 3, (2 * n) // 3, n]
    per_third = max(1, n_target // 3)
    picks: list[int] = []
    for t in range(3):
        lo, hi = third_edges[t], third_edges[t + 1]
        if hi <= lo:
            continue
        take = min(per_third, hi - lo)
        idx = rng.choice(np.arange(lo, hi), size=take, replace=False)
        picks.extend(idx.tolist())
    # Top up if we under-sampled due to small thirds.
    if len(picks) < n_target:
        remaining_pool = np.setdiff1d(np.arange(n), np.array(picks, dtype=np.int64))
        extra = min(n_target - len(picks), len(remaining_pool))
        if extra > 0:
            picks.extend(rng.choice(remaining_pool, size=extra, replace=False).tolist())
    picks = sorted(picks)
    return feats[picks]


def materialise_static(
    sessions: list,
    frames_per_session: int = 0,
    augment_mult: int = 0,
    seed: int = SEED,
    label_to_idx: dict | None = None,
    augment_mult_by_label: dict[str, int] | None = None,
) -> tuple:
    """Load features for a list of static session dicts.

    Args:
        sessions: list of session dicts from the inventory.
        frames_per_session: if > 0, randomly subsample this many frames per
            session (keeps all if the session has fewer). Caps per-session
            correlation which otherwise lets tree models memorize per-session
            sensor jitter.
        augment_mult: number of augmented copies to add per ORIGINAL frame
            (0 = no augmentation; the original is always kept). Should ONLY
            be > 0 for the train split.
        seed: RNG seed for subsampling + augmentation.
        label_to_idx: optional override for static labels outside A-Z.
        augment_mult_by_label: optional per-label override for augment_mult.

    Returns:
        X: np.ndarray (N, 26)
        y: np.ndarray (N,)
        user_ids: np.ndarray (N, object)
        session_ids: list[str]
    """
    rng = np.random.default_rng(seed)
    X_list, y_list, uid_list, sid_list = [], [], [], []

    for sess in sessions:
        pairs = pair_frames(Path(sess['path']))
        s, e = sess['settle_start'], sess['settle_end']
        settled_pairs = pairs[s:e + 1]
        if not settled_pairs:
            continue
        feats = extract_session_features(settled_pairs)

        if frames_per_session > 0 and len(feats) > frames_per_session:
            feats = _stratified_subsample(feats, frames_per_session, rng)

        label_map = label_to_idx or LETTER_TO_IDX
        letter_idx = label_map[sess['letter']]
        sid = Path(sess['path']).stem

        # Always include the originals
        X_list.append(feats)
        y_list.extend([letter_idx] * len(feats))
        uid_list.extend([sess['user_id']] * len(feats))
        sid_list.extend([sid] * len(feats))

        label_aug_mult = int((augment_mult_by_label or {}).get(sess['letter'], augment_mult))
        for k in range(label_aug_mult):
            aug = augment_features_frame(feats, rng, label=sess['letter'])
            # Re-zero broken-sensor channels that augmentation noise touched,
            # so train rows match what extract_features emits at inference.
            for row in aug:
                apply_sensor_mask(row)
            X_list.append(aug)
            y_list.extend([letter_idx] * len(aug))
            uid_list.extend([sess['user_id']] * len(aug))
            sid_list.extend([f"{sid}__aug{k}"] * len(aug))

    X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0, NUM_FEATURES))
    y = np.array(y_list, dtype=np.int32)
    user_ids = np.array(uid_list, dtype=object)
    return X, y, user_ids, sid_list


def materialise_dynamic(
    sessions: list,
    augment_mult: int = 0,
    seed: int = SEED,
    label_to_idx: dict | None = None,
    augment_mult_by_label: dict[str, int] | None = None,
) -> tuple:
    """Load features for a list of dynamic session dicts.

    Args:
        sessions: list of dynamic session dicts.
        augment_mult: number of augmented sequence copies per original
            (0 = no augmentation; original is always kept). Should ONLY be > 0
            for the train split.
        seed: RNG seed for augmentation.
        label_to_idx: optional override for dynamic labels outside A-Z.
        augment_mult_by_label: optional per-label override for augment_mult.

    Returns:
        X: np.ndarray (N, DYNAMIC_WINDOW_FRAMES, 26)
        y: np.ndarray (N,)
        user_ids: np.ndarray (N, object)
        session_ids: list[str]
    """
    rng = np.random.default_rng(seed)
    GRAVITY_WARMUP = 10  # frames to skip at start

    X_list, y_list, uid_list, sid_list = [], [], [], []

    for sess in sessions:
        pairs = pair_frames(Path(sess['path']))
        if len(pairs) <= GRAVITY_WARMUP:
            continue
        pairs = pairs[GRAVITY_WARMUP:]
        feats = extract_session_features(pairs)
        segment = extract_segment(feats, 0, len(feats) - 1, DYNAMIC_WINDOW_FRAMES)

        label_map = label_to_idx or LETTER_TO_IDX
        letter_idx = label_map[sess['letter']]
        sid = Path(sess['path']).stem

        X_list.append(segment)
        y_list.append(letter_idx)
        uid_list.append(sess['user_id'])
        sid_list.append(sid)

        label_aug_mult = int((augment_mult_by_label or {}).get(sess['letter'], augment_mult))
        for k in range(label_aug_mult):
            aug = augment_sequence(segment, rng, label=sess['letter'])
            # Re-zero broken-sensor channels frame-by-frame across the
            # sequence so train and inference are consistent.
            for row in aug:
                apply_sensor_mask(row)
            X_list.append(aug)
            y_list.append(letter_idx)
            uid_list.append(sess['user_id'])
            sid_list.append(f"{sid}__aug{k}")

    X = np.array(X_list, dtype=np.float64) if X_list else np.zeros((0, DYNAMIC_WINDOW_FRAMES, NUM_FEATURES))
    y = np.array(y_list, dtype=np.int32)
    user_ids = np.array(uid_list, dtype=object)
    return X, y, user_ids, sid_list


# =============================================================================
# Coverage report
# =============================================================================

def build_coverage_report(
    inventory: dict,
    static_splits: dict,
    dynamic_splits: dict,
) -> dict:
    """Build a human-readable coverage report dict."""
    users = load_users()
    user_names = {uid: info['name'] for uid, info in users.items()}

    static_letters = sorted(set(LETTER_TO_IDX.keys()) - DYNAMIC_LETTERS)

    # Per-split, per-letter, per-user session counts
    def count_sessions(sessions, key_fn):
        counts = defaultdict(lambda: defaultdict(int))
        for s in sessions:
            counts[key_fn(s)][s['user_id']] += 1
        return {k: dict(v) for k, v in counts.items()}

    static_per_letter_train = count_sessions(static_splits['train'], lambda s: s['letter'])
    static_per_letter_val = count_sessions(static_splits['val'], lambda s: s['letter'])
    static_per_letter_test = count_sessions(static_splits['test'], lambda s: s['letter'])

    return {
        'users': user_names,
        'total_sessions': {
            'static_kept': len(inventory['static']),
            'static_discarded': inventory['discarded'],
            'dynamic': len(inventory['dynamic']),
        },
        'static_splits': {
            'train': len(static_splits['train']),
            'val': len(static_splits['val']),
            'test': len(static_splits['test']),
        },
        'dynamic_splits': {
            'train': len(dynamic_splits['train']),
            'val': len(dynamic_splits['val']),
            'test': len(dynamic_splits['test']),
        },
        'static_letter_coverage': {
            'train': {l: static_per_letter_train.get(l, {}) for l in static_letters},
            'val': {l: static_per_letter_val.get(l, {}) for l in static_letters},
            'test': {l: static_per_letter_test.get(l, {}) for l in static_letters},
        },
    }


# =============================================================================
# Scaler-statistics sidecar (consumed by per-user runtime calibration)
# =============================================================================

def compute_scaler_stats(
    scaler,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Compute the JSON-serialisable scaler-statistics sidecar.

    Includes:
      - mean, scale: copies of scaler.mean_/scaler.scale_ for consumers that
        don't want to unpickle sklearn.
      - train_neutral_baseline: mean of training frames whose right- and
        left-hand openness are both below CALIBRATION['neutral_openness_threshold'].
        Used by the per-user calibration routine to compute an offset.
      - letter_flex_amplitudes: per-letter mean flex magnitude vs the neutral
        baseline, used to derive a per-flex amplitude scale during the
        optional reference-letter calibration step.

    The returned dict is plain Python (no ndarrays) so it can be `json.dump`ed.
    """
    flex_idx = np.array(FEATURE_GROUPS['flex'], dtype=np.int64)
    # Reconstruct the *unscaled* feature matrix from the scaled train set so
    # the baseline is in the same units the live pipeline emits before
    # scaler.transform.
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    X_unscaled = X_train * scale + mean

    open_thresh = float(CALIBRATION['neutral_openness_threshold'])
    neutral_mask = (X_unscaled[:, 22] < open_thresh) & (X_unscaled[:, 23] < open_thresh)
    if neutral_mask.any():
        neutral_baseline = X_unscaled[neutral_mask].mean(axis=0)
    else:
        # Fall back to the lowest-openness 5 % of frames if no row meets the threshold.
        score = X_unscaled[:, 22] + X_unscaled[:, 23]
        cutoff = np.quantile(score, 0.05)
        neutral_baseline = X_unscaled[score <= cutoff].mean(axis=0)

    # Per-letter mean of flex channels (unscaled). The calibration routine
    # divides the new user's flex amplitude (relative to their neutral) by this
    # training amplitude to derive a per-flex scale.
    letter_flex_amplitudes: dict[str, list[float]] = {}
    for letter, idx in sorted(LETTER_TO_IDX.items()):
        rows = y_train == idx
        if not rows.any():
            continue
        per_letter_mean = X_unscaled[rows][:, flex_idx].mean(axis=0)
        # Express as deviation from neutral so the consumer divides like with like.
        letter_flex_amplitudes[letter] = (per_letter_mean - neutral_baseline[flex_idx]).tolist()

    return {
        'mean': mean.tolist(),
        'scale': scale.tolist(),
        'train_neutral_baseline': neutral_baseline.tolist(),
        'letter_flex_amplitudes': letter_flex_amplitudes,
        'flex_indices': flex_idx.tolist(),
        'neutral_openness_threshold': open_thresh,
    }


# =============================================================================
# Main pipeline
# =============================================================================

def build(
    min_settled_seconds: float = 0.6,
    settle_threshold: float = SETTLE_THRESHOLD,
    seed: int = SEED,
    dry_run: bool = False,
    verbose: bool = True,
    split_mode: str = 'same_user_session',
    test_user: str | None = None,
    frames_per_session: int = 12,
    augment_mult: int = 4,
    dyn_augment_mult: int = 16,
    out_dir: Path | None = None,
):
    """Run the full build pipeline.

    Steps:
        1. Scan all user/letter sessions → session inventory
        2. Split 80/10/10 (or LOUO) at session level
        3. Materialise features for each split (subsample + augment train only)
        4. Fit StandardScaler on real training frames ONLY
        5. Scale all splits
        6. Save .npz files + scaler + coverage report
    """
    sep = '=' * 65

    print(sep)
    print('Build Real Dataset')
    print(sep)

    # --- Step 1: Session inventory ---
    print('\n[1/6] Scanning JSONL sessions...')
    inventory = build_session_inventory(min_settled_seconds, settle_threshold, verbose)

    if dry_run:
        print('\n[DRY RUN] Inventory only — not writing any files.')
        print(f"  Static sessions: {len(inventory['static'])}")
        print(f"  Dynamic sessions: {len(inventory['dynamic'])}")
        return inventory

    # --- Step 2: Split sessions ---
    print(f'\n[2/6] Splitting sessions (mode={split_mode})...')
    static_splits = split_sessions(
        inventory['static'], seed=seed, mode=split_mode, test_user=test_user,
    )
    dynamic_splits = split_sessions(
        inventory['dynamic'], seed=seed, mode=split_mode, test_user=test_user,
    )

    for split_name in ['train', 'val', 'test']:
        print(f"  Static  {split_name:5s}: {len(static_splits[split_name])} sessions")
        print(f"  Dynamic {split_name:5s}: {len(dynamic_splits[split_name])} sessions")

    # --- Step 3: Materialise features ---
    print(
        f'\n[3/6] Extracting features '
        f'(frames_per_session={frames_per_session}, augment_mult={augment_mult})...'
    )
    print('  Static train (augmented)...')
    X_train, y_train, uid_train, sid_train = materialise_static(
        static_splits['train'],
        frames_per_session=frames_per_session,
        augment_mult=augment_mult,
        seed=seed,
    )
    print(f'    {len(X_train):,} frames')

    print('  Static val...')
    X_val, y_val, uid_val, sid_val = materialise_static(
        static_splits['val'],
        frames_per_session=frames_per_session,
        augment_mult=0,
        seed=seed + 1,
    )
    print(f'    {len(X_val):,} frames')

    print('  Static test...')
    X_test, y_test, uid_test, sid_test = materialise_static(
        static_splits['test'],
        frames_per_session=frames_per_session,
        augment_mult=0,
        seed=seed + 2,
    )
    print(f'    {len(X_test):,} frames')

    print(f'  Dynamic train (augment_mult={dyn_augment_mult})...')
    Xd_train, yd_train, uid_dyn_train, sid_dyn_train = materialise_dynamic(
        dynamic_splits['train'], augment_mult=dyn_augment_mult, seed=seed,
    )
    print(f'    {len(Xd_train)} sequences')

    print('  Dynamic val...')
    Xd_val, yd_val, uid_dyn_val, sid_dyn_val = materialise_dynamic(
        dynamic_splits['val'], augment_mult=0, seed=seed + 1,
    )
    print(f'    {len(Xd_val)} sequences')

    print('  Dynamic test...')
    Xd_test, yd_test, uid_dyn_test, sid_dyn_test = materialise_dynamic(
        dynamic_splits['test'], augment_mult=0, seed=seed + 2,
    )
    print(f'    {len(Xd_test)} sequences')

    # Sanity checks
    assert X_train.shape[1] == NUM_FEATURES, \
        f"Feature count mismatch: {X_train.shape[1]} != {NUM_FEATURES}"
    assert not np.any(np.isnan(X_train)), "NaN in static train"
    assert not np.any(np.isnan(Xd_train)), "NaN in dynamic train"

    # --- Step 4: Fit scaler on training frames ONLY ---
    print('\n[4/6] Fitting StandardScaler on real train frames...')
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    print(f'  Fitted: mean shape={scaler.mean_.shape}')

    # --- Step 5: Scale val / test ---
    print('\n[5/6] Scaling val/test splits...')
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # Scale dynamic sequences (flatten, transform, reshape)
    def scale_sequences(X_seq):
        if X_seq.shape[0] == 0:
            return X_seq
        orig = X_seq.shape
        X_flat = X_seq.reshape(-1, NUM_FEATURES)
        X_flat_sc = scaler.transform(X_flat)
        return X_flat_sc.reshape(orig)

    Xd_train_sc = scale_sequences(Xd_train)
    Xd_val_sc = scale_sequences(Xd_val)
    Xd_test_sc = scale_sequences(Xd_test)

    # --- Step 6: Save ---
    print('\n[6/6] Saving to disk...')
    out = Path(out_dir) if out_dir is not None else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    def save_npz(path, X, y, **extra):
        np.savez_compressed(str(path), X=X, y=y, **extra)
        print(f'  Saved: {path}  ({X.shape})')

    save_npz(out / 'train.npz', X_train_sc, y_train,
             user_ids=uid_train)
    save_npz(out / 'val.npz', X_val_sc, y_val,
             user_ids=uid_val)
    save_npz(out / 'test.npz', X_test_sc, y_test,
             user_ids=uid_test)

    # Dynamic: store class_names for train_dynamic.py compatibility
    dyn_class_names = np.array(sorted(DYNAMIC_LETTERS))
    save_npz(out / 'dynamic_train.npz', Xd_train_sc, yd_train,
             user_ids=uid_dyn_train, class_names=dyn_class_names)
    save_npz(out / 'dynamic_val.npz', Xd_val_sc, yd_val,
             user_ids=uid_dyn_val, class_names=dyn_class_names)
    save_npz(out / 'dynamic_test.npz', Xd_test_sc, yd_test,
             user_ids=uid_dyn_test, class_names=dyn_class_names)

    # Scaler (pickled sklearn object)
    scaler_path = out / 'scaler.pkl'
    with open(str(scaler_path), 'wb') as f:
        pickle.dump(scaler, f)
    print(f'  Saved: {scaler_path}')

    # Scaler stats sidecar (JSON; consumed by per-user runtime calibration
    # without needing to unpickle sklearn).
    stats = compute_scaler_stats(scaler, X_train_sc, y_train)
    stats_path = out / 'scaler_stats.json'
    with open(str(stats_path), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'  Saved: {stats_path}')

    # When writing to the canonical OUT_DIR, also mirror the scaler to the
    # PATHS-declared location so existing consumers (predictor.py) keep working.
    if out == OUT_DIR:
        canonical_scaler = PROJECT_ROOT / PATHS['scaler']
        canonical_stats = PROJECT_ROOT / PATHS['scaler_stats']
        canonical_scaler.parent.mkdir(parents=True, exist_ok=True)
        if canonical_scaler.resolve() != scaler_path.resolve():
            with open(str(canonical_scaler), 'wb') as f:
                pickle.dump(scaler, f)
        if canonical_stats.resolve() != stats_path.resolve():
            with open(str(canonical_stats), 'w') as f:
                json.dump(stats, f, indent=2)

    # Coverage report
    coverage = build_coverage_report(inventory, static_splits, dynamic_splits)
    coverage_path = out / 'coverage_report.json'
    with open(str(coverage_path), 'w') as f:
        json.dump(coverage, f, indent=2)
    print(f'  Saved: {coverage_path}')

    # --- Summary ---
    print(f'\n{sep}')
    print('Build complete.')
    print(f'  Static  — train: {len(X_train_sc):,} frames | val: {len(X_val_sc):,} | test: {len(X_test_sc):,}')
    print(f'  Dynamic — train: {len(Xd_train_sc)} seqs  | val: {len(Xd_val_sc)} | test: {len(Xd_test_sc)}')
    print(f'  Classes in train: {len(np.unique(y_train))} static, {len(np.unique(yd_train))} dynamic')
    print(sep)

    return coverage


# =============================================================================
# LOUO orchestrator — run build once per held-out user
# =============================================================================

def build_louo_all(
    seed: int = SEED,
    min_settled_seconds: float = 0.6,
    settle_threshold: float = SETTLE_THRESHOLD,
    frames_per_session: int = 12,
    augment_mult: int = 4,
    dyn_augment_mult: int = 16,
) -> dict:
    """Run build() once for each user as the held-out test set.

    Writes per-user output under data/processed/real/louo/<user_id>/ and
    emits a summary listing the produced directories so downstream training
    (train_static.py / train_dynamic.py with --louo) can find them.
    """
    users = load_users()
    user_ids = sorted(users.keys())
    LOUO_ROOT.mkdir(parents=True, exist_ok=True)

    produced: dict[str, str] = {}
    for held in user_ids:
        out_subdir = LOUO_ROOT / held
        print(f"\n\n========== LOUO build: held-out user = {held} ==========")
        build(
            min_settled_seconds=min_settled_seconds,
            settle_threshold=settle_threshold,
            seed=seed,
            split_mode='leave_one_user_out',
            test_user=held,
            frames_per_session=frames_per_session,
            augment_mult=augment_mult,
            dyn_augment_mult=dyn_augment_mult,
            out_dir=out_subdir,
        )
        produced[held] = str(out_subdir)

    summary = {'users': produced, 'root': str(LOUO_ROOT)}
    summary_path = LOUO_ROOT / 'louo_build_summary.json'
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nLOUO build summary written: {summary_path}")
    return summary


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build real data processed splits for ML training.'
    )
    parser.add_argument(
        '--min-settled', type=float, default=0.6,
        help='Minimum settled window in seconds (default: 0.6)'
    )
    parser.add_argument(
        '--settle-threshold', type=float, default=SETTLE_THRESHOLD,
        help=f'Motion energy threshold for settled detection (default: {SETTLE_THRESHOLD} m/s^2)'
    )
    parser.add_argument(
        '--seed', type=int, default=SEED,
        help=f'Random seed for session splitting (default: {SEED})'
    )
    parser.add_argument(
        '--split-mode', choices=['same_user_session', 'leave_one_user_out'],
        default='same_user_session',
        help=(
            'same_user_session = current behaviour (80/10/10 per (user,letter); '
            'inflates test accuracy). leave_one_user_out = test on --test-user, '
            'train+val from the rest (honest cross-wearer metric).'
        ),
    )
    parser.add_argument(
        '--mode', choices=['standard', 'louo'], default='standard',
        help=(
            'standard = single build per CLI invocation (default). '
            'louo = run build once per user as the held-out test, writing to '
            'data/processed/real/louo/<user_id>/. Use with scripts/run_louo.py.'
        ),
    )
    parser.add_argument(
        '--test-user', type=str, default=None,
        help='user_id to hold out when --split-mode=leave_one_user_out',
    )
    parser.add_argument(
        '--frames-per-session', type=int, default=12,
        help=(
            'Subsample N frames per static session (default 12, stratified '
            'across early/middle/late thirds of the settled window). 0 = keep '
            'all (legacy behaviour; lets tree models memorize per-session jitter).'
        ),
    )
    parser.add_argument(
        '--augment-mult', type=int, default=4,
        help=(
            'Number of augmented STATIC frame copies per original in the train '
            'split (default 4 -> 5x train size). 0 disables augmentation.'
        ),
    )
    parser.add_argument(
        '--dyn-augment-mult', type=int, default=16,
        help=(
            'Number of augmented DYNAMIC sequence copies per original in the '
            'train split (default 16). 0 disables dynamic augmentation.'
        ),
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Scan sessions and print inventory without writing files'
    )
    args = parser.parse_args()

    if args.mode == 'louo':
        build_louo_all(
            seed=args.seed,
            min_settled_seconds=args.min_settled,
            settle_threshold=args.settle_threshold,
            frames_per_session=args.frames_per_session,
            augment_mult=args.augment_mult,
            dyn_augment_mult=args.dyn_augment_mult,
        )
    else:
        build(
            min_settled_seconds=args.min_settled,
            settle_threshold=args.settle_threshold,
            seed=args.seed,
            dry_run=args.dry_run,
            split_mode=args.split_mode,
            test_user=args.test_user,
            frames_per_session=args.frames_per_session,
            augment_mult=args.augment_mult,
            dyn_augment_mult=args.dyn_augment_mult,
        )
