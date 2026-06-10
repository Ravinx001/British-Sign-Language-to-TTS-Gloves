"""
Real-data loader: JSONL sessions → paired frames → auto-trimmed → 26-feature arrays.

Walks data/raw/real/ for each user, loads per-letter JSONL recording sessions,
pairs right/left hand frames by timestamp proximity, applies motion-energy-based
auto-trimming to remove transition noise (critical for Isuru's hand-from-table
collection style), and extracts 26-feature vectors via feature_engineering.py.

For static letters (A-Z excluding configured dynamic letters):
  - Finds the longest contiguous "settled" window where motion energy < threshold.
  - Discards sessions shorter than min_settled_seconds after trimming.

For configured dynamic letters:
  - Keeps the full session; optionally clips leading pre-motion frames.
  - Returns padded/truncated 100-frame sequences for the CNN-LSTM.

Usage:
    from ml.real_data import load_static_real_data, load_dynamic_real_data

    X, y, user_ids, session_ids = load_static_real_data()
    X_seq, y_seq, user_ids_seq, session_ids_seq = load_dynamic_real_data()
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ml.config import (
    DYNAMIC_LETTERS,
    DYNAMIC_WINDOW_FRAMES,
    GRAVITY_LPF_ALPHA,
    LETTER_TO_IDX,
    REAL_DATA_SOURCE,
    SAMPLE_RATE_HZ,
    SETTLE_THRESHOLD_TRAIN,
)
from ml.feature_engineering import extract_features
from ml.segmentation import update_gravity_estimate, extract_segment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Authoritative real-data location. Defaults to REAL_DATA_SOURCE (outside the
# repo). Falls back to the in-repo copy at data/raw/real if the external
# folder is unreachable (e.g., when running on a CI machine without the
# external drive mounted).
_EXT = Path(REAL_DATA_SOURCE)
_IN_REPO = PROJECT_ROOT / 'data' / 'raw' / 'real'
RAW_REAL_DIR = _EXT if _EXT.exists() else _IN_REPO

# Motion energy threshold for "settled" window detection during training data
# construction. Imported from config so live inference (MOTION_ENERGY_THRESHOLD)
# and training (SETTLE_THRESHOLD_TRAIN) stay in sync if either is retuned.
SETTLE_THRESHOLD = SETTLE_THRESHOLD_TRAIN
PAIR_WINDOW_MS = 25      # Max timestamp gap to consider frames a valid pair


# =============================================================================
# Users & Session Discovery
# =============================================================================

def load_users() -> dict:
    """Load users.json → {user_id: {user_id, name, folder_name, ...}}."""
    path = RAW_REAL_DIR / 'users.json'
    with open(path, 'r') as f:
        return json.load(f)


def discover_sessions(user_id: str, letter: str) -> List[Path]:
    """Return sorted list of JSONL paths for a given user and letter."""
    users = load_users()
    if user_id not in users:
        return []
    folder = RAW_REAL_DIR / users[user_id]['folder_name'] / letter
    if not folder.exists():
        return []
    return sorted(folder.glob('*.jsonl'))


# =============================================================================
# Frame Pairing
# =============================================================================

def pair_frames(jsonl_path: Path) -> List[dict]:
    """Load a JSONL session and pair right/left frames by timestamp proximity.

    Each returned dict has keys: 'right', 'left', 'timestamp_ms'.

    Right: hand_id=1, Left: hand_id=2.
    Frames are paired greedily: for each right frame, find the nearest left
    frame within PAIR_WINDOW_MS. Unpaired tail frames are dropped.

    Returns:
        List of {'right': <packet>, 'left': <packet>, 'timestamp_ms': int}.
        Empty list on error.
    """
    rights = []
    lefts = []

    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pkt = json.loads(line)
                if pkt.get('hand_id') == 1:
                    rights.append(pkt)
                elif pkt.get('hand_id') == 2:
                    lefts.append(pkt)
    except (json.JSONDecodeError, OSError):
        return []

    if not rights or not lefts:
        return []

    pairs = []
    left_idx = 0
    for r in rights:
        r_ts = r['timestamp_ms']
        # Advance left pointer to the closest timestamp
        while left_idx + 1 < len(lefts):
            curr_diff = abs(lefts[left_idx]['timestamp_ms'] - r_ts)
            next_diff = abs(lefts[left_idx + 1]['timestamp_ms'] - r_ts)
            if next_diff < curr_diff:
                left_idx += 1
            else:
                break

        l = lefts[left_idx]
        if abs(l['timestamp_ms'] - r_ts) <= PAIR_WINDOW_MS:
            pairs.append({
                'right': r,
                'left': l,
                'timestamp_ms': r_ts,
            })

    return pairs


# =============================================================================
# Motion Energy Computation (per-session)
# =============================================================================

def compute_motion_energy_series(pairs: List[dict]) -> np.ndarray:
    """Compute gravity-compensated motion energy for each paired frame.

    Uses the right-hand accelerometer (dominant hand, the one that moves).

    Returns:
        np.ndarray of shape (N,) — motion energy per frame in m/s².
    """
    gravity = np.array([0.0, 0.0, 9.81], dtype=np.float64)
    energies = []

    for pair in pairs:
        accel = np.asarray(pair['right']['accel'], dtype=np.float64)
        # Only update gravity when not in heavy motion (first few frames)
        gravity = update_gravity_estimate(gravity, accel, alpha=GRAVITY_LPF_ALPHA)
        linear = accel - gravity
        energies.append(float(np.linalg.norm(linear)))

    return np.array(energies, dtype=np.float64)


# =============================================================================
# Settled Window Detection (for static trimming)
# =============================================================================

def find_settled_window(
    energy: np.ndarray,
    threshold: float = SETTLE_THRESHOLD,
) -> Tuple[int, int]:
    """Find the longest contiguous run of frames where energy < threshold.

    Used to strip transition noise from the start (and end) of static
    recording sessions, particularly for Isuru's hand-from-table style.

    Args:
        energy: per-frame motion energy array.
        threshold: frames below this are considered "settled".

    Returns:
        (start_idx, end_idx) inclusive, both pointing into the original array.
        Returns (0, len(energy)-1) if no run is found (entire session kept).
    """
    n = len(energy)
    if n == 0:
        return (0, 0)

    best_start, best_end = 0, n - 1
    best_len = 0
    run_start = None

    for i in range(n):
        if energy[i] < threshold:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len > best_len:
                    best_len = run_len
                    best_start = run_start
                    best_end = i - 1
                run_start = None

    # Check run that reaches the end
    if run_start is not None:
        run_len = n - run_start
        if run_len > best_len:
            best_start = run_start
            best_end = n - 1

    # If nothing settled at all, return full range
    if best_len == 0:
        return (0, n - 1)

    return (best_start, best_end)


# =============================================================================
# Feature Extraction (single session)
# =============================================================================

def extract_session_features(pairs: List[dict]) -> np.ndarray:
    """Extract 26-feature vectors for all paired frames in a session.

    Returns:
        np.ndarray of shape (N, 26).
    """
    features = []
    for pair in pairs:
        r = pair['right']
        l = pair['left']
        raw_frame = {
            'R_flex': r['flex'],
            'L_flex': l['flex'],
            'R_fsr': r['fsr'],
            'L_fsr': l['fsr'],
            'R_quaternion': r['quaternion'],
            'L_quaternion': l['quaternion'],
        }
        features.append(extract_features(raw_frame))
    return np.array(features, dtype=np.float64)


# =============================================================================
# Static Data Loader
# =============================================================================

def load_static_real_data(
    min_settled_seconds: float = 0.6,
    settle_threshold: float = SETTLE_THRESHOLD,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load all static letters from all users with motion-energy trimming.

    Args:
        min_settled_seconds: Sessions whose settled window is shorter than
            this after trimming are discarded. 0.6s = 30 frames at 50 Hz.
        settle_threshold: Motion energy threshold for "settled" detection.
        verbose: Print per-letter statistics.

    Returns:
        X:           np.ndarray (N, 26) — feature vectors (unscaled)
        y:           np.ndarray (N,)    — letter indices (LETTER_TO_IDX)
        user_ids:    np.ndarray (N,)    — user_id string per sample
        session_ids: list[str] of length N — session filename per sample
    """
    users = load_users()
    static_letters = sorted(set(LETTER_TO_IDX.keys()) - DYNAMIC_LETTERS)
    min_settled_frames = int(min_settled_seconds * SAMPLE_RATE_HZ)

    X_list, y_list, uid_list, sid_list = [], [], [], []
    discard_count = 0
    total_sessions = 0

    for user_id, user_info in users.items():
        folder_name = user_info['folder_name']
        if verbose:
            print(f"  Loading user: {user_info['name']} ({user_id})")

        for letter in static_letters:
            sessions = discover_sessions(user_id, letter)
            letter_idx = LETTER_TO_IDX[letter]

            for sess_path in sessions:
                total_sessions += 1
                pairs = pair_frames(sess_path)
                if len(pairs) < min_settled_frames:
                    discard_count += 1
                    continue

                energy = compute_motion_energy_series(pairs)
                start, end = find_settled_window(energy, threshold=settle_threshold)
                settled_len = end - start + 1

                if settled_len < min_settled_frames:
                    discard_count += 1
                    continue

                settled_pairs = pairs[start:end + 1]
                features = extract_session_features(settled_pairs)

                X_list.append(features)
                y_list.extend([letter_idx] * len(features))
                uid_list.extend([user_id] * len(features))
                sid_list.extend([sess_path.stem] * len(features))

    X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0, 26))
    y = np.array(y_list, dtype=np.int32)
    user_ids = np.array(uid_list, dtype=object)
    session_ids = sid_list

    if verbose:
        print(f"\n  Static data: {len(X):,} frames from "
              f"{total_sessions - discard_count} sessions "
              f"({discard_count} discarded, "
              f"{total_sessions} total)")

    return X, y, user_ids, session_ids


# =============================================================================
# Dynamic Data Loader
# =============================================================================

def load_dynamic_real_data(
    window_frames: int = DYNAMIC_WINDOW_FRAMES,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load configured dynamic sign sessions as fixed-length sequences for CNN-LSTM.

    Each session is padded/truncated to window_frames (100 frames = 2s at 50Hz).
    No trimming — motion IS the signal. A leading gravity warm-up of 10 frames
    is skipped to let the gravity estimate settle.

    Returns:
        X:           np.ndarray (N, window_frames, 26) — sequences (unscaled)
        y:           np.ndarray (N,)                   — letter indices
        user_ids:    np.ndarray (N,)                   — user_id per sequence
        session_ids: list[str] of length N
    """
    users = load_users()
    GRAVITY_WARMUP = 10  # frames to skip at start for gravity LPF to settle

    X_list, y_list, uid_list, sid_list = [], [], [], []

    for user_id, user_info in users.items():
        if verbose:
            print(f"  Loading dynamic user: {user_info['name']} ({user_id})")

        for letter in sorted(DYNAMIC_LETTERS):
            sessions = discover_sessions(user_id, letter)
            letter_idx = LETTER_TO_IDX[letter]

            for sess_path in sessions:
                pairs = pair_frames(sess_path)
                if len(pairs) <= GRAVITY_WARMUP:
                    continue

                # Skip warm-up frames
                pairs = pairs[GRAVITY_WARMUP:]

                features = extract_session_features(pairs)
                # Pad or centre-crop to window_frames
                segment = extract_segment(
                    features, 0, len(features) - 1, window_frames
                )

                X_list.append(segment)
                y_list.append(letter_idx)
                uid_list.append(user_id)
                sid_list.append(sess_path.stem)

    X = np.array(X_list, dtype=np.float64) if X_list else np.zeros((0, window_frames, 26))
    y = np.array(y_list, dtype=np.int32)
    user_ids = np.array(uid_list, dtype=object)
    session_ids = sid_list

    if verbose:
        print(f"\n  Dynamic data: {len(X)} sequences")

    return X, y, user_ids, session_ids


# =============================================================================
# Session-level inventory (for splitting)
# =============================================================================

def get_session_inventory(
    min_settled_seconds: float = 0.6,
    settle_threshold: float = SETTLE_THRESHOLD,
) -> dict:
    """Return a dict of session metadata without loading features.

    Useful for building train/val/test splits before loading all data.

    Returns:
        {
          'static': [{'user_id', 'letter', 'path', 'n_settled_frames'}, ...],
          'dynamic': [{'user_id', 'letter', 'path', 'n_frames'}, ...],
        }
    """
    users = load_users()
    static_letters = sorted(set(LETTER_TO_IDX.keys()) - DYNAMIC_LETTERS)
    min_settled_frames = int(min_settled_seconds * SAMPLE_RATE_HZ)

    static_sessions = []
    dynamic_sessions = []

    for user_id, user_info in users.items():
        # Static
        for letter in static_letters:
            for sess_path in discover_sessions(user_id, letter):
                pairs = pair_frames(sess_path)
                if len(pairs) < min_settled_frames:
                    continue
                energy = compute_motion_energy_series(pairs)
                start, end = find_settled_window(energy, threshold=settle_threshold)
                settled_len = end - start + 1
                if settled_len >= min_settled_frames:
                    static_sessions.append({
                        'user_id': user_id,
                        'letter': letter,
                        'path': str(sess_path),
                        'n_settled_frames': settled_len,
                        'settle_start': start,
                        'settle_end': end,
                    })

        # Dynamic
        for letter in sorted(DYNAMIC_LETTERS):
            for sess_path in discover_sessions(user_id, letter):
                pairs = pair_frames(sess_path)
                if len(pairs) > 10:
                    dynamic_sessions.append({
                        'user_id': user_id,
                        'letter': letter,
                        'path': str(sess_path),
                        'n_frames': len(pairs),
                    })

    return {'static': static_sessions, 'dynamic': dynamic_sessions}
