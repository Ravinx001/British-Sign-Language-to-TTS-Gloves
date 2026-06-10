"""Full Ravindu A-Z + REST validation and deployment run.

Builds a fresh dataset from data/raw/real/ravindu-8dd5c02e only, using:
  - static classifier labels: all A-Z letters except configured dynamic letters,
    plus REST_1, REST_2, REST_3
  - dynamic classifier labels: configured DYNAMIC_LETTERS (currently H, J)

The script trains from scratch into isolated full_user_validation folders first.
If the held-out metrics meet the acceptance thresholds, it deploys the generated
artifacts to the canonical live paths consumed by the backend.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from ml.config import (
    DYNAMIC_LETTERS,
    DYNAMIC_WINDOW_FRAMES,
    FEATURE_NAMES,
    FLEX_MAX,
    FLEX_MIN,
    FSR_MAX,
    IDX_TO_LETTER,
    LETTERS,
    LETTER_TO_IDX,
    NUM_CLASSES,
    NUM_FEATURES,
    PATHS,
    REST_LABELS,
    REST_LABEL_TO_IDX,
    SAMPLE_RATE_HZ,
    SEED,
)
from ml.real_data import (
    compute_motion_energy_series,
    extract_session_features,
    find_settled_window,
    pair_frames,
)
from scripts.build_real_dataset import (
    SETTLE_THRESHOLD,
    compute_scaler_stats,
    materialise_dynamic,
    materialise_static,
    split_sessions,
)
from ml.train_dynamic import train_dynamic
from ml.train_static import load_static_model, train_all


PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_ID = "8dd5c02e"
RAW_USER_DIR = PROJECT_ROOT / "data" / "raw" / "real" / "ravindu-8dd5c02e"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "real" / "full_user_validation"
MODELS_DIR = PROJECT_ROOT / "data" / "models" / "full_user_validation"
FIGURES_DIR = PROJECT_ROOT / "data" / "figures" / "full_user_validation"
REPORT_PATH = MODELS_DIR / "full_user_validation_report.json"

STATIC_LETTERS = [letter for letter in LETTERS if letter not in DYNAMIC_LETTERS]
DYNAMIC_LABELS = sorted(DYNAMIC_LETTERS)
RESTS = list(REST_LABELS)
STATIC_LABELS = STATIC_LETTERS + RESTS
STATIC_LABEL_TO_IDX = {letter: LETTER_TO_IDX[letter] for letter in STATIC_LETTERS}
STATIC_LABEL_TO_IDX.update(REST_LABEL_TO_IDX)
STATIC_LABEL_NAMES_BY_IDX = {
    idx: label for label, idx in STATIC_LABEL_TO_IDX.items()
}

MIN_SETTLED_SECONDS = 0.6
FRAMES_PER_SESSION = 18
AUGMENT_MULT = 5
DYN_AUGMENT_MULT = 16
OTHER_DYNAMIC_LABEL = "OTHER"
OTHER_DYNAMIC_IDX = NUM_CLASSES + len(REST_LABELS)
DYNAMIC_OUTPUT_LABELS = DYNAMIC_LABELS + [OTHER_DYNAMIC_LABEL]
DYNAMIC_LABEL_TO_IDX = {
    **{label: LETTER_TO_IDX[label] for label in DYNAMIC_LABELS},
    OTHER_DYNAMIC_LABEL: OTHER_DYNAMIC_IDX,
}
STATIC_AUGMENT_MULT_BY_LABEL = {
    "I": 8,
    "L": 8,
    "M": 8,
    "N": 9,
    "O": 9,
    "U": 8,
    "T": 7,
    "V": 7,
}
DYNAMIC_OTHER_AUGMENT_MULT = 8
DYNAMIC_DISTRACTOR_SOURCE_LABELS = STATIC_LABELS
DYNAMIC_DISTRACTOR_QUOTA = {
    "train": 4,
    "val": 1,
    "test": 1,
}

STATIC_THRESHOLDS = {
    "accuracy": 0.90,
    "macro_f1": 0.90,
    "min_class_recall": 0.80,
    "rest_rejection_accuracy": 0.90,
}
DYNAMIC_ACCURACY_THRESHOLD = 0.88
DYNAMIC_BSL_MIN_RECALL_THRESHOLD = 0.80


def _summarize(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), 6),
        "p01": round(float(np.percentile(arr, 1)), 6),
        "median": round(float(np.median(arr)), 6),
        "p99": round(float(np.percentile(arr, 99)), 6),
        "max": round(float(np.max(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
    }


def _vector_magnitude(values) -> float | None:
    if not isinstance(values, list):
        return None
    nums = [float(v) for v in values[:3] if isinstance(v, (int, float))]
    if not nums:
        return None
    return math.sqrt(sum(v * v for v in nums))


def _quaternion_norm(values) -> float | None:
    if not isinstance(values, list) or len(values) != 4:
        return None
    if not all(isinstance(v, (int, float)) for v in values):
        return None
    return math.sqrt(sum(float(v) * float(v) for v in values))


def _bad_quaternion_count(pairs: list[dict]) -> int:
    count = 0
    for pair in pairs:
        for hand in ("right", "left"):
            norm = _quaternion_norm(pair[hand].get("quaternion"))
            if norm is None or norm < 1e-8 or not math.isfinite(norm):
                count += 1
    return count


def _reset_output_dirs() -> list[str]:
    removed: list[str] = []
    for path in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        if path.exists():
            removed.append(str(path))
            shutil.rmtree(path)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return removed


def analyze_raw_dataset(raw_user_dir: Path = RAW_USER_DIR) -> dict:
    if not raw_user_dir.exists():
        raise FileNotFoundError(f"Raw user directory not found: {raw_user_dir}")

    labels = STATIC_LABELS + DYNAMIC_LABELS
    raw_channels = {
        "flex": [],
        "fsr": [],
        "accel_magnitude": [],
        "gyro_magnitude": [],
        "quaternion_norm": [],
    }
    quality_totals = {
        "records": 0,
        "paired_frames": 0,
        "malformed_records": 0,
        "missing_fields": Counter(),
        "bad_dimensions": Counter(),
        "label_mismatches": 0,
        "non_numeric_values": Counter(),
        "flex_below_config_min": 0,
        "flex_above_config_max": 0,
        "flex_zero_values": 0,
        "fsr_saturated_values": 0,
        "accel_mag_lt_8": 0,
        "accel_mag_gt_13": 0,
        "gyro_mag_gt_100": 0,
        "quaternion_zero_norm": 0,
    }
    required = {
        "timestamp_ms": None,
        "hand_id": None,
        "flex": 5,
        "fsr": 3,
        "accel": 3,
        "gyro": 3,
        "quaternion": 4,
    }
    per_label = {}

    missing_dirs = [label for label in labels if not (raw_user_dir / label).exists()]
    if missing_dirs:
        raise FileNotFoundError(
            "Missing raw label folders: "
            + ", ".join(str(raw_user_dir / label) for label in missing_dirs)
        )

    for label in labels:
        files = sorted((raw_user_dir / label).glob("*.jsonl"))
        stats = {
            "sessions": len(files),
            "records": 0,
            "paired_frames": 0,
            "session_pair_frames": [],
            "malformed_records": 0,
            "missing_fields": Counter(),
            "bad_dimensions": Counter(),
            "label_mismatches": 0,
        }

        for path in files:
            line_count = 0
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                line_count += 1
                quality_totals["records"] += 1
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    stats["malformed_records"] += 1
                    quality_totals["malformed_records"] += 1
                    continue

                if rec.get("label") != label:
                    stats["label_mismatches"] += 1
                    quality_totals["label_mismatches"] += 1

                for field, expected_len in required.items():
                    if field not in rec:
                        stats["missing_fields"][field] += 1
                        quality_totals["missing_fields"][field] += 1
                        continue
                    if expected_len is None:
                        continue
                    value = rec.get(field)
                    if not isinstance(value, list) or len(value) != expected_len:
                        stats["bad_dimensions"][field] += 1
                        quality_totals["bad_dimensions"][field] += 1
                    elif not all(isinstance(v, (int, float)) for v in value):
                        quality_totals["non_numeric_values"][field] += 1

                flex = rec.get("flex", [])
                if isinstance(flex, list):
                    for v in flex:
                        if isinstance(v, (int, float)):
                            raw_channels["flex"].append(float(v))
                            quality_totals["flex_below_config_min"] += int(v < FLEX_MIN)
                            quality_totals["flex_above_config_max"] += int(v > FLEX_MAX)
                            quality_totals["flex_zero_values"] += int(v == 0)

                fsr = rec.get("fsr", [])
                if isinstance(fsr, list):
                    for v in fsr:
                        if isinstance(v, (int, float)):
                            raw_channels["fsr"].append(float(v))
                            quality_totals["fsr_saturated_values"] += int(v >= FSR_MAX)

                accel_mag = _vector_magnitude(rec.get("accel"))
                if accel_mag is not None:
                    raw_channels["accel_magnitude"].append(accel_mag)
                    quality_totals["accel_mag_lt_8"] += int(accel_mag < 8.0)
                    quality_totals["accel_mag_gt_13"] += int(accel_mag > 13.0)

                gyro_mag = _vector_magnitude(rec.get("gyro"))
                if gyro_mag is not None:
                    raw_channels["gyro_magnitude"].append(gyro_mag)
                    quality_totals["gyro_mag_gt_100"] += int(gyro_mag > 100.0)

                quat_norm = _quaternion_norm(rec.get("quaternion"))
                if quat_norm is not None:
                    raw_channels["quaternion_norm"].append(quat_norm)
                    quality_totals["quaternion_zero_norm"] += int(quat_norm < 1e-8)

            stats["records"] += line_count
            stats["paired_frames"] += line_count // 2
            stats["session_pair_frames"].append(line_count // 2)
            quality_totals["paired_frames"] += line_count // 2

        session_frames = stats.pop("session_pair_frames")
        stats["session_pair_frame_stats"] = _summarize(session_frames)
        stats["missing_fields"] = dict(stats["missing_fields"])
        stats["bad_dimensions"] = dict(stats["bad_dimensions"])
        per_label[label] = stats

    session_counts = {label: per_label[label]["sessions"] for label in labels}
    max_sessions = max(session_counts.values()) if session_counts else 0
    underrepresented = [
        label for label, count in session_counts.items()
        if count < max(1, int(0.8 * max_sessions))
    ]

    return {
        "raw_user_dir": str(raw_user_dir),
        "user_id": USER_ID,
        "static_letters": STATIC_LETTERS,
        "dynamic_letters": DYNAMIC_LABELS,
        "rest_labels": RESTS,
        "labels": labels,
        "session_counts": session_counts,
        "total_sessions": int(sum(session_counts.values())),
        "paired_frame_counts": {
            label: int(per_label[label]["paired_frames"]) for label in labels
        },
        "total_paired_frames": int(
            sum(per_label[label]["paired_frames"] for label in labels)
        ),
        "underrepresented_labels": underrepresented,
        "per_label_quality": per_label,
        "global_quality": {
            **{
                k: v for k, v in quality_totals.items()
                if not isinstance(v, Counter)
            },
            "missing_fields": dict(quality_totals["missing_fields"]),
            "bad_dimensions": dict(quality_totals["bad_dimensions"]),
            "non_numeric_values": dict(quality_totals["non_numeric_values"]),
        },
        "raw_sensor_stats": {
            name: _summarize(vals) for name, vals in raw_channels.items()
        },
    }


def build_inventory() -> tuple[dict, dict]:
    min_settled_frames = int(MIN_SETTLED_SECONDS * SAMPLE_RATE_HZ)
    static_sessions: list[dict] = []
    dynamic_sessions: list[dict] = []
    discarded = defaultdict(int)

    for label in STATIC_LABELS:
        for sess_path in sorted((RAW_USER_DIR / label).glob("*.jsonl")):
            pairs = pair_frames(sess_path)
            if len(pairs) < min_settled_frames:
                discarded["static_too_short_raw"] += 1
                continue
            energy = compute_motion_energy_series(pairs)
            start, end = find_settled_window(energy, threshold=SETTLE_THRESHOLD)
            n_settled = end - start + 1
            if n_settled < min_settled_frames:
                discarded["static_too_short_settled"] += 1
                continue
            static_sessions.append({
                "user_id": USER_ID,
                "letter": label,
                "path": str(sess_path),
                "settle_start": start,
                "settle_end": end,
                "n_settled_frames": n_settled,
            })

    for label in DYNAMIC_LABELS:
        for sess_path in sorted((RAW_USER_DIR / label).glob("*.jsonl")):
            pairs = pair_frames(sess_path)
            if len(pairs) <= 10:
                discarded["dynamic_too_short_raw"] += 1
                continue
            dynamic_sessions.append({
                "user_id": USER_ID,
                "letter": label,
                "path": str(sess_path),
                "n_frames": len(pairs),
            })

    return {
        "static": static_sessions,
        "dynamic": dynamic_sessions,
    }, dict(discarded)


def build_dynamic_distractor_splits(
    static_splits: dict,
    seed: int = SEED,
) -> dict[str, list[dict]]:
    """Build OTHER dynamic examples from Ravindu static/rest recordings.

    These full-session sequences model normal hand movement, settling, and
    non-H/J signing. The dynamic classifier can then learn to emit Unknown
    for non-dynamic motion instead of forcing every movement into H or J.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, list[dict]] = {}
    for split_name, split_sessions in static_splits.items():
        by_label: dict[str, list[dict]] = defaultdict(list)
        for sess in split_sessions:
            if sess["letter"] in DYNAMIC_DISTRACTOR_SOURCE_LABELS:
                by_label[sess["letter"]].append(sess)

        selected: list[dict] = []
        quota = int(DYNAMIC_DISTRACTOR_QUOTA.get(split_name, 0))
        for label in DYNAMIC_DISTRACTOR_SOURCE_LABELS:
            bucket = list(by_label.get(label, []))
            if not bucket or quota <= 0:
                continue
            rng.shuffle(bucket)
            for sess in bucket[: min(quota, len(bucket))]:
                selected.append({
                    **sess,
                    "letter": OTHER_DYNAMIC_LABEL,
                    "source_letter": sess["letter"],
                    "is_dynamic_distractor": True,
                })
        out[split_name] = selected
    return out


def _save_npz(path: Path, X: np.ndarray, y: np.ndarray, **extra) -> None:
    np.savez_compressed(str(path), X=X, y=y, **extra)
    print(f"  Saved: {path} ({X.shape})")


def _feature_stats(X: np.ndarray) -> dict:
    if X.size == 0:
        return {}
    result = {}
    for i, name in enumerate(FEATURE_NAMES):
        vals = X[:, i]
        result[name] = {
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals)), 6),
            "min": round(float(np.min(vals)), 6),
            "max": round(float(np.max(vals)), 6),
        }
    return result


def build_processed_dataset(seed: int = SEED) -> dict:
    print("\n=================================================================")
    print("Build Full A-Z + REST Validation Dataset")
    print("=================================================================")

    inventory, discarded = build_inventory()
    static_by_label = Counter(sess["letter"] for sess in inventory["static"])
    dynamic_by_label = Counter(sess["letter"] for sess in inventory["dynamic"])
    print(f"  Static sessions kept:  {len(inventory['static'])}")
    print(f"  Dynamic sessions kept: {len(inventory['dynamic'])}")
    print(f"  Discarded: {discarded}")
    for label in STATIC_LABELS:
        print(f"    static  {label}: {static_by_label.get(label, 0)} sessions")
    for label in DYNAMIC_LABELS:
        print(f"    dynamic {label}: {dynamic_by_label.get(label, 0)} sessions")

    static_splits = split_sessions(inventory["static"], seed=seed, mode="same_user_session")
    dynamic_letter_splits = split_sessions(inventory["dynamic"], seed=seed, mode="same_user_session")
    dynamic_distractor_splits = build_dynamic_distractor_splits(static_splits, seed=seed)
    dynamic_splits = {
        name: dynamic_letter_splits[name] + dynamic_distractor_splits.get(name, [])
        for name in ("train", "val", "test")
    }
    for split_name in ("train", "val", "test"):
        print(
            f"  {split_name:5s}: static={len(static_splits[split_name])} "
            f"dynamic_letters={len(dynamic_letter_splits[split_name])} "
            f"dynamic_other={len(dynamic_distractor_splits.get(split_name, []))}"
        )

    X_train, y_train, uid_train, sid_train = materialise_static(
        static_splits["train"],
        frames_per_session=FRAMES_PER_SESSION,
        augment_mult=AUGMENT_MULT,
        seed=seed,
        label_to_idx=STATIC_LABEL_TO_IDX,
        augment_mult_by_label=STATIC_AUGMENT_MULT_BY_LABEL,
    )
    X_val, y_val, uid_val, sid_val = materialise_static(
        static_splits["val"],
        frames_per_session=FRAMES_PER_SESSION,
        augment_mult=0,
        seed=seed + 1,
        label_to_idx=STATIC_LABEL_TO_IDX,
    )
    X_test, y_test, uid_test, sid_test = materialise_static(
        static_splits["test"],
        frames_per_session=FRAMES_PER_SESSION,
        augment_mult=0,
        seed=seed + 2,
        label_to_idx=STATIC_LABEL_TO_IDX,
    )

    Xd_train, yd_train, uid_dyn_train, sid_dyn_train = materialise_dynamic(
        dynamic_splits["train"],
        augment_mult=DYN_AUGMENT_MULT,
        seed=seed,
        label_to_idx=DYNAMIC_LABEL_TO_IDX,
        augment_mult_by_label={OTHER_DYNAMIC_LABEL: DYNAMIC_OTHER_AUGMENT_MULT},
    )
    Xd_val, yd_val, uid_dyn_val, sid_dyn_val = materialise_dynamic(
        dynamic_splits["val"],
        augment_mult=0,
        seed=seed + 1,
        label_to_idx=DYNAMIC_LABEL_TO_IDX,
    )
    Xd_test, yd_test, uid_dyn_test, sid_dyn_test = materialise_dynamic(
        dynamic_splits["test"],
        augment_mult=0,
        seed=seed + 2,
        label_to_idx=DYNAMIC_LABEL_TO_IDX,
    )

    for name, X in (("train", X_train), ("val", X_val), ("test", X_test)):
        if X.shape[1] != NUM_FEATURES:
            raise AssertionError(f"{name} feature count mismatch: {X.shape}")
        if not np.isfinite(X).all():
            raise AssertionError(f"{name} contains NaN/Inf")
    for name, X in (("dynamic_train", Xd_train), ("dynamic_val", Xd_val), ("dynamic_test", Xd_test)):
        if X.shape[1:] != (DYNAMIC_WINDOW_FRAMES, NUM_FEATURES):
            raise AssertionError(f"{name} shape mismatch: {X.shape}")
        if not np.isfinite(X).all():
            raise AssertionError(f"{name} contains NaN/Inf")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    def scale_sequences(X_seq):
        if X_seq.shape[0] == 0:
            return X_seq
        orig = X_seq.shape
        X_flat = X_seq.reshape(-1, NUM_FEATURES)
        return scaler.transform(X_flat).reshape(orig)

    Xd_train_sc = scale_sequences(Xd_train)
    Xd_val_sc = scale_sequences(Xd_val)
    Xd_test_sc = scale_sequences(Xd_test)

    _save_npz(PROCESSED_DIR / "train.npz", X_train_sc, y_train,
              user_ids=uid_train, session_ids=np.array(sid_train))
    _save_npz(PROCESSED_DIR / "val.npz", X_val_sc, y_val,
              user_ids=uid_val, session_ids=np.array(sid_val))
    _save_npz(PROCESSED_DIR / "test.npz", X_test_sc, y_test,
              user_ids=uid_test, session_ids=np.array(sid_test))
    dyn_class_names = np.array(DYNAMIC_OUTPUT_LABELS)
    _save_npz(PROCESSED_DIR / "dynamic_train.npz", Xd_train_sc, yd_train,
              user_ids=uid_dyn_train, session_ids=np.array(sid_dyn_train),
              class_names=dyn_class_names)
    _save_npz(PROCESSED_DIR / "dynamic_val.npz", Xd_val_sc, yd_val,
              user_ids=uid_dyn_val, session_ids=np.array(sid_dyn_val),
              class_names=dyn_class_names)
    _save_npz(PROCESSED_DIR / "dynamic_test.npz", Xd_test_sc, yd_test,
              user_ids=uid_dyn_test, session_ids=np.array(sid_dyn_test),
              class_names=dyn_class_names)

    with (PROCESSED_DIR / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
    scaler_stats = compute_scaler_stats(scaler, X_train_sc, y_train)
    (PROCESSED_DIR / "scaler_stats.json").write_text(
        json.dumps(scaler_stats, indent=2), encoding="utf-8",
    )

    build_report = {
        "seed": seed,
        "split_ratio": "80/10/10 session-level stratified by (user, label)",
        "user_id": USER_ID,
        "raw_user_dir": str(RAW_USER_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "static_letters": STATIC_LETTERS,
        "dynamic_letters": DYNAMIC_LABELS,
        "dynamic_output_labels": DYNAMIC_OUTPUT_LABELS,
        "dynamic_other_label": OTHER_DYNAMIC_LABEL,
        "dynamic_distractor_source_labels": DYNAMIC_DISTRACTOR_SOURCE_LABELS,
        "static_augment_mult": AUGMENT_MULT,
        "static_augment_mult_by_label": STATIC_AUGMENT_MULT_BY_LABEL,
        "dynamic_augment_mult": DYN_AUGMENT_MULT,
        "dynamic_other_augment_mult": DYNAMIC_OTHER_AUGMENT_MULT,
        "rest_labels": RESTS,
        "static_labels": STATIC_LABELS,
        "static_label_to_idx": STATIC_LABEL_TO_IDX,
        "dynamic_label_to_idx": DYNAMIC_LABEL_TO_IDX,
        "discarded_sessions": discarded,
        "kept_sessions": {
            "static": dict(sorted(static_by_label.items())),
            "dynamic": dict(sorted(dynamic_by_label.items())),
        },
        "split_sessions": {
            "static": {
                name: dict(sorted(Counter(sess["letter"] for sess in split).items()))
                for name, split in static_splits.items()
            },
            "dynamic": {
                name: dict(sorted(Counter(sess["letter"] for sess in split).items()))
                for name, split in dynamic_splits.items()
            },
            "dynamic_distractor_sources": {
                name: dict(sorted(Counter(
                    sess.get("source_letter", sess["letter"]) for sess in split
                ).items()))
                for name, split in dynamic_distractor_splits.items()
            },
        },
        "processed_samples": {
            "static_train": int(X_train_sc.shape[0]),
            "static_val": int(X_val_sc.shape[0]),
            "static_test": int(X_test_sc.shape[0]),
            "dynamic_train": int(Xd_train_sc.shape[0]),
            "dynamic_val": int(Xd_val_sc.shape[0]),
            "dynamic_test": int(Xd_test_sc.shape[0]),
        },
        "processed_samples_by_label": {
            "train": {label: int(np.sum(y_train == STATIC_LABEL_TO_IDX[label])) for label in STATIC_LABELS},
            "val": {label: int(np.sum(y_val == STATIC_LABEL_TO_IDX[label])) for label in STATIC_LABELS},
            "test": {label: int(np.sum(y_test == STATIC_LABEL_TO_IDX[label])) for label in STATIC_LABELS},
            "dynamic_train": {label: int(np.sum(yd_train == DYNAMIC_LABEL_TO_IDX[label])) for label in DYNAMIC_OUTPUT_LABELS},
            "dynamic_val": {label: int(np.sum(yd_val == DYNAMIC_LABEL_TO_IDX[label])) for label in DYNAMIC_OUTPUT_LABELS},
            "dynamic_test": {label: int(np.sum(yd_test == DYNAMIC_LABEL_TO_IDX[label])) for label in DYNAMIC_OUTPUT_LABELS},
        },
        "feature_stats_unscaled_static_train": _feature_stats(X_train),
        "feature_stats_scaled_static_train": _feature_stats(X_train_sc),
    }
    (PROCESSED_DIR / "full_build_report.json").write_text(
        json.dumps(build_report, indent=2), encoding="utf-8",
    )
    (PROCESSED_DIR / "coverage_report.json").write_text(
        json.dumps(build_report, indent=2), encoding="utf-8",
    )
    print(f"  Saved build report: {PROCESSED_DIR / 'full_build_report.json'}")
    return build_report


def evaluate_best_static_model(training_report: dict) -> dict:
    best_key = training_report["best_model"]
    model_path = training_report["models"][best_key]["model_path"]
    model, inv_label_map, weighted, label_names = load_static_model(
        model_path, include_label_names=True,
    )
    test = np.load(PROCESSED_DIR / "test.npz", allow_pickle=True)
    X_test = test["X"]
    y_test = test["y"]
    if weighted:
        from ml.config import FEATURE_WEIGHTS
        X_in = X_test * np.asarray(FEATURE_WEIGHTS, dtype=np.float64)
    else:
        X_in = X_test
    pred_train_labels = model.predict(X_in)
    y_pred = np.array(
        [int(inv_label_map.get(int(v), int(v))) for v in pred_train_labels],
        dtype=np.int32,
    )
    labels = [STATIC_LABEL_TO_IDX[label] for label in STATIC_LABELS]
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=STATIC_LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    letter_ids = [STATIC_LABEL_TO_IDX[label] for label in STATIC_LETTERS]
    rest_ids = [STATIC_LABEL_TO_IDX[label] for label in RESTS]
    letter_mask = np.isin(y_test, letter_ids)
    rest_mask = np.isin(y_test, rest_ids)
    letter_accuracy = float(np.mean(y_pred[letter_mask] == y_test[letter_mask])) if letter_mask.any() else 0.0
    rest_rejection_accuracy = float(np.mean(np.isin(y_pred[rest_mask], rest_ids))) if rest_mask.any() else 0.0
    return {
        "best_model": best_key,
        "model_path": model_path,
        "weighted": weighted,
        "label_names": {str(k): v for k, v in label_names.items()},
        "classification_report": report,
        "confusion_matrix_labels": STATIC_LABELS,
        "confusion_matrix": matrix.tolist(),
        "accuracy": round(float(report["accuracy"]), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "letter_accuracy": round(letter_accuracy, 6),
        "rest_rejection_accuracy": round(rest_rejection_accuracy, 6),
        "min_class_recall": round(float(min(report[label]["recall"] for label in STATIC_LABELS)), 6),
    }


def save_confusion_csv(path: Path, labels: list[str], matrix: list[list[int]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix, dtype=np.int64)
    with path.open("w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, arr):
            f.write(label + "," + ",".join(str(int(v)) for v in row) + "\n")
    return str(path)


def build_recommendation(static_eval: dict, dynamic_report: dict) -> dict:
    dynamic_acc = float(dynamic_report.get("test_accuracy", 0.0))
    dynamic_bsl_recalls = [
        float((dynamic_report.get("per_class") or {}).get(label, {}).get("recall", 0.0))
        for label in DYNAMIC_LABELS
    ]
    dynamic_bsl_min_recall = min(dynamic_bsl_recalls) if dynamic_bsl_recalls else 0.0
    concerns = []
    for metric, threshold in STATIC_THRESHOLDS.items():
        if float(static_eval[metric]) < threshold:
            concerns.append(f"static {metric} below {threshold:.2f}")
    if dynamic_acc < DYNAMIC_ACCURACY_THRESHOLD:
        concerns.append(f"dynamic accuracy below {DYNAMIC_ACCURACY_THRESHOLD:.2f}")
    if dynamic_bsl_min_recall < DYNAMIC_BSL_MIN_RECALL_THRESHOLD:
        concerns.append(
            f"dynamic H/J min recall below {DYNAMIC_BSL_MIN_RECALL_THRESHOLD:.2f}"
        )
    if not concerns:
        concerns.append("all held-out static/rest and dynamic acceptance metrics passed")
    return {
        "decision": "GO" if len(concerns) == 1 and concerns[0].startswith("all ") else "NO-GO",
        "thresholds": {
            **{f"static_{k}": v for k, v in STATIC_THRESHOLDS.items()},
            "dynamic_accuracy": DYNAMIC_ACCURACY_THRESHOLD,
            "dynamic_hj_min_recall": DYNAMIC_BSL_MIN_RECALL_THRESHOLD,
        },
        "dynamic_hj_min_recall": round(float(dynamic_bsl_min_recall), 6),
        "reasons": concerns,
    }


def deploy_artifacts(final_report: dict) -> dict:
    static_model_map = {
        "xgb": "xgboost_model",
        "rf": "rf_model",
        "svm": "svm_model",
        "mlp": "mlp_model",
    }
    copied = {}
    for key, path_key in static_model_map.items():
        src = MODELS_DIR / f"{key}.pkl"
        dst = PROJECT_ROOT / PATHS[path_key]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[str(dst)] = str(src)

    for name in ("train.npz", "val.npz", "test.npz", "dynamic_train.npz", "dynamic_val.npz", "dynamic_test.npz"):
        src = PROCESSED_DIR / name
        dst = PROJECT_ROOT / "data" / "processed" / "real" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[str(dst)] = str(src)

    for name in ("scaler.pkl", "scaler_stats.json", "coverage_report.json", "full_build_report.json"):
        src = PROCESSED_DIR / name
        dst = PROJECT_ROOT / "data" / "processed" / "real" / name
        shutil.copy2(src, dst)
        copied[str(dst)] = str(src)

    deploy_pairs = [
        (MODELS_DIR / "training_report.json", PROJECT_ROOT / PATHS["training_report_real"]),
        (MODELS_DIR / "cnn_lstm.pt", PROJECT_ROOT / PATHS["cnn_lstm_model"]),
        (MODELS_DIR / "dynamic_training_report.json", PROJECT_ROOT / PATHS["dynamic_training_report_real"]),
        (REPORT_PATH, PROJECT_ROOT / "data" / "models" / "full_user_validation_report.json"),
    ]
    for src, dst in deploy_pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[str(dst)] = str(src)

    deployment_report = {
        "deployed": True,
        "copied": copied,
        "best_static_model": final_report["static_evaluation"]["best_model"],
        "dynamic_letters": DYNAMIC_LABELS,
        "dynamic_output_labels": DYNAMIC_OUTPUT_LABELS,
    }
    (MODELS_DIR / "deployment_report.json").write_text(
        json.dumps(deployment_report, indent=2), encoding="utf-8",
    )
    return deployment_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full Ravindu A-Z + REST validation")
    parser.add_argument("--no-deploy", action="store_true", help="Train/evaluate without copying to live paths")
    parser.add_argument("--quick", action="store_true", help="Use quick static/dynamic training for smoke checks")
    parser.add_argument("--epochs", type=int, default=None, help="Override dynamic max epochs")
    args = parser.parse_args()

    print("=================================================================")
    print("Full A-Z + REST Fresh Real-Data Validation Run")
    print("=================================================================")
    print(f"Raw data: {RAW_USER_DIR}")
    print(f"Static labels: {STATIC_LABELS}")
    print(f"Dynamic labels: {DYNAMIC_LABELS}")
    print(f"Seed: {SEED}")

    removed = _reset_output_dirs()
    if removed:
        print("\nRemoved previous generated validation outputs:")
        for path in removed:
            print(f"  {path}")
    else:
        print("\nNo previous generated validation outputs to remove.")

    raw_analysis = analyze_raw_dataset()
    build_report = build_processed_dataset(seed=SEED)

    print("\n=================================================================")
    print("Training static classifiers from scratch")
    print("=================================================================")
    training_report, _, _ = train_all(
        quick=args.quick,
        models_to_train={"xgb", "rf", "svm", "mlp"},
        real=True,
        data_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR,
        report_path=MODELS_DIR / "training_report.json",
        label_names=STATIC_LABEL_NAMES_BY_IDX,
    )
    static_eval = evaluate_best_static_model(training_report)
    static_cm_path = save_confusion_csv(
        FIGURES_DIR / "static_confusion_matrix.csv",
        static_eval["confusion_matrix_labels"],
        static_eval["confusion_matrix"],
    )

    print("\n=================================================================")
    print("Training dynamic classifier from scratch")
    print("=================================================================")
    dynamic_report = train_dynamic(
        max_epochs=args.epochs,
        quick=args.quick,
        real=True,
        data_dir=PROCESSED_DIR,
        model_path=MODELS_DIR / "cnn_lstm.pt",
        report_path=MODELS_DIR / "dynamic_training_report.json",
        mixup_alpha=0.2,
    )
    dynamic_cm_path = save_confusion_csv(
        FIGURES_DIR / "dynamic_confusion_matrix.csv",
        list(dynamic_report["class_names"]),
        dynamic_report["confusion_matrix"],
    )

    recommendation = build_recommendation(static_eval, dynamic_report)
    final = {
        "raw_analysis": raw_analysis,
        "build": build_report,
        "static_training": training_report,
        "static_evaluation": static_eval,
        "dynamic_training": dynamic_report,
        "artifacts": {
            "static_confusion_matrix_csv": static_cm_path,
            "dynamic_confusion_matrix_csv": dynamic_cm_path,
        },
        "recommendation": recommendation,
    }
    REPORT_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")

    deployment = {"deployed": False, "reason": "NO-GO or --no-deploy"}
    if recommendation["decision"] == "GO" and not args.no_deploy:
        deployment = deploy_artifacts(final)
    final["deployment"] = deployment
    REPORT_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")

    print("\n=================================================================")
    print("Full Validation Summary")
    print("=================================================================")
    print(f"Best static model: {static_eval['best_model']}")
    print(f"Static accuracy: {static_eval['accuracy']:.4f}")
    print(f"Static macro F1: {static_eval['macro_f1']:.4f}")
    print(f"Static min class recall: {static_eval['min_class_recall']:.4f}")
    print(f"REST rejection accuracy: {static_eval['rest_rejection_accuracy']:.4f}")
    print(f"Dynamic classes: {dynamic_report['class_names']}")
    print(f"Dynamic test accuracy: {dynamic_report['test_accuracy']:.4f}")
    print(f"Decision: {recommendation['decision']}")
    print(f"Report: {REPORT_PATH}")
    print(f"Static confusion matrix: {static_cm_path}")
    print(f"Dynamic confusion matrix: {dynamic_cm_path}")
    if deployment.get("deployed"):
        print("Deployment: copied validation artifacts to canonical live paths")
    else:
        print(f"Deployment: skipped ({deployment.get('reason')})")
    return 0 if recommendation["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
