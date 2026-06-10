"""Fresh A-G validation run for the new Ravindu real-data subset.

This script intentionally bypasses the standard all-letter real-data builder.
It reads only data/raw/real/ravindu-8dd5c02e/A..G, creates isolated processed
splits, trains static classifiers from scratch, and writes an evaluation report.
"""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from ml.config import (
    FEATURE_NAMES,
    FLEX_MAX,
    FLEX_MIN,
    FSR_MAX,
    LETTER_TO_IDX,
    REST_LABELS,
    REST_LABEL_TO_IDX,
    SAMPLE_RATE_HZ,
    SEED,
)
from ml.feature_engineering import apply_sensor_mask
from ml.real_data import (
    compute_motion_energy_series,
    extract_session_features,
    find_settled_window,
    pair_frames,
)
from scripts.build_real_dataset import (
    SETTLE_THRESHOLD,
    compute_scaler_stats,
    materialise_static,
    split_sessions,
)
from ml.train_static import load_static_model, train_all


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LETTERS = list("ABCDEFG")
RESTS = list(REST_LABELS)
LABELS = LETTERS + RESTS
LABEL_TO_IDX = {letter: LETTER_TO_IDX[letter] for letter in LETTERS}
LABEL_TO_IDX.update(REST_LABEL_TO_IDX)
LABEL_NAMES_BY_IDX = {idx: label for label, idx in LABEL_TO_IDX.items()}
USER_ID = "8dd5c02e"
RAW_USER_DIR = PROJECT_ROOT / "data" / "raw" / "real" / "ravindu-8dd5c02e"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "real" / "ag_validation"
MODELS_DIR = PROJECT_ROOT / "data" / "models" / "ag_validation"
FIGURES_DIR = PROJECT_ROOT / "data" / "figures" / "ag_validation"
REPORT_PATH = MODELS_DIR / "ag_validation_report.json"

MIN_SETTLED_SECONDS = 0.6
FRAMES_PER_SESSION = 12
AUGMENT_MULT = 4


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


def analyze_raw_dataset(raw_user_dir: Path = RAW_USER_DIR) -> dict:
    if not raw_user_dir.exists():
        raise FileNotFoundError(f"Raw user directory not found: {raw_user_dir}")

    per_letter = {}
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

    missing_rest = [label for label in RESTS if not (raw_user_dir / label).exists()]
    if missing_rest:
        raise FileNotFoundError(
            "Missing resting-pose folders: "
            + ", ".join(str(raw_user_dir / label) for label in missing_rest)
            + ". Collect REST_1, REST_2, and REST_3 before A-G+rest training."
        )

    for letter in LABELS:
        letter_dir = raw_user_dir / letter
        files = sorted(letter_dir.glob("*.jsonl")) if letter_dir.exists() else []
        letter_stats = {
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
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            letter_stats["records"] += len(lines)
            letter_stats["paired_frames"] += len(lines) // 2
            letter_stats["session_pair_frames"].append(len(lines) // 2)
            quality_totals["records"] += len(lines)
            quality_totals["paired_frames"] += len(lines) // 2

            for line in lines:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    letter_stats["malformed_records"] += 1
                    quality_totals["malformed_records"] += 1
                    continue

                if rec.get("label") != letter:
                    letter_stats["label_mismatches"] += 1
                    quality_totals["label_mismatches"] += 1

                for field, expected_len in required.items():
                    if field not in rec:
                        letter_stats["missing_fields"][field] += 1
                        quality_totals["missing_fields"][field] += 1
                        continue
                    if expected_len is not None:
                        value = rec.get(field)
                        if not isinstance(value, list) or len(value) != expected_len:
                            letter_stats["bad_dimensions"][field] += 1
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

                quat = rec.get("quaternion", [])
                if isinstance(quat, list) and len(quat) == 4 and all(isinstance(v, (int, float)) for v in quat):
                    raw_channels["quaternion_norm"].append(math.sqrt(sum(float(v) * float(v) for v in quat)))

        session_frames = letter_stats.pop("session_pair_frames")
        letter_stats["session_pair_frame_stats"] = _summarize(session_frames)
        letter_stats["missing_fields"] = dict(letter_stats["missing_fields"])
        letter_stats["bad_dimensions"] = dict(letter_stats["bad_dimensions"])
        per_letter[letter] = letter_stats

    session_counts = {label: per_letter[label]["sessions"] for label in LABELS}
    paired_frame_counts = {label: per_letter[label]["paired_frames"] for label in LABELS}
    min_sessions = min(session_counts.values()) if session_counts else 0
    underrepresented = [
        letter for letter, count in session_counts.items()
        if count < max(1, int(0.8 * max(session_counts.values())))
    ]

    return {
        "raw_user_dir": str(raw_user_dir),
        "letters": LETTERS,
        "rest_labels": RESTS,
        "labels": LABELS,
        "session_counts": session_counts,
        "paired_frame_counts": paired_frame_counts,
        "total_sessions": int(sum(session_counts.values())),
        "total_paired_frames": int(sum(paired_frame_counts.values())),
        "min_sessions_per_letter": int(min_sessions),
        "underrepresented_letters": underrepresented,
        "per_letter_quality": per_letter,
        "global_quality": {
            **{
                k: v for k, v in quality_totals.items()
                if not isinstance(v, Counter)
            },
            "missing_fields": dict(quality_totals["missing_fields"]),
            "bad_dimensions": dict(quality_totals["bad_dimensions"]),
            "non_numeric_values": dict(quality_totals["non_numeric_values"]),
        },
        "raw_sensor_stats": {name: _summarize(vals) for name, vals in raw_channels.items()},
    }


def build_ag_inventory() -> tuple[list[dict], dict]:
    min_settled_frames = int(MIN_SETTLED_SECONDS * SAMPLE_RATE_HZ)
    sessions: list[dict] = []
    discarded = Counter()

    for letter in LABELS:
        for sess_path in sorted((RAW_USER_DIR / letter).glob("*.jsonl")):
            pairs = pair_frames(sess_path)
            if len(pairs) < min_settled_frames:
                discarded["too_short_raw"] += 1
                continue

            energy = compute_motion_energy_series(pairs)
            start, end = find_settled_window(energy, threshold=SETTLE_THRESHOLD)
            n_settled = end - start + 1
            if n_settled < min_settled_frames:
                discarded["too_short_settled"] += 1
                continue

            sessions.append({
                "user_id": USER_ID,
                "letter": letter,
                "path": str(sess_path),
                "settle_start": start,
                "settle_end": end,
                "n_settled_frames": n_settled,
            })

    return sessions, dict(discarded)


def _save_npz(path: Path, X: np.ndarray, y: np.ndarray, **extra) -> None:
    np.savez_compressed(str(path), X=X, y=y, **extra)
    print(f"  Saved: {path} ({X.shape})")


def _feature_stats(X: np.ndarray) -> dict:
    result = {}
    if X.size == 0:
        return result
    for i, name in enumerate(FEATURE_NAMES):
        vals = X[:, i]
        result[name] = {
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals)), 6),
            "min": round(float(np.min(vals)), 6),
            "max": round(float(np.max(vals)), 6),
        }
    return result


def build_processed_dataset(seed: int = SEED) -> tuple[dict, dict]:
    print("\n=================================================================")
    print("Build A-G + Rest Validation Dataset")
    print("=================================================================")

    inventory, discarded = build_ag_inventory()
    by_letter = Counter(sess["letter"] for sess in inventory)
    print(f"  Sessions kept: {len(inventory)}")
    print(f"  Discarded: {discarded}")
    for letter in LABELS:
        print(f"    {letter}: {by_letter.get(letter, 0)} sessions")

    splits = split_sessions(inventory, seed=seed, mode="same_user_session")
    for split_name in ("train", "val", "test"):
        split_counts = Counter(sess["letter"] for sess in splits[split_name])
        print(f"  {split_name:5s}: {len(splits[split_name])} sessions {dict(sorted(split_counts.items()))}")

    X_train, y_train, uid_train, sid_train = materialise_static(
        splits["train"],
        frames_per_session=FRAMES_PER_SESSION,
        augment_mult=AUGMENT_MULT,
        seed=seed,
        label_to_idx=LABEL_TO_IDX,
    )
    X_val, y_val, uid_val, sid_val = materialise_static(
        splits["val"],
        frames_per_session=FRAMES_PER_SESSION,
        augment_mult=0,
        seed=seed + 1,
        label_to_idx=LABEL_TO_IDX,
    )
    X_test, y_test, uid_test, sid_test = materialise_static(
        splits["test"],
        frames_per_session=FRAMES_PER_SESSION,
        augment_mult=0,
        seed=seed + 2,
        label_to_idx=LABEL_TO_IDX,
    )

    for name, X in (("train", X_train), ("val", X_val), ("test", X_test)):
        if X.shape[1] != len(FEATURE_NAMES):
            raise AssertionError(f"{name} feature count mismatch: {X.shape}")
        if not np.isfinite(X).all():
            raise AssertionError(f"{name} contains NaN/Inf")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    _save_npz(PROCESSED_DIR / "train.npz", X_train_sc, y_train, user_ids=uid_train, session_ids=np.array(sid_train))
    _save_npz(PROCESSED_DIR / "val.npz", X_val_sc, y_val, user_ids=uid_val, session_ids=np.array(sid_val))
    _save_npz(PROCESSED_DIR / "test.npz", X_test_sc, y_test, user_ids=uid_test, session_ids=np.array(sid_test))

    with (PROCESSED_DIR / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
    scaler_stats = compute_scaler_stats(scaler, X_train_sc, y_train)
    (PROCESSED_DIR / "scaler_stats.json").write_text(json.dumps(scaler_stats, indent=2), encoding="utf-8")

    build_report = {
        "seed": seed,
        "split_ratio": "80/10/10 session-level stratified by (user, letter)",
        "letters": LETTERS,
        "rest_labels": RESTS,
        "labels": LABELS,
        "label_to_idx": LABEL_TO_IDX,
        "user_id": USER_ID,
        "raw_user_dir": str(RAW_USER_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "discarded_sessions": discarded,
        "kept_sessions_by_letter": dict(sorted(by_letter.items())),
        "split_sessions": {
            name: dict(sorted(Counter(sess["letter"] for sess in split).items()))
            for name, split in splits.items()
        },
        "processed_samples": {
            "train": int(X_train_sc.shape[0]),
            "val": int(X_val_sc.shape[0]),
            "test": int(X_test_sc.shape[0]),
        },
        "processed_samples_by_letter": {
            "train": {label: int(np.sum(y_train == LABEL_TO_IDX[label])) for label in LABELS},
            "val": {label: int(np.sum(y_val == LABEL_TO_IDX[label])) for label in LABELS},
            "test": {label: int(np.sum(y_test == LABEL_TO_IDX[label])) for label in LABELS},
        },
        "feature_stats_unscaled_train": _feature_stats(X_train),
        "feature_stats_scaled_train": _feature_stats(X_train_sc),
    }
    (PROCESSED_DIR / "ag_build_report.json").write_text(json.dumps(build_report, indent=2), encoding="utf-8")

    print(f"  Saved build report: {PROCESSED_DIR / 'ag_build_report.json'}")
    return build_report, {
        "X_test": X_test_sc,
        "y_test": y_test,
    }


def evaluate_best_model(training_report: dict) -> dict:
    best_key = training_report["best_model"]
    model_path = training_report["models"][best_key]["model_path"]
    model, inv_label_map, weighted, label_names = load_static_model(
        model_path,
        include_label_names=True,
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
    y_pred = np.array([int(inv_label_map.get(int(v), int(v))) for v in pred_train_labels], dtype=np.int32)
    labels = [LABEL_TO_IDX[label] for label in LABELS]
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    letter_ids = [LABEL_TO_IDX[label] for label in LETTERS]
    rest_ids = [LABEL_TO_IDX[label] for label in RESTS]
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
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": matrix.tolist(),
        "accuracy": round(float(report["accuracy"]), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "letter_accuracy": round(letter_accuracy, 6),
        "rest_rejection_accuracy": round(rest_rejection_accuracy, 6),
        "min_class_recall": round(float(min(report[label]["recall"] for label in LABELS)), 6),
    }


def maybe_save_confusion_matrix(evaluation: dict) -> str | None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIGURES_DIR / "ag_confusion_matrix.csv"
    labels = evaluation["confusion_matrix_labels"]
    matrix = np.asarray(evaluation["confusion_matrix"], dtype=np.int64)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, matrix):
            f.write(label + "," + ",".join(str(int(v)) for v in row) + "\n")

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return str(csv_path)

    png_path = FIGURES_DIR / "ag_confusion_matrix.png"
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("A-G Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    return str(png_path)


def recommendation(evaluation: dict) -> dict:
    go = (
        evaluation["accuracy"] >= 0.90
        and evaluation["macro_f1"] >= 0.90
        and evaluation["min_class_recall"] >= 0.80
        and evaluation["letter_accuracy"] >= 0.90
        and evaluation["rest_rejection_accuracy"] >= 0.90
    )
    concerns = []
    if evaluation["accuracy"] < 0.90:
        concerns.append("test accuracy below 90%")
    if evaluation["macro_f1"] < 0.90:
        concerns.append("macro F1 below 0.90")
    if evaluation["min_class_recall"] < 0.80:
        concerns.append("at least one A-G/rest class recall below 0.80")
    if evaluation["letter_accuracy"] < 0.90:
        concerns.append("A-G letter-only accuracy below 90%")
    if evaluation["rest_rejection_accuracy"] < 0.90:
        concerns.append("rest rejection accuracy below 90%")
    if not concerns:
        concerns.append("no blocking metric concerns on the held-out A-G sessions")
    return {
        "decision": "GO" if go else "NO-GO",
        "thresholds": {
            "accuracy": ">= 0.90",
            "macro_f1": ">= 0.90",
            "min_class_recall": ">= 0.80",
            "letter_accuracy": ">= 0.90",
            "rest_rejection_accuracy": ">= 0.90",
        },
        "reasons": concerns,
    }


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("A-G + Rest Fresh Real-Data Validation Run")
    print("=================================================================")
    print(f"Raw data: {RAW_USER_DIR}")
    print(f"Labels: {LABELS}")
    print(f"Seed: {SEED}")

    try:
        raw_analysis = analyze_raw_dataset()
        build_report, _ = build_processed_dataset(seed=SEED)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        return 2

    print("\n=================================================================")
    print("Training static classifiers from scratch")
    print("=================================================================")
    training_report_path = MODELS_DIR / "training_report.json"
    training_report, _, _ = train_all(
        quick=False,
        models_to_train={"xgb", "rf", "svm", "mlp"},
        real=True,
        data_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR,
        report_path=training_report_path,
        label_names=LABEL_NAMES_BY_IDX,
    )

    evaluation = evaluate_best_model(training_report)
    figure_path = maybe_save_confusion_matrix(evaluation)
    final = {
        "raw_analysis": raw_analysis,
        "build": build_report,
        "training": training_report,
        "evaluation": evaluation,
        "confusion_matrix_artifact": figure_path,
        "recommendation": recommendation(evaluation),
    }
    REPORT_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")

    print("\n=================================================================")
    print("A-G Validation Summary")
    print("=================================================================")
    print(f"Best model: {evaluation['best_model']}")
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Macro F1: {evaluation['macro_f1']:.4f}")
    print(f"Letter accuracy: {evaluation['letter_accuracy']:.4f}")
    print(f"Rest rejection accuracy: {evaluation['rest_rejection_accuracy']:.4f}")
    print(f"Min class recall: {evaluation['min_class_recall']:.4f}")
    print(f"Decision: {final['recommendation']['decision']}")
    print(f"Report: {REPORT_PATH}")
    if figure_path:
        print(f"Confusion matrix artifact: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
