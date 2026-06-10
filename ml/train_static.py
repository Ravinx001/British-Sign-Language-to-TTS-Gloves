"""
Static classifier training: XGBoost, Random Forest, SVM via GridSearchCV.

Loads pre-scaled training data from data/processed/train.npz (scaler already
applied in synthetic_data.py — do NOT re-scale). Trains three classifiers
with 5-fold stratified cross-validation, compares on validation set, selects
best model, and saves all models + training report.

Usage:
    python -m ml.train_static                # Full grid search on synthetic data
    python -m ml.train_static --real         # Full grid search on real data
    python -m ml.train_static --quick        # Reduced grid for fast testing
    python -m ml.train_static --real --quick # Real data, reduced grid
    python -m ml.train_static --model xgb    # Train only XGBoost
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# sklearn 1.6 removed `cv='prefit'` in favour of wrapping the pre-fit
# estimator in FrozenEstimator. Fall back to the older API on earlier
# versions so this code works across both.
try:
    from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
    _HAS_FROZEN_ESTIMATOR = True
except ImportError:  # sklearn < 1.6
    FrozenEstimator = None  # type: ignore[assignment]
    _HAS_FROZEN_ESTIMATOR = False

from ml.config import (
    CV_FOLDS,
    CV_SCORING,
    FEATURE_WEIGHTS,
    IDX_TO_LETTER,
    NUM_CLASSES,
    NUM_STATIC_CLASSES,
    PATHS,
    RF_PARAM_GRID,
    SEED,
    SVM_PARAM_GRID,
    XGBOOST_PARAM_GRID,
)

# Per-feature reliability weights as a NumPy array. The XGBoost constructor
# requires strictly > 0 weights, so masked channels (weight 0) get a tiny
# epsilon — they're already zero in the data so they contribute no signal
# regardless of how XGB samples them.
_W_ARR = np.asarray(FEATURE_WEIGHTS, dtype=np.float64)
_XGB_FEATURE_WEIGHTS = np.where(_W_ARR > 0.0, _W_ARR, 1e-6)
UNKNOWN_EVAL_LABEL = -1

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_split(name, real=False, data_dir: Path | None = None):
    """Load a data split (train/val/test) from data/processed/.

    Args:
        name: Split name — 'train', 'val', or 'test'.
        real: If True, loads real_* paths (data/processed/real/).
        data_dir: If provided, loads ``<data_dir>/<name>.npz`` instead of
            consulting PATHS. Used by LOUO mode to read from per-user
            subdirectories under data/processed/real/louo/<user>/.
    """
    if data_dir is not None:
        path = Path(data_dir) / f'{name}.npz'
    else:
        key = f'real_{name}' if real else name
        path = PROJECT_ROOT / PATHS[key]
    d = np.load(str(path))
    return d['X'], d['y']


def make_quick_grids():
    """Reduced parameter grids for fast testing."""
    return {
        'xgb': {
            'n_estimators': [200],
            'max_depth': [6],
            'learning_rate': [0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [3],
            'reg_alpha': [0.1],
            'reg_lambda': [1.0],
        },
        'rf': {
            'n_estimators': [200],
            'max_depth': [12],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['sqrt'],
        },
        'svm': {
            'C': [1.0],
            'gamma': ['scale'],
            'kernel': ['rbf'],
        },
    }


def train_xgboost(X_train, y_train, param_grid, cv_folds, scoring,
                  n_classes=None, sample_weight=None):
    """Train XGBoost with GridSearchCV (optional class-balancing sample weights).

    Per-feature reliability weights (FEATURE_WEIGHTS) are passed via the
    ``feature_weights`` constructor argument so XGBoost biases its
    column-subsampling toward the more reliable channels. Masked channels
    are already zero in the data but get a tiny epsilon weight to satisfy
    XGBoost's >0 requirement.
    """
    print("\n--- XGBoost ---")
    if n_classes is None:
        n_classes = int(len(np.unique(y_train)))
    base = XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
        feature_weights=_XGB_FEATURE_WEIGHTS,
    )

    grid = GridSearchCV(
        base, param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED),
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    t0 = time.time()
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    grid.fit(X_train, y_train, **fit_kwargs)
    elapsed = time.time() - t0

    print(f"  Best CV score: {grid.best_score_:.4f}")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Training time: {elapsed:.1f}s")

    return grid.best_estimator_, {
        'best_cv_score': float(grid.best_score_),
        'best_params': grid.best_params_,
        'training_time_s': round(elapsed, 1),
    }


def train_random_forest(X_train, y_train, param_grid, cv_folds, scoring,
                         class_weight='balanced'):
    """Train Random Forest with GridSearchCV (balanced classes by default)."""
    print("\n--- Random Forest ---")
    base = RandomForestClassifier(
        random_state=SEED, n_jobs=-1, class_weight=class_weight,
    )

    grid = GridSearchCV(
        base, param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED),
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Best CV score: {grid.best_score_:.4f}")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Training time: {elapsed:.1f}s")

    return grid.best_estimator_, {
        'best_cv_score': float(grid.best_score_),
        'best_params': {k: v if v is not None else 'None' for k, v in grid.best_params_.items()},
        'training_time_s': round(elapsed, 1),
    }


def train_svm(X_train, y_train, param_grid, cv_folds, scoring,
              class_weight='balanced'):
    """Train SVM with GridSearchCV (balanced classes by default)."""
    print("\n--- SVM ---")
    base = SVC(
        probability=True, random_state=SEED, cache_size=1000,
        class_weight=class_weight,
    )

    grid = GridSearchCV(
        base, param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED),
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Best CV score: {grid.best_score_:.4f}")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Training time: {elapsed:.1f}s")

    return grid.best_estimator_, {
        'best_cv_score': float(grid.best_score_),
        'best_params': grid.best_params_,
        'training_time_s': round(elapsed, 1),
    }


def train_mlp(X_train, y_train, X_val, y_val, n_classes=None):
    """Train an MLP classifier with sigmoid (Platt) probability calibration.

    The MLP is the only candidate that natively respects FEATURE_WEIGHTS
    through input scaling — we pre-multiply X by the weights vector before
    fitting AND at inference time (the saved payload carries weighted=True so
    predict.py knows to multiply at inference). The pre-multiplication folds
    the reliability signal directly into the input distribution.

    Calibration uses sigmoid (Platt) — isotonic is fragile with only a handful
    of LOUO val samples per class, and Platt's two-parameter fit generalizes
    better in this regime.
    """
    print("\n--- MLP (weighted, sigmoid-calibrated) ---")
    if n_classes is None:
        n_classes = int(len(np.unique(y_train)))
    Xw_train = X_train * _W_ARR
    Xw_val   = X_val * _W_ARR

    base = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
    )
    t0 = time.time()
    base.fit(Xw_train, y_train)
    # sklearn 1.6 deprecated cv='prefit' in favour of FrozenEstimator. Both
    # paths are equivalent — wrap the already-fitted MLP so the calibrator
    # treats it as frozen.
    if _HAS_FROZEN_ESTIMATOR:
        cal = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')
    else:
        cal = CalibratedClassifierCV(base, cv='prefit', method='sigmoid')
    cal.fit(Xw_val, y_val)
    elapsed = time.time() - t0

    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Train iters: {getattr(base, 'n_iter_', '?')}")

    return cal, {
        'training_time_s': round(elapsed, 1),
        'hidden': [64, 32],
        'activation': 'relu',
        'calibration': 'sigmoid (prefit on val)',
        'n_iter': int(getattr(base, 'n_iter_', 0) or 0),
    }


def evaluate_on_split(model, X, y, split_name, weighted=False):
    """Evaluate a model on a data split, return accuracy.

    If ``weighted`` is True, X is multiplied by FEATURE_WEIGHTS before predict
    (matches what train_mlp did at fit time).
    """
    from sklearn.metrics import accuracy_score, f1_score
    X_in = X * _W_ARR if weighted else X
    y_pred = model.predict(X_in)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    print(f"  {split_name}: accuracy={acc:.4f}, f1_weighted={f1:.4f}")
    return {'accuracy': round(float(acc), 4), 'f1_weighted': round(float(f1), 4)}


def _remap_static_labels(y_train, y_val, y_test):
    """Map train labels to 0..N-1 and unknown eval labels to a miss sentinel."""
    train_labels = sorted(int(v) for v in np.unique(y_train))
    if not train_labels:
        raise ValueError("Training split has no labels.")

    eval_labels = sorted({
        int(v) for arr in (y_val, y_test) for v in np.unique(arr)
    })
    all_labels = sorted(set(train_labels).union(eval_labels))
    missing_train_labels = sorted(set(all_labels) - set(train_labels))

    label_map = {old: new for new, old in enumerate(train_labels)}
    inv_label_map = {new: old for old, new in label_map.items()}

    def map_train(arr):
        return np.array([label_map[int(v)] for v in arr], dtype=np.int32)

    def map_eval(arr):
        return np.array([
            label_map.get(int(v), UNKNOWN_EVAL_LABEL) for v in arr
        ], dtype=np.int32)

    return {
        'y_train': map_train(y_train),
        'y_val': map_eval(y_val),
        'y_test': map_eval(y_test),
        'label_map': label_map,
        'inv_label_map': inv_label_map,
        'train_labels': train_labels,
        'all_labels': all_labels,
        'missing_train_labels': missing_train_labels,
        'unknown_val_count': int(np.sum([
            int(v) not in label_map for v in y_val
        ])),
        'unknown_test_count': int(np.sum([
            int(v) not in label_map for v in y_test
        ])),
    }


def save_model(model, path, inv_label_map=None, weighted=False, label_names=None):
    """Pickle model (+ optional inv_label_map + weighted flag) to disk.

    Saved payload: {
        'model': <estimator>,
        'inv_label_map': {0: 0, 1: 1, ...},
        'weighted': bool,    # True if predict.py must multiply X by FEATURE_WEIGHTS
        'label_names': {0: 'A', ...},  # optional global label id -> display label
    }

    Args:
        model: trained estimator.
        path: either a key into PATHS (str) OR an absolute Path. Absolute paths
            are used by LOUO mode which writes under data/models/louo/<user>/.
        inv_label_map: maps contiguous training labels back to LETTER_TO_IDX.
        weighted: True iff this model expects pre-multiplied features.
        label_names: optional map for labels outside A-Z (for example REST_1).
    """
    if isinstance(path, Path):
        out_path = path
    else:
        out_path = PROJECT_ROOT / PATHS[path]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': model,
        'inv_label_map': inv_label_map or {},
        'weighted': bool(weighted),
        'label_names': label_names or {},
    }
    with open(str(out_path), 'wb') as f:
        pickle.dump(payload, f)
    print(f"  Saved: {out_path}")
    return str(out_path)


def load_static_model(path, include_label_names=False):
    """Load a static model saved by save_model().

    Returns (model, inv_label_map, weighted). If include_label_names=True,
    appends the optional global-label decoder map as a fourth return value.
    Backwards-compatible: if the pkl is a plain estimator (old v1 format),
    returns empty metadata.
    """
    with open(str(path), 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and 'model' in payload:
        result = (
            payload['model'],
            payload.get('inv_label_map', {}),
            bool(payload.get('weighted', False)),
        )
        if include_label_names:
            return (*result, payload.get('label_names', {}) or {})
        return result
    # Legacy format: raw estimator
    if include_label_names:
        return payload, {}, False, {}
    return payload, {}, False


def train_all(
    quick=False,
    models_to_train=None,
    real=False,
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    report_path: Path | None = None,
    label_names: dict[int, str] | None = None,
):
    """Run the full training pipeline.

    Args:
        quick: Use reduced grids for fast testing.
        models_to_train: Set of model keys to train ('xgb', 'rf', 'svm', 'mlp').
            None = train all four.
        real: If True, load real data splits (data/processed/real/) and write
            models to v2 paths.
        data_dir: When provided, load splits from this directory instead of
            PATHS. Used by LOUO mode.
        models_dir: When provided, write per-model pickles into this directory
            (filenames: rf.pkl, xgb.pkl, svm.pkl, mlp.pkl). Used by LOUO mode.
        report_path: Optional override for the training report destination.

    Returns:
        (report_dict, best_model, best_model_name)
    """
    if models_to_train is None:
        models_to_train = {'xgb', 'rf', 'svm', 'mlp'}

    # Model save targets — either PATHS keys (default) or absolute paths
    # inside models_dir (LOUO mode).
    if models_dir is not None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_paths = {
            'xgb': models_dir / 'xgb.pkl',
            'rf':  models_dir / 'rf.pkl',
            'svm': models_dir / 'svm.pkl',
            'mlp': models_dir / 'mlp.pkl',
        }
    else:
        model_paths = {
            'xgb': 'xgboost_model',
            'rf':  'rf_model',
            'svm': 'svm_model',
            'mlp': 'mlp_model',
        }
    report_key = 'training_report_real' if real else 'training_report'

    print("=" * 60)
    data_src = "Real" if real else "Synthetic"
    suffix = f" (LOUO @ {data_dir})" if data_dir is not None else ""
    print(f"Phase C — Static Classifier Training ({data_src} Data{suffix})")
    print("=" * 60)

    # Load data
    print("\nLoading data (pre-scaled)...")
    X_train, y_train = load_split('train', real=real, data_dir=data_dir)
    X_val,   y_val   = load_split('val',   real=real, data_dir=data_dir)
    X_test,  y_test  = load_split('test',  real=real, data_dir=data_dir)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Remap labels to train-contiguous 0..N-1 (required by XGBoost). If a
    # LOUO split has labels that only exist in val/test, mark them with the
    # unknown sentinel so evaluation counts them as misses.
    remapped = _remap_static_labels(y_train, y_val, y_test)
    y_train = remapped['y_train']
    y_val = remapped['y_val']
    y_test = remapped['y_test']
    label_map = remapped['label_map']
    inv_label_map = remapped['inv_label_map']
    train_labels = remapped['train_labels']
    all_labels = remapped['all_labels']
    missing_train_labels = remapped['missing_train_labels']
    n_classes_train = len(train_labels)
    n_classes_total = len(all_labels)
    label_names = {
        int(label): str((label_names or {}).get(int(label), IDX_TO_LETTER.get(int(label), int(label))))
        for label in all_labels
    }

    if label_map != {v: v for v in range(n_classes_train)}:
        print(f"  Labels remapped to train-contiguous 0..{n_classes_train-1}")
    if missing_train_labels:
        print(
            f"  [WARN] Train is missing {len(missing_train_labels)} class(es) "
            f"that appear in val/test (global LETTER_TO_IDX values: "
            f"{missing_train_labels}). Model cannot predict them; eval will "
            "count those rows as misses."
        )
        print(
            f"  Unknown eval rows: val={remapped['unknown_val_count']}, "
            f"test={remapped['unknown_test_count']}"
        )
    print(f"  Classes in train: {n_classes_train} / {n_classes_total} total")

    # Set up grids (MLP has no grid — fixed architecture)
    if quick:
        print("\n[QUICK MODE] Using reduced parameter grids")
        grids = make_quick_grids()
    else:
        grids = {
            'xgb': XGBOOST_PARAM_GRID,
            'rf': RF_PARAM_GRID,
            'svm': SVM_PARAM_GRID,
        }
    grids.setdefault('mlp', None)  # MLP doesn't use GridSearchCV

    trainers = {
        'xgb': (train_xgboost, model_paths['xgb']),
        'rf': (train_random_forest, model_paths['rf']),
        'svm': (train_svm, model_paths['svm']),
        'mlp': (train_mlp, model_paths['mlp']),
    }

    report = {
        'dataset': {
            'train_samples': int(X_train.shape[0]),
            'val_samples': int(X_val.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'n_features': int(X_train.shape[1]),
            'n_classes': n_classes_total,
            'n_train_classes': n_classes_train,
            'missing_train_labels': missing_train_labels,
            'unknown_eval_label': UNKNOWN_EVAL_LABEL,
            'unknown_val_samples': remapped['unknown_val_count'],
            'unknown_test_samples': remapped['unknown_test_count'],
        },
        # inv_label_map lets predict.py decode model output back to letter index
        'label_map': label_map,
        'inv_label_map': {str(k): v for k, v in inv_label_map.items()},
        'label_names': {str(k): v for k, v in label_names.items()},
        'models': {},
    }

    best_model = None
    best_model_name = None
    best_val_acc = -1.0

    # Class-balancing sample weights for XGBoost. RF and SVM use the
    # `class_weight='balanced'` constructor arg instead (set in their trainers).
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    print(
        f"  Class balancing — sample_weight range: "
        f"[{sample_weights.min():.3f}, {sample_weights.max():.3f}]"
    )

    for key in ['xgb', 'rf', 'svm', 'mlp']:
        if key not in models_to_train:
            continue

        train_fn, path_target = trainers[key]
        weighted = (key == 'mlp')

        # Per-trainer signatures diverge — branch instead of stuffing kwargs.
        if key == 'xgb':
            model, train_info = train_fn(
                X_train, y_train, grids[key], CV_FOLDS, CV_SCORING,
                n_classes=n_classes_train, sample_weight=sample_weights,
            )
        elif key == 'mlp':
            cal_mask = y_val != UNKNOWN_EVAL_LABEL
            X_val_cal = X_val[cal_mask]
            y_val_cal = y_val[cal_mask]
            if y_val_cal.size == 0:
                print(
                    "  [WARN] MLP calibration split has no train-known labels; "
                    "falling back to the training split for calibration."
                )
                X_val_cal = X_train
                y_val_cal = y_train
            model, train_info = train_fn(
                X_train, y_train, X_val_cal, y_val_cal,
                n_classes=n_classes_train,
            )
        else:
            model, train_info = train_fn(
                X_train, y_train, grids[key], CV_FOLDS, CV_SCORING,
            )

        # Evaluate on val and test (MLP needs weight pre-multiply at predict)
        print(f"\n  Evaluating {key}:")
        val_metrics = evaluate_on_split(model, X_val, y_val, 'val', weighted=weighted)
        test_metrics = evaluate_on_split(model, X_test, y_test, 'test', weighted=weighted)

        # Save model (with inv_label_map + weighted flag bundled for predict.py)
        model_path = save_model(
            model, path_target, inv_label_map=inv_label_map, weighted=weighted,
            label_names=label_names,
        )

        report['models'][key] = {
            'training': train_info,
            'val': val_metrics,
            'test': test_metrics,
            'model_path': model_path,
            'weighted': weighted,
        }

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model = model
            best_model_name = key

    # Summary
    report['best_model'] = best_model_name
    report['best_val_accuracy'] = best_val_acc

    print(f"\n{'=' * 60}")
    print(f"Best model: {best_model_name} (val acc: {best_val_acc:.4f})")
    print(f"{'=' * 60}")

    # Save report
    report['data_source'] = 'real' if real else 'synthetic'
    out_report = Path(report_path) if report_path is not None else PROJECT_ROOT / PATHS[report_key]
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_report), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {out_report}")

    return report, best_model, best_model_name


def train_all_louo(quick: bool = False, models_to_train=None) -> dict:
    """Train per-held-out-user models across the LOUO splits.

    For each subdirectory under data/processed/real/louo/<user>/, train all
    candidate models, write them to data/models/louo/<user>/, and aggregate
    the per-user accuracies into data/models/louo_report.json + louo_summary.json.
    """
    louo_root = PROJECT_ROOT / PATHS['louo_root']
    models_root = PROJECT_ROOT / PATHS['louo_models_root']
    if not louo_root.exists():
        raise SystemExit(
            f"LOUO data root not found: {louo_root}\n"
            f"Run `python -m scripts.build_real_dataset --mode louo` first."
        )

    user_dirs = sorted(d for d in louo_root.iterdir() if d.is_dir())
    if not user_dirs:
        raise SystemExit(f"No per-user subdirectories under {louo_root}.")

    summary: dict[str, dict] = {}
    per_user_reports: dict[str, dict] = {}

    for udir in user_dirs:
        user_id = udir.name
        models_dir = models_root / user_id
        report_path = models_dir / 'training_report.json'
        report, _, best = train_all(
            quick=quick,
            models_to_train=models_to_train,
            real=True,
            data_dir=udir,
            models_dir=models_dir,
            report_path=report_path,
        )
        per_user_reports[user_id] = report
        summary[user_id] = {
            'best_model': best,
            'val_accuracy': report.get('best_val_accuracy'),
            'per_model_test_accuracy': {
                k: v['test']['accuracy'] for k, v in report['models'].items()
            },
        }

    # Aggregate
    mean_acc_by_model: dict[str, float] = {}
    for u, info in summary.items():
        for m, acc in info['per_model_test_accuracy'].items():
            mean_acc_by_model.setdefault(m, []).append(acc)
    aggregate = {m: float(np.mean(v)) for m, v in mean_acc_by_model.items()}

    out = {
        'per_user': summary,
        'mean_test_accuracy_by_model': aggregate,
        'best_model_overall': max(aggregate, key=aggregate.get) if aggregate else None,
    }
    models_root.mkdir(parents=True, exist_ok=True)
    summary_path = PROJECT_ROOT / PATHS['louo_summary']
    with open(str(summary_path), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nLOUO summary saved: {summary_path}")
    print("LOUO mean test accuracy by model:")
    for m, a in aggregate.items():
        print(f"  {m:5s}  {a:.4f}")
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train static classifiers')
    parser.add_argument('--quick', action='store_true',
                        help='Fast mode with reduced grids')
    parser.add_argument('--model', choices=['xgb', 'rf', 'svm', 'mlp'],
                        help='Train only one model')
    parser.add_argument('--real', action='store_true',
                        help='Load real data splits (data/processed/real/) '
                             'and save v2 models')
    parser.add_argument('--louo', action='store_true',
                        help='Train one model set per held-out user under '
                             'data/processed/real/louo/<user>/. Implies --real.')
    args = parser.parse_args()

    models = {args.model} if args.model else None
    if args.louo:
        train_all_louo(quick=args.quick, models_to_train=models)
    else:
        train_all(quick=args.quick, models_to_train=models, real=args.real)
