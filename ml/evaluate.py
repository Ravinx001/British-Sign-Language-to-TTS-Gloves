"""
Evaluation module: confusion matrix, per-sign metrics, SHAP importance.

Generates:
  - 26×26 confusion matrix heatmap (PNG)
  - Per-sign precision/recall/F1 table
  - SHAP feature importance (global + per-class top features)
  - JSON report with all metrics
  - Confused-pair analysis

Usage:
    python -m ml.evaluate                          # Evaluate best model
    python -m ml.evaluate --model xgb              # Evaluate specific model
    python -m ml.evaluate --skip-shap              # Skip SHAP (faster)
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ml.config import (
    CONFUSED_PAIRS,
    FEATURE_NAMES,
    IDX_TO_LETTER,
    LETTERS,
    LETTER_TO_IDX,
    NUM_CLASSES,
    NUM_FEATURES,
    PATHS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / 'data' / 'figures'
MODEL_DIR = PROJECT_ROOT / 'data' / 'models'


def load_split(name):
    """Load a data split from data/processed/."""
    path = PROJECT_ROOT / PATHS[name]
    d = np.load(str(path))
    return d['X'], d['y']


def load_model(key):
    """Load a pickled model by config key."""
    path = PROJECT_ROOT / PATHS[key]
    with open(str(path), 'rb') as f:
        return pickle.load(f)


def load_training_report():
    """Load the training report JSON."""
    path = PROJECT_ROOT / PATHS['training_report']
    if not path.exists():
        return None
    with open(str(path)) as f:
        return json.load(f)


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix',
                          save_path=None):
    """Generate and save a 26×26 confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax, square=True, linewidths=0.5,
        cbar_kws={'shrink': 0.8},
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
        print(f"  Saved: {save_path}")

    plt.close(fig)
    return cm


def plot_per_class_f1(precisions, recalls, f1s, labels, save_path=None):
    """Bar chart of per-class F1 scores with precision/recall overlay."""
    fig, ax = plt.subplots(figsize=(16, 5))

    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, precisions, width, label='Precision', color='steelblue', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', color='coral', alpha=0.8)
    ax.bar(x + width, f1s, width, label='F1', color='seagreen', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.80, color='red', linestyle='--', alpha=0.4, label='80% threshold')
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
        print(f"  Saved: {save_path}")

    plt.close(fig)


def analyze_confused_pairs(cm, labels):
    """Analyze the known confused pairs from the confusion matrix.

    Returns list of dicts with pair info and confusion counts.
    """
    results = []
    for a, b in CONFUSED_PAIRS:
        if a not in LETTER_TO_IDX or b not in LETTER_TO_IDX:
            continue
        i, j = LETTER_TO_IDX[a], LETTER_TO_IDX[b]
        total_a = cm[i].sum()
        total_b = cm[j].sum()
        # Confusion rate: what fraction of A was predicted as B, and vice versa
        a_as_b = cm[i, j] / max(total_a, 1)
        b_as_a = cm[j, i] / max(total_b, 1)
        results.append({
            'pair': f'{a}/{b}',
            f'{a}_predicted_as_{b}': int(cm[i, j]),
            f'{b}_predicted_as_{a}': int(cm[j, i]),
            f'{a}_as_{b}_rate': round(float(a_as_b), 4),
            f'{b}_as_{a}_rate': round(float(b_as_a), 4),
            'max_confusion_rate': round(float(max(a_as_b, b_as_a)), 4),
        })
    return results


def compute_shap_importance(model, X_sample, model_name='model'):
    """Compute SHAP feature importance.

    Returns:
        global_importance: dict mapping feature name → mean |SHAP|
        shap_values: raw SHAP values array
    """
    import shap

    print(f"  Computing SHAP values ({model_name})...")
    t0 = time.time()

    if hasattr(model, 'feature_importances_'):
        # Tree-based: use TreeExplainer (fast)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        # SVM / other: use KernelExplainer (slower, use smaller sample)
        background = shap.kmeans(X_sample, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample[:200])

    elapsed = time.time() - t0
    print(f"  SHAP computation: {elapsed:.1f}s")

    # For multi-class, shap_values is list of arrays (one per class)
    # or 3D array (samples × features × classes)
    if isinstance(shap_values, list):
        # Average absolute SHAP across all classes
        abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        abs_shap = np.mean(np.abs(shap_values), axis=2)
    else:
        abs_shap = np.abs(shap_values)

    # Global importance: mean across samples
    global_imp = abs_shap.mean(axis=0)
    importance_dict = {
        FEATURE_NAMES[i]: round(float(global_imp[i]), 6)
        for i in range(min(len(global_imp), NUM_FEATURES))
    }

    return importance_dict, shap_values


def plot_feature_importance(importance_dict, save_path=None, top_n=20):
    """Bar chart of top-N SHAP feature importances."""
    sorted_feats = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top = sorted_feats[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [f[0] for f in top]
    values = [f[1] for f in top]

    ax.barh(range(len(top)), values, color='steelblue', edgecolor='navy', alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Top {top_n} Feature Importances (SHAP)')
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150)
        print(f"  Saved: {save_path}")

    plt.close(fig)


def evaluate_model(model, model_name, X_test, y_test, skip_shap=False):
    """Full evaluation of a single model.

    Returns:
        eval_report: dict with all metrics and paths to generated artifacts.
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 60}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)

    # Overall metrics
    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"\n  Test accuracy:    {acc:.4f}")
    print(f"  Test F1 weighted: {f1_w:.4f}")
    print(f"  Test F1 macro:    {f1_macro:.4f}")

    # Per-class metrics
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    per_class = {}
    low_precision_letters = []
    for i in range(NUM_CLASSES):
        letter = IDX_TO_LETTER[i]
        per_class[letter] = {
            'precision': round(float(precisions[i]), 4),
            'recall': round(float(recalls[i]), 4),
            'f1': round(float(f1s[i]), 4),
            'support': int(supports[i]),
        }
        if precisions[i] < 0.80 and supports[i] > 0:
            low_precision_letters.append(letter)

    if low_precision_letters:
        print(f"\n  WARNING: Letters below 80% precision: {low_precision_letters}")
    else:
        print(f"\n  All letters ≥80% precision")

    # Confusion matrix
    print("\n  Generating confusion matrix...")
    cm = plot_confusion_matrix(
        y_test, y_pred, LETTERS,
        title=f'Confusion Matrix — {model_name} (acc={acc:.3f})',
        save_path=FIG_DIR / f'confusion_matrix_{model_name}.png',
    )

    # Also save to the canonical path
    plot_confusion_matrix(
        y_test, y_pred, LETTERS,
        title=f'Confusion Matrix — {model_name} (acc={acc:.3f})',
        save_path=PROJECT_ROOT / PATHS['confusion_matrix'],
    )

    # Per-class F1 chart
    plot_per_class_f1(
        precisions, recalls, f1s, LETTERS,
        save_path=FIG_DIR / f'per_class_f1_{model_name}.png',
    )

    # Confused pairs analysis
    print("\n  Analyzing confused pairs...")
    confused = analyze_confused_pairs(cm, LETTERS)
    high_confusion = [c for c in confused if c['max_confusion_rate'] > 0.10]
    for c in confused:
        status = " ⚠️" if c['max_confusion_rate'] > 0.10 else " ✓"
        print(f"    {c['pair']}: max confusion rate = {c['max_confusion_rate']:.2%}{status}")

    # SHAP
    shap_importance = None
    if not skip_shap:
        print("\n  SHAP feature importance...")
        # Use a subsample for SHAP (faster)
        n_shap = min(1000, len(X_test))
        rng = np.random.default_rng(42)
        shap_idx = rng.choice(len(X_test), n_shap, replace=False)
        X_shap = X_test[shap_idx]

        shap_importance, _ = compute_shap_importance(model, X_shap, model_name)
        plot_feature_importance(
            shap_importance,
            save_path=FIG_DIR / f'shap_importance_{model_name}.png',
        )

        # Save feature importance JSON
        imp_path = PROJECT_ROOT / PATHS['feature_importance']
        with open(str(imp_path), 'w') as f:
            json.dump(shap_importance, f, indent=2)
        print(f"  Saved: {imp_path}")

    # Build report
    eval_report = {
        'model_name': model_name,
        'test_accuracy': round(float(acc), 4),
        'test_f1_weighted': round(float(f1_w), 4),
        'test_f1_macro': round(float(f1_macro), 4),
        'per_class': per_class,
        'low_precision_letters': low_precision_letters,
        'confused_pairs': confused,
        'high_confusion_pairs': [c['pair'] for c in high_confusion],
        'n_high_confusion_pairs': len(high_confusion),
        'meets_92_target': acc >= 0.92,
        'meets_80_precision_target': len(low_precision_letters) == 0,
        'meets_confusion_target': len(high_confusion) <= 5,
    }

    if shap_importance:
        # Top 10 most important features
        sorted_feats = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
        eval_report['top_10_features'] = [
            {'feature': f, 'importance': v} for f, v in sorted_feats[:10]
        ]

    return eval_report


def run_evaluation(model_key=None, skip_shap=False):
    """Run full evaluation pipeline.

    If model_key is None, evaluates the best model from training_report.json.
    """
    # Load test data
    print("Loading test data...")
    X_test, y_test = load_split('test')
    print(f"  Test set: {X_test.shape}, {len(np.unique(y_test))} classes")

    # Determine which model to evaluate
    if model_key is None:
        report = load_training_report()
        if report and 'best_model' in report:
            model_key = report['best_model']
            print(f"  Best model from training: {model_key}")
        else:
            model_key = 'xgb'
            print(f"  No training report found, defaulting to: {model_key}")

    path_map = {
        'xgb': 'xgboost_model',
        'rf': 'rf_model',
        'svm': 'svm_model',
    }

    config_key = path_map.get(model_key, model_key)
    model = load_model(config_key)
    print(f"  Loaded model: {config_key}")

    # Run evaluation
    eval_report = evaluate_model(model, model_key, X_test, y_test, skip_shap)

    # Save evaluation report
    eval_path = MODEL_DIR / 'evaluation_report.json'
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(eval_path), 'w') as f:
        json.dump(eval_report, f, indent=2)
    print(f"\nEvaluation report saved: {eval_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Model:              {eval_report['model_name']}")
    print(f"  Test accuracy:      {eval_report['test_accuracy']:.4f}")
    print(f"  Test F1 (weighted): {eval_report['test_f1_weighted']:.4f}")
    print(f"  ≥92% accuracy:      {'PASS' if eval_report['meets_92_target'] else 'FAIL'}")
    print(f"  All letters ≥80%:   {'PASS' if eval_report['meets_80_precision_target'] else 'FAIL'}")
    print(f"  Confused pairs ≤5:  {'PASS' if eval_report['meets_confusion_target'] else 'FAIL'}")

    if not eval_report['meets_92_target']:
        print(f"\n  *** TARGET NOT MET — iterate: adjust definitions → regenerate → retrain ***")

    return eval_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate static classifier')
    parser.add_argument('--model', choices=['xgb', 'rf', 'svm'],
                        help='Model to evaluate (default: best from training)')
    parser.add_argument('--skip-shap', action='store_true',
                        help='Skip SHAP computation (faster)')
    args = parser.parse_args()

    run_evaluation(model_key=args.model, skip_shap=args.skip_shap)
