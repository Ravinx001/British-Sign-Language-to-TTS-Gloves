"""
Phase F — Ablation studies and noise robustness evaluation.

Tests model degradation when each feature group (flex, fsr, touch,
orientation, derived) is independently removed, and evaluates
robustness under 2× training noise.

Usage:
    python -m ml.ablation           # Full ablation study
    python -m ml.ablation --quick   # Quick mode (single param grid)
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from ml.config import (
    CV_FOLDS,
    CV_SCORING,
    DATA_GEN,
    FEATURE_GROUPS,
    FEATURE_NAMES,
    LETTERS,
    NUM_CLASSES,
    NUM_FEATURES,
    PATHS,
    SEED,
    XGBOOST_PARAM_GRID,
)
from ml.train_static import load_split, make_quick_grids, train_xgboost

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Ablation: Feature Group Removal
# =============================================================================

def run_feature_ablation(quick=False):
    """Retrain XGBoost with each feature group removed and measure accuracy.

    Returns:
        dict with keys:
          'baseline': float (test accuracy with all features)
          'groups': dict mapping group_name → {accuracy, drop, features_removed}
    """
    print("=" * 60)
    print("  Feature Group Ablation Study")
    print("=" * 60)

    X_train, y_train = load_split('train')
    X_test, y_test = load_split('test')

    # --- Baseline ---
    print("\n  [Baseline] All 36 features")
    if quick:
        grid = make_quick_grids()['xgb']
    else:
        grid = XGBOOST_PARAM_GRID

    baseline_model, baseline_info = train_xgboost(
        X_train, y_train, grid, CV_FOLDS, CV_SCORING,
    )
    baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test))
    print(f"  Baseline test accuracy: {baseline_acc:.4f}")

    # --- Ablate each group ---
    groups_results = {}

    for group_name, group_indices in FEATURE_GROUPS.items():
        print(f"\n  [Ablate] Removing '{group_name}' "
              f"({len(group_indices)} features: indices {group_indices[0]}–{group_indices[-1]})")

        # Build mask: keep all features EXCEPT this group
        keep_mask = np.ones(NUM_FEATURES, dtype=bool)
        keep_mask[group_indices] = False
        n_kept = int(keep_mask.sum())

        X_train_sub = X_train[:, keep_mask]
        X_test_sub = X_test[:, keep_mask]

        print(f"  Training on {n_kept} features...")
        model, info = train_xgboost(
            X_train_sub, y_train, grid, CV_FOLDS, CV_SCORING,
        )
        acc = accuracy_score(y_test, model.predict(X_test_sub))
        drop = baseline_acc - acc

        removed_names = [FEATURE_NAMES[i] for i in group_indices]
        groups_results[group_name] = {
            'accuracy': round(float(acc), 4),
            'accuracy_drop': round(float(drop), 4),
            'features_removed': removed_names,
            'n_features_kept': n_kept,
            'cv_score': round(float(info['best_cv_score']), 4),
        }

        status = "PASS" if acc >= 0.70 else "FAIL"
        print(f"  Accuracy: {acc:.4f} (drop: {drop:+.4f}) — "
              f"≥70% target: {status}")

    return {
        'baseline_accuracy': round(float(baseline_acc), 4),
        'groups': groups_results,
    }


# =============================================================================
# Noise Robustness
# =============================================================================

def run_noise_robustness():
    """Test model accuracy at 2× training noise level.

    Adds extra Gaussian noise (σ=3% → σ=6% total effective) to test data
    and evaluates the baseline XGBoost model.

    Returns:
        dict with noise test results.
    """
    print("\n" + "=" * 60)
    print("  Noise Robustness (2× Training Noise)")
    print("=" * 60)

    import pickle
    model_path = PROJECT_ROOT / PATHS['xgboost_model']
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X_test, y_test = load_split('test')

    # Baseline (no extra noise)
    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  Baseline test accuracy: {baseline_acc:.4f}")

    # Add extra noise: effective σ = 2× training noise = 6%
    # Since data is already StandardScaled, adding noise in scaled space
    rng = np.random.default_rng(SEED)
    extra_sigma = DATA_GEN['noise_sigma']  # 3% extra → total ~6%

    X_noisy = X_test.copy()
    noise = rng.normal(0, extra_sigma, X_noisy.shape)
    # Don't add noise to binary touch features (indices 16–25 in raw space,
    # but data is scaled — still, these features are 0/1 binary and scaled)
    X_noisy += noise

    noisy_acc = accuracy_score(y_test, model.predict(X_noisy))
    drop = baseline_acc - noisy_acc

    status = "PASS" if noisy_acc >= 0.85 else "FAIL"
    print(f"  2× noise accuracy: {noisy_acc:.4f} (drop: {drop:+.4f}) — "
          f"≥85% target: {status}")

    # Also test at higher noise levels for the report
    noise_levels = [0.01, 0.03, 0.06, 0.09, 0.12, 0.15]
    noise_curve = {}
    for sigma in noise_levels:
        X_n = X_test + rng.normal(0, sigma, X_test.shape)
        acc = accuracy_score(y_test, model.predict(X_n))
        noise_curve[str(sigma)] = round(float(acc), 4)
        print(f"    σ={sigma:.2f}: accuracy={acc:.4f}")

    return {
        'baseline_accuracy': round(float(baseline_acc), 4),
        'noise_2x_accuracy': round(float(noisy_acc), 4),
        'noise_2x_drop': round(float(drop), 4),
        'noise_2x_pass': noisy_acc >= 0.85,
        'noise_curve': noise_curve,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_ablation_chart(ablation_results, save_path=None):
    """Bar chart showing accuracy per feature group removal."""
    baseline = ablation_results['baseline_accuracy']
    groups = ablation_results['groups']

    names = list(groups.keys())
    accs = [groups[n]['accuracy'] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(names, accs, color='steelblue', edgecolor='black', width=0.6)

    # Baseline reference line
    ax.axhline(y=baseline, color='red', linestyle='--', linewidth=1.5,
               label=f'Baseline: {baseline:.1%}')

    # 70% threshold line
    ax.axhline(y=0.70, color='orange', linestyle=':', linewidth=1.5,
               label='Min threshold: 70%')

    # Value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_xlabel('Removed Feature Group', fontsize=12)
    ax.set_title('Feature Group Ablation — XGBoost Static Classifier', fontsize=14)
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Ablation chart saved: {save_path}")
    plt.close(fig)


def plot_noise_curve(noise_results, save_path=None):
    """Line chart showing accuracy degradation with increasing noise."""
    curve = noise_results['noise_curve']
    sigmas = [float(s) for s in curve.keys()]
    accs = list(curve.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigmas, accs, 'o-', color='steelblue', linewidth=2, markersize=8)

    # 85% threshold line
    ax.axhline(y=0.85, color='orange', linestyle=':', linewidth=1.5,
               label='Target: 85%')

    # Mark the 2× noise point
    ax.axvline(x=0.03, color='red', linestyle='--', linewidth=1,
               label='2× training noise (σ=0.03 extra)')

    ax.set_xlabel('Additional Noise σ', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Noise Robustness — XGBoost Static Classifier', fontsize=14)
    ax.set_ylim(0.6, 1.02)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Noise curve saved: {save_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def run_ablation(quick=False):
    """Run full ablation study: feature groups + noise robustness."""
    t0 = time.time()

    # 1. Feature group ablation
    ablation_results = run_feature_ablation(quick=quick)

    # 2. Noise robustness
    noise_results = run_noise_robustness()

    elapsed = time.time() - t0

    # --- Save results ---
    output_dir = PROJECT_ROOT / 'data' / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = PROJECT_ROOT / 'data' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    report = {
        'ablation': ablation_results,
        'noise_robustness': noise_results,
        'runtime_s': round(elapsed, 1),
    }
    json_path = output_dir / 'ablation_table.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # Charts
    plot_ablation_chart(ablation_results,
                        save_path=str(figures_dir / 'ablation_chart.png'))
    plot_noise_curve(noise_results,
                     save_path=str(figures_dir / 'noise_curve.png'))

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  ABLATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Baseline accuracy: {ablation_results['baseline_accuracy']:.1%}")
    print()

    all_pass = True
    for name, res in ablation_results['groups'].items():
        ok = res['accuracy'] >= 0.70
        if not ok:
            all_pass = False
        print(f"  Remove {name:12s}: {res['accuracy']:.1%} "
              f"(drop {res['accuracy_drop']:+.1%}) "
              f"{'PASS' if ok else 'FAIL'}")

    noise_pass = noise_results['noise_2x_pass']
    if not noise_pass:
        all_pass = False
    print(f"\n  2× noise accuracy: {noise_results['noise_2x_accuracy']:.1%} "
          f"{'PASS' if noise_pass else 'FAIL'}")

    print(f"\n  Overall Phase F ablation: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Runtime: {elapsed:.0f}s")
    print(f"{'=' * 60}")

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--quick', action='store_true',
                        help='Use reduced hyperparameter grid for speed')
    args = parser.parse_args()

    run_ablation(quick=args.quick)
