"""End-to-end LOUO (Leave-One-User-Out) orchestrator.

Runs the full evaluation pipeline so a single command produces an honest
cross-wearer accuracy metric:

    1) Build per-held-out-user datasets under data/processed/real/louo/<user>/
       (calls scripts.build_real_dataset.build_louo_all).
    2) Train static models (RF, XGB, SVM, MLP) per LOUO split
       (calls ml.train_static.train_all_louo).
    3) Train CNN-LSTM per LOUO split
       (calls ml.train_dynamic.train_dynamic_louo).
    4) Aggregate per-user accuracies and write data/models/louo_summary.json.

Success criteria (from the implementation plan):
    LOUO static  mean test accuracy ≥ 0.75
    LOUO dynamic mean test accuracy ≥ 0.70

Usage:
    python -m scripts.run_louo                          # full run
    python -m scripts.run_louo --skip-build             # reuse existing data
    python -m scripts.run_louo --skip-dynamic           # static only
    python -m scripts.run_louo --quick                  # fast smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import PATHS, SEED


def main() -> int:
    parser = argparse.ArgumentParser(description='LOUO end-to-end orchestrator')
    parser.add_argument('--skip-build', action='store_true',
                        help='Reuse existing data/processed/real/louo/ output.')
    parser.add_argument('--skip-static', action='store_true',
                        help='Skip the static-classifier LOUO pass.')
    parser.add_argument('--skip-dynamic', action='store_true',
                        help='Skip the CNN-LSTM LOUO pass.')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (reduced grids + fewer epochs).')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    if not args.skip_build:
        from scripts.build_real_dataset import build_louo_all
        print('=' * 70)
        print('Step 1/3 — Building per-held-out-user datasets')
        print('=' * 70)
        build_louo_all(seed=args.seed)

    static_summary = None
    if not args.skip_static:
        from ml.train_static import train_all_louo
        print('\n' + '=' * 70)
        print('Step 2/3 — Training static classifiers per LOUO split')
        print('=' * 70)
        static_summary = train_all_louo(quick=args.quick)

    dynamic_summary = None
    if not args.skip_dynamic:
        from ml.train_dynamic import train_dynamic_louo
        print('\n' + '=' * 70)
        print('Step 3/3 — Training CNN-LSTM per LOUO split')
        print('=' * 70)
        dynamic_summary = train_dynamic_louo(quick=args.quick)

    # Aggregate
    out = {
        'static': static_summary,
        'dynamic': dynamic_summary,
        'targets': {'static_mean_test_acc': 0.75, 'dynamic_mean_test_acc': 0.70},
        'pass': {
            'static': bool(
                static_summary
                and static_summary.get('best_model_overall')
                and max(static_summary['mean_test_accuracy_by_model'].values()) >= 0.75
            ),
            'dynamic': bool(
                dynamic_summary
                and dynamic_summary.get('mean_test_accuracy', 0) >= 0.70
            ),
        },
    }
    summary_path = PROJECT_ROOT / PATHS['louo_summary']
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print('\n' + '=' * 70)
    print('LOUO end-to-end summary')
    print('=' * 70)
    if static_summary:
        print('Static mean test accuracy:')
        for m, a in (static_summary.get('mean_test_accuracy_by_model') or {}).items():
            target = ' ✓' if a >= 0.75 else ' ✗ (target 0.75)'
            print(f'  {m:5s}  {a:.4f}{target}')
    if dynamic_summary:
        a = dynamic_summary.get('mean_test_accuracy', 0)
        target = ' ✓' if a >= 0.70 else ' ✗ (target 0.70)'
        print(f'Dynamic CNN-LSTM mean test accuracy: {a:.4f}{target}')
    print(f'\nSummary written: {summary_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
