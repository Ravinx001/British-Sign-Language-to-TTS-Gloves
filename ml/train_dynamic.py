"""
Dynamic classifier training: CNN-LSTM for configured dynamic letters.

Architecture (from plan):
    Input(100,36) → Conv1D(64,k=5) → BN → MaxPool(2)
                  → Conv1D(128,k=3) → BN → MaxPool(2)
                  → LSTM(128,return_seq) → Dropout(0.3)
                  → LSTM(64) → Dropout(0.3)
                  → Dense(64,relu) → Dropout(0.2)
                  → Dense(n_classes,softmax)

Implemented in PyTorch (TensorFlow unavailable on Python 3.14).
Uses Adam optimizer, early stopping, learning rate reduction on plateau.

Usage:
    python -m ml.train_dynamic                # Full training on synthetic data
    python -m ml.train_dynamic --real         # Full training on real dynamic data
    python -m ml.train_dynamic --quick        # Quick test (10 epochs)
    python -m ml.train_dynamic --real --quick # Real data, quick mode
    python -m ml.train_dynamic --epochs 50    # Custom epoch count
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.config import CNN_LSTM, DYNAMIC_LETTERS, DYNAMIC_WINDOW_FRAMES, IDX_TO_LETTER, NUM_FEATURES, PATHS, SEED

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Model Architecture
# =============================================================================

class CnnLstmClassifier(nn.Module):
    """CNN-LSTM for dynamic gesture classification.

    Conv1D layers extract local temporal patterns, then LSTM layers
    capture sequential dependencies across the full gesture window.
    """

    def __init__(self, n_features, n_classes, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = CNN_LSTM

        # Conv block 1: Conv1D(64, k=5) → BN → ReLU → MaxPool(2)
        self.conv1 = nn.Conv1d(n_features, cfg['conv1_filters'], cfg['conv1_kernel'], padding='same')
        self.bn1 = nn.BatchNorm1d(cfg['conv1_filters'])
        self.pool1 = nn.MaxPool1d(cfg['pool_size'])

        # Conv block 2: Conv1D(128, k=3) → BN → ReLU → MaxPool(2)
        self.conv2 = nn.Conv1d(cfg['conv1_filters'], cfg['conv2_filters'], cfg['conv2_kernel'], padding='same')
        self.bn2 = nn.BatchNorm1d(cfg['conv2_filters'])
        self.pool2 = nn.MaxPool1d(cfg['pool_size'])

        # LSTM layers
        self.lstm1 = nn.LSTM(cfg['conv2_filters'], cfg['lstm1_units'],
                             batch_first=True, dropout=0.0)
        self.drop_lstm1 = nn.Dropout(cfg['lstm_dropout'])
        self.lstm2 = nn.LSTM(cfg['lstm1_units'], cfg['lstm2_units'],
                             batch_first=True, dropout=0.0)
        self.drop_lstm2 = nn.Dropout(cfg['lstm_dropout'])

        # Dense classifier
        self.fc1 = nn.Linear(cfg['lstm2_units'], cfg['dense_units'])
        self.drop_fc = nn.Dropout(cfg['dense_dropout'])
        self.fc2 = nn.Linear(cfg['dense_units'], n_classes)

    def forward(self, x):
        # x: (batch, seq_len, features) — e.g. (B, 100, 36)
        # Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))

        # Back to (batch, seq_len, channels) for LSTM
        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)
        x = self.drop_lstm1(x)
        x, _ = self.lstm2(x)
        x = self.drop_lstm2(x[:, -1, :])  # Take last time step

        x = torch.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x  # Raw logits; CrossEntropyLoss applies softmax internally


# =============================================================================
# Data Loading
# =============================================================================

def load_dynamic_split(name, real=False, data_dir: Path | None = None):
    """Load a dynamic data split. Returns (X, y, class_names).

    Args:
        name: Split name — 'train', 'val', or 'test'.
        real: If True, loads real_dynamic_* paths (data/processed/real/).
        data_dir: If provided, loads ``<data_dir>/dynamic_<name>.npz``. Used
            by LOUO mode.
    """
    if data_dir is not None:
        path = Path(data_dir) / f'dynamic_{name}.npz'
        d = np.load(str(path), allow_pickle=True)
        X, y = d['X'], d['y']
        unique_labels = sorted(np.unique(y))
        label_map = {int(old): new for new, old in enumerate(unique_labels)}
        y = np.array([label_map[int(v)] for v in y], dtype=np.int32)
        if 'class_names' in d:
            class_names = d['class_names']
        else:
            class_names = np.array([IDX_TO_LETTER[int(l)] for l in unique_labels])
        return X, y, class_names
    if real:
        path = PROJECT_ROOT / PATHS[f'real_dynamic_{name}']
        d = np.load(str(path), allow_pickle=True)
        X, y = d['X'], d['y']
        # Always remap y to contiguous 0..N-1.
        # Real data stores global LETTER_TO_IDX labels (J=9, Z=25) which are
        # non-contiguous and out-of-bounds for a 2-class PyTorch model.
        unique_labels = sorted(np.unique(y))
        label_map = {int(old): new for new, old in enumerate(unique_labels)}
        y = np.array([label_map[int(v)] for v in y], dtype=np.int32)
        # class_names: prefer stored names, else reconstruct
        if 'class_names' in d:
            class_names = d['class_names']
        else:
            class_names = np.array([IDX_TO_LETTER[int(l)] for l in unique_labels])
        return X, y, class_names
    else:
        path = PROJECT_ROOT / f'data/processed/dynamic_{name}.npz'
        d = np.load(str(path), allow_pickle=True)
        return d['X'], d['y'], d['class_names']


def make_dataloader(X, y, batch_size, shuffle=True):
    """Create a PyTorch DataLoader from numpy arrays."""
    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_class_weights(y: np.ndarray, n_classes: int, device) -> torch.Tensor:
    """Balanced class weights for dynamic training splits with distractors."""
    counts = np.bincount(y.astype(np.int64), minlength=n_classes).astype(np.float64)
    total = float(counts.sum())
    weights = np.ones(n_classes, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = total / (float(n_classes) * counts[nonzero])
    return torch.FloatTensor(weights).to(device)


# =============================================================================
# Training Loop
# =============================================================================

def _mixup_batch(X, y, alpha: float, rng: np.random.Generator):
    """Sample a single λ ~ Beta(α, α) and produce a mixed batch.

    Returns (X_mix, y_a, y_b, lam). With α=0.2 the Beta is U-shaped, so most
    samples are nearly unmixed — gentle regularization that helps when the
    LOUO held-out set is small.
    """
    lam = float(rng.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    X_mix = lam * X + (1.0 - lam) * X[idx]
    return X_mix, y, y[idx], lam


def train_one_epoch(model, loader, criterion, optimizer, device,
                    mixup_alpha: float = 0.0,
                    rng: np.random.Generator | None = None):
    """Train for one epoch. Returns (loss, accuracy).

    If mixup_alpha > 0, mixes each batch and uses the interpolated CE loss.
    Accuracy is computed against the dominant target (y_a) for reporting only.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        if mixup_alpha > 0 and rng is not None:
            X_mix, y_a, y_b, lam = _mixup_batch(X_batch, y_batch, mixup_alpha, rng)
            logits = model(X_mix)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            preds = logits.argmax(dim=1)
            correct += (preds == y_a).sum().item()  # report against dominant target
        else:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        total += len(y_batch)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on a data split. Returns (loss, accuracy, all_preds, all_targets)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_targets)


def train_dynamic(
    max_epochs=None,
    quick=False,
    real=False,
    data_dir: Path | None = None,
    model_path: Path | None = None,
    report_path: Path | None = None,
    mixup_alpha: float = 0.2,
):
    """Train the CNN-LSTM dynamic gesture classifier.

    Args:
        max_epochs: Override max epochs. None uses config default.
        quick: Quick test mode (10 epochs, no patience).
        real: If True, load real_dynamic_* splits and write report to
              dynamic_training_report_real.json.
        data_dir: When provided, load dynamic_<split>.npz from here. Used by
            LOUO mode to read from data/processed/real/louo/<user>/.
        model_path: Override for the saved-model destination (LOUO writes to
            data/models/louo/<user>/cnn_lstm.pt).
        report_path: Override for the JSON report destination.
        mixup_alpha: Beta α for batch-level mixup (0 = disabled). Default
            0.2 = gentle regularisation that helps when LOUO val is small.

    Returns:
        dict with training report.
    """
    cfg = CNN_LSTM
    if max_epochs is None:
        max_epochs = 10 if quick else cfg['max_epochs']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    data_src = "Real" if real else "Synthetic"
    suffix = f" (LOUO @ {data_dir})" if data_dir is not None else ""
    print(f"Phase D — Dynamic Classifier Training (CNN-LSTM) — {data_src}{suffix}")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Mixup alpha: {mixup_alpha}")

    # Seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # Load data
    print("\nLoading dynamic data...")
    X_train, y_train, class_names = load_dynamic_split('train', real=real, data_dir=data_dir)
    X_val, y_val, _ = load_dynamic_split('val', real=real, data_dir=data_dir)
    X_test, y_test, _ = load_dynamic_split('test', real=real, data_dir=data_dir)
    n_classes = len(class_names)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes ({n_classes}): {list(class_names)}")

    # DataLoaders
    batch_size = cfg['batch_size']
    train_loader = make_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, batch_size, shuffle=False)

    # Model
    model = CnnLstmClassifier(NUM_FEATURES, n_classes, cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {param_count:,}")

    # Loss, optimizer, scheduler
    class_weights = make_class_weights(y_train, n_classes, device)
    print(f"  Class weights: {[round(float(w), 3) for w in class_weights.detach().cpu()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=cfg['lr_reduce_factor'],
        patience=cfg['lr_reduce_patience'],
    )

    # Early stopping
    patience = cfg['early_stop_patience'] if not quick else max_epochs
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nTraining...")
    t0 = time.time()

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=mixup_alpha, rng=rng,
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{max_epochs}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Best val loss: {best_val_loss:.4f}, Best val acc: {best_val_acc:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Test evaluation
    print("\n  Evaluating on test set...")
    test_loss, test_acc, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )
    print(f"  Test accuracy: {test_acc:.4f}")

    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    metric_labels = list(range(n_classes))
    report_text = classification_report(
        test_targets,
        test_preds,
        labels=metric_labels,
        target_names=list(class_names),
        zero_division=0,
    )
    print(f"\n{report_text}")

    cm = confusion_matrix(test_targets, test_preds, labels=metric_labels)

    # Save model
    out_model_path = Path(model_path) if model_path is not None else PROJECT_ROOT / PATHS['cnn_lstm_model']
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features': NUM_FEATURES,
        'n_classes': n_classes,
        'class_names': list(class_names),
        'cfg': cfg,
    }, str(out_model_path))
    print(f"\n  Model saved: {out_model_path}")

    # Save training report
    report = {
        'model': 'CNN-LSTM (PyTorch)',
        'device': str(device),
        'n_params': param_count,
        'n_classes': n_classes,
        'class_names': list(class_names),
        'epochs_trained': len(history['train_loss']),
        'max_epochs': max_epochs,
        'training_time_s': round(elapsed, 1),
        'best_val_loss': round(best_val_loss, 4),
        'best_val_acc': round(best_val_acc, 4),
        'test_accuracy': round(test_acc, 4),
        'test_loss': round(test_loss, 4),
        'confusion_matrix': cm.tolist(),
        'per_class': {},
        'dataset': {
            'train': int(X_train.shape[0]),
            'val': int(X_val.shape[0]),
            'test': int(X_test.shape[0]),
            'seq_len': int(X_train.shape[1]),
            'n_features': int(X_train.shape[2]),
        },
        'meets_88_target': test_acc >= 0.88,
    }

    # Per-class accuracy
    from sklearn.metrics import precision_recall_fscore_support
    precs, recs, f1s, supps = precision_recall_fscore_support(
        test_targets, test_preds, labels=list(range(n_classes)), zero_division=0
    )
    for i, name in enumerate(class_names):
        report['per_class'][str(name)] = {
            'precision': round(float(precs[i]), 4),
            'recall': round(float(recs[i]), 4),
            'f1': round(float(f1s[i]), 4),
            'support': int(supps[i]),
        }

    report_path_key = 'dynamic_training_report_real' if real else 'dynamic_training_report'
    out_report_path = Path(report_path) if report_path is not None else PROJECT_ROOT / PATHS[report_path_key]
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_report_path), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {out_report_path}")

    # Plot confusion matrix
    _plot_dynamic_confusion(cm, list(class_names))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Test accuracy:    {test_acc:.4f}")
    target_names = "/".join(str(name) for name in class_names)
    print(f"  {target_names} target >=88%: {'PASS' if test_acc >= 0.88 else 'FAIL'}")
    print(f"{'=' * 60}")

    return report


def _plot_dynamic_confusion(cm, class_names):
    """Save dynamic classifier confusion matrix."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, square=True)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Dynamic Classifier — Confusion Matrix')
        fig.tight_layout()

        fig_dir = PROJECT_ROOT / 'data' / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(fig_dir / 'dynamic_confusion_matrix.png'), dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {fig_dir / 'dynamic_confusion_matrix.png'}")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")


# =============================================================================
# Model Loading Utility
# =============================================================================

def load_cnn_lstm_model(device=None):
    """Load a trained CNN-LSTM model from disk.

    Returns:
        (model, class_names, cfg) tuple.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = PROJECT_ROOT / PATHS['cnn_lstm_model']
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    model = CnnLstmClassifier(
        checkpoint['n_features'],
        checkpoint['n_classes'],
        checkpoint['cfg'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint['class_names'], checkpoint['cfg']


# =============================================================================
# CLI
# =============================================================================

def train_dynamic_louo(max_epochs=None, quick: bool = False,
                        mixup_alpha: float = 0.2) -> dict:
    """Train one CNN-LSTM per held-out user from data/processed/real/louo/<user>/.

    Writes models to data/models/louo/<user>/cnn_lstm.pt and reports to the
    same directory. Emits an aggregated summary under
    data/models/louo_dynamic_summary.json.
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

    per_user: dict[str, dict] = {}
    for udir in user_dirs:
        user_id = udir.name
        out_dir = models_root / user_id
        report = train_dynamic(
            max_epochs=max_epochs,
            quick=quick,
            real=True,
            data_dir=udir,
            model_path=out_dir / 'cnn_lstm.pt',
            report_path=out_dir / 'dynamic_training_report.json',
            mixup_alpha=mixup_alpha,
        )
        per_user[user_id] = {
            'test_accuracy': report.get('test_accuracy'),
            'best_val_acc': report.get('best_val_acc'),
        }

    mean_test = float(np.mean([r['test_accuracy'] for r in per_user.values()]))
    summary = {'per_user': per_user, 'mean_test_accuracy': mean_test}
    out_path = models_root / 'louo_dynamic_summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nLOUO dynamic summary saved: {out_path}")
    print(f"LOUO dynamic mean test accuracy: {mean_test:.4f}")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train dynamic CNN-LSTM classifier')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test (10 epochs)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--real', action='store_true',
                        help='Load real data splits (data/processed/real/) '
                             'and save v2 model')
    parser.add_argument('--louo', action='store_true',
                        help='Train one CNN-LSTM per held-out user under '
                             'data/processed/real/louo/<user>/. Implies --real.')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='Beta alpha for batch-level mixup (0 disables).')
    args = parser.parse_args()

    if args.louo:
        train_dynamic_louo(max_epochs=args.epochs, quick=args.quick,
                           mixup_alpha=args.mixup_alpha)
    else:
        train_dynamic(max_epochs=args.epochs, quick=args.quick, real=args.real,
                      mixup_alpha=args.mixup_alpha)
