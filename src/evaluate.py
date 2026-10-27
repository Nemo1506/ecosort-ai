"""Step 4 - Evaluate the saved model on the held-out test set.

Produces in reports/:
    metrics.json, classification_report.txt, confusion_matrix.png
Supports fast batched evaluation for rapid benchmarking.
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from src.predict import load_model_and_classes


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    top3_correct: int = 0,
) -> Tuple[Dict, str, np.ndarray]:
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total_samples = len(y_true)
    acc = float(np.mean(y_true == y_pred)) if total_samples > 0 else 0.0
    top3_acc = float(top3_correct / total_samples) if total_samples > 0 else 0.0

    precisions = []
    recalls = []
    f1s = []
    supports = []

    report_lines = [
        f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}",
        "-" * 60,
    ]

    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

        report_lines.append(f"{cls:<15} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>10d}")

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    weighted_p = float(sum(p * s for p, s in zip(precisions, supports)) / total_samples) if total_samples > 0 else 0.0
    weighted_r = float(sum(r * s for r, s in zip(recalls, supports)) / total_samples) if total_samples > 0 else 0.0
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_samples) if total_samples > 0 else 0.0

    report_lines.append("-" * 60)
    report_lines.append(f"{'Accuracy (Top-1)':<15} {'':>10} {'':>10} {acc:>10.4f} {total_samples:>10d}")
    report_lines.append(f"{'Accuracy (Top-3)':<15} {'':>10} {'':>10} {top3_acc:>10.4f} {total_samples:>10d}")
    report_lines.append(f"{'Macro Avg':<15} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total_samples:>10d}")
    report_lines.append(f"{'Weighted Avg':<15} {weighted_p:>10.4f} {weighted_r:>10.4f} {weighted_f1:>10.4f} {total_samples:>10d}")

    report_str = "\n".join(report_lines)

    metrics = {
        "test_samples": total_samples,
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_p,
        "recall_weighted": weighted_r,
        "f1_weighted": weighted_f1,
    }

    return metrics, report_str, cm


def plot_confusion_matrices(cm: np.ndarray, class_names: List[str], save_path: Path):
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    # Counts
    im0 = axes[0].imshow(cm, cmap="Blues", interpolation="nearest")
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Row-normalized
    im1 = axes[1].imshow(cm_norm, cmap="Greens", interpolation="nearest")
    axes[1].set_title("Confusion Matrix (Row-Normalized)", fontsize=14, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    n = len(class_names)
    for ax, data, is_norm in zip(axes, [cm, cm_norm], [False, True]):
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Class", fontsize=11, fontweight="bold")

        max_val = data.max()
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                txt = f"{val:.2f}" if is_norm else str(int(val))
                thresh = 0.5 if is_norm else (max_val / 2.0)
                color = "white" if val > thresh else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def evaluate_test_set(batch_size: int = 64) -> Tuple[Dict, str]:
    """Execute complete test set evaluation and save metrics reports and plots."""
    config.ensure_dirs()
    model, class_names = load_model_and_classes()
    print(f"Loaded model ({model.backend} backend) with {len(class_names)} classes.")

    test_samples = []
    for idx, cls in enumerate(class_names):
        cls_dir = config.TEST_DIR / cls
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.iterdir():
            if p.suffix.lower() in config.IMAGE_EXTENSIONS:
                test_samples.append((p, idx))

    if not test_samples:
        raise FileNotFoundError("No test samples found in data/processed/test. Run: python src/prepare_data.py")

    print(f"Evaluating {len(test_samples)} test images using batched inference (batch_size={batch_size})...")
    t0 = time.time()
    y_true = []
    y_pred = []
    top3_correct = 0

    # Process in batches
    for i in range(0, len(test_samples), batch_size):
        chunk = test_samples[i : i + batch_size]
        chunk_imgs = []
        chunk_targets = []
        for path, true_label in chunk:
            try:
                with Image.open(path) as img:
                    chunk_imgs.append(img.copy())
                    chunk_targets.append(true_label)
            except Exception:
                continue

        if not chunk_imgs:
            continue

        probs_batch = model.predict_batch(chunk_imgs)
        preds_top1 = np.argmax(probs_batch, axis=1)

        for target, pred_cls, probs in zip(chunk_targets, preds_top1, probs_batch):
            y_true.append(target)
            y_pred.append(pred_cls)
            top3_classes = np.argsort(probs)[::-1][:3]
            if target in top3_classes:
                top3_correct += 1

        if (i + batch_size) % 500 < batch_size or (i + batch_size) >= len(test_samples):
            evaluated = min(i + batch_size, len(test_samples))
            print(f"  Processed {evaluated}/{len(test_samples)} images ({evaluated/len(test_samples):.1%})...")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    duration = time.time() - t0

    metrics, report_str, cm = compute_metrics(y_true, y_pred, class_names, top3_correct=top3_correct)
    print("\n" + report_str)
    print(f"\nEvaluation Duration:    {duration:.1f}s ({duration/len(y_true)*1000:.1f}ms/sample)")
    print(f"Overall Test Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Top-3 Test Accuracy:    {metrics['top3_accuracy']:.2%}")
    print(f"Macro F1 Score:         {metrics['f1_macro']:.2%}")

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(config.REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

    plot_confusion_matrices(cm, class_names, config.REPORT_DIR / "confusion_matrix.png")
    print(f"Saved evaluation metrics and confusion matrix to {config.REPORT_DIR}")

    return metrics, report_str


def main():
    evaluate_test_set()


if __name__ == "__main__":
    main()
