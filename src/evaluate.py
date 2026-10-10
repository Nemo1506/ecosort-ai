"""Step 4 - Evaluate the saved model on the held-out test set.

Produces in reports/:
    metrics.json, classification_report.txt, confusion_matrix.png
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from src.predict import load_model_and_classes, predict_image


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Tuple[Dict, str, np.ndarray]:
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    acc = float(np.mean(y_true == y_pred))

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

    total_samples = len(y_true)
    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    weighted_p = float(sum(p * s for p, s in zip(precisions, supports)) / total_samples) if total_samples > 0 else 0.0
    weighted_r = float(sum(r * s for r, s in zip(recalls, supports)) / total_samples) if total_samples > 0 else 0.0
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_samples) if total_samples > 0 else 0.0

    report_lines.append("-" * 60)
    report_lines.append(f"{'Accuracy':<15} {'':>10} {'':>10} {acc:>10.4f} {total_samples:>10d}")
    report_lines.append(f"{'Macro Avg':<15} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total_samples:>10d}")
    report_lines.append(f"{'Weighted Avg':<15} {weighted_p:>10.4f} {weighted_r:>10.4f} {weighted_f1:>10.4f} {total_samples:>10d}")

    report_str = "\n".join(report_lines)

    metrics = {
        "test_samples": total_samples,
        "accuracy": acc,
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

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

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
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
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
        print("No test samples found in data/processed/test. Run: python src/prepare_data.py")
        sys.exit(1)

    print(f"Evaluating {len(test_samples)} test images...")
    y_true = []
    y_pred = []

    for idx, (path, true_label) in enumerate(test_samples):
        if idx > 0 and idx % 200 == 0:
            print(f"  Evaluated {idx}/{len(test_samples)} ({idx/len(test_samples):.1%})...")
        try:
            with Image.open(path) as img:
                res = predict_image(model, class_names, img, top_k=1)
                pred_label = class_names.index(res["label"])
                y_true.append(true_label)
                y_pred.append(pred_label)
        except Exception as e:
            continue

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics, report_str, cm = compute_metrics(y_true, y_pred, class_names)
    print("\n" + report_str)
    print(f"\nOverall Test Accuracy: {metrics['accuracy']:.2%}")
    print(f"Macro F1 Score:       {metrics['f1_macro']:.2%}")

    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(config.REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)

    plot_confusion_matrices(cm, class_names, config.REPORT_DIR / "confusion_matrix.png")
    print(f"Saved evaluation metrics and confusion matrix to {config.REPORT_DIR}")


if __name__ == "__main__":
    main()
