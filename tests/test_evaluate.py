"""Tests for src/evaluate.py metrics and evaluation calculations."""
import numpy as np
import config
from src.evaluate import compute_metrics


def test_compute_metrics_accuracy_and_f1():
    class_names = list(config.CLASS_NAMES)
    n = len(class_names)

    # Perfect prediction test
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    metrics, report_str, cm = compute_metrics(y_true, y_pred, class_names, top3_correct=12)

    assert metrics["test_samples"] == 12
    assert abs(metrics["accuracy"] - 1.0) < 1e-6
    assert abs(metrics["top3_accuracy"] - 1.0) < 1e-6
    assert abs(metrics["f1_macro"] - 1.0) < 1e-6
    assert cm.shape == (n, n)
    assert np.trace(cm) == 12


def test_compute_metrics_imperfect():
    class_names = list(config.CLASS_NAMES)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])  # 3 correct, 1 wrong

    metrics, report_str, cm = compute_metrics(y_true, y_pred, class_names, top3_correct=4)

    assert metrics["accuracy"] == 0.75
    assert metrics["top3_accuracy"] == 1.0
    assert 0.0 < metrics["f1_macro"] < 1.0
