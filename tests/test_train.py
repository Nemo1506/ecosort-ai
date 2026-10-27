"""Tests for train.py dataset validation and class weighting."""
from unittest.mock import patch

import pytest

import config
from src.train import compute_class_weights_torch


def test_compute_class_weights_torch():
    weights = compute_class_weights_torch("cpu")
    assert weights.shape[0] == len(config.CLASS_NAMES)
    assert (weights > 0).all()


def test_empty_dataset_error_handling(tmp_path):
    from src.train import train_pytorch
    empty_train = tmp_path / "train"
    empty_val = tmp_path / "val"
    empty_raw = tmp_path / "raw"
    empty_train.mkdir(parents=True)
    empty_val.mkdir(parents=True)
    empty_raw.mkdir(parents=True)

    with patch.object(config, "TRAIN_DIR", empty_train), \
         patch.object(config, "VAL_DIR", empty_val), \
         patch.object(config, "RAW_DIR", empty_raw), \
         patch("src.download_dataset.detect_local_zip", return_value=None):
        with pytest.raises((ValueError, FileNotFoundError)) as exc_info:
            train_pytorch(epochs=1, quick=True)
        msg = str(exc_info.value).lower()
        assert "0 image samples" in msg or "no training images" in msg
