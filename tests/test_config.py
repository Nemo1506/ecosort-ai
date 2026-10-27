"""Tests for config.py configuration, paths, and constants."""
import config


def test_classes_integrity():
    assert len(config.CLASS_NAMES) == 12
    assert config.NUM_CLASSES == 12
    expected = [
        "battery", "biological", "brown-glass", "cardboard", "clothes",
        "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
    ]
    assert config.CLASS_NAMES == expected


def test_split_ratios():
    assert len(config.SPLIT) == 3
    assert abs(sum(config.SPLIT) - 1.0) < 1e-6
    train_f, val_f, test_f = config.SPLIT
    assert train_f > 0.5
    assert val_f > 0.1
    assert test_f > 0.1


def test_image_extensions():
    assert ".jpg" in config.IMAGE_EXTENSIONS
    assert ".png" in config.IMAGE_EXTENSIONS
    assert ".jpeg" in config.IMAGE_EXTENSIONS


def test_image_size():
    assert config.IMG_SIZE == (224, 224)


def test_hyperparameters():
    assert config.LR_HEAD > 0
    assert config.LR_FINE > 0
    assert config.DROPOUT >= 0 and config.DROPOUT < 1
    assert config.LABEL_SMOOTHING >= 0 and config.LABEL_SMOOTHING < 0.5


def test_custom_split_ratios():
    custom = (0.80, 0.10, 0.10)
    assert abs(sum(custom) - 1.0) < 1e-4
    assert len(custom) == 3
