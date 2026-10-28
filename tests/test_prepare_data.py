import shutil
from pathlib import Path
import pytest
import config
from src.prepare_data import _safe_rmtree, split_dataset


def test_split_ratios_assertion():
    """Verify that split ratios that do not sum to 1.0 raise an AssertionError."""
    with pytest.raises(AssertionError):
        split_dataset(split_ratios=(0.6, 0.2, 0.1))  # sums to 0.9, not 1.0


def test_safe_rmtree(tmp_path):
    """Verify that _safe_rmtree safely handles directory deletion."""
    test_dir = tmp_path / "test_deletion_dir"
    test_dir.mkdir()
    sample_file = test_dir / "sample.txt"
    sample_file.write_text("test content", encoding="utf-8")

    assert test_dir.exists()
    _safe_rmtree(test_dir)
    assert not test_dir.exists()

    # Calling on non-existent directory should not raise an error
    _safe_rmtree(test_dir)


def test_processed_split_structure():
    """Verify that data/processed contains train, val, and test partitions."""
    if config.PROCESSED_DIR.exists():
        for split in ("train", "val", "test"):
            split_dir = config.PROCESSED_DIR / split
            assert split_dir.exists(), f"Missing split directory {split_dir}"


def test_split_dataset_partial_classes(tmp_path, monkeypatch):
    """Verify that missing classes (like battery) do not crash split_dataset when other classes have images."""
    mock_raw = tmp_path / "raw"
    mock_proc = tmp_path / "processed"
    mock_raw.mkdir(parents=True)
    mock_proc.mkdir(parents=True)

    # Only create cardboard and plastic (omit battery and other 10 classes)
    cb_dir = mock_raw / "cardboard"
    cb_dir.mkdir()
    pl_dir = mock_raw / "plastic"
    pl_dir.mkdir()

    # Create dummy valid JPEG images (minimal JPEG header)
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
    for i in range(5):
        (cb_dir / f"cardboard_{i}.jpg").write_bytes(jpeg_bytes)
        (pl_dir / f"plastic_{i}.jpg").write_bytes(jpeg_bytes)

    monkeypatch.setattr(config, "RAW_DIR", mock_raw)
    monkeypatch.setattr(config, "PROCESSED_DIR", mock_proc)
    monkeypatch.setattr(config, "TRAIN_DIR", mock_proc / "train")
    monkeypatch.setattr(config, "VAL_DIR", mock_proc / "val")
    monkeypatch.setattr(config, "TEST_DIR", mock_proc / "test")

    # Should not raise FileNotFoundError for missing 'battery'
    res = split_dataset(force=True)
    assert res["train"] > 0
    assert (mock_proc / "train" / "cardboard").exists()
    assert (mock_proc / "train" / "battery").exists()  # Structure exists even if 0 images


def test_split_dataset_empty_raises(tmp_path, monkeypatch):
    """Verify that completely empty raw directory raises FileNotFoundError."""
    mock_raw = tmp_path / "empty_raw"
    mock_proc = tmp_path / "empty_proc"
    mock_raw.mkdir(parents=True)
    mock_proc.mkdir(parents=True)

    monkeypatch.setattr(config, "RAW_DIR", mock_raw)
    monkeypatch.setattr(config, "PROCESSED_DIR", mock_proc)
    monkeypatch.setattr(config, "TRAIN_DIR", mock_proc / "train")
    monkeypatch.setattr(config, "VAL_DIR", mock_proc / "val")
    monkeypatch.setattr(config, "TEST_DIR", mock_proc / "test")

    with pytest.raises(FileNotFoundError):
        split_dataset(force=True)
