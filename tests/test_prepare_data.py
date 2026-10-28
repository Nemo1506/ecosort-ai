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
