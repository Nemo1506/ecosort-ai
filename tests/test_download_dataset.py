import io
import zipfile
from pathlib import Path
import config
from src.download_dataset import extract_dataset_zip


def test_extract_dataset_zip_in_memory(tmp_path):
    """Test extracting an in-memory zip archive with normalized folder names."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # Create a mock image in a class directory with underscores and uppercase
        zf.writestr("archive/Brown_Glass/test_image.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        zf.writestr("archive/plastic/item.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    zip_buffer.seek(0)
    target_dir = tmp_path / "raw"
    counts = extract_dataset_zip(zip_buffer, target_dir=target_dir, overwrite=True)

    assert isinstance(counts, dict)
    assert counts.get("brown-glass", 0) == 1
    assert counts.get("plastic", 0) == 1
    assert (target_dir / "brown-glass" / "test_image.jpg").exists()
    assert (target_dir / "plastic" / "item.png").exists()
