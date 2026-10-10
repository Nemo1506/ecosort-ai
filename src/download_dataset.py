"""Step 1 - Download or extract the 12-class garbage dataset into data/raw/<class>/.

Usage:
    python src/download_dataset.py                         # auto-detect zip or download via kagglehub
    python src/download_dataset.py --from-zip PATH         # extract from a .zip file
    python src/download_dataset.py --from-dir PATH         # use a folder you downloaded manually
"""
import argparse
import io
import shutil
import sys
import zipfile
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

KAGGLE_HANDLE = "mostafaabla/garbage-classification"


def find_class_root(path: Path) -> Path:
    """Return the folder that directly contains all 12 class folders."""
    path = Path(path)
    candidates = [path] + [p for p in path.rglob("*") if p.is_dir()]
    for p in candidates:
        subdirs = {d.name.lower() for d in p.iterdir() if d.is_dir()}
        if set(config.CLASS_NAMES).issubset(subdirs):
            return p
    raise FileNotFoundError(
        f"Could not find the 12 class folders {config.CLASS_NAMES} under {path}"
    )


def extract_dataset_zip(
    zip_source: Union[str, Path, BinaryIO, io.BytesIO],
    target_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, int]:
    """Extract a dataset zip file into target_dir (default: config.RAW_DIR).

    Handles any directory hierarchy inside the zip as long as it contains
    the class folders. Supports both file paths and in-memory file objects
    (e.g., Streamlit UploadedFile).
    """
    target = Path(target_dir or config.RAW_DIR)
    target.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {cls: 0 for cls in config.CLASS_NAMES}
    class_set = set(config.CLASS_NAMES)

    with zipfile.ZipFile(zip_source, "r") as zf:
        members = zf.namelist()

        # Check if the zip contains another zip (like archive(2).zip containing archive(1).zip)
        nested_zips = [m for m in members if m.lower().endswith(".zip") and not m.startswith("__MACOSX")]
        if nested_zips and not any(any(cls in m.split("/") for cls in class_set) for m in members):
            nested_name = nested_zips[0]
            print(f"Detected nested zip archive: {nested_name}, reading inner archive...")
            inner_bytes = io.BytesIO(zf.read(nested_name))
            return extract_dataset_zip(inner_bytes, target_dir=target, overwrite=overwrite)

        for member in members:
            # Skip macOS hidden metadata and directories
            if member.startswith("__MACOSX") or member.endswith("/"):
                continue

            parts = [p.strip() for p in member.split("/") if p.strip()]
            if len(parts) < 2:
                continue

            # Identify if any folder part matches one of our class names
            matched_cls = None
            filename = parts[-1]
            ext = Path(filename).suffix.lower()

            if ext not in config.IMAGE_EXTENSIONS:
                continue

            for i, part in enumerate(parts[:-1]):
                if part.lower() in class_set:
                    matched_cls = part.lower()
                    break

            if matched_cls is not None:
                cls_dir = target / matched_cls
                cls_dir.mkdir(parents=True, exist_ok=True)
                dest_file = cls_dir / filename

                if dest_file.exists() and not overwrite:
                    counts[matched_cls] += 1
                    continue

                with zf.open(member) as src_f, open(dest_file, "wb") as dst_f:
                    shutil.copyfileobj(src_f, dst_f)
                counts[matched_cls] += 1

    return counts


def copy_classes(root: Path) -> None:
    for cls in config.CLASS_NAMES:
        dst = config.RAW_DIR / cls
        if dst.exists() and any(dst.iterdir()):
            print(f"  {cls}: already present, skipping")
            continue
        shutil.copytree(root / cls, dst, dirs_exist_ok=True)
        print(f"  {cls}: copied")


def summarize() -> Dict[str, int]:
    print("\nImages per class in", config.RAW_DIR)
    total = 0
    counts = {}
    for cls in config.CLASS_NAMES:
        cls_path = config.RAW_DIR / cls
        if cls_path.exists():
            n = sum(
                1
                for f in cls_path.iterdir()
                if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS
            )
        else:
            n = 0
        counts[cls] = n
        total += n
        print(f"  {cls:<12} {n:>5}")
    print(f"  {'TOTAL':<12} {total:>5}")
    return counts


def detect_local_zip() -> Optional[Path]:
    """Search for downloaded dataset zip files in parent/Downloads or workspace."""
    search_dirs = [
        config.BASE_DIR,
        config.BASE_DIR.parent,
        Path.home() / "Downloads",
    ]
    candidate_names = [
        "archive(1).zip",
        "archive.zip",
        "archive(2).zip",
        "garbage-classification.zip",
        "garbage_classification.zip",
    ]

    for d in search_dirs:
        for name in candidate_names:
            p = d / name
            if p.is_file():
                return p
        # Check any zip matching archive*.zip
        for p in d.glob("archive*.zip"):
            if p.is_file():
                return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-zip", type=Path, default=None,
                        help="Path to dataset .zip file (e.g. archive(1).zip)")
    parser.add_argument("--from-dir", type=Path, default=None,
                        help="Folder containing the 12 class folders (skips download)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted files")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.from_zip is not None:
        zip_path = Path(args.from_zip)
        if not zip_path.is_file():
            print(f"Error: Zip file not found at {zip_path}")
            sys.exit(1)
        print(f"Extracting dataset from zip: {zip_path} ...")
        counts = extract_dataset_zip(zip_path, overwrite=args.force)
        print(f"Extracted {sum(counts.values())} images across {len(counts)} classes.")
        summarize()
        print("\nDone. Next: python src/prepare_data.py")
        return

    if args.from_dir is not None:
        source = args.from_dir
        root = find_class_root(source)
        print("Dataset root:", root)
        copy_classes(root)
        summarize()
        print("\nDone. Next: python src/prepare_data.py")
        return

    # Check if a zip already exists locally
    local_zip = detect_local_zip()
    if local_zip:
        print(f"Found local dataset zip file: {local_zip}")
        print("Extracting directly from local zip...")
        counts = extract_dataset_zip(local_zip, overwrite=args.force)
        print(f"Extracted {sum(counts.values())} images across {len(counts)} classes.")
        summarize()
        print("\nDone. Next: python src/prepare_data.py")
        return

    # Fall back to kagglehub
    try:
        import kagglehub

        print(f"Downloading {KAGGLE_HANDLE} from Kaggle ...")
        source = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
        root = find_class_root(source)
        print("Dataset root:", root)
        copy_classes(root)
        summarize()
        print("\nDone. Next: python src/prepare_data.py")
    except Exception as exc:
        print(f"Kaggle download failed: {exc}")
        print("Please provide a zip file using: python src/download_dataset.py --from-zip <path_to_zip>")
        sys.exit(1)


if __name__ == "__main__":
    main()

