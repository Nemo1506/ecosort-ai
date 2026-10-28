"""Step 2 - Validate images, split into train/val/test, and build tf.data loaders.

Usage:
    python src/prepare_data.py            # split data/raw -> data/processed
    python src/prepare_data.py --force    # redo the split from scratch

Other modules import get_datasets() and compute_class_weights() from here.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random  # noqa: E402

from PIL import Image  # noqa: E402

import config  # noqa: E402


# ------------------------------------------------------------------ split
def _is_valid_image(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _list_valid_images(class_dir: Path) -> list:
    files = sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )
    valid = [p for p in files if _is_valid_image(p)]
    skipped = len(files) - len(valid)
    if skipped:
        print(f"  {class_dir.name}: skipped {skipped} corrupt file(s)")
    return valid


def _safe_rmtree(target_dir: Path, retries: int = 3, delay: float = 0.2) -> None:
    """Safely remove a directory tree on Windows, handling read-only flags and file locks."""
    if not target_dir.exists():
        return
    import os
    import stat
    import time

    def _remove_readonly(func, path, _):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    for attempt in range(retries):
        try:
            shutil.rmtree(target_dir, onerror=_remove_readonly)
            break
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                for root, dirs, files in os.walk(str(target_dir), topdown=False):
                    for name in files:
                        p = Path(root) / name
                        try:
                            os.chmod(p, stat.S_IWRITE)
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass
                    for name in dirs:
                        p = Path(root) / name
                        try:
                            p.rmdir()
                        except Exception:
                            pass


def split_dataset(force: bool = False, split_ratios: tuple = None) -> dict:
    """Stratified (per-class) split of data/raw into data/processed/{train,val,test}."""
    ratios = tuple(split_ratios) if split_ratios is not None else config.SPLIT
    assert abs(sum(ratios) - 1.0) < 1e-4, f"split_ratios must sum to 1.0 (got {sum(ratios)})"
    config.ensure_dirs()

    # Auto-discover raw directory if nested inside RAW_DIR (e.g. data/raw/garbage_classification/<class>)
    raw_source = config.RAW_DIR
    if raw_source.exists() and not any((raw_source / cls).is_dir() for cls in config.CLASS_NAMES):
        for candidate in raw_source.iterdir():
            if candidate.is_dir() and any((candidate / cls).is_dir() for cls in config.CLASS_NAMES):
                raw_source = candidate
                print(f"Notice: Auto-detected nested raw directory at {candidate}")
                break

    if config.TRAIN_DIR.exists() and any(config.TRAIN_DIR.iterdir()):
        has_train_images = any(
            f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS
            for f in config.TRAIN_DIR.rglob("*")
        )
        if has_train_images and not force:
            print("data/processed already exists. Use --force to redo the split.")
            return {"status": "skipped", "message": "already_exists"}
        _safe_rmtree(config.PROCESSED_DIR)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_frac, val_frac, test_frac = ratios
    summary = {}
    rnd = random.Random(config.SEED)

    # Collect valid images per class
    class_images = {}
    for cls in config.CLASS_NAMES:
        class_dir = raw_source / cls
        if class_dir.is_dir():
            class_images[cls] = _list_valid_images(class_dir)
        else:
            class_images[cls] = []

    total_valid = sum(len(imgs) for imgs in class_images.values())
    if total_valid == 0:
        raise FileNotFoundError(
            f"{config.RAW_DIR / 'battery'} not found. Run: python src/download_dataset.py"
        )

    # Warn about missing classes if partial dataset
    missing = [cls for cls, imgs in class_images.items() if len(imgs) == 0]
    if missing:
        print(f"Notice: {len(missing)} of 12 classes not found in raw data ({', '.join(missing)}). Empty partition folders initialized.")

    for cls in config.CLASS_NAMES:
        files = class_images[cls]
        shuffled = list(files)
        rnd.shuffle(shuffled)
        n = len(shuffled)

        if n == 0:
            train_files, val_files, test_files = [], [], []
        elif n < 3:
            train_files = shuffled
            val_files = shuffled[:1]
            test_files = shuffled[:1]
        else:
            n_train = max(1, int(round(n * train_frac)))
            n_val = max(1, int(round(n * val_frac)))
            if n - n_train - n_val < 1:
                if n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
            train_files = shuffled[:n_train]
            val_files = shuffled[n_train:n_train + n_val]
            test_files = shuffled[n_train + n_val:]

        for split_name, split_files in (
            ("train", train_files), ("val", val_files), ("test", test_files)
        ):
            out_dir = config.PROCESSED_DIR / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out_dir / f.name)

        summary[cls] = (len(train_files), len(val_files), len(test_files))

    print(f"\n{'class':<12} {'train':>6} {'val':>6} {'test':>6}")
    totals = [0, 0, 0]
    for cls, counts in summary.items():
        print(f"{cls:<12} {counts[0]:>6} {counts[1]:>6} {counts[2]:>6}")
        totals = [t + c for t, c in zip(totals, counts)]
    print(f"{'TOTAL':<12} {totals[0]:>6} {totals[1]:>6} {totals[2]:>6}")
    return {"train": totals[0], "val": totals[1], "test": totals[2], "by_class": summary}


# ---------------------------------------------------------------- loaders
def get_datasets(batch_size: int = None):
    """Return (train_loader, val_loader, test_loader) as PyTorch DataLoaders."""
    return get_torch_dataloaders(batch_size=batch_size)


def get_torch_dataloaders(batch_size: int = None, num_workers: int = 0):
    """Return (train_loader, val_loader, test_loader) as PyTorch DataLoader instances."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    batch_size = batch_size or config.BATCH_SIZE

    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class PyTorchGarbageDataset(Dataset):
        def __init__(self, root_dir: Path, transform=None):
            self.samples = []
            self.transform = transform
            for idx, cls in enumerate(config.CLASS_NAMES):
                cls_dir = root_dir / cls
                if not cls_dir.is_dir():
                    continue
                for p in cls_dir.iterdir():
                    if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS:
                        self.samples.append((p, idx))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, target = self.samples[idx]
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    return img, target
            except Exception:
                dummy = torch.zeros(3, config.IMG_SIZE[0], config.IMG_SIZE[1])
                return dummy, target

    train_ds = PyTorchGarbageDataset(config.TRAIN_DIR, transform=transform)
    val_ds = PyTorchGarbageDataset(config.VAL_DIR, transform=transform)
    test_ds = PyTorchGarbageDataset(config.TEST_DIR, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader



def compute_class_weights() -> dict:
    """Balanced class weights {class_index: weight} from the training folder."""
    counts = []
    for cls in config.CLASS_NAMES:
        d = config.TRAIN_DIR / cls
        c = sum(1 for f in d.iterdir() if f.suffix.lower() in config.IMAGE_EXTENSIONS) if d.exists() else 0
        counts.append(max(c, 1))
    total = sum(counts)
    n = len(counts)
    return {i: total / (n * c) for i, c in enumerate(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Redo the split")
    args = parser.parse_args()
    split_dataset(force=args.force)
    print("\nDone. Next: python src/train.py")


if __name__ == "__main__":
    main()
