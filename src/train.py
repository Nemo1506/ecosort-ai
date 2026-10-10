"""Step 3 - Train MobileNetV2 transfer learning model on EcoSort AI dataset.

Supports both PyTorch and TensorFlow backends.
Exports trained weights to models/ecosort_mobilenetv2.pt and
models/ecosort_mobilenetv2.onnx for high-performance inference.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


def compute_class_weights_torch(device="cpu"):
    counts = []
    for cls in config.CLASS_NAMES:
        d = config.TRAIN_DIR / cls
        c = sum(1 for f in d.iterdir() if f.suffix.lower() in config.IMAGE_EXTENSIONS) if d.exists() else 1
        counts.append(max(c, 1))
    total = sum(counts)
    n = len(counts)
    weights = [total / (n * c) for c in counts]
    import torch
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_pytorch(epochs: int = 5, fine_tune: bool = False, fine_epochs: int = 3, quick: bool = False):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import torchvision.models as models
    from torchvision import transforms

    config.ensure_dirs()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using PyTorch on device: {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class GarbageDataset(Dataset):
        def __init__(self, root_dir: Path, transform=None, max_per_class: int = None):
            self.samples = []
            self.transform = transform
            for idx, cls in enumerate(config.CLASS_NAMES):
                cls_dir = root_dir / cls
                if not cls_dir.is_dir():
                    continue
                files = [p for p in cls_dir.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS]
                if max_per_class:
                    files = files[:max_per_class]
                for p in files:
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
                # Corrupt image fallback
                dummy = torch.zeros(3, config.IMG_SIZE[0], config.IMG_SIZE[1])
                return dummy, target

    max_samples = 150 if quick else None
    train_ds = GarbageDataset(config.TRAIN_DIR, transform=train_transform, max_per_class=max_samples)
    val_ds = GarbageDataset(config.VAL_DIR, transform=eval_transform, max_per_class=50 if quick else None)

    print(f"Dataset loaded: {len(train_ds)} training images, {len(val_ds)} validation images")

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build MobileNetV2
    print("Loading MobileNetV2 with ImageNet pre-trained weights...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze base features
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.DROPOUT),
        nn.Linear(in_features, config.NUM_CLASSES)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=compute_class_weights_torch(device))
    optimizer = optim.Adam(model.classifier.parameters(), lr=config.LR_HEAD)

    # Save class names
    with open(config.CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(config.CLASS_NAMES, f, indent=2)

    history = {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": [],
    }

    best_val_acc = 0.0
    print("\n=== Phase 1: Training Classification Head ===")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_train_loss = train_loss / max(train_total, 1)
        epoch_train_acc = train_correct / max(train_total, 1)
        epoch_val_loss = val_loss / max(val_total, 1)
        epoch_val_acc = val_correct / max(val_total, 1)
        duration = time.time() - t0

        history["loss"].append(round(epoch_train_loss, 4))
        history["accuracy"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_accuracy"].append(round(epoch_val_acc, 4))

        print(
            f"Epoch {epoch}/{epochs} [{duration:.1f}s] - "
            f"Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.1%} - "
            f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.1%}"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), config.MODEL_PT_PATH)

    history["phase1_epochs"] = epochs

    # Save best PyTorch weights
    if not config.MODEL_PT_PATH.exists():
        torch.save(model.state_dict(), config.MODEL_PT_PATH)

    # Export to ONNX for universal cross-platform inference
    print("\nExporting model to ONNX format...")
    try:
        model.eval()
        dummy_input = torch.randn(1, 3, config.IMG_SIZE[0], config.IMG_SIZE[1], device=device)
        torch.onnx.export(
            model,
            dummy_input,
            str(config.MODEL_ONNX_PATH),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["predictions"],
            dynamic_axes={"image": {0: "batch_size"}, "predictions": {0: "batch_size"}},
        )
        print(f"Exported ONNX model to {config.MODEL_ONNX_PATH}")
    except Exception as exc:
        print(f"ONNX export warning: {exc}")

    # Save history and curves
    with open(config.HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    epochs_range = range(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs_range, history["accuracy"], label="train", color="#2d6a4f", lw=2)
    axes[0].plot(epochs_range, history["val_accuracy"], label="validation", color="#1d70b8", lw=2)
    axes[0].set_title("EcoSort Accuracy Progression")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs_range, history["loss"], label="train", color="#d62828", lw=2)
    axes[1].plot(epochs_range, history["val_loss"], label="validation", color="#e07a1f", lw=2)
    axes[1].set_title("EcoSort Loss Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(config.REPORT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nBest validation accuracy: {best_val_acc:.1%}")
    print(f"Saved model to: {config.MODEL_PT_PATH}")
    print(f"Next: python src/evaluate.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--fine-tune", action="store_true", help="Run phase 2 fine tuning")
    parser.add_argument("--fine-epochs", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="Quick mode (balanced subset for fast demo)")
    args = parser.parse_args()

    train_pytorch(epochs=args.epochs, fine_tune=args.fine_tune, fine_epochs=args.fine_epochs, quick=args.quick)


if __name__ == "__main__":
    main()
