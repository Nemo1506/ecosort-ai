"""Step 3 - Train MobileNetV2 transfer learning model on EcoSort AI dataset.

Supports both Phase 1 (Classifier Head Training) and Phase 2 (Fine-Tuning
top convolutional feature layers) using PyTorch with differential learning rates,
Cosine Annealing LR scheduling, and Label Smoothing regularized loss.

Exports trained weights to models/ecosort_mobilenetv2.pt and
models/ecosort_mobilenetv2.onnx for high-performance universal inference.
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
    """Compute smoothed inverse frequency class weights to balance class gradients."""
    counts = []
    for cls in config.CLASS_NAMES:
        d = config.TRAIN_DIR / cls
        c = sum(1 for f in d.iterdir() if f.suffix.lower() in config.IMAGE_EXTENSIONS) if d.exists() else 1
        counts.append(max(c, 1))
    total = sum(counts)
    n = len(counts)
    # Smoothed inverse frequency (square-root dampened)
    raw_weights = [(total / (n * c)) ** 0.5 for c in counts]
    mean_w = sum(raw_weights) / len(raw_weights)
    norm_weights = [w / mean_w for w in raw_weights]
    import torch
    return torch.tensor(norm_weights, dtype=torch.float32, device=device)


def export_model_to_onnx(model, device):
    """Export PyTorch MobileNetV2 model to ONNX format with verification."""
    print("\nExporting model to ONNX format...")
    try:
        import torch
        import onnx
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
            dynamo=False,
        )
        # Validate ONNX model structure
        onnx_model = onnx.load(str(config.MODEL_ONNX_PATH))
        onnx.checker.check_model(onnx_model)
        size_mb = config.MODEL_ONNX_PATH.stat().st_size / (1024 * 1024)
        print(f"Exported verified ONNX model to {config.MODEL_ONNX_PATH} ({size_mb:.2f} MB)")
        return True
    except Exception as exc:
        print(f"ONNX export notice: {exc}")
        return False


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

    # Enhanced Data Augmentation Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
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
                dummy = torch.zeros(3, config.IMG_SIZE[0], config.IMG_SIZE[1])
                return dummy, target

    max_samples = 150 if quick else None
    train_ds = GarbageDataset(config.TRAIN_DIR, transform=train_transform, max_per_class=max_samples)
    val_ds = GarbageDataset(config.VAL_DIR, transform=eval_transform, max_per_class=50 if quick else None)

    print(f"Dataset loaded: {len(train_ds)} training images, {len(val_ds)} validation images")

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build MobileNetV2 with ImageNet pre-trained weights
    print("Loading MobileNetV2 with ImageNet pre-trained weights...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze base features for Phase 1
    for param in model.features.parameters():
        param.requires_grad = False

    # Classification Head with Dropout and Linear Layer
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=config.DROPOUT),
        nn.Linear(in_features, config.NUM_CLASSES)
    )
    model.to(device)

    # Label-smoothing regularized cross-entropy loss
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights_torch(device),
        label_smoothing=getattr(config, "LABEL_SMOOTHING", 0.08),
    )

    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=config.LR_HEAD,
        weight_decay=getattr(config, "WEIGHT_DECAY", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Persist class names registry
    with open(config.CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(config.CLASS_NAMES, f, indent=2)

    history = {
        "accuracy": [],
        "val_accuracy": [],
        "loss": [],
        "val_loss": [],
        "phase1_epochs": epochs,
        "phase2_epochs": 0,
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

        scheduler.step()

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
            f"Phase 1 Epoch {epoch}/{epochs} [{duration:.1f}s] - "
            f"Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.1%} - "
            f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.1%}"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), config.MODEL_PT_PATH)

    # === Phase 2: Fine-Tuning Top Convolutional Layers ===
    if fine_tune and fine_epochs > 0:
        print(f"\n=== Phase 2: Fine-Tuning Top Feature Layers ({fine_epochs} Epochs) ===")
        # Reload best checkpoint from Phase 1
        if config.MODEL_PT_PATH.exists():
            model.load_state_dict(torch.load(config.MODEL_PT_PATH, map_location=device))

        # Unfreeze top feature blocks (blocks 14 to 18 of MobileNetV2 features)
        for i, block in enumerate(model.features):
            if i >= 14:
                for param in block.parameters():
                    param.requires_grad = True
            else:
                for param in block.parameters():
                    param.requires_grad = False

        # Differential learning rates: conservative on backbone, higher on classifier
        fine_optimizer = optim.AdamW([
            {"params": [p for i, block in enumerate(model.features) if i >= 14 for p in block.parameters()], "lr": config.LR_FINE},
            {"params": model.classifier.parameters(), "lr": config.LR_HEAD * 0.1},
        ], weight_decay=getattr(config, "WEIGHT_DECAY", 1e-4))

        fine_scheduler = optim.lr_scheduler.CosineAnnealingLR(fine_optimizer, T_max=fine_epochs, eta_min=1e-6)

        for epoch in range(1, fine_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                fine_optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                fine_optimizer.step()

                train_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            fine_scheduler.step()

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
                f"Phase 2 Epoch {epoch}/{fine_epochs} [{duration:.1f}s] - "
                f"Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.1%} - "
                f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.1%}"
            )

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(model.state_dict(), config.MODEL_PT_PATH)

        history["phase2_epochs"] = fine_epochs

    # Guarantee model weights exist
    if not config.MODEL_PT_PATH.exists():
        torch.save(model.state_dict(), config.MODEL_PT_PATH)

    # Load best weights into model before ONNX export
    try:
        model.load_state_dict(torch.load(config.MODEL_PT_PATH, map_location=device))
    except Exception:
        pass

    # Export to ONNX
    export_model_to_onnx(model, device)

    # Save history json
    with open(config.HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Render publication-grade training progression curves
    epochs_range = range(1, len(history["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Accuracy Plot
    axes[0].plot(epochs_range, history["accuracy"], label="Train Acc", color="#10b981", lw=2.2)
    axes[0].plot(epochs_range, history["val_accuracy"], label="Val Acc", color="#3b82f6", lw=2.2)
    if history.get("phase2_epochs", 0) > 0:
        split_x = history["phase1_epochs"] + 0.5
        axes[0].axvline(x=split_x, color="#f59e0b", linestyle="--", alpha=0.8, label="Fine-Tune Start")
    axes[0].set_title("EcoSort Accuracy Progression", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Accuracy", fontsize=10)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower right")

    # Loss Plot
    axes[1].plot(epochs_range, history["loss"], label="Train Loss", color="#ef4444", lw=2.2)
    axes[1].plot(epochs_range, history["val_loss"], label="Val Loss", color="#f97316", lw=2.2)
    if history.get("phase2_epochs", 0) > 0:
        split_x = history["phase1_epochs"] + 0.5
        axes[1].axvline(x=split_x, color="#f59e0b", linestyle="--", alpha=0.8, label="Fine-Tune Start")
    axes[1].set_title("EcoSort Loss Convergence", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=10)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(config.REPORT_DIR / "training_curves.png", dpi=160)
    plt.close(fig)

    print(f"\nTraining completed successfully! Best Validation Accuracy: {best_val_acc:.1%}")
    print(f"Saved PyTorch weights: {config.MODEL_PT_PATH}")
    print(f"Saved ONNX model:    {config.MODEL_ONNX_PATH}")
    print("Next step: python src/evaluate.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3, help="Phase 1 head training epochs")
    parser.add_argument("--fine-tune", action="store_true", help="Run Phase 2 fine tuning")
    parser.add_argument("--fine-epochs", type=int, default=2, help="Phase 2 fine tuning epochs")
    parser.add_argument("--quick", action="store_true", help="Quick mode (balanced subset for fast demo)")
    args = parser.parse_args()

    train_pytorch(epochs=args.epochs, fine_tune=args.fine_tune, fine_epochs=args.fine_epochs, quick=args.quick)


if __name__ == "__main__":
    main()
