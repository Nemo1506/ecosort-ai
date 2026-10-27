"""Step 5 - Load the saved model and classify single or batch images.

Supports ONNX Runtime, PyTorch, and TensorFlow backends.
Used by the Streamlit app and command line:
    python src/predict.py path/to/photo.jpg
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageOps

import config


class ModelWrapper:
    """Unified inference wrapper across ONNX Runtime, PyTorch, and Keras backends."""

    def __init__(self, model_obj, backend: str, class_names: list):
        self.model = model_obj
        self.backend = backend
        self.class_names = class_names
        self._torch_model = None

    def get_torch_model(self):
        """Retrieve or lazy-load the PyTorch model (required for Grad-CAM explainability)."""
        if self.backend == "torch":
            return self.model
        if self._torch_model is not None:
            return self._torch_model

        if config.MODEL_PT_PATH.exists():
            import torch
            import torch.nn as nn
            import torchvision.models as models

            m = models.mobilenet_v2(weights=None)
            state_dict = torch.load(config.MODEL_PT_PATH, map_location="cpu")
            num_classes = len(self.class_names)
            if "classifier.5.weight" in state_dict:
                hidden_dim = state_dict["classifier.1.weight"].shape[0]
                m.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(1280, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(hidden_dim, num_classes),
                )
            else:
                in_features = m.classifier[1].in_features
                m.classifier = nn.Sequential(
                    nn.Dropout(p=config.DROPOUT),
                    nn.Linear(in_features, num_classes),
                )
            m.load_state_dict(state_dict)
            m.eval()
            self._torch_model = m
            return self._torch_model
        return None

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = ImageOps.exif_transpose(image).convert("RGB")
        resample = getattr(Image, "Resampling", Image).BILINEAR
        resized = image.resize((config.IMG_SIZE[1], config.IMG_SIZE[0]), resample)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (arr - mean) / std
        return np.transpose(norm, (2, 0, 1))  # (3, 224, 224)

    def predict_probs(self, image: Image.Image) -> np.ndarray:
        """Compute class probabilities for a single PIL image."""
        if self.backend == "onnx":
            inp = np.expand_dims(self._preprocess_image(image), axis=0)  # (1, 3, 224, 224)
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: inp})[0][0]
            exp_logits = np.exp(outputs - np.max(outputs))
            return exp_logits / exp_logits.sum()

        elif self.backend == "torch":
            import torch
            inp = torch.tensor(np.expand_dims(self._preprocess_image(image), axis=0), dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(inp).numpy()[0]
            exp_logits = np.exp(logits - np.max(logits))
            return exp_logits / exp_logits.sum()

        elif self.backend == "keras":
            image = ImageOps.exif_transpose(image).convert("RGB")
            resized = image.resize((config.IMG_SIZE[1], config.IMG_SIZE[0]))
            arr = np.asarray(resized, dtype=np.float32)
            inp = np.expand_dims(arr, axis=0)
            probs = self.model.predict(inp, verbose=0)[0]
            return probs

        raise ValueError(f"Unknown backend: {self.backend}")

    def predict_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Compute class probabilities for a list of PIL images with batching."""
        if not images:
            return np.empty((0, len(self.class_names)), dtype=np.float32)

        if self.backend == "onnx":
            batch_arr = np.stack([self._preprocess_image(img) for img in images], axis=0)
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: batch_arr})[0]
            exp_logits = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)

        elif self.backend == "torch":
            import torch
            batch_arr = np.stack([self._preprocess_image(img) for img in images], axis=0)
            inp = torch.tensor(batch_arr, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(inp).numpy()
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)

        else:
            return np.array([self.predict_probs(img) for img in images])


def load_model_and_classes(force_backend: Optional[str] = None) -> Tuple[ModelWrapper, List[str]]:
    """Return (model_wrapper, class_names). Raises FileNotFoundError if not trained yet."""
    if config.CLASS_NAMES_PATH.exists():
        with open(config.CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = list(config.CLASS_NAMES)

    # 1. Check ONNX model (highest execution speed & cross-platform)
    if (force_backend is None or force_backend == "onnx") and config.MODEL_ONNX_PATH.exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(config.MODEL_ONNX_PATH), providers=["CPUExecutionProvider"])
            return ModelWrapper(session, "onnx", class_names), class_names
        except Exception as exc:
            print(f"Warning: Failed to load ONNX model ({exc}), checking other formats...")

    # 2. Check PyTorch model
    if (force_backend is None or force_backend == "torch") and config.MODEL_PT_PATH.exists():
        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models

            model = models.mobilenet_v2(weights=None)
            state_dict = torch.load(config.MODEL_PT_PATH, map_location="cpu")
            if "classifier.5.weight" in state_dict:
                hidden_dim = state_dict["classifier.1.weight"].shape[0]
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(1280, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(hidden_dim, len(class_names)),
                )
            else:
                in_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=config.DROPOUT),
                    nn.Linear(in_features, len(class_names))
                )
            model.load_state_dict(state_dict)
            model.eval()
            return ModelWrapper(model, "torch", class_names), class_names
        except Exception as exc:
            print(f"Warning: Failed to load PyTorch model ({exc}), checking other formats...")

    # 3. Check Keras model
    if (force_backend is None or force_backend == "keras") and config.MODEL_PATH.exists():
        try:
            from tensorflow import keras
            model = keras.models.load_model(config.MODEL_PATH)
            return ModelWrapper(model, "keras", class_names), class_names
        except Exception as exc:
            print(f"Warning: Failed to load Keras model: {exc}")

    raise FileNotFoundError(
        "No trained model found in models/. Please run: python src/train.py"
    )


def predict_image(model_wrapper: ModelWrapper, class_names: List[str], image: Image.Image, top_k: int = 3) -> Dict:
    """Return {'label', 'confidence', 'top_k': [(label, prob), ...], 'probs': {label: prob}}."""
    probs = model_wrapper.predict_probs(image)
    order = np.argsort(probs)[::-1]
    top = [(class_names[i], float(probs[i])) for i in order[:top_k]]
    return {
        "label": top[0][0],
        "confidence": float(top[0][1]),
        "top_k": top,
        "probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def load_disposal_guide() -> Dict:
    with open(config.DISPOSAL_GUIDE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)
    model, names = load_model_and_classes()
    result = predict_image(model, names, Image.open(sys.argv[1]))
    guide = load_disposal_guide().get(result["label"], {})
    print(f"Prediction : {result['label']}  ({result['confidence']:.1%})")
    for label, p in result["top_k"]:
        print(f"  {label:<12} {p:.1%}")
    if guide:
        print(f"Bin        : {guide.get('bin', 'N/A')}")
        for step in guide.get("steps", []):
            print(f"  - {step}")
