"""Tests for src/predict.py inference, model wrappers, and probability outputs."""
import numpy as np
from PIL import Image

import config
from src.predict import (
    ModelWrapper,
    load_disposal_guide,
    load_model_and_classes,
    predict_image,
)


def test_disposal_guide_coverage():
    guide = load_disposal_guide()
    assert isinstance(guide, dict)
    for cls in config.CLASS_NAMES:
        assert cls in guide, f"Missing disposal guide definition for: {cls}"
        item = guide[cls]
        assert "bin" in item
        assert "steps" in item
        assert len(item["steps"]) > 0


def test_model_loading_and_prediction():
    model, class_names = load_model_and_classes()
    assert isinstance(model, ModelWrapper)
    assert len(class_names) == 12

    # Create synthetic test image
    synthetic_arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(synthetic_arr)

    # Test single image prediction
    probs = model.predict_probs(img)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 12
    assert abs(float(np.sum(probs)) - 1.0) < 1e-4

    # Test predict_image helper
    result = predict_image(model, class_names, img, top_k=3)
    assert "label" in result
    assert result["label"] in class_names
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "top_k" in result
    assert len(result["top_k"]) == 3
    assert "probs" in result
    assert len(result["probs"]) == 12


def test_batch_prediction():
    model, class_names = load_model_and_classes()
    imgs = [
        Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        for _ in range(3)
    ]
    batch_probs = model.predict_batch(imgs)
    assert batch_probs.shape == (3, 12)
    for row in batch_probs:
        assert abs(float(np.sum(row)) - 1.0) < 1e-4
