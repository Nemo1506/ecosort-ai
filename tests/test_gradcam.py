"""Tests for src/gradcam.py Explainable AI attention heatmap generation."""
import numpy as np
from PIL import Image
from src.predict import load_model_and_classes
from src.gradcam import GradCAMExplainer


def test_gradcam_heatmap_generation():
    model, class_names = load_model_and_classes()
    torch_model = model.get_torch_model()
    assert torch_model is not None, "Failed to load PyTorch model for Grad-CAM"

    explainer = GradCAMExplainer(torch_model)
    assert len(explainer.hooks) == 2, "Expected 2 hooks for forward and backward"

    # Synthetic image
    synthetic_arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(synthetic_arr)

    superimposed, raw_heatmap, target_idx = explainer.generate_heatmap(img, alpha=0.4)

    assert isinstance(superimposed, Image.Image)
    assert superimposed.size == img.size
    assert isinstance(raw_heatmap, np.ndarray)
    assert raw_heatmap.shape == (7, 7)
    assert 0 <= target_idx < len(class_names)
    assert raw_heatmap.min() >= 0.0
    assert raw_heatmap.max() <= 1.0 + 1e-6

    # Test clean hook removal
    explainer.remove_hooks()
    assert len(explainer.hooks) == 0
