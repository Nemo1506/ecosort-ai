"""Grad-CAM (Gradient-Weighted Class Activation Mapping) for EcoSort AI.

Provides Explainable AI (XAI) attention heatmaps for MobileNetV2,
highlighting the spatial regions of an image that contributed most
to the classification decision.
"""
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.cm as cm

import config


class GradCAMExplainer:
    """Computes and overlays Grad-CAM heatmaps for PyTorch MobileNetV2."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()

        self.transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _register_hooks(self):
        """Hook into the final convolutional layer of MobileNetV2."""
        # In MobileNetV2, model.features[-1] is the last 1280-dim Conv2dNormActivation block
        target_layer = self.model.features[-1]

        def forward_hook(module, inp, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def generate_heatmap(
        self,
        image: Image.Image,
        target_class_idx: Optional[int] = None,
        colormap_name: str = "jet",
        alpha: float = 0.45,
    ) -> Tuple[Image.Image, np.ndarray, int]:
        """Generate superimposed Grad-CAM visualization.

        Returns:
            (superimposed_pil_image, raw_heatmap_2d_array, target_class_idx)
        """
        orig_img = ImageOps.exif_transpose(image).convert("RGB")
        tensor_img = self.transform(orig_img).unsqueeze(0)
        tensor_img.requires_grad = True

        self.model.zero_grad()
        logits = self.model(tensor_img)

        if target_class_idx is None:
            target_class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, target_class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients/activations.")

        # Global average pooling of gradients over spatial dimensions (H, W)
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])  # (C,)
        activations = self.activations.clone()

        # Weight activations by gradients
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_grads[i]

        # ReLU on linear combination of weighted feature maps
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        # Normalize heatmap to [0, 1]
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / (max_val + 1e-8)

        heatmap_np = heatmap.detach().cpu().numpy()

        # Upscale heatmap to original image resolution
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap_np)).resize(
            orig_img.size, resample=Image.Resampling.BILINEAR
        )
        heatmap_norm = np.array(heatmap_img, dtype=np.float32) / 255.0

        # Apply colormap
        cmap = getattr(cm, colormap_name, cm.jet)
        colored_heatmap = cmap(heatmap_norm)[:, :, :3]  # RGB floats in [0, 1]

        # Blend with original image
        orig_arr = np.array(orig_img, dtype=np.float32) / 255.0
        blended = (1.0 - alpha) * orig_arr + alpha * colored_heatmap
        blended = np.uint8(255 * np.clip(blended, 0.0, 1.0))

        superimposed_img = Image.fromarray(blended)
        return superimposed_img, heatmap_np, target_class_idx
