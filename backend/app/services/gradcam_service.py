"""
GradCAM (Gradient-weighted Class Activation Mapping) service for XADE.

Generates heatmaps that visualize which facial regions the deepfake detection
model focused on, enabling grounded VLM explanations.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from typing import Optional

# third party
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


@dataclass
class RegionMetadata:
    """Metadata about activated regions in the GradCAM heatmap."""

    centroid_x: float  # Normalized [0, 1] — horizontal center of mass
    centroid_y: float  # Normalized [0, 1] — vertical center of mass
    top_region_area_pct: float  # % of image area occupied by high-activation region (>0.5)
    max_activation: float  # Peak activation value (always 1.0 after normalization)
    mean_activation: float  # Mean activation across the heatmap


class GradCAMGenerator:
    """
    Generates GradCAM heatmaps for a DeepfakeDetector model.

    Uses PyTorch forward/backward hooks on the last convolutional block of
    EfficientNet-B4 (`model.model.features[-1]`).

    Usage:
        generator = GradCAMGenerator(model)
        heatmap = generator.generate(image_tensor, target_class=0)
        overlay = generator.create_overlay(original_image, heatmap)
        metadata = generator.extract_region_metadata(heatmap)
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._target_layer = model.model.features[-1]
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module: nn.Module, input: tuple, output: torch.Tensor) -> None:
            self._activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            self._gradients = grad_output[0].detach()

        self._hooks.append(self._target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self._target_layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self) -> None:
        """Remove all registered hooks to prevent memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a GradCAM heatmap for the given image tensor.

        Args:
            image_tensor: Preprocessed image tensor of shape (1, C, H, W).
                          Must already be on the correct device.
            target_class: Class index to generate heatmap for.
                          If None, uses the predicted class (argmax).

        Returns:
            Normalized heatmap as a float32 numpy array of shape (H, W)
            with values in [0, 1].
        """
        was_training = self.model.training
        self.model.eval()

        self._register_hooks()

        try:
            # Ensure gradients are enabled for this forward pass
            image_tensor = image_tensor.requires_grad_(False)

            # Forward pass with gradient tracking on the model parameters
            outputs = self.model(image_tensor)

            if target_class is None:
                target_class = int(outputs.argmax(dim=1).item())

            # Zero existing gradients
            self.model.zero_grad()

            # Backward pass on the target class logit
            one_hot = torch.zeros_like(outputs)
            one_hot[0, target_class] = 1.0
            outputs.backward(gradient=one_hot)

            # Retrieve captured activations and gradients
            assert self._activations is not None, "Forward hook did not fire"
            assert self._gradients is not None, "Backward hook did not fire"

            activations = self._activations[0]  # (C, h, w)
            gradients = self._gradients[0]  # (C, h, w)

            # Global average pooling of gradients → importance weights
            weights = gradients.mean(dim=(1, 2))  # (C,)

            # Weighted combination of feature maps
            cam = torch.einsum("c,chw->hw", weights, activations)  # (h, w)

            # ReLU — keep only positive contributions
            cam = torch.clamp(cam, min=0)

            # Normalize to [0, 1]
            cam_np = cam.cpu().numpy()
            if cam_np.max() > 0:
                cam_np = cam_np / cam_np.max()

            # Resize to match the input spatial dimensions
            _, _, H, W = image_tensor.shape
            cam_resized = cv2.resize(cam_np, (W, H), interpolation=cv2.INTER_LINEAR)

            return cam_resized.astype(np.float32)

        finally:
            self._remove_hooks()
            self._activations = None
            self._gradients = None
            if was_training:
                self.model.train()

    def create_overlay(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """
        Blend a GradCAM heatmap onto the original image.

        Args:
            original_image: PIL RGB image (any size).
            heatmap: Normalized float32 heatmap of shape (H, W) in [0, 1].
            alpha: Heatmap blend weight (0 = no heatmap, 1 = only heatmap).
            colormap: OpenCV colormap constant (default: COLORMAP_JET).

        Returns:
            PIL RGB image with heatmap overlay.
        """
        # Resize heatmap to match original image
        orig_w, orig_h = original_image.size
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Convert heatmap to uint8 and apply colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend with original image
        original_np = np.array(original_image.convert("RGB")).astype(np.float32)
        overlay = (1 - alpha) * original_np + alpha * heatmap_rgb.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return Image.fromarray(overlay)

    def extract_region_metadata(self, heatmap: np.ndarray) -> RegionMetadata:
        """
        Extract spatial metadata from a GradCAM heatmap.

        Computes the center of mass of activations and the area fraction
        occupied by the high-activation region (threshold > 0.5).

        Args:
            heatmap: Normalized float32 heatmap of shape (H, W) in [0, 1].

        Returns:
            RegionMetadata with centroid coordinates and area statistics.
        """
        H, W = heatmap.shape

        # Center of mass (weighted by activation intensity)
        y_coords, x_coords = np.mgrid[0:H, 0:W]
        total_weight = heatmap.sum()

        if total_weight > 0:
            centroid_x = float((x_coords * heatmap).sum() / total_weight) / W
            centroid_y = float((y_coords * heatmap).sum() / total_weight) / H
        else:
            centroid_x = 0.5
            centroid_y = 0.5

        # High-activation region (>50% of max)
        high_activation_mask = heatmap > 0.5
        top_region_area_pct = float(high_activation_mask.sum()) / (H * W) * 100.0

        return RegionMetadata(
            centroid_x=round(centroid_x, 4),
            centroid_y=round(centroid_y, 4),
            top_region_area_pct=round(top_region_area_pct, 2),
            max_activation=float(heatmap.max()),
            mean_activation=round(float(heatmap.mean()), 4),
        )
