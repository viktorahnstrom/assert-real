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
        regions = generator.extract_evidence_regions(original_image, heatmap)
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
            image_tensor = image_tensor.requires_grad_(False)

            outputs = self.model(image_tensor)

            if target_class is None:
                target_class = int(outputs.argmax(dim=1).item())

            self.model.zero_grad()

            one_hot = torch.zeros_like(outputs)
            one_hot[0, target_class] = 1.0
            outputs.backward(gradient=one_hot)

            assert self._activations is not None, "Forward hook did not fire"
            assert self._gradients is not None, "Backward hook did not fire"

            activations = self._activations[0]  # (C, h, w)
            gradients = self._gradients[0]  # (C, h, w)

            weights = gradients.mean(dim=(1, 2))  # (C,)
            cam = torch.einsum("c,chw->hw", weights, activations)  # (h, w)
            cam = torch.clamp(cam, min=0)

            cam_np = cam.cpu().numpy()
            if cam_np.max() > 0:
                cam_np = cam_np / cam_np.max()

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
        orig_w, orig_h = original_image.size
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        original_np = np.array(original_image.convert("RGB")).astype(np.float32)
        overlay = (1 - alpha) * original_np + alpha * heatmap_rgb.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return Image.fromarray(overlay)

    def extract_region_metadata(self, heatmap: np.ndarray) -> RegionMetadata:
        """
        Extract spatial metadata from a GradCAM heatmap.

        Args:
            heatmap: Normalized float32 heatmap of shape (H, W) in [0, 1].

        Returns:
            RegionMetadata with centroid coordinates and area statistics.
        """
        H, W = heatmap.shape

        y_coords, x_coords = np.mgrid[0:H, 0:W]
        total_weight = heatmap.sum()

        if total_weight > 0:
            centroid_x = float((x_coords * heatmap).sum() / total_weight) / W
            centroid_y = float((y_coords * heatmap).sum() / total_weight) / H
        else:
            centroid_x = 0.5
            centroid_y = 0.5

        high_activation_mask = heatmap > 0.5
        top_region_area_pct = float(high_activation_mask.sum()) / (H * W) * 100.0

        return RegionMetadata(
            centroid_x=round(centroid_x, 4),
            centroid_y=round(centroid_y, 4),
            top_region_area_pct=round(top_region_area_pct, 2),
            max_activation=float(heatmap.max()),
            mean_activation=round(float(heatmap.mean()), 4),
        )

    def extract_evidence_regions(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        top_n: int = 3,
        min_region_size: float = 0.01,
    ) -> list[dict]:
        """
        Extract the top-N highest activation regions from the heatmap as cropped images.
        Always returns at least one region using the peak activation point as a fallback.

        Args:
            original_image: The original PIL image.
            heatmap: Normalized heatmap array (H x W) with values in [0, 1].
            top_n: Number of regions to extract.
            min_region_size: Minimum region size as fraction of image area.

        Returns:
            List of dicts with keys: image (PIL), label (str), activation_score (float).
        """
        width, height = original_image.size
        heatmap_resized = cv2.resize(heatmap, (width, height))

        # Threshold at 60% of max activation to find hot regions
        threshold = heatmap_resized.max() * 0.4
        binary = (heatmap_resized >= threshold).astype(np.uint8) * 255

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        min_area = int(width * height * min_region_size)
        regions = []

        for i in range(1, num_labels):  # skip background (0)
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue

            pad = int(min(width, height) * 0.05)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(width, x + w + pad)
            y2 = min(height, y + h + pad)

            crop = original_image.crop((x1, y1, x2, y2))
            mean_activation = float(heatmap_resized[y : y + h, x : x + w].mean())

            regions.append(
                {
                    "image": crop,
                    "activation_score": mean_activation,
                    "bbox": (x1, y1, x2, y2),
                }
            )

        # Fallback: if no regions passed the filter, use the peak activation point
        if not regions:
            peak_y, peak_x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
            crop_size = int(min(width, height) * 0.35)
            x1 = max(0, peak_x - crop_size // 2)
            y1 = max(0, peak_y - crop_size // 2)
            x2 = min(width, x1 + crop_size)
            y2 = min(height, y1 + crop_size)

            regions.append(
                {
                    "image": original_image.crop((x1, y1, x2, y2)),
                    "activation_score": float(heatmap_resized[peak_y, peak_x]),
                    "bbox": (x1, y1, x2, y2),
                }
            )

        # Sort by activation score, take top N
        regions.sort(key=lambda r: r["activation_score"], reverse=True)
        top_regions = regions[:top_n]

        # Label each region by its position on the face
        for region in top_regions:
            x1, y1, x2, y2 = region["bbox"]
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            region["label"] = _label_region(cx, cy)

        return top_regions


def _label_region(cx: float, cy: float) -> str:
    """Map normalized centroid coordinates to a facial region label."""
    if cy < 0.25:
        return "Forehead and hairline region"
    elif cy < 0.45:
        if cx < 0.4:
            return "Left eye region"
        elif cx > 0.6:
            return "Right eye region"
        else:
            return "Eye and nose bridge region"
    elif cy < 0.65:
        if cx < 0.35:
            return "Left cheek region"
        elif cx > 0.65:
            return "Right cheek region"
        else:
            return "Nose and mid-face region"
    else:
        if cx < 0.4:
            return "Left jaw region"
        elif cx > 0.6:
            return "Right jaw region"
        else:
            return "Chin and jawline region"
