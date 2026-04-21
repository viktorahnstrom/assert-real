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
        max_regions: int = 3,
        face_bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> list[dict]:
        """
        Extract up to max_regions distinct crops from the GradCAM heatmap.

        Uses connected-component analysis on the top-50% activation mask to
        find spatially separate high-activation blobs. Each blob becomes one
        evidence region with a tight, zoomed-in crop and its own label and
        activation score.

        Falls back to a single centroid-based crop if no distinct components
        are found above the size threshold.

        Args:
            original_image: The original PIL image.
            heatmap: Normalized heatmap array (H x W) with values in [0, 1].
            max_regions: Maximum number of regions to return (default 3).
            face_bbox: Optional (x1, y1, x2, y2) bounding box of the detected
                face. When provided, heatmap activations outside this box are
                zeroed before connected-component analysis so that background
                regions are never returned as evidence crops.

        Returns:
            List of up to max_regions dicts, each with keys:
              image (PIL), label (str), activation_score (float), bbox (tuple).
            Sorted by activation_score descending.
        """
        width, height = original_image.size
        heatmap_resized = cv2.resize(heatmap, (width, height))

        # Mask out background activations when a face bbox is available
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            mask = np.zeros_like(heatmap_resized)
            mask[fy1:fy2, fx1:fx2] = 1.0
            heatmap_resized = heatmap_resized * mask

        # Two-pass thresholding: use a tighter threshold (50%) to find distinct
        # blobs, then expand each blob's crop boundary down to 35% to capture
        # the full activation region. This avoids merging separate face areas
        # (e.g. forehead + nose) into one large centroid-averaged blob.
        threshold_blob = heatmap_resized.max() * 0.50
        binary = (heatmap_resized >= threshold_blob).astype(np.uint8)

        # Connected components — each distinct blob is a candidate region
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        # Minimum blob area: 1% of image area
        min_area = width * height * 0.010
        # Padding around each blob's tight bbox
        pad = max(20, int(min(width, height) * 0.06))

        # Minimum crop dimensions — ensure every crop shows enough face context.
        # When we have a face bbox use 28% of face dimensions; otherwise 18% of image.
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            face_w = fx2 - fx1
            face_h = fy2 - fy1
            min_crop_half_w = int(face_w * 0.28)
            min_crop_half_h = int(face_h * 0.28)
        else:
            min_crop_half_w = int(width * 0.18)
            min_crop_half_h = int(height * 0.18)

        components = []
        for i in range(1, num_labels):  # 0 is background
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            mask = labels == i
            activation_score = float(heatmap_resized[mask].mean())

            # Use the PEAK activation pixel (not the geometric centroid) for
            # labeling and crop centering. The peak is always where the model
            # actually focused most, so the label and crop will match the
            # hottest part of the heatmap even when the blob extends further.
            blob_activations = heatmap_resized * mask
            peak_idx = np.unravel_index(np.argmax(blob_activations), blob_activations.shape)
            cx_px, cy_px = int(peak_idx[1]), int(peak_idx[0])  # (col, row)

            # Tight bbox from the blob itself, then add padding
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])

            x1 = max(0, bx - pad)
            y1 = max(0, by - pad)
            x2 = min(width, bx + bw + pad)
            y2 = min(height, by + bh + pad)

            # Expand to minimum size centered on peak pixel so crops always
            # show the actual focal area rather than an averaged midpoint.
            if (x2 - x1) < min_crop_half_w * 2:
                x1 = max(0, cx_px - min_crop_half_w)
                x2 = min(width, cx_px + min_crop_half_w)
            if (y2 - y1) < min_crop_half_h * 2:
                y1 = max(0, cy_px - min_crop_half_h)
                y2 = min(height, cy_px + min_crop_half_h)

            components.append(
                {
                    # Peak-based coordinates for label and crop center
                    "cx_norm": cx_px / width,
                    "cy_norm": cy_px / height,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "activation_score": activation_score,
                }
            )

        # Sort by activation score, strongest first
        components.sort(key=lambda c: c["activation_score"], reverse=True)

        # Fallback: no blobs passed the size filter — use single peak centroid
        if not components:
            cy_px, cx_px = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
            cx_px, cy_px = int(cx_px), int(cy_px)
            activation_score = float(heatmap_resized[cy_px, cx_px])
            crop_half = max(min_crop_half_w, min_crop_half_h)
            x1 = max(0, cx_px - crop_half)
            y1 = max(0, cy_px - crop_half)
            x2 = min(width, cx_px + crop_half)
            y2 = min(height, cy_px + crop_half)
            components = [
                {
                    "cx_norm": cx_px / width,
                    "cy_norm": cy_px / height,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "activation_score": activation_score,
                }
            ]

        regions = []
        for comp in components[:max_regions]:
            x1, y1, x2, y2 = comp["x1"], comp["y1"], comp["x2"], comp["y2"]
            crop = original_image.crop((x1, y1, x2, y2))
            label = _label_region(comp["cx_norm"], comp["cy_norm"])
            regions.append(
                {
                    "image": crop,
                    "label": label,
                    "activation_score": round(comp["activation_score"], 4),
                    "bbox": (x1, y1, x2, y2),
                }
            )

        return regions


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
