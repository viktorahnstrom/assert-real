"""
Error Level Analysis (ELA).

ELA re-saves an image at a known JPEG quality and measures the per-pixel
difference between the original and the re-saved version.  Regions that were
previously compressed (authentic) show low residual error; regions that were
synthetically generated or heavily manipulated often show anomalously high or
anomalously uniform error patterns compared to the rest of the face.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def compute_ela(image: Image.Image, quality: int = 95, scale: int = 10) -> Image.Image:
    """Produce an ELA residual image.

    The input image is re-saved at *quality* and the absolute per-channel
    difference to the original is scaled by *scale* for visual clarity.

    Args:
        image: Source PIL image (any mode; converted to RGB internally).
        quality: JPEG re-save quality level (1–95).  Higher values produce
            smaller residuals, making genuine-vs-manipulated contrast easier
            to detect.  95 is the standard forensic setting.
        scale: Multiplicative amplification of the residual before returning.
            Does not affect the numeric intensity used by
            :func:`ela_intensity_per_region` (which works on raw residuals).

    Returns:
        PIL RGB image of the same size as *image* where pixel brightness
        corresponds to the re-compression residual (amplified by *scale*).
    """
    rgb = image.convert("RGB")

    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig_arr = np.array(rgb, dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)

    residual = np.abs(orig_arr - recomp_arr) * scale
    residual = np.clip(residual, 0, 255).astype(np.uint8)
    return Image.fromarray(residual, mode="RGB")


def create_ela_overlay(
    original: Image.Image,
    ela_map: Image.Image,
    alpha: float = 0.55,
) -> Image.Image:
    """Blend an ELA residual map faintly over the original image.

    Mirrors the GradCAM overlay pattern so the VLM receives a visually
    consistent set of evidence images. The ELA map is converted to a single
    luminance channel and recoloured with a hot colormap (dark = no residual,
    bright yellow/white = strong residual) so anomalies pop visually.

    Args:
        original: Source PIL image (any mode; converted to RGB internally).
        ela_map: ELA residual image from :func:`compute_ela`.
        alpha: Overlay blend weight (0 = original only, 1 = ELA only).

    Returns:
        PIL RGB image of the same size as *original* with the ELA residual
        rendered as a faint hot colormap on top.
    """
    rgb = original.convert("RGB")
    orig_arr = np.array(rgb, dtype=np.float32)

    # Use luminance of the residual as scalar intensity, then map to a
    # red-yellow gradient so high-residual areas are visually obvious.
    ela_resized = ela_map.convert("L").resize(rgb.size, Image.BILINEAR)
    intensity = np.array(ela_resized, dtype=np.float32) / 255.0
    intensity = np.clip(intensity, 0.0, 1.0)

    coloured = np.zeros_like(orig_arr)
    coloured[..., 0] = 255.0 * np.minimum(1.0, intensity * 2.0)  # R
    coloured[..., 1] = 255.0 * np.clip(intensity * 2.0 - 0.5, 0.0, 1.0)  # G
    coloured[..., 2] = 0.0  # B

    overlay = (1.0 - alpha) * orig_arr + alpha * coloured
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGB")


def ela_intensity_per_region(
    ela_map: Image.Image,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute mean ELA residual intensity for each facial region mask.

    Args:
        ela_map: ELA residual image (output of :func:`compute_ela` called
            with the *unscaled* residual, or any consistent scaling).
        masks: Mapping from UI category id to boolean mask of shape ``(H, W)``
            matching the original image dimensions.  Typically
            ``FaceParsingResult.masks_ui``.

    Returns:
        Mapping from category id to mean pixel intensity (float, ≥ 0) of the
        ELA residual inside that region.  Categories with no active pixels
        receive 0.0.
    """
    ela_arr = np.array(ela_map.convert("L"), dtype=np.float64)

    result: dict[str, float] = {}
    for category_id, mask in masks.items():
        if mask.shape != ela_arr.shape:
            result[category_id] = 0.0
            continue
        pixels = ela_arr[mask]
        result[category_id] = float(pixels.mean()) if pixels.size > 0 else 0.0

    return result
