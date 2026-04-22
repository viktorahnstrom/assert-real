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
