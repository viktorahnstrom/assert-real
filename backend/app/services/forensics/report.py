"""
Forensic report aggregator.

Runs all three forensic signals (sharpness, HF spectrum energy, ELA) over
every facial region mask and returns a structured report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app.services.forensics.ela import compute_ela, ela_intensity_per_region
from app.services.forensics.sharpness import laplacian_variance
from app.services.forensics.spectrum import hf_energy


@dataclass
class RegionForensics:
    """Per-region forensic metrics.

    Attributes:
        laplacian_variance: Sharpness score — higher means sharper / more
            textured.  Authentic regions typically score higher.
        hf_energy: Fraction of spectral energy in the high-frequency band.
            Authentic regions typically retain more HF energy.
        ela_intensity: Mean ELA residual intensity inside the region.
            Synthetically generated regions often show anomalously high or
            uniform ELA intensity.
    """

    laplacian_variance: float
    hf_energy: float
    ela_intensity: float


@dataclass
class ForensicsReport:
    """Forensic feature report for a single image.

    Attributes:
        regions: Mapping from UI category id to per-region metrics.
        image_size: (width, height) of the analysed image.
    """

    regions: dict[str, RegionForensics] = field(default_factory=dict)
    image_size: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "regions": {
                cat_id: {
                    "laplacian_variance": r.laplacian_variance,
                    "hf_energy": r.hf_energy,
                    "ela_intensity": r.ela_intensity,
                }
                for cat_id, r in self.regions.items()
            },
        }


def _extract_region_pixels(image_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the subset of image pixels that fall inside *mask*.

    Returns an (N, C) or (N,) array of pixel values, or an empty array when
    the mask has no active pixels.
    """
    if mask.sum() == 0:
        return np.empty((0,), dtype=image_arr.dtype)
    if image_arr.ndim == 3:
        return image_arr[mask]  # shape (N, C)
    return image_arr[mask]  # shape (N,)


def extract(image: Image.Image, ui_masks: dict[str, np.ndarray]) -> ForensicsReport:
    """Run all forensic signals over every region and return a ForensicsReport.

    Args:
        image: Full-face PIL image (any mode; converted to RGB internally).
        ui_masks: Mapping from UI category id to boolean ``np.ndarray`` mask
            of shape ``(H, W)``, e.g. ``FaceParsingResult.masks_ui``.

    Returns:
        A :class:`ForensicsReport` with one :class:`RegionForensics` entry per
        key in *ui_masks*.
    """
    rgb = image.convert("RGB")
    image_arr = np.array(rgb, dtype=np.uint8)  # (H, W, 3)

    # ELA is computed once over the full image, then sampled per region
    ela_map = compute_ela(rgb, quality=95, scale=1)
    ela_per_region = ela_intensity_per_region(ela_map, ui_masks)

    regions: dict[str, RegionForensics] = {}

    for cat_id, mask in ui_masks.items():
        region_pixels = _extract_region_pixels(image_arr, mask)

        if region_pixels.size == 0:
            regions[cat_id] = RegionForensics(
                laplacian_variance=0.0,
                hf_energy=0.0,
                ela_intensity=ela_per_region.get(cat_id, 0.0),
            )
            continue

        # Sharpness: needs a 2-D spatial arrangement — rebuild a tight bounding
        # box crop masked to the region so spatial gradients are meaningful.
        rows, cols = np.where(mask)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        crop = image_arr[r0:r1, c0:c1]  # (crop_H, crop_W, 3)
        crop_mask = mask[r0:r1, c0:c1]  # (crop_H, crop_W) bool

        # Zero-out background pixels inside the bounding box
        masked_crop = crop.copy()
        masked_crop[~crop_mask] = 0

        sharp = laplacian_variance(masked_crop)
        hf = hf_energy(masked_crop)

        regions[cat_id] = RegionForensics(
            laplacian_variance=sharp,
            hf_energy=hf,
            ela_intensity=ela_per_region.get(cat_id, 0.0),
        )

    return ForensicsReport(regions=regions, image_size=rgb.size)
