"""
Sharpness estimation via Laplacian variance.

A high Laplacian variance means the region contains strong edges and fine
texture — typical of genuine photographs.  Deepfake-generated regions tend
to be over-smoothed, yielding a lower variance.
"""

from __future__ import annotations

import numpy as np


def laplacian_variance(region_pixels: np.ndarray) -> float:
    """Compute the Laplacian variance of a masked region.

    Args:
        region_pixels: 2-D (H, W) or 3-D (H, W, C) uint8 array of pixel
            values that belong to the region (background pixels should be
            excluded before calling this function).

    Returns:
        Variance of the discrete Laplacian over all pixels in the region,
        or 0.0 if the region contains fewer than 4 pixels.
    """
    if region_pixels.size < 4:
        return 0.0

    # Convert to float grayscale if needed
    if region_pixels.ndim == 3:
        # Weighted luminance: ITU-R BT.601
        gray = (
            0.299 * region_pixels[..., 0].astype(np.float64)
            + 0.587 * region_pixels[..., 1].astype(np.float64)
            + 0.114 * region_pixels[..., 2].astype(np.float64)
        )
    else:
        gray = region_pixels.astype(np.float64)

    # Discrete Laplacian kernel applied via finite differences on interior pixels
    # lap[i,j] = gray[i-1,j] + gray[i+1,j] + gray[i,j-1] + gray[i,j+1] - 4*gray[i,j]
    lap = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]
    )

    if lap.size == 0:
        return 0.0

    return float(np.var(lap))
