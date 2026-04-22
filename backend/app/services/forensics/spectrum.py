"""
High-frequency energy estimation via 2-D FFT.

Real photographs preserve high-frequency texture (fine detail, noise grain).
Deepfake generators often suppress high-frequency content due to upsampling
and smoothing operations, resulting in lower HF energy relative to the total
spectral energy.
"""

from __future__ import annotations

import numpy as np


def hf_energy(region_pixels: np.ndarray, k_pct: float = 0.25) -> float:
    """Fraction of spectral energy in the outer high-frequency band.

    The 2-D DFT magnitude spectrum is computed for the region.  The spectrum
    is divided into a low-frequency centre circle (radius = (1 - k_pct) of
    the Nyquist limit) and a high-frequency outer ring.  The function returns
    the ratio  HF_energy / total_energy, so the result is always in [0, 1].

    Args:
        region_pixels: 2-D (H, W) or 3-D (H, W, C) uint8 array containing
            only the pixels that belong to the region.
        k_pct: Fraction of Nyquist frequency above which components count as
            high-frequency.  Default 0.25 keeps the outer 25 % of the radius.

    Returns:
        HF energy fraction in [0, 1], or 0.0 if the region is too small to
        compute a meaningful spectrum (fewer than 16 pixels).
    """
    if region_pixels.size < 16:
        return 0.0

    # Grayscale conversion
    if region_pixels.ndim == 3:
        gray = (
            0.299 * region_pixels[..., 0].astype(np.float64)
            + 0.587 * region_pixels[..., 1].astype(np.float64)
            + 0.114 * region_pixels[..., 2].astype(np.float64)
        )
    else:
        gray = region_pixels.astype(np.float64)

    h, w = gray.shape
    spectrum = np.abs(np.fft.fft2(gray)) ** 2
    spectrum = np.fft.fftshift(spectrum)

    # Build a radius map (normalised so the corner = 1.0)
    cy, cx = h / 2.0, w / 2.0
    y_idx, x_idx = np.ogrid[:h, :w]
    r = np.sqrt(((y_idx - cy) / cy) ** 2 + ((x_idx - cx) / cx) ** 2)

    hf_mask = r >= (1.0 - k_pct)
    total_energy = spectrum.sum()
    if total_energy == 0.0:
        return 0.0

    return float(spectrum[hf_mask].sum() / total_energy)
