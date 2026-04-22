"""
Unit tests for :mod:`app.services.forensics`.

Tests use synthetic images to verify directional properties of each signal
without requiring any model weights or external files.

Directional expectations (real vs fake proxies):
  - Sharp images (lots of high-frequency detail) → higher laplacian_variance
  - Sharp images → higher hf_energy
  - Freshly-rendered (never JPEG-compressed) images → higher ela_intensity
    compared to repeatedly-compressed images, because the generator-produced
    pixel values don't align with JPEG quantisation grids.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.services.forensics.ela import compute_ela, ela_intensity_per_region
from app.services.forensics.report import ForensicsReport, RegionForensics, extract
from app.services.forensics.sharpness import laplacian_variance
from app.services.forensics.spectrum import hf_energy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _checkerboard(size: int = 64, tile: int = 4) -> np.ndarray:
    """High-frequency uint8 image (alternating black/white tiles)."""
    arr = np.zeros((size, size), dtype=np.uint8)
    for r in range(size):
        for c in range(size):
            if ((r // tile) + (c // tile)) % 2 == 0:
                arr[r, c] = 255
    return arr


def _smooth(size: int = 64) -> np.ndarray:
    """Low-frequency uint8 image (gentle linear gradient)."""
    row = np.linspace(0, 255, size, dtype=np.float32)
    return np.tile(row, (size, 1)).astype(np.uint8)


def _full_mask(h: int, w: int) -> np.ndarray:
    """Boolean mask covering the entire image."""
    return np.ones((h, w), dtype=bool)


# ---------------------------------------------------------------------------
# sharpness.laplacian_variance
# ---------------------------------------------------------------------------


class TestLaplacianVariance:
    def test_sharp_greater_than_smooth(self):
        sharp_arr = _checkerboard()
        smooth_arr = _smooth()
        assert laplacian_variance(sharp_arr) > laplacian_variance(smooth_arr)

    def test_uniform_image_is_zero(self):
        uniform = np.full((32, 32), 128, dtype=np.uint8)
        assert laplacian_variance(uniform) == pytest.approx(0.0, abs=1e-6)

    def test_too_small_returns_zero(self):
        assert laplacian_variance(np.array([[1, 2], [3, 4]], dtype=np.uint8)) == 0.0

    def test_rgb_input(self):
        rgb = np.stack([_checkerboard()] * 3, axis=-1)
        val = laplacian_variance(rgb)
        assert val > 0.0

    def test_returns_float(self):
        result = laplacian_variance(_checkerboard())
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# spectrum.hf_energy
# ---------------------------------------------------------------------------


class TestHfEnergy:
    def test_sharp_has_more_hf_than_smooth(self):
        sharp_val = hf_energy(_checkerboard())
        smooth_val = hf_energy(_smooth())
        assert sharp_val > smooth_val

    def test_uniform_image_no_hf(self):
        uniform = np.full((32, 32), 128, dtype=np.uint8)
        val = hf_energy(uniform)
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_result_in_unit_interval(self):
        val = hf_energy(_checkerboard())
        assert 0.0 <= val <= 1.0

    def test_too_small_returns_zero(self):
        small = np.zeros((2, 2), dtype=np.uint8)
        assert hf_energy(small) == 0.0

    def test_rgb_input(self):
        rgb = np.stack([_checkerboard()] * 3, axis=-1)
        val = hf_energy(rgb)
        assert 0.0 < val <= 1.0


# ---------------------------------------------------------------------------
# ela.compute_ela / ela_intensity_per_region
# ---------------------------------------------------------------------------


class TestComputeEla:
    def test_output_size_matches_input(self):
        img = Image.fromarray(_checkerboard(), mode="L").convert("RGB")
        ela = compute_ela(img)
        assert ela.size == img.size

    def test_output_is_rgb(self):
        img = Image.fromarray(_checkerboard(), mode="L").convert("RGB")
        ela = compute_ela(img)
        assert ela.mode == "RGB"

    def test_never_compressed_has_higher_ela_than_recompressed(self):
        """
        A freshly-generated image re-saved once should show more ELA residual
        than the same image already saved at the target quality (which has
        adapted to the JPEG quantisation grid).
        """
        import io

        # Create a source image from a checkerboard (never JPEG-compressed)
        raw_img = Image.fromarray(_checkerboard(), mode="L").convert("RGB")

        # Pre-compress at quality=95 so its coefficients align with the grid
        buf = io.BytesIO()
        raw_img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        pre_compressed = Image.open(buf).convert("RGB")

        ela_raw = compute_ela(raw_img, quality=95, scale=1)
        ela_pre = compute_ela(pre_compressed, quality=95, scale=1)

        mean_raw = np.array(ela_raw).mean()
        mean_pre = np.array(ela_pre).mean()

        # Already-compressed image has lower ELA residual
        assert mean_pre < mean_raw


class TestElaIntensityPerRegion:
    def test_returns_key_for_every_mask(self):
        img = Image.fromarray(_checkerboard(), mode="L").convert("RGB")
        ela = compute_ela(img)
        masks = {
            "eyes_pupils": _full_mask(64, 64),
            "skin_texture": np.zeros((64, 64), dtype=bool),
        }
        result = ela_intensity_per_region(ela, masks)
        assert set(result.keys()) == {"eyes_pupils", "skin_texture"}

    def test_empty_mask_gives_zero(self):
        img = Image.fromarray(_smooth(), mode="L").convert("RGB")
        ela = compute_ela(img)
        empty_mask = np.zeros((64, 64), dtype=bool)
        result = ela_intensity_per_region(ela, {"region": empty_mask})
        assert result["region"] == 0.0

    def test_full_mask_gives_positive_intensity(self):
        img = Image.fromarray(_checkerboard(), mode="L").convert("RGB")
        ela = compute_ela(img, quality=95, scale=10)
        full_mask = _full_mask(64, 64)
        result = ela_intensity_per_region(ela, {"region": full_mask})
        assert result["region"] >= 0.0


# ---------------------------------------------------------------------------
# report.extract  (integration)
# ---------------------------------------------------------------------------


class TestExtract:
    def _make_image_and_masks(self, size: int = 64) -> tuple[Image.Image, dict[str, np.ndarray]]:
        arr = _checkerboard(size)
        img = Image.fromarray(arr, mode="L").convert("RGB")
        h, w = size, size
        # Two non-overlapping half-image masks
        top_mask = np.zeros((h, w), dtype=bool)
        top_mask[: h // 2, :] = True
        bottom_mask = np.zeros((h, w), dtype=bool)
        bottom_mask[h // 2 :, :] = True
        masks = {"eyes_pupils": top_mask, "skin_texture": bottom_mask}
        return img, masks

    def test_returns_forensics_report(self):
        img, masks = self._make_image_and_masks()
        report = extract(img, masks)
        assert isinstance(report, ForensicsReport)

    def test_all_regions_present(self):
        img, masks = self._make_image_and_masks()
        report = extract(img, masks)
        assert set(report.regions.keys()) == {"eyes_pupils", "skin_texture"}

    def test_region_has_correct_type(self):
        img, masks = self._make_image_and_masks()
        report = extract(img, masks)
        for region in report.regions.values():
            assert isinstance(region, RegionForensics)

    def test_all_metrics_non_negative(self):
        img, masks = self._make_image_and_masks()
        report = extract(img, masks)
        for region in report.regions.values():
            assert region.laplacian_variance >= 0.0
            assert region.hf_energy >= 0.0
            assert region.ela_intensity >= 0.0

    def test_image_size_recorded(self):
        img, masks = self._make_image_and_masks(64)
        report = extract(img, masks)
        assert report.image_size == (64, 64)

    def test_empty_mask_gives_zeros(self):
        img = Image.fromarray(_checkerboard(), mode="L").convert("RGB")
        empty_mask = np.zeros((64, 64), dtype=bool)
        report = extract(img, {"empty_region": empty_mask})
        r = report.regions["empty_region"]
        assert r.laplacian_variance == 0.0
        assert r.hf_energy == 0.0
        assert r.ela_intensity == 0.0

    def test_to_dict_is_json_safe(self):
        import json

        img, masks = self._make_image_and_masks()
        report = extract(img, masks)
        d = report.to_dict()
        # Must not raise
        serialised = json.dumps(d)
        assert "eyes_pupils" in serialised

    def test_sharp_image_higher_laplacian_than_smooth(self):
        """Directional test: sharp region → higher laplacian than smooth region."""
        size = 64
        sharp_img = Image.fromarray(_checkerboard(size), mode="L").convert("RGB")
        smooth_img = Image.fromarray(_smooth(size), mode="L").convert("RGB")
        full_mask = {"region": _full_mask(size, size)}

        sharp_report = extract(sharp_img, full_mask)
        smooth_report = extract(smooth_img, full_mask)

        assert (
            sharp_report.regions["region"].laplacian_variance
            > smooth_report.regions["region"].laplacian_variance
        )

    def test_sharp_image_higher_hf_energy_than_smooth(self):
        """Directional test: sharp region → higher HF energy than smooth region."""
        size = 64
        sharp_img = Image.fromarray(_checkerboard(size), mode="L").convert("RGB")
        smooth_img = Image.fromarray(_smooth(size), mode="L").convert("RGB")
        full_mask = {"region": _full_mask(size, size)}

        sharp_report = extract(sharp_img, full_mask)
        smooth_report = extract(smooth_img, full_mask)

        assert sharp_report.regions["region"].hf_energy > smooth_report.regions["region"].hf_energy
