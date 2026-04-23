"""
Unit tests for :mod:`app.services.region_ranker`.

The ranker fuses CAM activations with forensic z-scores, so the fixtures here
build synthetic heatmaps, masks, and :class:`ForensicsReport` objects directly
— no model weights or reference distribution on disk required.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import app.services.forensics.zscore as zscore_mod
from app.services.forensics.report import ForensicsReport, RegionForensics
from app.services.region_ranker import RankedRegion, rank

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_zscore_cache():
    """Clear the module-level cache between tests."""
    zscore_mod._distribution = None
    zscore_mod._load_error = None
    yield
    zscore_mod._distribution = None
    zscore_mod._load_error = None


@pytest.fixture()
def synthetic_distribution(tmp_path, monkeypatch):
    """Install a minimal reference distribution covering two regions."""
    dist = {
        "meta": {"dataset": "test", "n_sampled": 10, "n_processed": 10, "seed": 42},
        "regions": {
            "eyes_pupils": {
                "laplacian_variance": {"mean": 100.0, "std": 20.0, "n": 10},
                "hf_energy": {"mean": 0.30, "std": 0.05, "n": 10},
                "ela_intensity": {"mean": 5.0, "std": 1.0, "n": 10},
            },
            "skin_texture": {
                "laplacian_variance": {"mean": 80.0, "std": 10.0, "n": 10},
                "hf_energy": {"mean": 0.20, "std": 0.04, "n": 10},
                "ela_intensity": {"mean": 4.0, "std": 0.5, "n": 10},
            },
        },
    }
    path = tmp_path / "real_distribution.json"
    path.write_text(json.dumps(dist), encoding="utf-8")
    monkeypatch.setattr(zscore_mod, "_DIST_PATH", path)
    return path


@pytest.fixture()
def masks_two_regions():
    """Two non-overlapping half-image masks at 32×32."""
    h = w = 32
    left = np.zeros((h, w), dtype=bool)
    left[:, : w // 2] = True
    right = np.zeros((h, w), dtype=bool)
    right[:, w // 2 :] = True
    return {"eyes_pupils": left, "skin_texture": right}


def _flat_heatmap(value: float = 1.0, size: int = 32) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float32)


def _split_heatmap(left_val: float, right_val: float, size: int = 32) -> np.ndarray:
    arr = np.zeros((size, size), dtype=np.float32)
    arr[:, : size // 2] = left_val
    arr[:, size // 2 :] = right_val
    return arr


def _normal_report() -> ForensicsReport:
    """Report whose metrics sit exactly at the reference means (z=0)."""
    return ForensicsReport(
        regions={
            "eyes_pupils": RegionForensics(
                laplacian_variance=100.0, hf_energy=0.30, ela_intensity=5.0
            ),
            "skin_texture": RegionForensics(
                laplacian_variance=80.0, hf_energy=0.20, ela_intensity=4.0
            ),
        },
        image_size=(32, 32),
    )


def _anomalous_eye_report() -> ForensicsReport:
    """Report where ``eyes_pupils`` is far outside the reference distribution."""
    return ForensicsReport(
        regions={
            "eyes_pupils": RegionForensics(
                laplacian_variance=20.0,  # z = (20 - 100) / 20 = -4
                hf_energy=0.30,
                ela_intensity=5.0,
            ),
            "skin_texture": RegionForensics(
                laplacian_variance=80.0, hf_energy=0.20, ela_intensity=4.0
            ),
        },
        image_size=(32, 32),
    )


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestOutputStructure:
    def test_empty_masks_returns_empty_list(self, synthetic_distribution):
        result = rank(_flat_heatmap(), {}, forensics_report=None)
        assert result == []

    def test_returns_list_of_ranked_region(self, synthetic_distribution, masks_two_regions):
        result = rank(_flat_heatmap(), masks_two_regions, forensics_report=None)
        assert all(isinstance(r, RankedRegion) for r in result)

    def test_ranked_region_has_all_fields(self, synthetic_distribution, masks_two_regions):
        result = rank(_flat_heatmap(), masks_two_regions, forensics_report=_normal_report())
        r = result[0]
        assert isinstance(r.category_id, str)
        assert isinstance(r.cam_score, float)
        assert isinstance(r.forensic_score, float)
        assert isinstance(r.suspicion_score, float)
        assert set(r.z_scores.keys()) == {"laplacian_variance", "hf_energy", "ela_intensity"}

    def test_sorted_by_suspicion_desc(self, synthetic_distribution, masks_two_regions):
        result = rank(
            _split_heatmap(0.1, 0.9), masks_two_regions, forensics_report=_normal_report()
        )
        scores = [r.suspicion_score for r in result]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


class TestScoring:
    def test_cam_score_normalised_to_peak_one(self, synthetic_distribution, masks_two_regions):
        """Strongest region's cam_score should be 1.0 exactly."""
        result = rank(
            _split_heatmap(0.2, 0.8), masks_two_regions, forensics_report=_normal_report()
        )
        top = max(result, key=lambda r: r.cam_score)
        assert top.cam_score == pytest.approx(1.0)

    def test_normal_forensics_give_zero_score(self, synthetic_distribution, masks_two_regions):
        """Metrics equal to reference means produce z=0 → forensic_score=0."""
        result = rank(_flat_heatmap(), masks_two_regions, forensics_report=_normal_report())
        for r in result:
            assert r.forensic_score == pytest.approx(0.0)

    def test_anomalous_forensics_boost_that_region(self, synthetic_distribution, masks_two_regions):
        """The region with an outlier metric should score higher on forensics."""
        result = rank(_flat_heatmap(), masks_two_regions, forensics_report=_anomalous_eye_report())
        eyes = next(r for r in result if r.category_id == "eyes_pupils")
        skin = next(r for r in result if r.category_id == "skin_texture")
        assert eyes.forensic_score > skin.forensic_score

    def test_z_clamped_to_one(self, synthetic_distribution, masks_two_regions):
        """|z| = 4 → forensic_score = 1.0 exactly."""
        result = rank(_flat_heatmap(), masks_two_regions, forensics_report=_anomalous_eye_report())
        eyes = next(r for r in result if r.category_id == "eyes_pupils")
        assert eyes.forensic_score == pytest.approx(1.0)

    def test_alpha_half_averages(self, synthetic_distribution, masks_two_regions):
        """With α=0.5 the suspicion score equals the mean of the two components."""
        result = rank(
            _flat_heatmap(),
            masks_two_regions,
            forensics_report=_anomalous_eye_report(),
            alpha=0.5,
            threshold=0.0,
        )
        for r in result:
            expected = 0.5 * r.cam_score + 0.5 * r.forensic_score
            assert r.suspicion_score == pytest.approx(expected, abs=1e-4)

    def test_alpha_one_uses_cam_only(self, synthetic_distribution, masks_two_regions):
        result = rank(
            _split_heatmap(0.1, 0.9),
            masks_two_regions,
            forensics_report=_anomalous_eye_report(),
            alpha=1.0,
            threshold=0.0,
        )
        for r in result:
            assert r.suspicion_score == pytest.approx(r.cam_score, abs=1e-4)

    def test_alpha_zero_uses_forensics_only(self, synthetic_distribution, masks_two_regions):
        result = rank(
            _split_heatmap(0.1, 0.9),
            masks_two_regions,
            forensics_report=_anomalous_eye_report(),
            alpha=0.0,
            threshold=0.0,
        )
        for r in result:
            assert r.suspicion_score == pytest.approx(r.forensic_score, abs=1e-4)


# ---------------------------------------------------------------------------
# Filtering and fallback behaviour
# ---------------------------------------------------------------------------


class TestFilteringAndFallback:
    def test_threshold_filters_low_regions(self, synthetic_distribution, masks_two_regions):
        """Regions strictly below threshold are dropped when enough remain."""
        result = rank(
            _split_heatmap(0.05, 0.9),
            masks_two_regions,
            forensics_report=_normal_report(),
            threshold=0.5,
        )
        assert len(result) == 1
        assert result[0].category_id == "skin_texture"

    def test_at_least_one_returned_when_all_below_threshold(
        self, synthetic_distribution, masks_two_regions
    ):
        """If every region is below threshold, top one is still returned."""
        result = rank(
            _split_heatmap(0.05, 0.1),
            masks_two_regions,
            forensics_report=_normal_report(),
            threshold=0.99,
        )
        assert len(result) == 1

    def test_missing_forensics_falls_back_to_cam_only(
        self, synthetic_distribution, masks_two_regions
    ):
        """forensics_report=None forces α=1 regardless of env setting."""
        result = rank(
            _split_heatmap(0.1, 0.9),
            masks_two_regions,
            forensics_report=None,
            threshold=0.0,
        )
        for r in result:
            assert r.forensic_score == 0.0
            assert r.suspicion_score == pytest.approx(r.cam_score, abs=1e-4)

    def test_region_missing_from_report_gets_zero_forensic(
        self, synthetic_distribution, masks_two_regions
    ):
        """A region present in masks but absent from the report scores 0 forensically."""
        partial_report = ForensicsReport(
            regions={
                "eyes_pupils": RegionForensics(
                    laplacian_variance=100.0, hf_energy=0.30, ela_intensity=5.0
                )
            },
            image_size=(32, 32),
        )
        result = rank(
            _flat_heatmap(),
            masks_two_regions,
            forensics_report=partial_report,
            threshold=0.0,
        )
        skin = next(r for r in result if r.category_id == "skin_texture")
        assert skin.forensic_score == 0.0
        assert all(z == 0.0 for z in skin.z_scores.values())

    def test_empty_mask_produces_zero_cam_score(self, synthetic_distribution):
        """Zero-pixel masks don't divide by zero; they just score 0."""
        masks = {
            "eyes_pupils": np.ones((16, 16), dtype=bool),
            "skin_texture": np.zeros((16, 16), dtype=bool),
        }
        result = rank(
            _flat_heatmap(size=16),
            masks,
            forensics_report=_normal_report(),
            threshold=0.0,
        )
        skin = next(r for r in result if r.category_id == "skin_texture")
        assert skin.cam_score == 0.0


# ---------------------------------------------------------------------------
# Dimension handling
# ---------------------------------------------------------------------------


class TestHeatmapResizing:
    def test_heatmap_smaller_than_masks(self, synthetic_distribution, masks_two_regions):
        """A 16×16 heatmap is bilinearly upsampled to the 32×32 masks."""
        small = np.full((16, 16), 0.5, dtype=np.float32)
        result = rank(small, masks_two_regions, forensics_report=None, threshold=0.0)
        assert len(result) >= 1
        assert all(0.0 <= r.cam_score <= 1.0 for r in result)

    def test_heatmap_larger_than_masks(self, synthetic_distribution, masks_two_regions):
        """A 64×64 heatmap is bilinearly downsampled to the 32×32 masks."""
        big = np.full((64, 64), 0.5, dtype=np.float32)
        result = rank(big, masks_two_regions, forensics_report=None, threshold=0.0)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_alpha_out_of_range_raises(
        self, synthetic_distribution, masks_two_regions, monkeypatch
    ):
        monkeypatch.setenv("REGION_RANKER_ALPHA", "1.5")
        with pytest.raises(ValueError, match="REGION_RANKER_ALPHA"):
            rank(_flat_heatmap(), masks_two_regions, forensics_report=None)

    def test_min_greater_than_max_raises(
        self, synthetic_distribution, masks_two_regions, monkeypatch
    ):
        monkeypatch.setenv("REGION_RANKER_MIN_REGIONS", "5")
        monkeypatch.setenv("REGION_RANKER_MAX_REGIONS", "3")
        with pytest.raises(ValueError, match="MIN_REGIONS"):
            rank(_flat_heatmap(), masks_two_regions, forensics_report=None)
