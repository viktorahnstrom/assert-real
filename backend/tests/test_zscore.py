"""
Unit tests for :mod:`app.services.forensics.zscore`.

Uses a tmp-file fixture to inject a synthetic distribution without requiring
the real dataset to have been processed.
"""

from __future__ import annotations

import json

import pytest

import app.services.forensics.zscore as zscore_mod


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear the module-level cache before each test."""
    zscore_mod._distribution = None
    zscore_mod._load_error = None
    yield
    zscore_mod._distribution = None
    zscore_mod._load_error = None


@pytest.fixture()
def fake_dist_file(tmp_path, monkeypatch):
    """Write a minimal distribution JSON and point the module at it."""
    dist = {
        "meta": {"dataset": "test", "n_sampled": 10, "n_processed": 10, "seed": 42},
        "regions": {
            "skin_texture": {
                "laplacian_variance": {"mean": 100.0, "std": 20.0, "n": 10},
                "hf_energy": {"mean": 0.30, "std": 0.05, "n": 10},
                "ela_intensity": {"mean": 5.0, "std": 1.0, "n": 10},
            },
            "eyes_pupils": {
                "laplacian_variance": {"mean": 200.0, "std": 0.0, "n": 10},
                "hf_energy": {"mean": 0.40, "std": 0.10, "n": 10},
                "ela_intensity": {"mean": 8.0, "std": 2.0, "n": 10},
            },
        },
    }
    dist_file = tmp_path / "real_distribution.json"
    dist_file.write_text(json.dumps(dist), encoding="utf-8")
    monkeypatch.setattr(zscore_mod, "_DIST_PATH", dist_file)
    return dist_file


class TestZScore:
    def test_positive_z_when_above_mean(self, fake_dist_file):
        result = zscore_mod.z_score("skin_texture", "laplacian_variance", 140.0)
        assert result == pytest.approx(2.0)

    def test_negative_z_when_below_mean(self, fake_dist_file):
        result = zscore_mod.z_score("skin_texture", "laplacian_variance", 60.0)
        assert result == pytest.approx(-2.0)

    def test_zero_when_at_mean(self, fake_dist_file):
        result = zscore_mod.z_score("skin_texture", "laplacian_variance", 100.0)
        assert result == pytest.approx(0.0)

    def test_zero_std_returns_zero(self, fake_dist_file):
        # eyes_pupils laplacian_variance has std=0
        result = zscore_mod.z_score("eyes_pupils", "laplacian_variance", 9999.0)
        assert result == 0.0

    def test_unknown_region_returns_zero(self, fake_dist_file):
        result = zscore_mod.z_score("nonexistent_region", "laplacian_variance", 50.0)
        assert result == 0.0

    def test_unknown_metric_returns_zero(self, fake_dist_file):
        result = zscore_mod.z_score("skin_texture", "nonexistent_metric", 50.0)
        assert result == 0.0

    def test_hf_energy_z_score(self, fake_dist_file):
        result = zscore_mod.z_score("skin_texture", "hf_energy", 0.35)
        assert result == pytest.approx(1.0)

    def test_ela_intensity_z_score(self, fake_dist_file):
        result = zscore_mod.z_score("eyes_pupils", "ela_intensity", 4.0)
        assert result == pytest.approx(-2.0)

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(zscore_mod, "_DIST_PATH", tmp_path / "missing.json")
        with pytest.raises(RuntimeError, match="real_distribution.json not found"):
            zscore_mod.z_score("skin_texture", "laplacian_variance", 100.0)

    def test_distribution_meta_returns_dict(self, fake_dist_file):
        meta = zscore_mod.distribution_meta()
        assert meta["n_sampled"] == 10
        assert meta["dataset"] == "test"

    def test_module_import_exposes_z_score(self):
        from app.services.forensics import z_score

        assert callable(z_score)
