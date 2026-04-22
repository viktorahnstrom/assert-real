"""
Z-score helper for forensic metrics.

Loads the pre-built real-face reference distribution from
``real_distribution.json`` and exposes :func:`z_score` for converting a raw
metric value into standard deviations from the real-face mean.

A large positive z-score means the value is *above* the real-face average;
a large negative z-score means it is *below*.  For sharpness and HF energy
(where real > fake), a strongly negative score is suspicious.  For ELA
intensity the relationship depends on the image's compression history.
"""

from __future__ import annotations

import json
from pathlib import Path

_DIST_PATH = Path(__file__).parent / "real_distribution.json"

# Lazily loaded; None means not yet attempted.
_distribution: dict | None = None
_load_error: str | None = None


def _load() -> dict:
    global _distribution, _load_error
    if _distribution is not None:
        return _distribution
    if _load_error is not None:
        raise RuntimeError(_load_error)
    if not _DIST_PATH.exists():
        _load_error = (
            f"real_distribution.json not found at {_DIST_PATH}. "
            "Run scripts/build_reference_distribution.py first."
        )
        raise RuntimeError(_load_error)
    with open(_DIST_PATH, encoding="utf-8") as f:
        _distribution = json.load(f)
    return _distribution


def z_score(region: str, metric: str, value: float) -> float:
    """Return the z-score of *value* for the given region and metric.

    z = (value − μ) / σ

    Args:
        region: UI category id, e.g. ``"eyes_pupils"``.
        metric: One of ``"laplacian_variance"``, ``"hf_energy"``,
            ``"ela_intensity"``.
        value: The raw forensic metric value for this image.

    Returns:
        Z-score as a float.  Returns 0.0 when σ = 0 (all reference samples
        were identical) or when the region/metric is absent from the
        distribution (graceful degradation before the distribution is built).

    Raises:
        RuntimeError: If ``real_distribution.json`` does not exist.
    """
    dist = _load()
    regions = dist.get("regions", {})
    region_data = regions.get(region)
    if region_data is None:
        return 0.0
    metric_data = region_data.get(metric)
    if metric_data is None:
        return 0.0
    mu = metric_data.get("mean", 0.0)
    sigma = metric_data.get("std", 0.0)
    if sigma == 0.0:
        return 0.0
    return (value - mu) / sigma


def distribution_meta() -> dict:
    """Return the metadata block from the distribution file.

    Useful for logging which dataset and sample size were used.

    Raises:
        RuntimeError: If ``real_distribution.json`` does not exist.
    """
    return _load().get("meta", {})
