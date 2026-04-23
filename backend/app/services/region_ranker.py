"""
Region ranker combining LayerCAM attention with forensic anomaly scores.

The CAM heatmap shows where the detector looked; the forensic report shows
where the pixels are anomalous relative to real faces.  The ranker fuses both
signals so the region surfaced to the VLM reflects *model-faithful* and
*data-faithful* evidence at once.

For each of the six UI facial regions we compute:

``cam_score``
    Mean LayerCAM activation inside the region's BiSeNet mask, normalised so
    the most-attended region scores 1.0.

``forensic_score``
    Max absolute z-score across the three forensic metrics (Laplacian
    variance, HF energy, ELA intensity), clamped to 4 σ and divided by 4 so
    the result falls in [0, 1].

``suspicion_score``
    ``α · cam_score + (1 − α) · forensic_score``.  α is config-driven
    (environment variable ``REGION_RANKER_ALPHA``, default 0.5).

The ranker returns the top 1-3 regions above a configurable minimum
``REGION_RANKER_THRESHOLD`` (default 0.35).  When every region scores below
the threshold the top region is still returned, so downstream consumers
never see an empty evidence list.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.services.forensics import ForensicsReport, z_score
from app.services.forensics.report import RegionForensics

logger = logging.getLogger(__name__)

_METRICS: tuple[str, ...] = ("laplacian_variance", "hf_energy", "ela_intensity")
_Z_CLAMP: float = 4.0


@dataclass
class RankedRegion:
    """One region in the final suspicion ranking.

    Attributes:
        category_id: UI category identifier, e.g. ``"eyes_pupils"``.
        cam_score: Model-attention score in [0, 1].
        forensic_score: Data-anomaly score in [0, 1].
        suspicion_score: Fused score used for ranking; in [0, 1].
        z_scores: Raw z-score per forensic metric for downstream display.
    """

    category_id: str
    cam_score: float
    forensic_score: float
    suspicion_score: float
    z_scores: dict[str, float] = field(default_factory=dict)


def _load_config() -> tuple[float, float, int, int]:
    """Read ranker knobs from environment variables."""
    alpha = float(os.getenv("REGION_RANKER_ALPHA", "0.5"))
    threshold = float(os.getenv("REGION_RANKER_THRESHOLD", "0.35"))
    max_regions = int(os.getenv("REGION_RANKER_MAX_REGIONS", "3"))
    min_regions = int(os.getenv("REGION_RANKER_MIN_REGIONS", "1"))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"REGION_RANKER_ALPHA must be in [0, 1], got {alpha}")
    if min_regions > max_regions:
        raise ValueError(f"REGION_RANKER_MIN_REGIONS ({min_regions}) exceeds MAX ({max_regions})")
    return alpha, threshold, max_regions, min_regions


def _resize_heatmap(heatmap: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Bilinear-resize ``heatmap`` (H, W) to ``target_shape`` (H, W)."""
    if heatmap.shape == target_shape:
        return heatmap
    h, w = target_shape
    return cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)


def _cam_means(heatmap: np.ndarray, ui_masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean CAM activation inside each region mask (0 for empty masks)."""
    result: dict[str, float] = {}
    for cat_id, mask in ui_masks.items():
        if mask.sum() == 0:
            result[cat_id] = 0.0
            continue
        result[cat_id] = float(heatmap[mask].mean())
    return result


def _normalise_cam(cam_means: dict[str, float]) -> dict[str, float]:
    """Scale so the strongest region = 1.0; others proportional."""
    peak = max(cam_means.values(), default=0.0)
    if peak <= 0.0:
        return dict.fromkeys(cam_means, 0.0)
    return {k: v / peak for k, v in cam_means.items()}


def _forensic_score_for_region(
    category_id: str, region: RegionForensics
) -> tuple[float, dict[str, float]]:
    """Return ``(forensic_score, z_scores)`` for a single region."""
    z_by_metric: dict[str, float] = {}
    for metric in _METRICS:
        value = getattr(region, metric, 0.0)
        z_by_metric[metric] = z_score(category_id, metric, value)

    max_abs = max((abs(z) for z in z_by_metric.values()), default=0.0)
    clamped = min(max_abs, _Z_CLAMP) / _Z_CLAMP
    return clamped, z_by_metric


def rank(
    cam_heatmap: np.ndarray,
    ui_masks: dict[str, np.ndarray],
    forensics_report: ForensicsReport | None,
    *,
    alpha: float | None = None,
    threshold: float | None = None,
) -> list[RankedRegion]:
    """Rank facial regions by combined CAM + forensic suspicion.

    Args:
        cam_heatmap: Float32 heatmap of shape ``(H, W)`` with values in
            ``[0, 1]``.  Resized to match the mask dimensions when different.
        ui_masks: Mapping ``{ui_id: (H, W) bool}`` — usually
            :attr:`FaceParsingResult.masks_ui`.
        forensics_report: Optional :class:`ForensicsReport`.  When ``None``
            the ranker falls back to pure CAM scoring (α forced to 1.0).
        alpha: Override for the fusion weight.  Defaults to
            ``REGION_RANKER_ALPHA`` (0.5).
        threshold: Override for the minimum suspicion score.  Defaults to
            ``REGION_RANKER_THRESHOLD`` (0.35).

    Returns:
        Between ``REGION_RANKER_MIN_REGIONS`` (1) and
        ``REGION_RANKER_MAX_REGIONS`` (3) :class:`RankedRegion` objects
        sorted by descending ``suspicion_score``.  Empty list only when
        ``ui_masks`` is empty.
    """
    cfg_alpha, cfg_threshold, max_regions, min_regions = _load_config()
    effective_alpha = cfg_alpha if alpha is None else alpha
    effective_threshold = cfg_threshold if threshold is None else threshold

    if not ui_masks:
        return []

    if forensics_report is None:
        if effective_alpha < 1.0:
            logger.debug("No forensics report, forcing alpha=1.0")
        effective_alpha = 1.0

    # Masks live in original image coordinates; resize the heatmap to match.
    any_mask = next(iter(ui_masks.values()))
    heatmap = _resize_heatmap(cam_heatmap, any_mask.shape)

    cam_norm = _normalise_cam(_cam_means(heatmap, ui_masks))

    ranked: list[RankedRegion] = []
    for cat_id in ui_masks:
        cam_s = cam_norm[cat_id]

        if forensics_report is not None and cat_id in forensics_report.regions:
            for_s, z_by_metric = _forensic_score_for_region(
                cat_id, forensics_report.regions[cat_id]
            )
        else:
            for_s = 0.0
            z_by_metric = dict.fromkeys(_METRICS, 0.0)

        suspicion = effective_alpha * cam_s + (1.0 - effective_alpha) * for_s

        ranked.append(
            RankedRegion(
                category_id=cat_id,
                cam_score=round(cam_s, 4),
                forensic_score=round(for_s, 4),
                suspicion_score=round(suspicion, 4),
                z_scores={m: round(z, 4) for m, z in z_by_metric.items()},
            )
        )

    ranked.sort(key=lambda r: r.suspicion_score, reverse=True)

    above = [r for r in ranked if r.suspicion_score >= effective_threshold]
    if len(above) >= min_regions:
        return above[:max_regions]
    return ranked[:min_regions]
