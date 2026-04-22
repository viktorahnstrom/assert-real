"""
Build the real-face reference distribution for forensic z-scoring.

Samples N real-face images from the training split, runs BiSeNet face parsing
and the three forensic signals (sharpness, HF spectrum energy, ELA) on each,
then writes per-(region, metric) μ and σ to:

    backend/app/services/forensics/real_distribution.json

Usage
-----
From the backend/ directory, with the venv active:

    python scripts/build_reference_distribution.py

Options
-------
--dataset-dir   Root of the labelled dataset.
                Default: data/140k-real-fake/train/real
--n-samples     Number of images to sample (default: 200).
--seed          Random seed for reproducible sampling (default: 42).
--out           Output JSON path (default: app/services/forensics/real_distribution.json).

Dataset used for v1
-------------------
140k Real and Fake Faces (Kaggle), training split, real class.
~69 000 images available; 200 randomly sampled with seed=42.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Make sure the backend package is importable when run from backend/ ──────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.face_parser import FaceParser  # noqa: E402
from app.services.forensics.report import extract  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METRICS = ("laplacian_variance", "hf_energy", "ela_intensity")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/140k-real-fake/train/real",
        help="Directory containing real-face JPEG/PNG images.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of images to sample (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    parser.add_argument(
        "--out",
        default="app/services/forensics/real_distribution.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error("Dataset directory not found: %s", dataset_dir.resolve())
        sys.exit(1)

    all_images = sorted(
        p for p in dataset_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not all_images:
        logger.error("No JPEG/PNG images found in %s", dataset_dir.resolve())
        sys.exit(1)

    rng = random.Random(args.seed)
    sample = rng.sample(all_images, min(args.n_samples, len(all_images)))
    logger.info("Sampled %d / %d images (seed=%d)", len(sample), len(all_images), args.seed)

    face_parser = FaceParser()

    # Accumulate raw metric values: {region_id: {metric: [values]}}
    accum: dict[str, dict[str, list[float]]] = {}

    failed = 0
    for img_path in tqdm(sample, desc="Processing"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning("Could not open %s: %s", img_path.name, e)
            failed += 1
            continue

        try:
            parsing = face_parser.parse(image)
        except Exception as e:
            logger.warning("Face parsing failed for %s: %s", img_path.name, e)
            failed += 1
            continue

        if not parsing.masks_ui:
            failed += 1
            continue

        try:
            report = extract(image, parsing.masks_ui)
        except Exception as e:
            logger.warning("Forensics failed for %s: %s", img_path.name, e)
            failed += 1
            continue

        for region_id, region in report.regions.items():
            if region_id not in accum:
                accum[region_id] = {m: [] for m in METRICS}
            accum[region_id]["laplacian_variance"].append(region.laplacian_variance)
            accum[region_id]["hf_energy"].append(region.hf_energy)
            accum[region_id]["ela_intensity"].append(region.ela_intensity)

    if failed:
        logger.warning("%d image(s) skipped due to errors.", failed)

    # Build distribution dict
    distribution: dict[str, object] = {
        "meta": {
            "dataset": str(dataset_dir),
            "n_sampled": len(sample),
            "n_processed": len(sample) - failed,
            "seed": args.seed,
            "metrics": list(METRICS),
        },
        "regions": {},
    }

    for region_id, metrics in accum.items():
        distribution["regions"][region_id] = {}  # type: ignore[index]
        for metric, values in metrics.items():
            arr = np.array(values, dtype=np.float64)
            n = int(arr.size)
            mu = float(arr.mean()) if n > 0 else 0.0
            sigma = float(arr.std(ddof=1)) if n > 1 else 0.0
            distribution["regions"][region_id][metric] = {  # type: ignore[index]
                "mean": mu,
                "std": sigma,
                "n": n,
            }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(distribution, f, indent=2)

    logger.info("Wrote distribution to %s", out_path.resolve())
    logger.info(
        "Regions: %d  |  Metrics per region: %d",
        len(accum),
        len(METRICS),
    )


if __name__ == "__main__":
    main()
