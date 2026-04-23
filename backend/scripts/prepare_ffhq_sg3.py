"""
Split the FFHQ+SG3 raw set into train/val/test folders.

Reads from ``backend/data/ffhq_sg3/raw/{real,fake}/`` (populated by
``download_ffhq.py`` and ``generate_sg3.py``) and produces an 80/10/10
split into ``backend/data/ffhq_sg3/{train,val,test}/{real,fake}/``.

The split is deterministic (``random_state=42`` on a sorted filename list),
so re-running with the same raw contents always produces the same
assignments.  This matters for Issue 9's methodology: the 12 study images
are drawn from ``test/``, never from ``train/`` or ``val/``, so the
fine-tune never sees them.

By default, files are hardlinked rather than copied — saves ~8 GB on the
full run.  Pass ``--copy`` to fall back to a real copy (required when
destination is on a different volume than the source).

Usage:
    python -m backend.scripts.prepare_ffhq_sg3
    python -m backend.scripts.prepare_ffhq_sg3 --copy
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "backend" / "data" / "ffhq_sg3" / "raw"
DEFAULT_DEST = REPO_ROOT / "backend" / "data" / "ffhq_sg3"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")


def collect_images(directory: Path) -> list[Path]:
    """Return a sorted list of image paths under ``directory``."""
    images: list[Path] = []
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
    images.sort()
    return images


def place_file(src: Path, dest: Path, *, copy: bool) -> None:
    """Hardlink (default) or copy ``src`` to ``dest``, skipping if present."""
    if dest.exists():
        return
    if copy:
        shutil.copy2(src, dest)
        return
    try:
        os.link(src, dest)
    except OSError:
        # Different volume, or filesystem rejects hardlinks — fall back.
        shutil.copy2(src, dest)


def split_label(
    images: list[Path],
    label: str,
    dest_root: Path,
    *,
    copy: bool,
    seed: int,
) -> dict[str, int]:
    """Split one label's images into train/val/test under ``dest_root``.

    Returns a ``{split_name: count}`` mapping.
    """
    train, temp = train_test_split(images, test_size=0.2, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)

    counts: dict[str, int] = {}
    for split_name, split_images in (("train", train), ("val", val), ("test", test)):
        split_dir = dest_root / split_name / label
        split_dir.mkdir(parents=True, exist_ok=True)
        for img in tqdm(split_images, desc=f"  {split_name}/{label}", leave=False):
            place_file(img, split_dir / img.name, copy=copy)
        counts[split_name] = len(split_images)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Raw image root with real/ and fake/ subdirs (default {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination root for the split (default {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of hardlinking (required across volumes).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split (default 42).",
    )
    args = parser.parse_args()

    real_src = args.source / "real"
    fake_src = args.source / "fake"
    if not real_src.exists() or not fake_src.exists():
        raise FileNotFoundError(
            f"Expected {real_src} and {fake_src} — run download_ffhq.py and generate_sg3.py first."
        )

    real_images = collect_images(real_src)
    fake_images = collect_images(fake_src)
    print(f"Found {len(real_images)} real and {len(fake_images)} fake images")
    print(f"Source: {args.source}")
    print(f"Dest:   {args.dest}")
    print(f"Mode:   {'copy' if args.copy else 'hardlink'} (seed={args.seed})")

    real_counts = split_label(real_images, "real", args.dest, copy=args.copy, seed=args.seed)
    fake_counts = split_label(fake_images, "fake", args.dest, copy=args.copy, seed=args.seed)

    print("\n=== Split Summary ===")
    for split in SPLITS:
        real_on_disk = len(list((args.dest / split / "real").glob("*.*")))
        fake_on_disk = len(list((args.dest / split / "fake").glob("*.*")))
        print(
            f"{split:5s}: {real_counts[split]:>5d} real (on disk: {real_on_disk}), "
            f"{fake_counts[split]:>5d} fake (on disk: {fake_on_disk})"
        )


if __name__ == "__main__":
    main()
