"""
Download a subset of the FFHQ-1024 dataset from HuggingFace.

Pulls N shards (1000 images each at 1024x1024, WebP) from
`gaunernst/ffhq-1024-wds` and extracts the individual images into
``backend/data/ffhq_sg3/raw/real/``.

Each shard is downloaded once into the HuggingFace cache and can be
re-extracted without re-downloading.  Images already present in the output
directory are skipped, so the script is safe to re-run.

Usage:
    python -m backend.scripts.download_ffhq --shards 4
    python -m backend.scripts.download_ffhq --shards 5 --output backend/data/ffhq_sg3/raw/real
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "gaunernst/ffhq-1024-wds"
REPO_TYPE = "dataset"
SHARD_STEP = 1000  # shards are named 00000.tar, 01000.tar, 02000.tar, ...

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "ffhq_sg3" / "raw" / "real"


def shard_name(index: int) -> str:
    """Return the tar filename for the Nth shard (0-indexed)."""
    return f"{index * SHARD_STEP:05d}.tar"


def extract_shard(tar_path: Path, output_dir: Path) -> int:
    """Extract all .webp members from a shard into output_dir.

    Returns the number of images newly written (existing files are skipped).
    """
    new_count = 0
    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".webp")]
        for member in tqdm(members, desc=f"  extract {tar_path.name}", leave=False):
            dest = output_dir / Path(member.name).name
            if dest.exists():
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            dest.write_bytes(extracted.read())
            new_count += 1
    return new_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards",
        type=int,
        default=4,
        help="Number of 1000-image shards to download (default 4 = ~4000 images)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of the first shard to download (default 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output}")
    print(f"Pulling {args.shards} shard(s) starting at index {args.start}")

    total_new = 0
    for i in range(args.start, args.start + args.shards):
        filename = shard_name(i)
        print(f"\n[{i - args.start + 1}/{args.shards}] {filename}")
        try:
            tar_path = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    filename=filename,
                )
            )
        except Exception as exc:
            print(f"  skip {filename}: {exc}")
            continue
        new = extract_shard(tar_path, args.output)
        total_new += new
        print(f"  +{new} new images")

    existing = len(list(args.output.glob("*.webp")))
    print(f"\nDone. {total_new} new image(s) written, {existing} total in {args.output}.")


if __name__ == "__main__":
    main()
