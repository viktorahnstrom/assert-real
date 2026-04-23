"""
Generate synthetic faces with StyleGAN3-T (FFHQ-U, 1024x1024).

Clones the NVlabs/stylegan3 repository and downloads the official pretrained
``stylegan3-t-ffhqu-1024x1024.pkl`` checkpoint on first run, then generates
N images at the requested truncation psi and saves them as WebP into
``backend/data/ffhq_sg3/raw/fake/``.

The NVlabs custom CUDA ops (bias_act, upfirdn2d, filtered_lrelu) need a C++
compiler and CUDA toolkit to build; instead of requiring those we
monkey-patch the `_init` function of each op to force the pure-PyTorch
reference implementation.  The generator still runs fully on GPU through
PyTorch's bundled CUDA runtime, just ~3-5x slower than the compiled path
— which is fine for a one-time batch generation.

Usage:
    python -m backend.scripts.generate_sg3 --count 1000 --truncation 0.7
    python -m backend.scripts.generate_sg3 --count 100 --truncation 1.0 --seed-offset 10000

Generations are deterministic per ``seed``; re-running with the same
``--seed-offset`` and ``--count`` will produce identical images (and skip
ones already on disk).
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SG3_ROOT = REPO_ROOT / "backend" / "data" / "stylegan3"
SG3_REPO_URL = "https://github.com/NVlabs/stylegan3.git"
SG3_CHECKPOINT_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/"
    "versions/1/files/stylegan3-t-ffhqu-1024x1024.pkl"
)
CHECKPOINT_PATH = SG3_ROOT / "stylegan3-t-ffhqu-1024x1024.pkl"

DEFAULT_OUTPUT = REPO_ROOT / "backend" / "data" / "ffhq_sg3" / "raw" / "fake"


def ensure_repo() -> None:
    """Clone NVlabs/stylegan3 if it's not already on disk."""
    if (SG3_ROOT / "torch_utils").exists():
        return
    SG3_ROOT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {SG3_REPO_URL} -> {SG3_ROOT}")
    subprocess.run(
        ["git", "clone", "--depth", "1", SG3_REPO_URL, str(SG3_ROOT)],
        check=True,
    )


def ensure_checkpoint() -> Path:
    """Download the pretrained StyleGAN3-T FFHQ checkpoint (~300 MB)."""
    if CHECKPOINT_PATH.exists():
        return CHECKPOINT_PATH
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint -> {CHECKPOINT_PATH}")
    with urllib.request.urlopen(SG3_CHECKPOINT_URL) as response:
        total = int(response.headers.get("Content-Length", 0))
        with (
            open(CHECKPOINT_PATH, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc="  weights") as bar,
        ):
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    return CHECKPOINT_PATH


def force_reference_ops() -> None:
    """Make StyleGAN3's custom CUDA ops fall back to pure-PyTorch reference code.

    Overriding each ``_init`` to return False skips plugin compilation (which
    would need nvcc + MSVC on Windows) and routes every op through its
    bundled ``*_ref`` implementation instead.
    """
    from torch_utils.ops import bias_act, filtered_lrelu, upfirdn2d

    bias_act._init = lambda: False
    upfirdn2d._init = lambda: False
    filtered_lrelu._init = lambda: False


def load_generator(device: torch.device) -> torch.nn.Module:
    """Load the StyleGAN3 generator from the pretrained pickle."""
    checkpoint = ensure_checkpoint()
    with open(checkpoint, "rb") as f:
        network = pickle.load(f)
    generator = network["G_ema"].to(device)
    generator.eval()
    return generator


def generate_image(
    generator: torch.nn.Module,
    seed: int,
    truncation: float,
    device: torch.device,
) -> Image.Image:
    """Generate one 1024x1024 RGB image from a fixed seed."""
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, generator.z_dim)).to(device)
    with torch.no_grad():
        img = generator(z, None, truncation_psi=truncation, noise_mode="const")
    img = (img.clamp(-1, 1) + 1) * (255.0 / 2.0)
    array = img[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    return Image.fromarray(array, "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Number of images to generate")
    parser.add_argument(
        "--truncation",
        type=float,
        default=0.7,
        help="Truncation psi. 0.7 produces near-mean faces (sharp, few artifacts); "
        "1.0 produces the raw distribution (more diverse, more artifacts).",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Starting seed. Use different offsets to generate disjoint sets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="WebP quality (default 95, near-lossless).",
    )
    args = parser.parse_args()

    ensure_repo()
    if str(SG3_ROOT) not in sys.path:
        sys.path.insert(0, str(SG3_ROOT))

    force_reference_ops()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: running on CPU — generation will be extremely slow.")

    generator = load_generator(device)
    print(f"Generator loaded: z_dim={generator.z_dim}, img_resolution={generator.img_resolution}")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output}")
    print(
        f"Generating {args.count} image(s) at psi={args.truncation}, "
        f"seeds {args.seed_offset}..{args.seed_offset + args.count - 1}"
    )

    new_count = 0
    skipped = 0
    psi_tag = f"psi{int(round(args.truncation * 100)):03d}"
    for i in tqdm(range(args.count), desc="generating"):
        seed = args.seed_offset + i
        dest = args.output / f"sg3_{psi_tag}_seed{seed:07d}.webp"
        if dest.exists():
            skipped += 1
            continue
        img = generate_image(generator, seed, args.truncation, device)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=args.quality, method=4)
        dest.write_bytes(buf.getvalue())
        new_count += 1

    existing = len(list(args.output.glob("*.webp")))
    print(
        f"\nDone. {new_count} generated, {skipped} skipped (already present), "
        f"{existing} total in {args.output}."
    )


if __name__ == "__main__":
    main()
