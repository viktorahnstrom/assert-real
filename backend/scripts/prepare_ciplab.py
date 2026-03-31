import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(directory: Path) -> list[Path]:
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(directory.rglob(f"*{ext}"))
    return images


def prepare_ciplab(source_dir: str, dest_dir: str) -> None:
    source = Path(source_dir)
    dest = Path(dest_dir)

    real_src = source / "training_real"
    fake_src = source / "training_fake"

    if not real_src.exists():
        raise FileNotFoundError(f"Expected training_real folder at: {real_src}")
    if not fake_src.exists():
        raise FileNotFoundError(f"Expected training_fake folder at: {fake_src}")

    real_dest = dest / "real"
    fake_dest = dest / "fake"
    real_dest.mkdir(parents=True, exist_ok=True)
    fake_dest.mkdir(parents=True, exist_ok=True)

    # Copy real images
    real_images = collect_images(real_src)
    print(f"Found {len(real_images)} real images")
    for img in tqdm(real_images, desc="Copying real"):
        shutil.copy2(img, real_dest / img.name)

    # Copy all fake subdirs (easy/mid/hard) into single fake/ folder
    fake_images = collect_images(fake_src)
    print(f"Found {len(fake_images)} fake images (all difficulties)")

    seen_names: set[str] = set()
    for img in tqdm(fake_images, desc="Copying fake"):
        # Prefix with parent dir name to avoid filename collisions across easy/mid/hard
        unique_name = f"{img.parent.name}_{img.name}"
        shutil.copy2(img, fake_dest / unique_name)
        seen_names.add(unique_name)

    print("\n=== CIPLAB Prep Summary ===")
    print(f"Real images: {len(list(real_dest.iterdir()))}")
    print(f"Fake images: {len(list(fake_dest.iterdir()))}")
    print(f"Output dir:  {dest.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=True, help="Path to raw CIPLAB folder (real_and_fake_face/)"
    )
    parser.add_argument("--dest", default="backend/data/ciplab", help="Output directory")
    args = parser.parse_args()

    prepare_ciplab(args.source, args.dest)
