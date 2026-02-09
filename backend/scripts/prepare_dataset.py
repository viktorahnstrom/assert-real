import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def organize_dataset(source_dir: str, dest_dir: str):
    """Split dataset into train/val/test with 80/10/10 split"""

    source = Path(source_dir)
    dest = Path(dest_dir)

    # Find all images
    real_images = list(source.glob("real/*.jpg")) + list(source.glob("real/*.png"))
    fake_images = list(source.glob("fake/*.jpg")) + list(source.glob("fake/*.png"))

    print(f"Found {len(real_images)} real and {len(fake_images)} fake images")

    for images, label in [(real_images, "real"), (fake_images, "fake")]:
        # First split: 80% train, 20%temp
        train, temp = train_test_split(images, test_size=0.2, random_state=42)
        # Second split: 10 %val, 10% test
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        # Copy files
        for split_name, split_images in [("train", train), ("val", val), ("test", test)]:
            dest_path = dest / split_name / label
            dest_path.mkdir(parents=True, exist_ok=True)

            print(f"Copying {len(split_images)} to {split_name}/{label}...")
            for img_path in tqdm(split_images):
                shutil.copy2(img_path, dest_path / img_path.name)

    # Print Summary
    print("\n=== Dataset Summary ===")
    for split in ["train", "val", "test"]:
        real_count = len(list((dest / split / "real").glob("*.*")))
        fake_count = len(list((dest / split / "fake").glob("*.*")))
        print(f"{split}: {real_count} real, {fake_count} fake")


if __name__ == "__main__":
    organize_dataset(source_dir="data/raw", dest_dir="data")
