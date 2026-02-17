"""
Utility to download and load model from Hugging Face
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


HF_REPO_ID = "viktorahnstrom/xade-deepfake-detector"
MODEL_FILENAME = "best_model.pt"


def download_model_from_hf(force_download: bool = False) -> Path:
    """
    Download model from Hugging Face Hub if not already present locally.

    Args:
        force_download: If True, re-download even if file exists locally

    Returns:
        Path to the downloaded model file
    """

    local_model_path = Path("checkpoints/best_model.pt")

    # If model exists locally and we're not forcing download, use it
    if local_model_path.exists() and not force_download:
        print(f"✓ Model found locally at {local_model_path}")
        return local_model_path

    # Download from Hugging Face
    print(f"📥 Downloading model from Hugging Face: {HF_REPO_ID}")
    print("   This may take a minute (~260MB)...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir="checkpoints/.cache",  # Cache in checkpoints folder
        )

        # Copy to expected location
        local_model_path.parent.mkdir(exist_ok=True)

        # If downloaded to cache, create symlink or copy
        import shutil

        shutil.copy2(downloaded_path, local_model_path)

        print(f"✓ Model downloaded successfully to {local_model_path}")
        return local_model_path

    except Exception as e:
        print(f"❌ Error downloading model from Hugging Face: {e}")
        raise


def load_model_checkpoint(model_path: Path = None) -> dict:
    """
    Load model checkpoint from file.

    Args:
        model_path: Path to model file (if None, uses default location)

    Returns:
        Model checkpoint dictionary
    """

    if model_path is None:
        model_path = Path("checkpoints/best_model.pt")

    if not model_path.exists():
        print(f"⚠️  Model not found at {model_path}")
        print("   Attempting to download from Hugging Face...")
        model_path = download_model_from_hf()

    print(f"📂 Loading model from {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        print(f"✓ Model loaded successfully")
        print(f"   Trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"   Validation accuracy: {checkpoint.get('val_acc', 0):.2f}%")
        return checkpoint

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
