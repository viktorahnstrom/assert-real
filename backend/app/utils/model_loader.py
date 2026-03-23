"""
XADE Model Loader

Handles downloading and loading the EfficientNet-B4 checkpoint from
Hugging Face Hub. Uses weights_only=True on all torch.load calls to
prevent arbitrary code execution via pickle deserialization.
"""

import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

HF_REPO_ID = "viktorahnstrom/xade-deepfake-detector"
HF_FILENAME = "best_model.pt"
LOCAL_CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "checkpoints"
LOCAL_CHECKPOINT_PATH = LOCAL_CHECKPOINT_DIR / HF_FILENAME


def _download_model_checkpoint() -> Path:
    """
    Downloads the model checkpoint from Hugging Face Hub if not already present.
    Returns the local path to the checkpoint file.
    """
    if LOCAL_CHECKPOINT_PATH.exists():
        logger.info("Checkpoint already exists at %s, skipping download.", LOCAL_CHECKPOINT_PATH)
        return LOCAL_CHECKPOINT_PATH

    logger.info("Checkpoint not found locally. Downloading from Hugging Face Hub...")
    LOCAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=str(LOCAL_CHECKPOINT_DIR),
    )

    logger.info("Checkpoint downloaded to %s", downloaded_path)
    return Path(downloaded_path)


def load_model_checkpoint() -> dict:
    """
    Downloads (if needed) and loads the model checkpoint safely.

    Uses weights_only=True to prevent arbitrary code execution via pickle.
    This is the only approved way to load torch checkpoints in XADE.

    Returns:
        The checkpoint dict containing model_state_dict, val_acc,
        class_names, and training metadata.

    Raises:
        RuntimeError: If the checkpoint cannot be loaded.
    """
    checkpoint_path = _download_model_checkpoint()

    logger.info("Loading checkpoint from %s", checkpoint_path)

    try:
        # weights_only=True prevents pickle-based arbitrary code execution.
        # If this raises an UnpicklingError, the checkpoint was saved with
        # non-tensor objects — contact the model author to re-export safely.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint from {checkpoint_path}. "
            f"If you see an UnpicklingError, the .pt file may contain non-tensor "
            f"objects incompatible with weights_only=True. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "Checkpoint loaded. val_acc=%.2f%%, classes=%s",
        checkpoint.get("val_acc", 0.0),
        checkpoint.get("class_names", []),
    )
    return checkpoint
