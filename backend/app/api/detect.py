"""
Detection API endpoints for XADE deepfake detection framework.

Provides image-based deepfake detection using EfficientNet-B4,
with optional GradCAM heatmap generation.
"""

from __future__ import annotations

# stdlib
import io
import logging
from pathlib import Path
from typing import Optional

# third-party
import torch
import torch.nn as nn
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class DeepfakeDetector(nn.Module):
    """EfficientNet-B4 deepfake detector (matches training architecture)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        self.model = efficientnet_b4(weights=weights)

        # Unfreeze last 30%
        total_layers = len(list(self.model.features.parameters()))
        freeze_until = int(total_layers * 0.7)

        for idx, param in enumerate(self.model.features.parameters()):
            param.requires_grad = idx >= freeze_until

        # Classifier
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Module-level state (set by main.py lifespan)
# ---------------------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
class_names = ["fake", "real"]
model: Optional[DeepfakeDetector] = None
vlm_factory = None  # Set by main.py lifespan after VLM init

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ---------------------------------------------------------------------------
# Model loading — called by main.py lifespan
# ---------------------------------------------------------------------------


def load_detection_model() -> None:
    """Load the EfficientNet-B4 checkpoint. Called once at startup by main.py."""
    global model
    try:
        from app.utils.model_loader import download_model_from_hf

        model_path = download_model_from_hf()
        detector = DeepfakeDetector()
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        detector.load_state_dict(checkpoint["model_state_dict"])
        detector.to(device)
        detector.eval()
        logger.info("Model loaded successfully on %s", device)
        print(f"✓ Detection model loaded on {device}")
        model = detector
    except Exception as exc:
        logger.warning("Could not load model checkpoint: %s", exc)
        print(f"✗ Could not load model checkpoint: {exc}")
        model = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class DetectionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    gradcam_heatmap_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(default=True, description="Generate GradCAM heatmap overlay"),
) -> DetectionResponse:
    """
    Detect whether an uploaded image is a deepfake.

    Args:
        file: Image file (JPEG, PNG, WEBP, etc.)
        include_gradcam: If True, generate and return a GradCAM heatmap URL.

    Returns:
        - prediction: "fake" or "real"
        - confidence: Model confidence (0-1)
        - probabilities: Per-class probabilities
        - gradcam_heatmap_url: Local file URL to overlay image (if requested)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Detection model not loaded. Ensure best_model.pt exists in backend/checkpoints/",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image, got: {file.content_type}",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Inference — no_grad is fine here, GradCAM runs a separate pass below
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)

        prediction = class_names[predicted.item()]
        confidence_value = float(confidence.item())
        probs = probabilities[0].cpu().numpy()
        target_class = int(predicted.item())

        gradcam_heatmap_url: Optional[str] = None

        if include_gradcam:
            try:
                from app.services.gradcam_service import GradCAMGenerator
                from app.services.gradcam_storage import (
                    get_local_heatmap_url,
                    save_heatmap_locally,
                )

                generator = GradCAMGenerator(model)
                heatmap = generator.generate(image_tensor, target_class=target_class)
                overlay = generator.create_overlay(image, heatmap)

                filepath = save_heatmap_locally(overlay)
                gradcam_heatmap_url = get_local_heatmap_url(filepath)

            except Exception as gradcam_exc:
                import traceback

                traceback.print_exc()
                logger.warning("GradCAM generation failed (non-fatal): %s", gradcam_exc)
                gradcam_heatmap_url = None

        return DetectionResponse(
            prediction=prediction,
            confidence=confidence_value,
            probabilities={"fake": float(probs[0]), "real": float(probs[1])},
            gradcam_heatmap_url=gradcam_heatmap_url,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(exc)}") from exc


@router.get("/model-info")
async def get_model_info() -> dict:
    """Get information about the loaded detection model."""
    if model is None:
        return {"status": "not_loaded", "message": "Detection model not available"}

    model_path = Path(__file__).parent.parent.parent / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    return {
        "status": "loaded",
        "architecture": "EfficientNet-B4",
        "validation_accuracy": f"{checkpoint['val_acc']:.2f}%",
        "training_samples": checkpoint["train_samples"],
        "classes": checkpoint["class_names"],
        "device": str(device),
        "parameters": f"{sum(p.numel() for p in model.parameters()):,}",
    }
