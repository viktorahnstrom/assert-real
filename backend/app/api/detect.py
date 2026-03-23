# backend/app/api/detect.py
"""
Deepfake detection endpoint
Loads EfficientNet-B4 model and provides inference API
"""

import io
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from app.dependencies.auth import require_auth
from app.utils.model_loader import load_model_checkpoint

logger = logging.getLogger(__name__)

router = APIRouter()

model: Optional[nn.Module] = None
device: Optional[torch.device] = None
transform: Optional[transforms.Compose] = None
class_names = ["fake", "real"]


class DeepfakeDetector(nn.Module):
    """EfficientNet-B4 deepfake detector (matches training architecture)"""

    def __init__(self, num_classes=2):
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

    def forward(self, x):
        return self.model(x)


class DetectionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model: str = "EfficientNet-B4"
    accuracy: str = "98.48%"


def load_detection_model():
    """Load the trained deepfake detection model (called on startup)"""
    global model, device, transform, class_names

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔧 Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🔧 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("🔧 Using CPU")

    print("🔧 Loading detection model...")

    try:
        checkpoint = load_model_checkpoint()

        model = DeepfakeDetector().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        class_names = checkpoint.get("class_names", ["fake", "real"])

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        print("✅ Detection model loaded successfully")
        print(f"   Validation accuracy: {checkpoint.get('val_acc', 0):.2f}%")
        print(f"   Classes: {class_names}")
        print(f"   Device: {device}")

    except Exception as e:
        print(f"❌ Failed to load detection model: {e}")
        print("   Detection endpoints will not be available")
        model = None
        device = None
        transform = None
        class_names = None


@router.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_auth),  # 🔒 requires valid JWT
):
    """
    Detect if an uploaded image is a deepfake.
    Requires a valid Bearer token in the Authorization header.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Detection model not loaded. Ensure best_model.pt exists in backend/checkpoints/",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail=f"File must be an image, got: {file.content_type}"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)

        prediction = class_names[predicted.item()]
        confidence_value = float(confidence.item())
        probs = probabilities[0].cpu().numpy()

        return DetectionResponse(
            prediction=prediction,
            confidence=confidence_value,
            probabilities={"fake": float(probs[0]), "real": float(probs[1])},
        )

    except Exception as e:
        logger.exception("Unhandled error during deepfake detection")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during detection. Please try again.",
        ) from e


@router.get("/model-info")
async def get_model_info(current_user: dict = Depends(require_auth)):  # 🔒 also protected
    """Get information about the loaded detection model."""
    if model is None:
        return {
            "status": "not_loaded",
            "message": "Detection model not available",
        }

    model_path = Path(__file__).parent.parent.parent / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    return {
        "status": "loaded",
        "architecture": "EfficientNet-B4",
        "validation_accuracy": f"{checkpoint['val_acc']:.2f}%",
        "classes": checkpoint["class_names"],
        "device": str(device),
    }
