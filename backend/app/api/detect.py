"""
Deepfake detection endpoint
Loads EfficientNet-B4 model and provides inference API
With VLM explanation generation support
"""

import io
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from app.services.vlm import DetectionContext, VLMProviderFactory
from app.utils.model_loader import load_model_checkpoint

router = APIRouter()

model: Optional[nn.Module] = None
device: Optional[torch.device] = None
transform: Optional[transforms.Compose] = None
class_names = ["fake", "real"]

# VLM factory reference — set by main.py on startup
vlm_factory: Optional[VLMProviderFactory] = None


class ExplanationResponse(BaseModel):
    """VLM-generated explanation included in detection response."""

    summary: str
    detailed_analysis: str
    technical_notes: Optional[str] = None
    provider: str
    model: str
    processing_time_ms: int
    estimated_cost_usd: float


class DetectionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model: str = "EfficientNet-B4"
    accuracy: str = "98.48%"
    explanation: Optional[ExplanationResponse] = None


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


def load_detection_model():
    """Load the trained deepfake detection model (called on startup)"""
    global model, device, transform, class_names

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Download model if needed, then load
    print("Loading detection model...")

    try:
        # This will auto-download from Hugging Face if not found locally
        checkpoint = load_model_checkpoint()

        # Initialize model architecture
        model = DeepfakeDetector().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Get class names
        class_names = checkpoint.get("class_names", ["fake", "real"])

        # Setup transforms
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        print("Detection model loaded successfully")
        print(f"   Validation accuracy: {checkpoint.get('val_acc', 0):.2f}%")
        print(f"   Classes: {class_names}")
        print(f"   Device: {device}")

    except Exception as e:
        print(f"Failed to load detection model: {e}")
        print("   Detection endpoints will not be available")
        model = None
        device = None
        transform = None
        class_names = None


def _generate_placeholder_heatmap(image: Image.Image) -> bytes:
    """
    Generate a placeholder heatmap image.

    TODO: Replace with actual GradCAM implementation.
    For now, returns a solid overlay so the VLM pipeline can be tested end-to-end.
    """
    import numpy as np

    # Create a simple gradient heatmap matching image dimensions
    width, height = image.size
    heatmap = np.zeros((height, width, 3), dtype=np.uint8)

    # Create a center-weighted gradient (simulates face-focused attention)
    for y in range(height):
        for x in range(width):
            cx, cy = width // 2, height // 2
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            max_dist = (cx**2 + cy**2) ** 0.5
            intensity = max(0, 1.0 - dist / max_dist)
            heatmap[y, x] = [
                int(255 * intensity),  # Red channel
                int(100 * intensity),  # Green channel
                0,  # Blue channel
            ]

    # Convert to PNG bytes
    heatmap_img = Image.fromarray(heatmap, "RGB")

    # Blend with original image (50% opacity)
    blended = Image.blend(image.resize((width, height)), heatmap_img, alpha=0.4)

    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    return buffer.getvalue()


@router.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(
    file: UploadFile = File(...),
    vlm_provider: Optional[str] = Query(
        None,
        description="VLM provider for explanation generation: 'google', 'openai', 'mock', or None to skip",
    ),
    explain: bool = Query(
        True,
        description="Whether to generate a VLM explanation (set to false for detection only)",
    ),
):
    """
    Detect if an uploaded image is a deepfake.

    Optionally generates a human-readable explanation using a Vision-Language Model
    grounded in the GradCAM heatmap visualization.

    **Args:**
    - file: Image file (JPG, PNG, etc.)
    - vlm_provider: Which VLM to use for explanation (default: configured default)
    - explain: Whether to generate explanation (default: true)

    **Returns:**
    - prediction: "fake" or "real"
    - confidence: Model confidence (0-1)
    - probabilities: Probability for each class
    - explanation: VLM-generated explanation (if requested)
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Detection model not loaded. Ensure best_model.pt exists in backend/checkpoints/",
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail=f"File must be an image, got: {file.content_type}"
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Transform
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities_tensor = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities_tensor.max(1)

        # Results
        prediction = class_names[predicted.item()]
        confidence_value = float(confidence.item())
        probs = probabilities_tensor[0].cpu().numpy()
        prob_dict = {"fake": float(probs[0]), "real": float(probs[1])}

        # Build response
        response = DetectionResponse(
            prediction=prediction,
            confidence=confidence_value,
            probabilities=prob_dict,
        )

        # Generate VLM explanation if requested and factory is available
        if explain and vlm_factory is not None:
            try:
                detection_context = DetectionContext(
                    classification=prediction,
                    confidence=confidence_value,
                    model_used="EfficientNet-B4",
                    probabilities=prob_dict,
                )

                # Generate heatmap (placeholder for now, will be replaced with GradCAM)
                heatmap_bytes = _generate_placeholder_heatmap(image)

                # Get explanation from VLM
                vlm_explanation = await vlm_factory.generate_explanation(
                    provider_id=vlm_provider,
                    image_bytes=contents,
                    heatmap_bytes=heatmap_bytes,
                    detection=detection_context,
                )

                response.explanation = ExplanationResponse(
                    summary=vlm_explanation.summary,
                    detailed_analysis=vlm_explanation.detailed_analysis,
                    technical_notes=vlm_explanation.technical_notes,
                    provider=vlm_explanation.provider,
                    model=vlm_explanation.model,
                    processing_time_ms=vlm_explanation.processing_time_ms,
                    estimated_cost_usd=vlm_explanation.estimated_cost_usd,
                )

            except ValueError as e:
                # Provider not configured — still return detection without explanation
                response.explanation = ExplanationResponse(
                    summary=f"Explanation unavailable: {e}",
                    detailed_analysis="The detection result above is still valid.",
                    provider=vlm_provider or "unknown",
                    model="n/a",
                    processing_time_ms=0,
                    estimated_cost_usd=0.0,
                )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}") from e


@router.get("/model-info")
async def get_model_info():
    """Get information about the loaded detection model"""

    if model is None:
        return {
            "status": "not_loaded",
            "message": "Detection model not available",
        }

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
