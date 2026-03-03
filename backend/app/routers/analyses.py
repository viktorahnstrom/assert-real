"""
Analysis routes for XADE.
Runs deepfake detection, generates VLM explanations, and stores results in database.
"""

import io
import os
import time

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

load_dotenv()

router = APIRouter(prefix="/api/v1/analyses", tags=["Analyses"])

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


# ============================================
# Request/Response Models
# ============================================
class AnalysisRequest(BaseModel):
    image_id: str
    user_id: str
    vlm_provider: str | None = None


class ExplanationData(BaseModel):
    summary: str
    detailed_analysis: str
    technical_notes: str | None = None
    provider: str
    model: str
    processing_time_ms: int
    estimated_cost_usd: float


class AnalysisResponse(BaseModel):
    id: str
    image_id: str
    user_id: str
    status: str
    deepfake_score: float | None = None
    classification: str | None = None
    model_used: str | None = None
    vlm_explanation: str | None = None
    vlm_model_used: str | None = None
    processing_time_ms: int | None = None
    created_at: str
    completed_at: str | None = None
    explanation: ExplanationData | None = None


class AnalysisListResponse(BaseModel):
    analyses: list[AnalysisResponse]
    count: int


# ============================================
# Helper functions
# ============================================
def get_db_headers():
    """Headers for Supabase Database API."""
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def get_storage_headers():
    """Headers for Supabase Storage API."""
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }


def _generate_placeholder_heatmap(image):
    """Generate a placeholder heatmap. TODO: Replace with real GradCAM."""
    import numpy as np
    from PIL import Image as PILImage

    width, height = image.size
    heatmap = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            cx, cy = width // 2, height // 2
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            max_dist = (cx**2 + cy**2) ** 0.5
            intensity = max(0, 1.0 - dist / max_dist)
            heatmap[y, x] = [int(255 * intensity), int(100 * intensity), 0]

    heatmap_img = PILImage.fromarray(heatmap, "RGB")
    blended = PILImage.blend(image.resize((width, height)), heatmap_img, alpha=0.4)

    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_analysis_response(
    a: dict, explanation: ExplanationData | None = None
) -> AnalysisResponse:
    """Build AnalysisResponse from database row."""
    return AnalysisResponse(
        id=a["id"],
        image_id=a["image_id"],
        user_id=a["user_id"],
        status=a["status"],
        deepfake_score=a.get("deepfake_score"),
        classification=a.get("classification"),
        model_used=a.get("model_used"),
        vlm_explanation=a.get("vlm_explanation"),
        vlm_model_used=a.get("vlm_model_used"),
        processing_time_ms=a.get("processing_time_ms"),
        created_at=a["created_at"],
        completed_at=a.get("completed_at"),
        explanation=explanation,
    )


# ============================================
# Routes
# ============================================
@router.post("/", response_model=AnalysisResponse)
async def create_analysis(request: AnalysisRequest):
    """
    Create a new analysis for an uploaded image.
    1. Creates analysis record with 'processing' status
    2. Fetches the image from storage
    3. Runs detection model
    4. Generates VLM explanation (if factory available)
    5. Updates analysis with results
    """
    import torch
    from PIL import Image

    from app.api.detect import class_names, device, model, transform, vlm_factory
    from app.services.vlm import DetectionContext

    if model is None:
        raise HTTPException(status_code=503, detail="Detection model not loaded")

    async with httpx.AsyncClient() as client:
        # Get image metadata from database
        img_response = await client.get(
            f"{SUPABASE_URL}/rest/v1/images?id=eq.{request.image_id}&select=*",
            headers=get_db_headers(),
        )

        if img_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image")

        images = img_response.json()
        if not images:
            raise HTTPException(status_code=404, detail="Image not found")

        image_data = images[0]
        storage_path = image_data["storage_path"]

        # Create analysis record with processing status
        analysis_data = {
            "image_id": request.image_id,
            "user_id": request.user_id,
            "status": "processing",
            "model_used": "efficientnet-b4",
        }

        create_response = await client.post(
            f"{SUPABASE_URL}/rest/v1/analyses",
            headers=get_db_headers(),
            json=analysis_data,
        )

        if create_response.status_code not in [200, 201]:
            error = create_response.json()
            raise HTTPException(
                status_code=create_response.status_code,
                detail=error.get("message", "Failed to create analysis"),
            )

        analysis = create_response.json()[0]
        analysis_id = analysis["id"]

        # Download image from storage
        start_time = time.time()

        img_download = await client.get(
            f"{SUPABASE_URL}/storage/v1/object/images/{storage_path}",
            headers=get_storage_headers(),
        )

        if img_download.status_code != 200:
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/analyses?id=eq.{analysis_id}",
                headers=get_db_headers(),
                json={"status": "failed", "error_message": "Failed to download image"},
            )
            raise HTTPException(status_code=500, detail="Failed to download image from storage")

        # Run detection
        try:
            image = Image.open(io.BytesIO(img_download.content)).convert("RGB")
            image_bytes = img_download.content
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = probabilities.max(1)

            prediction = class_names[predicted.item()]
            confidence_value = float(confidence.item())
            fake_prob = float(probabilities[0][0].cpu().numpy())
            real_prob = float(probabilities[0][1].cpu().numpy())

            processing_time = int((time.time() - start_time) * 1000)

            if prediction == "fake":
                classification = "fake"
            elif prediction == "real":
                classification = "real"
            else:
                classification = "uncertain"

            # Generate VLM explanation
            explanation_data = None
            vlm_explanation_text = None
            vlm_model_used = None

            if vlm_factory is not None:
                try:
                    detection_context = DetectionContext(
                        classification=prediction,
                        confidence=confidence_value,
                        model_used="EfficientNet-B4",
                        probabilities={"fake": fake_prob, "real": real_prob},
                    )

                    heatmap_bytes = _generate_placeholder_heatmap(image)

                    vlm_explanation = await vlm_factory.generate_explanation(
                        provider_id=request.vlm_provider,
                        image_bytes=image_bytes,
                        heatmap_bytes=heatmap_bytes,
                        detection=detection_context,
                    )

                    # Build structured explanation for API response
                    explanation_data = ExplanationData(
                        summary=vlm_explanation.summary,
                        detailed_analysis=vlm_explanation.detailed_analysis,
                        technical_notes=vlm_explanation.technical_notes,
                        provider=vlm_explanation.provider,
                        model=vlm_explanation.model,
                        processing_time_ms=vlm_explanation.processing_time_ms,
                        estimated_cost_usd=vlm_explanation.estimated_cost_usd,
                    )

                    # Store combined explanation text for the database
                    vlm_explanation_text = (
                        f"[SUMMARY]\n{vlm_explanation.summary}\n\n"
                        f"[DETAILED]\n{vlm_explanation.detailed_analysis}"
                    )
                    if vlm_explanation.technical_notes:
                        vlm_explanation_text += (
                            f"\n\n[TECHNICAL]\n{vlm_explanation.technical_notes}"
                        )

                    vlm_model_used = f"{vlm_explanation.provider}/{vlm_explanation.model}"

                except Exception as e:
                    # VLM failure should not fail the analysis
                    import logging

                    logging.getLogger(__name__).warning(f"VLM explanation failed: {e}")

            # Update analysis with results
            update_data = {
                "status": "completed",
                "deepfake_score": round(fake_prob, 4),
                "classification": classification,
                "processing_time_ms": processing_time,
                "completed_at": "now()",
            }

            if vlm_explanation_text:
                update_data["vlm_explanation"] = vlm_explanation_text
            if vlm_model_used:
                update_data["vlm_model_used"] = vlm_model_used

            update_response = await client.patch(
                f"{SUPABASE_URL}/rest/v1/analyses?id=eq.{analysis_id}",
                headers=get_db_headers(),
                json=update_data,
            )

            if update_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to update analysis")

            # Fetch the completed analysis
            final_response = await client.get(
                f"{SUPABASE_URL}/rest/v1/analyses?id=eq.{analysis_id}&select=*",
                headers=get_db_headers(),
            )

            final_analysis = final_response.json()[0]
            return _build_analysis_response(final_analysis, explanation=explanation_data)

        except HTTPException:
            raise
        except Exception as e:
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/analyses?id=eq.{analysis_id}",
                headers=get_db_headers(),
                json={"status": "failed", "error_message": str(e)},
            )
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.get("/", response_model=AnalysisListResponse)
async def list_analyses(user_id: str | None = None):
    """List all analyses, optionally filtered by user."""
    async with httpx.AsyncClient() as client:
        url = f"{SUPABASE_URL}/rest/v1/analyses?select=*&order=created_at.desc"
        if user_id:
            url += f"&user_id=eq.{user_id}"

        response = await client.get(url, headers=get_db_headers())

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch analyses")

        analyses_data = response.json()
        analyses = [_build_analysis_response(a) for a in analyses_data]
        return AnalysisListResponse(analyses=analyses, count=len(analyses))


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str):
    """Get a specific analysis by ID."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/analyses?id=eq.{analysis_id}&select=*",
            headers=get_db_headers(),
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch analysis")

        analyses = response.json()
        if not analyses:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return _build_analysis_response(analyses[0])


@router.get("/image/{image_id}", response_model=AnalysisListResponse)
async def get_analyses_for_image(image_id: str):
    """Get all analyses for a specific image."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/analyses?image_id=eq.{image_id}&select=*&order=created_at.desc",
            headers=get_db_headers(),
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch analyses")

        analyses_data = response.json()
        analyses = [_build_analysis_response(a) for a in analyses_data]
        return AnalysisListResponse(analyses=analyses, count=len(analyses))
