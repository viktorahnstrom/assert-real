"""
Study routes for XADE user study.

Runs deepfake detection + all three VLM providers in parallel for Phase 2
explanation comparison. Also persists anonymised participant results.
"""

import asyncio
import io
import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/study", tags=["Study"])

# Path to the Vite public directory where precomputed assets are served
_FRONTEND_PUBLIC = Path(__file__).parent.parent.parent.parent / "desktop" / "public"

# Fixed image list — must match ALL_IMAGES in DeepfakeTest.tsx
_STUDY_IMAGES = [
    {"id": 1, "filename": "sg3_psi070_seed0002378.webp", "label": "fake"},
    {"id": 2, "filename": "sg3_psi070_seed0002384.webp", "label": "fake"},
    {"id": 3, "filename": "sg3_psi070_seed0002763.webp", "label": "fake"},
    {"id": 4, "filename": "sg3_psi070_seed0003361.webp", "label": "fake"},
    {"id": 5, "filename": "sg3_psi070_seed0003379.webp", "label": "fake"},
    {"id": 6, "filename": "sg3_psi070_seed0003409.webp", "label": "fake"},
    {"id": 7, "filename": "00093.webp", "label": "real"},
    {"id": 8, "filename": "00408.webp", "label": "real"},
    {"id": 9, "filename": "00764.webp", "label": "real"},
    {"id": 10, "filename": "00818.webp", "label": "real"},
    {"id": 11, "filename": "01770.webp", "label": "real"},
    {"id": 12, "filename": "02213.webp", "label": "real"},
]

# Phase 3 retest image list — must match RETEST_IMAGES in DeepfakeTest.tsx.
# These are shown ONLY to participants who misclassified at least one
# Phase 1 image, after they've seen the Phase 2 explanations. Must be
# disjoint from _STUDY_IMAGES so participants see them for the first time.
# Retest images do NOT need precomputed explanations (no VLM analysis runs
# on them), so they're not iterated by /api/v1/study/precompute.
_RETEST_IMAGES = [
    {"id": 13, "filename": "sg3_psi070_seed0001025.webp", "label": "fake"},
    {"id": 14, "filename": "sg3_psi070_seed0001242.webp", "label": "fake"},
    {"id": 15, "filename": "00999.webp", "label": "real"},
]

# Sanity-check at import time that the two image sets do not overlap.
_overlap = {img["filename"] for img in _STUDY_IMAGES} & {img["filename"] for img in _RETEST_IMAGES}
if _overlap:
    raise RuntimeError(f"_STUDY_IMAGES and _RETEST_IMAGES must be disjoint, found: {_overlap}")

# ============================================
# Models
# ============================================


class StudyExplanation(BaseModel):
    provider: str
    model: str
    summary: str
    detailed_analysis: str
    technical_notes: str | None = None
    processing_time_ms: int
    error: str | None = None


class StudyAnalysisResponse(BaseModel):
    deepfake_score: float
    classification: str
    confidence: float
    gradcam_url: str | None = None
    ela_heatmap_url: str | None = None
    evidence_regions: list[dict] = []
    explanations: dict[str, StudyExplanation]


class StudyResults(BaseModel):
    participant_id: str
    self_confidence_rating: int
    baseline_accuracy: float
    total_images: int
    correct_count: int
    incorrect_count: int
    explanation_answers: list[dict]
    # Phase 3 retest. Empty when the participant got 100% on Phase 1 and
    # the retest was skipped. The #118 timer work will add per-entry
    # time_ms / idle_discarded fields.
    retest_answers: list[dict] = []
    trust_rating: int
    willingness_to_use: str  # "yes" | "no" | "maybe"
    # 1–5 rating shown only to participants who took the retest. None
    # when the retest was skipped.
    explanations_helped_in_retest: int | None = None
    comments: str
    completed_at: str


# ============================================
# Helpers
# ============================================


def _results_file() -> Path:
    results_dir = Path(tempfile.gettempdir()) / "xade_study_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / "study_results.jsonl"


def _run_detection_pipeline(image, image_tensor, target_class) -> dict:
    """Run face_bbox + GradCAM + face parsing + region ranker.

    Mirrors the production /analyses path so study explanations carry the
    same evidence_regions (with z_scores) the desktop ResultView renders.
    Returns dict with heatmap_bytes, gradcam_url, evidence_regions, crops,
    region_categories, region_labels.
    """
    from app.api.detect import face_category_mapper, face_parser
    from app.routers.analyses import _build_ranked_evidence, _run_gradcam
    from app.services.categories import FACE_CATEGORIES
    from app.services.vlm.base import RegionWithCategory

    face_bbox = None
    if face_category_mapper is not None:
        try:
            face_bbox = face_category_mapper.detect_face_bbox(image)
        except Exception as e:
            logger.warning("Face bbox detection failed: %s", e)

    heatmap, heatmap_bytes, gradcam_url, evidence_regions, crops = _run_gradcam(
        image, image_tensor, target_class, face_bbox=face_bbox
    )

    parsing_result = None
    if face_parser is not None:
        try:
            parsing_result = face_parser.parse(image)
        except Exception as e:
            logger.warning("Face parsing failed: %s", e)

    ranker_used = False
    region_categories: list[RegionWithCategory] = []

    if parsing_result is not None and heatmap is not None:
        ranked = _build_ranked_evidence(image, heatmap, parsing_result, face_bbox=face_bbox)
        if ranked is not None:
            evidence_regions, crops, _ = ranked
            ranker_used = True
            for ev_region in evidence_regions:
                cat_id = ev_region.get("category_id")
                face_cat = FACE_CATEGORIES.get(cat_id) if cat_id else None
                if face_cat is not None:
                    region_categories.append(
                        RegionWithCategory(
                            label=face_cat.label,
                            category_id=face_cat.id,
                            category_label=face_cat.label,
                            common_artifacts=face_cat.common_artifacts[:3],
                            activation_score=ev_region.get("activation_score", 0.0),
                        )
                    )

    if not ranker_used and face_category_mapper is not None and crops:
        try:
            categorized = face_category_mapper.map_regions(
                image, crops, parsing_result=parsing_result
            )
            region_categories = [r.to_region_with_category() for r in categorized]
            for ev_region, cat in zip(evidence_regions, categorized, strict=True):
                ev_region["category_id"] = cat.category_id
                ev_region["category_label"] = cat.category_label
                face_cat = FACE_CATEGORIES.get(cat.category_id)
                ev_region["common_artifacts"] = (
                    list(face_cat.common_artifacts[:3]) if face_cat else []
                )
        except Exception as e:
            logger.warning("Face category mapping failed: %s", e)

    region_labels = [r["label"] for r in evidence_regions] if evidence_regions else []

    return {
        "heatmap_bytes": heatmap_bytes,
        "gradcam_url": gradcam_url,
        "evidence_regions": evidence_regions or [],
        "crops": crops or [],
        "region_categories": region_categories,
        "region_labels": region_labels,
    }


def _build_ela_overlay_bytes(image) -> bytes | None:
    """Compute an ELA overlay PNG for the given PIL image, or None on failure."""
    try:
        from app.services.forensics.ela import compute_ela, create_ela_overlay

        ela_map = compute_ela(image, quality=95, scale=10)
        overlay = create_ela_overlay(image, ela_map)
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        logger.warning("ELA overlay generation failed: %s", exc)
        return None


async def _run_openai_and_attach_region_comments(
    vlm_factory,
    image_bytes: bytes,
    heatmap_bytes: bytes | None,
    ela_bytes: bytes | None,
    detection_context,
    evidence_regions: list[dict],
    crops: list[dict],
    img_id: str,
) -> dict:
    """Call OpenAI, attach per-region comments to evidence_regions in place,
    and return the StudyExplanation payload as a dict.

    Mirrors the production analyses.py flow so the per-region commentary
    drives the Facial Regions panel instead of bleeding into the main
    explanation text.
    """
    from app.routers.analyses import _attach_region_comments

    # Convert region crop PIL images to JPEG bytes for VLM
    region_image_bytes: list[bytes] = []
    for crop in crops:
        buf = io.BytesIO()
        crop["image"].save(buf, format="JPEG", quality=90)
        region_image_bytes.append(buf.getvalue())

    try:
        vlm_result = await vlm_factory.generate_explanation(
            provider_id="openai",
            image_bytes=image_bytes,
            heatmap_bytes=heatmap_bytes if heatmap_bytes else image_bytes,
            detection=detection_context,
            gradcam_available=heatmap_bytes is not None,
            region_image_bytes=region_image_bytes if region_image_bytes else None,
            ela_bytes=ela_bytes,
        )

        # Attach per-region commentary so Facial Regions cards show their
        # own observations instead of leaving them empty.
        if (vlm_result.region_comments or vlm_result.structured_regions) and evidence_regions:
            _attach_region_comments(
                evidence_regions,
                vlm_result.region_comments or {},
                structured_regions=vlm_result.structured_regions,
            )

        return {
            "provider": vlm_result.provider,
            "model": vlm_result.model,
            "summary": vlm_result.summary,
            "detailed_analysis": vlm_result.detailed_analysis,
            "technical_notes": vlm_result.technical_notes,
            "processing_time_ms": vlm_result.processing_time_ms,
            "error": None,
        }
    except Exception as exc:
        logger.warning("OpenAI VLM failed for image %s: %s", img_id, exc)
        return {
            "provider": "openai",
            "model": "unavailable",
            "summary": "Explanation unavailable for this provider.",
            "detailed_analysis": "This provider could not generate an explanation.",
            "technical_notes": None,
            "processing_time_ms": 0,
            "error": str(exc),
        }


# ============================================
# Routes
# ============================================


@router.post("/analyze", response_model=StudyAnalysisResponse)
async def analyze_for_study(file: UploadFile = File(...)):
    """
    Run deepfake detection + GradCAM + all VLM providers in parallel.
    Used by the user study Phase 2 to generate the three anonymised explanations.
    """
    import torch
    from PIL import Image

    from app.api.detect import (
        class_names,
        device,
        model,
        transform,
        vlm_factory,
    )
    from app.services.vlm import DetectionContext

    if model is None:
        raise HTTPException(status_code=503, detail="Detection model not loaded")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    prediction = class_names[predicted.item()]
    confidence_value = float(confidence.item())
    fake_prob = float(probabilities[0][0].cpu().numpy())
    real_prob = float(probabilities[0][1].cpu().numpy())
    target_class = int(predicted.item())
    classification = prediction if prediction in ("fake", "real") else "uncertain"

    pipeline = _run_detection_pipeline(image, image_tensor, target_class)
    heatmap_bytes = pipeline["heatmap_bytes"]
    gradcam_url = pipeline["gradcam_url"]
    evidence_regions = pipeline["evidence_regions"]
    crops = pipeline["crops"]

    detection_context = DetectionContext(
        classification=prediction,
        confidence=confidence_value,
        model_used="EfficientNet-B4",
        probabilities={"fake": fake_prob, "real": real_prob},
        region_labels=pipeline["region_labels"],
        region_categories=pipeline["region_categories"],
    )

    # Build ELA overlay once — used by the VLM AND served back to the frontend
    ela_bytes = _build_ela_overlay_bytes(image)
    ela_heatmap_url: str | None = None
    if ela_bytes is not None:
        from app.services.gradcam_storage import save_ela_locally

        ela_heatmap_url = save_ela_locally(ela_bytes)

    explanations: dict[str, StudyExplanation] = {}

    if vlm_factory is not None:
        result = await _run_openai_and_attach_region_comments(
            vlm_factory,
            image_bytes,
            heatmap_bytes,
            ela_bytes,
            detection_context,
            evidence_regions,
            crops,
            "live",
        )
        explanations["openai"] = StudyExplanation(**result)
    else:
        explanations["openai"] = StudyExplanation(
            provider="openai",
            model="mock",
            summary="VLM service is not configured on this instance.",
            detailed_analysis="No explanation could be generated because no VLM API keys are set.",
            processing_time_ms=0,
        )

    return StudyAnalysisResponse(
        deepfake_score=round(fake_prob, 4),
        classification=classification,
        confidence=confidence_value,
        gradcam_url=gradcam_url,
        ela_heatmap_url=ela_heatmap_url,
        evidence_regions=evidence_regions or [],
        explanations=explanations,
    )


@router.post("/results")
async def save_study_results(results: StudyResults):
    """Append one participant's anonymised results to the JSONL log file."""
    entry = results.model_dump()
    entry["saved_at"] = datetime.now(UTC).isoformat()

    with open(_results_file(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("Study result saved for participant %s", results.participant_id)
    return {"status": "saved", "participant_id": results.participant_id}


@router.post("/precompute")
async def precompute_study_analyses():
    """
    One-time researcher endpoint: run detection + GradCAM + all VLM providers
    for every study image and save results to desktop/public/study-analyses.json.
    Heatmap images are saved to desktop/public/quiz-heatmaps/ so they are served
    by the Vite dev server as static assets.

    Run once before participant sessions start.
    """
    import torch
    from PIL import Image

    from app.api.detect import (
        class_names,
        device,
        model,
        transform,
        vlm_factory,
    )
    from app.services.vlm import DetectionContext

    if model is None:
        raise HTTPException(status_code=503, detail="Detection model not loaded")

    heatmaps_dir = _FRONTEND_PUBLIC / "quiz-heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    analyses: dict[str, dict] = {}

    for img_meta in _STUDY_IMAGES:
        img_id = str(img_meta["id"])
        img_path = _FRONTEND_PUBLIC / "quiz-images" / img_meta["filename"]

        logger.info("Precomputing image %s (%s)", img_id, img_meta["filename"])

        try:
            image = Image.open(img_path).convert("RGB")
            image_bytes = img_path.read_bytes()
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = probabilities.max(1)

            prediction = class_names[predicted.item()]
            confidence_value = float(confidence.item())
            fake_prob = float(probabilities[0][0].cpu().numpy())
            real_prob = float(probabilities[0][1].cpu().numpy())
            target_class = int(predicted.item())
            classification = prediction if prediction in ("fake", "real") else "uncertain"

            pipeline = _run_detection_pipeline(image, image_tensor, target_class)
            heatmap_bytes = pipeline["heatmap_bytes"]
            evidence_regions = pipeline["evidence_regions"]
            crops = pipeline["crops"]

            heatmap_public_url: str | None = None
            if heatmap_bytes:
                heatmap_filename = f"heatmap_{img_id}.jpg"
                (heatmaps_dir / heatmap_filename).write_bytes(heatmap_bytes)
                heatmap_public_url = f"/quiz-heatmaps/{heatmap_filename}"

            # ELA overlay → public Vite asset
            ela_public_url: str | None = None
            ela_bytes = _build_ela_overlay_bytes(image)
            if ela_bytes:
                ela_filename = f"ela_{img_id}.png"
                (heatmaps_dir / ela_filename).write_bytes(ela_bytes)
                ela_public_url = f"/quiz-heatmaps/{ela_filename}"

            # Re-save evidence crops as public Vite assets so the deployed
            # study uses /quiz-heatmaps/... instead of localhost:8000/gradcam/...
            for i, (region, crop) in enumerate(zip(evidence_regions, crops, strict=True)):
                crop_filename = f"region_{img_id}_{i}.jpg"
                crop["image"].save(heatmaps_dir / crop_filename, format="JPEG", quality=90)
                region["url"] = f"/quiz-heatmaps/{crop_filename}"

            detection_context = DetectionContext(
                classification=prediction,
                confidence=confidence_value,
                model_used="EfficientNet-B4",
                probabilities={"fake": fake_prob, "real": real_prob},
                region_labels=pipeline["region_labels"],
                region_categories=pipeline["region_categories"],
            )

            if vlm_factory is not None:
                # Retry on rate-limit fallbacks. OpenAI's per-minute caps are
                # easy to hit when precomputing 12 images back-to-back; the
                # provider returns a fallback summary rather than raising,
                # so we detect that here and back off.
                result = None
                for attempt in range(3):
                    candidate = await _run_openai_and_attach_region_comments(
                        vlm_factory,
                        image_bytes,
                        heatmap_bytes,
                        ela_bytes,
                        detection_context,
                        evidence_regions,
                        crops,
                        img_id,
                    )
                    if not candidate["summary"].startswith("Explanation unavailable"):
                        result = candidate
                        break
                    logger.warning(
                        "Image %s OpenAI fallback (%s) — attempt %d/3, backing off",
                        img_id,
                        candidate["summary"],
                        attempt + 1,
                    )
                    await asyncio.sleep(15 * (attempt + 1))
                if result is None:
                    result = candidate
                explanations = {"openai": result}
            else:
                explanations = {
                    "openai": {
                        "provider": "openai",
                        "model": "mock",
                        "summary": "VLM not configured.",
                        "detailed_analysis": "No API keys set.",
                        "technical_notes": None,
                        "processing_time_ms": 0,
                        "error": None,
                    }
                }

            analyses[img_id] = {
                "deepfake_score": round(fake_prob, 4),
                "classification": classification,
                "confidence": round(confidence_value, 4),
                "gradcam_url": heatmap_public_url,
                "ela_heatmap_url": ela_public_url,
                "evidence_regions": evidence_regions or [],
                "explanations": explanations,
            }

        except Exception as exc:
            logger.error("Failed to precompute image %s: %s", img_id, exc)
            analyses[img_id] = {"error": str(exc)}

    output = {
        "generated_at": datetime.now(UTC).isoformat(),
        "analyses": analyses,
    }

    output_path = _FRONTEND_PUBLIC / "study-analyses.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Precomputed analyses saved to %s", output_path)

    return {
        "status": "done",
        "images_processed": len(analyses),
        "output_path": str(output_path),
    }


@router.get("/results")
async def list_study_results():
    """Retrieve all saved participant results (researcher endpoint)."""
    path = _results_file()
    if not path.exists():
        return {"results": [], "count": 0}

    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return {"results": results, "count": len(results)}
