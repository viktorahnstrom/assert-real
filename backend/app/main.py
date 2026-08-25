"""Assert Real — backend API."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import detect, vlm
from app.db import get_postgrest_client
from app.rate_limit import limiter
from app.routers import analyses, auth, images, study
from app.services.gradcam_storage import GRADCAM_DIR
from app.services.vlm import VLMProviderFactory, get_vlm_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Assert Real backend")

    try:
        client = get_postgrest_client()
        client.from_("profiles").select("id").limit(1).execute()
        logger.info("Database connection successful")
    except Exception as e:
        logger.warning("Database connection failed: %s", e)

    # Load detection model
    detect.load_detection_model()

    # Initialize face category mapper (MediaPipe Face Mesh)
    try:
        from app.services.face_category_mapper import FaceCategoryMapper

        detect.face_category_mapper = FaceCategoryMapper()
        logger.info("Face category mapper initialised (MediaPipe Face Mesh)")
    except Exception as e:
        logger.warning("Face category mapper failed to initialise: %s", e)

    # Initialize BiSeNet face parser (pixel-accurate masks for 19 face classes)
    try:
        from app.services.face_parser import FaceParser

        detect.face_parser = FaceParser()
        logger.info("Face parser initialised (BiSeNet)")
    except Exception as e:
        logger.warning("Face parser failed to initialise: %s", e)

    # Initialize VLM provider factory
    try:
        vlm_config = get_vlm_config()
        factory = VLMProviderFactory(vlm_config)

        detect.vlm_factory = factory
        vlm.vlm_factory = factory

        logger.info("VLM service initialized (default: %s)", vlm_config.default_provider)

        for p in factory.list_providers():
            status = "available" if p.available else "not configured"
            logger.info("  %s: %s — %s", p.id, p.name, status)

        logger.info(
            "  Limits: %d req/day, $%.2f/month",
            vlm_config.max_requests_per_day,
            vlm_config.max_monthly_cost_usd,
        )

    except Exception as e:
        logger.warning("VLM service initialization failed: %s", e)

    yield

    logger.info("Shutting down Assert Real backend")
    if detect.face_category_mapper is not None:
        detect.face_category_mapper.close()


app = FastAPI(
    title="Assert Real API",
    description="Deepfake detection and explainability API",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS_ORIGINS: comma-separated list of allowed origins.
# Local dev defaults are always included; set CORS_ORIGINS in production
# to the public domain, e.g. "https://assertreal.dev".
_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8081",
]
_extra = os.getenv("CORS_ORIGINS", "")
if _extra:
    _CORS_ORIGINS.extend(o.strip() for o in _extra.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve GradCAM heatmaps as static files
app.mount("/gradcam", StaticFiles(directory=str(GRADCAM_DIR)), name="gradcam")

# Include routers
app.include_router(auth.router)
app.include_router(images.router)
app.include_router(detect.router, prefix="/api", tags=["detection"])
app.include_router(analyses.router)
app.include_router(vlm.router, prefix="/api", tags=["vlm"])

if os.getenv("ENABLE_STUDY_ROUTER", "false").lower() in ("true", "1", "yes"):
    app.include_router(study.router)


# Serve the built React frontend when running in single-container mode
# (e.g. Hugging Face Spaces). Disabled by default so local dev and the
# VPS path (separate nginx container) are unaffected.
_FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", "/app/frontend"))
if os.getenv("SERVE_FRONTEND", "false").lower() in ("true", "1", "yes") and _FRONTEND_DIR.is_dir():

    @app.get("/")
    async def serve_index():
        return FileResponse(_FRONTEND_DIR / "index.html")

    # SPA fallback: any path not matched by API routers or /gradcam returns
    # index.html so React Router can handle client-side routes.
    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        file_path = _FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_FRONTEND_DIR / "index.html")

else:

    @app.get("/")
    async def read_root():
        return {
            "name": "Assert Real API",
            "version": "0.1.0",
            "status": "running",
        }


@app.get("/health")
async def liveness():
    """Lightweight liveness probe for Docker HEALTHCHECK / load balancers."""
    return {"status": "ok"}


@app.get("/api/v1/health")
async def readiness():
    """Readiness check — reports dependency status for deploy verification."""
    try:
        client = get_postgrest_client()
        client.from_("profiles").select("id").limit(1).execute()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    model_status = "loaded" if detect.model is not None else "not_loaded"
    vlm_status = "initialized" if detect.vlm_factory is not None else "not_initialized"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "detection_model": model_status,
        "vlm_service": vlm_status,
    }
