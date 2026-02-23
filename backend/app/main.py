"""
XADE Backend API
eXplainable Automated Deepfake Evaluation
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import detect, vlm
from app.db import get_postgrest_client
from app.routers import auth
from app.services.vlm import VLMProviderFactory, get_vlm_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("🚀 Starting XADE Backend...")

    # Database connection (optional)
    try:
        client = get_postgrest_client()
        client.from_("profiles").select("id").limit(1).execute()
        print("✓ Database connection successful")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("  (This is OK if you haven't set up .env yet)")

    # Load detection model
    detect.load_detection_model()

    # Initialize VLM provider factory
    try:
        vlm_config = get_vlm_config()
        factory = VLMProviderFactory(vlm_config)

        # Share factory with API modules
        detect.vlm_factory = factory
        vlm.vlm_factory = factory

        print(f"✓ VLM service initialized (default: {vlm_config.default_provider})")

        # Log available providers
        providers = factory.list_providers()
        for p in providers:
            status = "✓ available" if p.available else "✗ not configured"
            print(f"  {p.id}: {p.name} — {status}")

        print(f"  Limits: {vlm_config.max_requests_per_day} req/day, "
              f"${vlm_config.max_monthly_cost_usd:.2f}/month")

    except Exception as e:
        print(f"✗ VLM service initialization failed: {e}")
        print("  (Explanations will not be available)")

    yield
    print("👋 Shutting down XADE backend...")


app = FastAPI(
    title="XADE Backend API",
    description="API for eXplainable Automated Deepfake Evaluation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8081",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(detect.router, prefix="/api", tags=["detection"])
app.include_router(vlm.router, prefix="/api", tags=["vlm"])


@app.get("/")
async def read_root():
    return {
        "name": "XADE Backend API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    # Check database
    try:
        client = get_postgrest_client()
        client.from_("profiles").select("id").limit(1).execute()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    # Check detection model
    model_status = "loaded" if detect.model is not None else "not_loaded"

    # Check VLM service
    vlm_status = "initialized" if detect.vlm_factory is not None else "not_initialized"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "detection_model": model_status,
        "vlm_service": vlm_status,
    }
