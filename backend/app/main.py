"""
XADE Backend API
eXplainable Automated Deepfake Evaluation
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import detect


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: verify database connection (optional)
    try:
        from app.db import get_postgrest_client

        client = get_postgrest_client()
        client.from_("profiles").select("id").limit(1).execute()
        print("✓ Database connection successful")
    except ValueError:
        print("✗ Database not configured (missing .env)")
        print("  Detection endpoints will work without database")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")

    # Load detection model
    detect.load_detection_model()

    yield
    print("Shutting down XADE backend...")


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

# Include detection router
app.include_router(detect.router, prefix="/api", tags=["detection"])


@app.get("/")
async def read_root():
    return {
        "name": "XADE Backend API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "detect": "/api/detect",
            "model_info": "/api/model-info",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint - verifies API, database, and model status."""
    # Database check (won't crash if not configured)
    db_status = "not_configured"
    try:
        from app.db import get_postgrest_client

        client = get_postgrest_client()
        client.from_("profiles").select("id").limit(1).execute()
        db_status = "healthy"
    except ValueError:
        db_status = "not_configured"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"

    # Model check
    model_status = "loaded" if detect.model is not None else "not_loaded"

    return {
        "status": "operational",
        "database": db_status,
        "detection_model": model_status,
    }


# TODO: Add routers for:
# - /api/v1/auth (login, register, logout)
# - /api/v1/images (upload, list, delete)
# - /api/v1/analyses (create, get, list)
# - /api/v1/preferences (get, update)
