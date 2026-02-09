"""
XADE Backend API
eXplainable Automated Deepfake Evaluation
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from postgrest import SyncPostgrestClient

from app.db import get_db, get_postgrest_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: verify database connection
    try:
        client = get_postgrest_client()
        client.from_("profiles").select("id").limit(1).execute()
        print("✓ Database connection successful")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("  (This is OK if you haven't set up .env yet)")
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
        "http://localhost:3000",  # Desktop dev
        "http://localhost:5173",  # Vite dev
        "http://localhost:8081",  # Expo dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {
        "name": "XADE Backend API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health")
async def health_check(db: SyncPostgrestClient = Depends(get_db)):
    """Health check endpoint - verifies API and database connectivity."""
    try:
        db.from_("profiles").select("id").limit(1).execute()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
    }


# TODO: Add routers for:
# - /api/v1/auth (login, register, logout)
# - /api/v1/images (upload, list, delete)
# - /api/v1/analyses (create, get, list)
# - /api/v1/preferences (get, update)
