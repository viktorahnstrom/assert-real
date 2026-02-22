"""
XADE Backend API
eXplainable Automated Deepfake Evaluation
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import get_postgrest_client
from app.routers import auth, images


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
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
app.include_router(images.router)


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
    try:
        client = get_postgrest_client()
        client.from_("profiles").select("id").limit(1).execute()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
    }
