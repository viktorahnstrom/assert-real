from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="XADE Backend API",
    description="API for the XADE backend services.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
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
async def health_check():
    return {"status": "healthy"}