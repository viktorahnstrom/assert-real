"""
Image upload and management routes for XADE.
"""

import hashlib
import os
import uuid

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

load_dotenv()

router = APIRouter(prefix="/api/v1/images", tags=["Images"])

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Allowed image types
ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


# ============================================
# Response Models
# ============================================
class ImageResponse(BaseModel):
    id: str
    filename: str
    storage_path: str
    file_size: int
    mime_type: str
    uploaded_at: str


class ImageListResponse(BaseModel):
    images: list[ImageResponse]
    count: int


# ============================================
# Helper functions
# ============================================
def get_storage_headers(access_token: str | None = None):
    """Headers for Supabase Storage API."""
    token = access_token or SUPABASE_SERVICE_ROLE_KEY
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
    }


def get_db_headers(access_token: str | None = None):
    """Headers for Supabase Database API."""
    token = access_token or SUPABASE_SERVICE_ROLE_KEY
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def calculate_checksum(data: bytes) -> str:
    """Calculate SHA-256 checksum of file data."""
    return hashlib.sha256(data).hexdigest()


# ============================================
# Routes
# ============================================
@router.post("/upload", response_model=ImageResponse)
async def upload_image(
    file: UploadFile = File(...),
    user_id: str | None = None,  # Will come from auth token later
):
    """
    Upload an image for deepfake analysis.
    - Validates file type and size
    - Uploads to Supabase Storage
    - Saves metadata to database
    """
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB",
        )

    # Generate unique filename
    file_ext = file.filename.split(".")[-1] if file.filename else "jpg"
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    storage_path = f"uploads/{unique_filename}"

    # Calculate checksum for deduplication
    checksum = calculate_checksum(content)

    # For now, use a placeholder user_id if not provided
    # In production, this comes from the authenticated user
    if not user_id:
        user_id = "00000000-0000-0000-0000-000000000000"

    async with httpx.AsyncClient() as client:
        # Upload to Supabase Storage
        upload_response = await client.post(
            f"{SUPABASE_URL}/storage/v1/object/images/{storage_path}",
            headers={
                **get_storage_headers(),
                "Content-Type": file.content_type,
            },
            content=content,
        )

        if upload_response.status_code not in [200, 201]:
            error = upload_response.json()
            raise HTTPException(
                status_code=upload_response.status_code,
                detail=error.get("message", "Failed to upload image"),
            )

        # Save metadata to database
        image_data = {
            "user_id": user_id,
            "storage_path": storage_path,
            "original_filename": file.filename or "unknown",
            "file_size_bytes": len(content),
            "mime_type": file.content_type,
            "checksum": checksum,
        }

        db_response = await client.post(
            f"{SUPABASE_URL}/rest/v1/images",
            headers=get_db_headers(),
            json=image_data,
        )

        if db_response.status_code not in [200, 201]:
            # Try to clean up the uploaded file
            await client.delete(
                f"{SUPABASE_URL}/storage/v1/object/images/{storage_path}",
                headers=get_storage_headers(),
            )
            error = db_response.json()
            raise HTTPException(
                status_code=db_response.status_code,
                detail=error.get("message", "Failed to save image metadata"),
            )

        saved_image = db_response.json()[0]

        return ImageResponse(
            id=saved_image["id"],
            filename=saved_image["original_filename"],
            storage_path=saved_image["storage_path"],
            file_size=saved_image["file_size_bytes"],
            mime_type=saved_image["mime_type"],
            uploaded_at=saved_image["uploaded_at"],
        )


@router.get("/", response_model=ImageListResponse)
async def list_images(user_id: str | None = None):
    """List all images for a user."""
    async with httpx.AsyncClient() as client:
        # Build query
        url = f"{SUPABASE_URL}/rest/v1/images?select=*&order=uploaded_at.desc"
        if user_id:
            url += f"&user_id=eq.{user_id}"

        response = await client.get(url, headers=get_db_headers())

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch images")

        images_data = response.json()

        images = [
            ImageResponse(
                id=img["id"],
                filename=img["original_filename"],
                storage_path=img["storage_path"],
                file_size=img["file_size_bytes"],
                mime_type=img["mime_type"],
                uploaded_at=img["uploaded_at"],
            )
            for img in images_data
        ]

        return ImageListResponse(images=images, count=len(images))


@router.get("/{image_id}", response_model=ImageResponse)
async def get_image(image_id: str):
    """Get a specific image by ID."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/images?id=eq.{image_id}&select=*",
            headers=get_db_headers(),
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch image")

        images = response.json()
        if not images:
            raise HTTPException(status_code=404, detail="Image not found")

        img = images[0]
        return ImageResponse(
            id=img["id"],
            filename=img["original_filename"],
            storage_path=img["storage_path"],
            file_size=img["file_size_bytes"],
            mime_type=img["mime_type"],
            uploaded_at=img["uploaded_at"],
        )


@router.delete("/{image_id}")
async def delete_image(image_id: str):
    """Delete an image and its storage file."""
    async with httpx.AsyncClient() as client:
        # First, get the image to find storage path
        get_response = await client.get(
            f"{SUPABASE_URL}/rest/v1/images?id=eq.{image_id}&select=*",
            headers=get_db_headers(),
        )

        if get_response.status_code != 200:
            raise HTTPException(
                status_code=get_response.status_code, detail="Failed to fetch image"
            )

        images = get_response.json()
        if not images:
            raise HTTPException(status_code=404, detail="Image not found")

        storage_path = images[0]["storage_path"]

        # Delete from storage
        await client.delete(
            f"{SUPABASE_URL}/storage/v1/object/images/{storage_path}",
            headers=get_storage_headers(),
        )

        # Delete from database
        db_response = await client.delete(
            f"{SUPABASE_URL}/rest/v1/images?id=eq.{image_id}",
            headers=get_db_headers(),
        )

        if db_response.status_code not in [200, 204]:
            raise HTTPException(
                status_code=db_response.status_code,
                detail="Failed to delete image record",
            )

        return {"message": "Image deleted successfully"}
