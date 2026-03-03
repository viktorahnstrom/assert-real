"""
GradCAM heatmap storage utilities for XADE.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

_TEMP_DIR = Path(tempfile.gettempdir()) / "xade_gradcam"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)

GRADCAM_SERVE_URL = "http://localhost:8000/gradcam"


def save_heatmap_locally(
    overlay_image: Image.Image,
    image_id: Optional[str] = None,
    fmt: str = "JPEG",
) -> str:
    """Save a GradCAM overlay image to the local temp directory."""
    file_id = image_id or str(uuid.uuid4())
    ext = "jpg" if fmt.upper() in ("JPEG", "JPG") else fmt.lower()
    filename = f"gradcam_{file_id}.{ext}"
    filepath = _TEMP_DIR / filename
    overlay_image.save(filepath, format=fmt, quality=90)
    return str(filepath)


def get_local_heatmap_url(filepath: str) -> str:
    """Return an HTTP URL to the heatmap served via FastAPI static files."""
    filename = Path(filepath).name
    return f"{GRADCAM_SERVE_URL}/{filename}"


async def upload_heatmap_to_supabase(
    overlay_image: Image.Image,
    image_id: str,
    bucket: str = "heatmaps",
) -> str:
    raise NotImplementedError("Supabase Storage upload not yet implemented.")
