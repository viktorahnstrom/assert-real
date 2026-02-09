"""
XADE Pydantic Schemas

These schemas define the data structures for API requests/responses
and match the Supabase database tables.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# ============================================
# Enums
# ============================================
class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Classification(str, Enum):
    REAL = "real"
    FAKE = "fake"
    UNCERTAIN = "uncertain"


class VLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class Theme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


class ClientType(str, Enum):
    DESKTOP = "desktop"
    MOBILE = "mobile"
    API = "api"
    WEB = "web"


# ============================================
# Profile Schemas
# ============================================
class ProfileBase(BaseModel):
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None


class ProfileCreate(ProfileBase):
    email: EmailStr


class ProfileUpdate(ProfileBase):
    pass


class Profile(ProfileBase):
    id: UUID
    email: EmailStr
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================
# Image Schemas
# ============================================
class ImageBase(BaseModel):
    original_filename: str
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None


class ImageCreate(ImageBase):
    storage_path: str
    checksum: Optional[str] = None


class Image(ImageBase):
    id: UUID
    user_id: UUID
    storage_path: str
    checksum: Optional[str]
    uploaded_at: datetime

    class Config:
        from_attributes = True


# ============================================
# Analysis Schemas
# ============================================
class AnalysisBase(BaseModel):
    model_used: Optional[str] = "efficientnet-b4"


class AnalysisCreate(AnalysisBase):
    image_id: UUID


class AnalysisUpdate(BaseModel):
    status: Optional[AnalysisStatus] = None
    deepfake_score: Optional[Decimal] = Field(None, ge=0, le=1)
    classification: Optional[Classification] = None
    gradcam_path: Optional[str] = None
    vlm_explanation: Optional[str] = None
    vlm_model_used: Optional[str] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None


class Analysis(AnalysisBase):
    id: UUID
    image_id: UUID
    user_id: UUID
    status: AnalysisStatus
    deepfake_score: Optional[Decimal]
    classification: Optional[Classification]
    gradcam_path: Optional[str]
    vlm_explanation: Optional[str]
    vlm_model_used: Optional[str]
    processing_time_ms: Optional[int]
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ============================================
# Analysis Response (with image info)
# ============================================
class AnalysisWithImage(Analysis):
    """Analysis response including image metadata."""
    image: Image


# ============================================
# User Preferences Schemas
# ============================================
class UserPreferencesBase(BaseModel):
    default_model: Optional[str] = "efficientnet-b4"
    vlm_provider: Optional[VLMProvider] = VLMProvider.ANTHROPIC
    theme: Optional[Theme] = Theme.SYSTEM
    notifications_enabled: Optional[bool] = True


class UserPreferencesUpdate(UserPreferencesBase):
    pass


class UserPreferences(UserPreferencesBase):
    id: UUID
    user_id: UUID
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================
# API Log Schemas (mainly for internal use)
# ============================================
class APILogCreate(BaseModel):
    endpoint: str
    method: str
    status_code: Optional[int] = None
    response_time_ms: Optional[int] = None
    client_type: Optional[ClientType] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class APILog(APILogCreate):
    id: UUID
    user_id: Optional[UUID]
    created_at: datetime

    class Config:
        from_attributes = True
