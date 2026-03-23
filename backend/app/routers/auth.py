"""
Authentication routes for XADE.
Uses Supabase Auth via HTTP.
"""

import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr

from app.dependencies.auth import require_auth

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

_security = HTTPBearer()


# ============================================
# Request/Response Models
# ============================================
class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class AuthResponse(BaseModel):
    message: str
    user_id: str | None = None
    email: str | None = None
    access_token: str | None = None


# ============================================
# Helper
# ============================================
def get_auth_headers():
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }


# ============================================
# Routes
# ============================================
@router.post("/signup", response_model=AuthResponse)
async def sign_up(request: SignUpRequest):
    """Create a new user account."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/auth/v1/signup",
            headers=get_auth_headers(),
            json={
                "email": request.email,
                "password": request.password,
                "data": {"display_name": request.display_name},
            },
        )

        if response.status_code == 200:
            data = response.json()
            return AuthResponse(
                message="Account created successfully. Please check your email to verify.",
                user_id=data.get("user", {}).get("id"),
                email=data.get("user", {}).get("email"),
                # Do NOT return access_token — require email verification before login
                access_token=None,
            )
        else:
            error = response.json()
            raise HTTPException(
                status_code=response.status_code,
                detail=error.get("error_description", error.get("msg", "Signup failed")),
            )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """Login with email and password."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            headers=get_auth_headers(),
            json={
                "email": request.email,
                "password": request.password,
            },
        )

        if response.status_code == 200:
            data = response.json()
            return AuthResponse(
                message="Login successful",
                user_id=data.get("user", {}).get("id"),
                email=data.get("user", {}).get("email"),
                access_token=data.get("access_token"),
            )
        else:
            error = response.json()
            raise HTTPException(
                status_code=401,
                detail=error.get("error_description", "Invalid credentials"),
            )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
):
    """
    Invalidates the session server-side via Supabase.
    The client should also discard the token locally.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/auth/v1/logout",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {credentials.credentials}",
                "Content-Type": "application/json",
            },
        )

    # Supabase returns 204 No Content on success
    if response.status_code not in (200, 204):
        logger.warning("Supabase logout returned unexpected status: %s", response.status_code)
        # Still return success to the client — the token may already be expired
    
    return {"message": "Logged out successfully."}


@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    """Send a password reset email."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/auth/v1/recover",
            headers=get_auth_headers(),
            json={"email": request.email},
        )

        if response.status_code == 200:
            return {"message": "Password reset email sent. Please check your inbox."}
        else:
            error = response.json()
            raise HTTPException(
                status_code=response.status_code,
                detail=error.get("error_description", "Failed to send reset email"),
            )


@router.get("/me")
async def get_current_user(current_user: dict = Depends(require_auth)):
    """
    Returns the authenticated user's profile.
    Requires a valid Bearer token — returns 401 otherwise.
    """
    return {
        "user_id": current_user.get("id"),
        "email": current_user.get("email"),
        "display_name": current_user.get("user_metadata", {}).get("display_name"),
        "email_confirmed": current_user.get("email_confirmed_at") is not None,
    }
