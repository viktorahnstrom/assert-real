"""
Authentication routes for XADE.
Uses Supabase Auth via HTTP.
"""

import os

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

load_dotenv()

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")


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
# Helper function
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
                access_token=data.get("access_token"),
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
async def logout():
    """Logout - client should discard the token."""
    return {"message": "Logged out successfully. Please discard your access token."}


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
async def get_current_user(authorization: str | None = None):
    """Get the currently logged in user. Pass token in Authorization header."""
    # For now, return a message explaining how to use this endpoint
    return {
        "message": "Pass your access_token in the Authorization header as 'Bearer <token>'",
        "example": "Authorization: Bearer eyJhbGciOiJIUzI1NiIs...",
    }
