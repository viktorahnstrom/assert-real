"""
Authentication dependency for XADE.

Provides a reusable FastAPI dependency that validates a Supabase JWT
on every request. Import `require_auth` and add it as a Depends() argument
to any endpoint that should be protected.
"""

import logging
import os

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_security = HTTPBearer()


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> dict:
    """
    FastAPI dependency — validates a Bearer token against Supabase Auth.

    Usage:
        @router.post("/my-endpoint")
        async def my_endpoint(current_user: dict = Depends(require_auth)):
            user_id = current_user["id"]

    Returns the Supabase user payload dict on success.
    Raises HTTP 401 if the token is missing, expired, or invalid.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_anon_key:
        logger.error("SUPABASE_URL or SUPABASE_ANON_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not configured.",
        )

    token = credentials.credentials

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{supabase_url}/auth/v1/user",
                headers={
                    "apikey": supabase_anon_key,
                    "Authorization": f"Bearer {token}",
                },
            )
    except httpx.RequestError as exc:
        logger.error("Failed to reach Supabase Auth during token validation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unreachable. Please try again.",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return response.json()
