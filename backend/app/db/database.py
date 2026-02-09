"""
XADE Database Configuration

Uses PostgREST client for database operations with Supabase.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from postgrest import SyncPostgrestClient

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL", "")
        self.supabase_anon_key = os.getenv("SUPABASE_ANON_KEY", "")
        self.supabase_service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        self.environment = os.getenv("ENVIRONMENT", "development")

        if not self.supabase_url or not self.supabase_anon_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_postgrest_client() -> SyncPostgrestClient:
    """
    Get PostgREST client for database operations.
    Uses anon key - respects Row Level Security.
    """
    settings = get_settings()
    rest_url = f"{settings.supabase_url}/rest/v1"
    return SyncPostgrestClient(
        base_url=rest_url,
        headers={
            "apikey": settings.supabase_anon_key,
            "Authorization": f"Bearer {settings.supabase_anon_key}",
        },
    )


def get_postgrest_admin_client() -> SyncPostgrestClient:
    """
    Get PostgREST client with service role key.
    Bypasses Row Level Security - use only for admin operations.
    """
    settings = get_settings()
    rest_url = f"{settings.supabase_url}/rest/v1"
    return SyncPostgrestClient(
        base_url=rest_url,
        headers={
            "apikey": settings.supabase_service_role_key,
            "Authorization": f"Bearer {settings.supabase_service_role_key}",
        },
    )


# Dependency for FastAPI
async def get_db() -> SyncPostgrestClient:
    """FastAPI dependency that provides a PostgREST client."""
    return get_postgrest_client()
