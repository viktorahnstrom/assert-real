"""
VLM provider management endpoints.

Provides API routes for listing available VLM providers,
checking usage stats, and health checking providers.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.vlm import VLMProviderFactory

router = APIRouter()

# VLM factory reference — set by main.py on startup
vlm_factory: Optional[VLMProviderFactory] = None


# ============================================
# Response Models
# ============================================


class ProviderResponse(BaseModel):
    id: str
    name: str
    model: str
    available: bool
    latency_estimate_ms: Optional[int] = None
    cost_per_1m_input_tokens: Optional[float] = None
    cost_per_1m_output_tokens: Optional[float] = None


class ProvidersListResponse(BaseModel):
    providers: list[ProviderResponse]
    default_provider: str


class UsageResponse(BaseModel):
    daily_requests: dict[str, int]
    daily_total: int
    daily_limit: int
    monthly_cost_usd: dict[str, float]
    monthly_cost_total_usd: float
    monthly_cost_limit_usd: float
    total_requests_all_time: dict[str, int]


# ============================================
# Endpoints
# ============================================


@router.get("/vlm-providers", response_model=ProvidersListResponse)
async def list_vlm_providers():
    """
    List all available VLM providers and their configuration.

    Returns provider IDs, models, availability status, and pricing info.
    The frontend uses this to populate the provider selector dropdown.
    """
    if vlm_factory is None:
        raise HTTPException(status_code=503, detail="VLM service not initialized")

    providers = vlm_factory.list_providers()

    return ProvidersListResponse(
        providers=[
            ProviderResponse(
                id=p.id,
                name=p.name,
                model=p.model,
                available=p.available,
                latency_estimate_ms=p.latency_estimate_ms,
                cost_per_1m_input_tokens=p.cost_per_1m_input_tokens,
                cost_per_1m_output_tokens=p.cost_per_1m_output_tokens,
            )
            for p in providers
        ],
        default_provider=vlm_factory._config.default_provider,
    )


@router.get("/vlm-usage", response_model=UsageResponse)
async def get_vlm_usage():
    """
    Get current VLM usage statistics.

    Shows daily request counts, monthly costs, and configured limits.
    Useful for monitoring spending and debugging rate limit issues.
    """
    if vlm_factory is None:
        raise HTTPException(status_code=503, detail="VLM service not initialized")

    summary = vlm_factory.get_usage_summary()
    return UsageResponse(**summary)


@router.get("/vlm-health/{provider_id}")
async def check_vlm_health(provider_id: str):
    """
    Health check a specific VLM provider.

    Makes a minimal API call to verify the provider is reachable
    and properly configured. Returns status and any error details.
    """
    if vlm_factory is None:
        raise HTTPException(status_code=503, detail="VLM service not initialized")

    try:
        provider = vlm_factory.get_provider(provider_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    is_healthy = await provider.health_check()
    info = provider.get_provider_info()

    return {
        "provider_id": provider_id,
        "name": info.name,
        "model": info.model,
        "healthy": is_healthy,
        "status": "ok" if is_healthy else "unreachable",
    }
