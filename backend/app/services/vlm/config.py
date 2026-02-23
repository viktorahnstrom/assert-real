"""
VLM Configuration

Loads providers settings, API keys, and usage limits frome environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProviderConfig:
    """Configuration for a single VLM provider."""

    api_key: Optional[str] = None
    model: str = ""
    enabled: bool = False
    timeout_seconds: int = 30


@dataclass
class VLMConfig:
    """Global VLM configuration loaded from enironment variables."""

    # Default provider to use when none is specified
    default_provider: str = "google"

    # Usage limits
    max_requests_per_day: int = 500
    max_monthly_cost_usd: float = 5.00

    # Provider-specific configs
    google: ProviderConfig = field(default_factory=ProviderConfig)
    openai: ProviderConfig = field(default_factory=ProviderConfig)
    anthropic: ProviderConfig = field(default_factory=ProviderConfig)


def get_vlm_config() -> VLMConfig:
    """Load VLM configuration from environment avariables."""

    config = VLMConfig(
        default_provider=os.getenv("VLM_DEFAULT_PROVIDER", "google"),
        max_requests_per_day=int(os.getenv("VLM_MAX_REQUESTS_PER_DAY", "500")),
        max_monthly_cost_usd=float(os.getenv("VLM_MAX_MONTHLY_COST_USD", "5.00")),
    )

    # Google Gemini
    google_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    config.google = ProviderConfig(
        api_key=google_key,
        model=os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash"),
        enabled=google_key is not None,
        timeout_seconds=int(os.getenv("GOOGLE_GEMINI_TIMEOUT_SECONDS", "30")),
    )

    # OpenAI
    openapi_key = os.getenv("OPENAI_API_KEY")
    config.openai = ProviderConfig(
        api_key=openapi_key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        enabled=openapi_key is not None,
        timeout_seconds=int(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
    )

    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    config.anthropic = ProviderConfig(
        api_key=anthropic_key,
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        enabled=anthropic_key is not None,
        timeout_seconds=int(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "30")),
    )

    return config
