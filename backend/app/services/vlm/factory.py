"""
VLM Provider Factory

Central registry that creates, caches, and manages VLM provider instances.
This is the main entry point for the rest of the application to interact
with VLM providers.

Usage:
    factory = VLMProviderFactory(config)
    explanation = await factory.generate_explanation("google", image, heatmap, detection)
"""

import logging
from typing import Optional

from app.services.vlm.base import BaseVLMProvider, DetectionContext, ProviderInfo, VLMExplanation
from app.services.vlm.config import VLMConfig
from app.services.vlm.providers.mock import MockProvider
from app.services.vlm.usage_tracker import UsageLimitExceeded, UsageTracker

logger = logging.getLogger(__name__)


class VLMProviderFactory:
    """
    Factory for creating and managing VLM provider instances.

    Lazily initializes providers on first use and caches them.
    Integrates with UsageTracker to enforce rate and cost limits.
    """

    def __init__(self, config: VLMConfig):
        self._config = config
        self._providers: dict[str, BaseVLMProvider] = {}
        self._usage_tracker = UsageTracker(config)

        # Mock provider is always available
        self._providers["mock"] = MockProvider()

    def get_provider(self, provider_id: Optional[str] = None) -> BaseVLMProvider:
        """
        Get a VLM provider by ID, creating it if needed.

        Args:
            provider_id: Provider ID ("google", "openai", "mock").
                         Falls back to configured default if None.

        Returns:
            BaseVLMProvider instance

        Raises:
            ValueError: If provider is not available or not configured
        """
        if provider_id is None:
            provider_id = self._config.default_provider

        # Return cached provider if exists
        if provider_id in self._providers:
            return self._providers[provider_id]

        # Lazily create the provider
        provider = self._create_provider(provider_id)
        self._providers[provider_id] = provider
        return provider

    def _create_provider(self, provider_id: str) -> BaseVLMProvider:
        """Create a new provider instance."""

        if provider_id == "google":
            if not self._config.google.enabled:
                raise ValueError(
                    "Google Gemini provider is not configured. "
                    "Set GOOGLE_GEMINI_API_KEY environment variable."
                )
            from app.services.vlm.providers.gemini import GeminiProvider

            return GeminiProvider(self._config.google)

        elif provider_id == "openai":
            if not self._config.openai.enabled:
                raise ValueError(
                    "OpenAI provider is not configured. "
                    "Set OPENAI_API_KEY environment variable."
                )
            from app.services.vlm.providers.openai import OpenAIProvider

            return OpenAIProvider(self._config.openai)

        elif provider_id == "mock":
            return MockProvider()

        else:
            raise ValueError(
                f"Unknown VLM provider: '{provider_id}'. "
                f"Available providers: google, openai, mock"
            )

    async def generate_explanation(
        self,
        provider_id: Optional[str],
        image_bytes: bytes,
        heatmap_bytes: bytes,
        detection: DetectionContext,
    ) -> VLMExplanation:
        """
        Generate an explanation using the specified provider.

        This is the main method the rest of the application should call.
        It handles provider selection, usage tracking, and error handling.

        Args:
            provider_id: Which provider to use (None = default)
            image_bytes: Original image as bytes
            heatmap_bytes: GradCAM heatmap as bytes
            detection: Detection results

        Returns:
            VLMExplanation with structured explanation
        """
        resolved_id = provider_id or self._config.default_provider

        # Check usage limits before making the call
        if resolved_id != "mock":
            try:
                self._usage_tracker.check_limits(resolved_id)
            except UsageLimitExceeded as e:
                logger.warning(f"Usage limit exceeded for {resolved_id}: {e}")
                return VLMExplanation(
                    summary=f"Explanation unavailable: {e.limit_type} limit reached.",
                    detailed_analysis=(
                        f"The VLM usage limit has been reached "
                        f"({e.limit_type}: {e.current:.2f}/{e.maximum:.2f}). "
                        f"The detection result above is still valid. "
                        f"Try again later or use a different provider."
                    ),
                    technical_notes=None,
                    provider=resolved_id,
                    model="n/a",
                    processing_time_ms=0,
                    estimated_cost_usd=0.0,
                )

        # Get the provider and generate explanation
        provider = self.get_provider(resolved_id)
        explanation = await provider.generate_explanation(
            image_bytes, heatmap_bytes, detection
        )

        # Record usage after successful call
        if resolved_id != "mock":
            self._usage_tracker.record_usage(resolved_id, explanation.estimated_cost_usd)

        return explanation

    def list_providers(self) -> list[ProviderInfo]:
        """List all available providers and their status."""
        providers = []

        # Always include mock
        providers.append(MockProvider().get_provider_info())

        # Check configured providers
        if self._config.google.enabled:
            try:
                provider = self.get_provider("google")
                providers.append(provider.get_provider_info())
            except ValueError:
                pass

        if self._config.openai.enabled:
            try:
                provider = self.get_provider("openai")
                providers.append(provider.get_provider_info())
            except ValueError:
                pass

        return providers

    def get_usage_summary(self) -> dict:
        """Get current usage stats from the tracker."""
        return self._usage_tracker.get_usage_summary()
