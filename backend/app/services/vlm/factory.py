"""
VLM Provider Factory

Central registry that creates, caches, and manages VLM provider instances.
"""

import logging
from typing import Optional

from app.services.vlm.base import BaseVLMProvider, DetectionContext, ProviderInfo, VLMExplanation
from app.services.vlm.config import VLMConfig
from app.services.vlm.providers.mock import MockProvider
from app.services.vlm.providers.rule_based import RuleBasedProvider
from app.services.vlm.usage_tracker import UsageLimitExceeded, UsageTracker

logger = logging.getLogger(__name__)


class VLMProviderFactory:
    def __init__(self, config: VLMConfig):
        self._config = config
        self._providers: dict[str, BaseVLMProvider] = {}
        self._usage_tracker = UsageTracker(config)
        self._providers["mock"] = MockProvider()
        self._providers["rule_based"] = RuleBasedProvider()

    def get_provider(self, provider_id: Optional[str] = None) -> BaseVLMProvider:
        if provider_id is None:
            provider_id = self._config.default_provider

        if provider_id in self._providers:
            return self._providers[provider_id]

        provider = self._create_provider(provider_id)
        self._providers[provider_id] = provider
        return provider

    def _create_provider(self, provider_id: str) -> BaseVLMProvider:
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
                    "OpenAI provider is not configured. Set OPENAI_API_KEY environment variable."
                )
            from app.services.vlm.providers.openai import OpenAIProvider

            return OpenAIProvider(self._config.openai)

        elif provider_id == "anthropic":
            if not self._config.anthropic.enabled:
                raise ValueError(
                    "Anthropic provider is not configured. "
                    "Set ANTHROPIC_API_KEY environment variable."
                )
            from app.services.vlm.providers.anthropic import AnthropicProvider

            return AnthropicProvider(self._config.anthropic)

        elif provider_id == "mock":
            return MockProvider()

        elif provider_id == "rule_based":
            return RuleBasedProvider()

        else:
            raise ValueError(
                f"Unknown VLM provider: '{provider_id}'. "
                f"Available: google, openai, anthropic, mock, rule_based"
            )

    async def generate_explanation(
        self,
        provider_id: Optional[str],
        image_bytes: bytes,
        heatmap_bytes: bytes,
        detection: DetectionContext,
        gradcam_available: bool = True,
        region_image_bytes: list[bytes] | None = None,
        ela_bytes: bytes | None = None,
    ) -> VLMExplanation:
        """
        Generate an explanation using the specified provider.

        Args:
            provider_id: Which provider to use (None = default)
            image_bytes: Original image as bytes
            heatmap_bytes: GradCAM heatmap as bytes (only used when gradcam_available=True)
            detection: Detection results
            gradcam_available: Whether a real heatmap was generated
            region_image_bytes: Zoomed-in crop images for each detected region,
                                 sent to the VLM for close-up artifact inspection
            ela_bytes: Optional ELA overlay PNG bytes — passed alongside
                       detection.forensics_report so the VLM has both the
                       visual map and the per-region z-scores.
        """
        resolved_id = provider_id or self._config.default_provider

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

        provider = self.get_provider(resolved_id)
        explanation = await provider.generate_explanation(
            image_bytes,
            heatmap_bytes,
            detection,
            gradcam_available=gradcam_available,
            region_image_bytes=region_image_bytes,
            ela_bytes=ela_bytes,
        )

        if resolved_id != "mock":
            self._usage_tracker.record_usage(resolved_id, explanation.estimated_cost_usd)

        return explanation

    def list_providers(self) -> list[ProviderInfo]:
        providers = [
            MockProvider().get_provider_info(),
            RuleBasedProvider().get_provider_info(),
        ]

        if self._config.google.enabled:
            try:
                providers.append(self.get_provider("google").get_provider_info())
            except ValueError:
                pass

        if self._config.openai.enabled:
            try:
                providers.append(self.get_provider("openai").get_provider_info())
            except ValueError:
                pass

        if self._config.anthropic.enabled:
            try:
                providers.append(self.get_provider("anthropic").get_provider_info())
            except ValueError:
                pass

        return providers

    def get_usage_summary(self) -> dict:
        return self._usage_tracker.get_usage_summary()
