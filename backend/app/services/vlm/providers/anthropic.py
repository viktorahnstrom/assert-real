"""
Anthropic Claude VLM Provider
"""

import base64
import logging
import time

import anthropic

from app.services.vlm.base import BaseVLMProvider, DetectionContext, ProviderInfo, VLMExplanation
from app.services.vlm.config import ProviderConfig
from app.services.vlm.prompt_builder import (
    SYSTEM_PROMPT_WITH_GRADCAM,
    SYSTEM_PROMPT_WITHOUT_GRADCAM,
    build_explanation_prompt,
    parse_explanation_response,
)

logger = logging.getLogger(__name__)

# claude-haiku-4-5 pricing (per 1M tokens)
_INPUT_PRICE = 0.80
_OUTPUT_PRICE = 4.00


class AnthropicProvider(BaseVLMProvider):
    def __init__(self, config: ProviderConfig):
        self._config = config
        self._client = anthropic.Anthropic(api_key=config.api_key, timeout=config.timeout_seconds)
        self._model = config.model or "claude-haiku-4-5-20251001"

    async def generate_explanation(
        self,
        image_bytes: bytes,
        heatmap_bytes: bytes,
        detection: DetectionContext,
        gradcam_available: bool = True,
    ) -> VLMExplanation:
        start_time = time.time()

        system_prompt = (
            SYSTEM_PROMPT_WITH_GRADCAM if gradcam_available else SYSTEM_PROMPT_WITHOUT_GRADCAM
        )
        user_prompt = build_explanation_prompt(detection, gradcam_available=gradcam_available)

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        content: list = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            },
        ]
        if gradcam_available:
            heatmap_b64 = base64.b64encode(heatmap_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": heatmap_b64,
                    },
                }
            )
        content.append({"type": "text", "text": user_prompt})

        try:
            response = self._client.messages.create(
                model=self._model,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
                max_tokens=2000,
                temperature=0.3,
            )

            raw_text = response.content[0].text if response.content else ""
            processing_time = int((time.time() - start_time) * 1000)

            input_tokens = response.usage.input_tokens if response.usage else None
            output_tokens = response.usage.output_tokens if response.usage else None

            estimated_cost = self.estimate_cost(input_tokens or 0, output_tokens or 0)
            sections = parse_explanation_response(raw_text)

            return VLMExplanation(
                summary=sections["summary"],
                detailed_analysis=sections["detailed_analysis"],
                technical_notes=sections["technical_notes"] or None,
                region_comments=sections["region_comments"] or None,
                provider="anthropic",
                model=self._model,
                processing_time_ms=processing_time,
                estimated_cost_usd=estimated_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_response=raw_text,
            )

        except anthropic.RateLimitError as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"Anthropic rate limit exceeded: {e}")
            return VLMExplanation(
                summary="Explanation unavailable: Anthropic rate limit exceeded.",
                detailed_analysis="The explanation service is temporarily rate-limited. "
                "The detection result above is still valid. Please try again shortly.",
                technical_notes=None,
                region_comments=None,
                provider="anthropic",
                model=self._model,
                processing_time_ms=processing_time,
                estimated_cost_usd=0.0,
                raw_response=str(e),
            )
        except anthropic.APITimeoutError as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"Anthropic API timeout: {e}")
            return VLMExplanation(
                summary="Explanation unavailable: request timed out.",
                detailed_analysis="The explanation service took too long to respond. "
                "The detection result above is still valid.",
                technical_notes=None,
                region_comments=None,
                provider="anthropic",
                model=self._model,
                processing_time_ms=processing_time,
                estimated_cost_usd=0.0,
                raw_response=str(e),
            )
        except (anthropic.APIError, Exception) as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"Anthropic provider error: {e}")
            return VLMExplanation(
                summary="Explanation unavailable due to an error.",
                detailed_analysis="The explanation service encountered an error. "
                "The detection result above is still valid.",
                technical_notes=None,
                region_comments=None,
                provider="anthropic",
                model=self._model,
                processing_time_ms=processing_time,
                estimated_cost_usd=0.0,
                raw_response=str(e),
            )

    def get_provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            id="anthropic",
            name=f"Anthropic {self._model}",
            model=self._model,
            available=self._config.enabled,
            latency_estimate_ms=3000,
            cost_per_1m_input_tokens=_INPUT_PRICE,
            cost_per_1m_output_tokens=_OUTPUT_PRICE,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * _INPUT_PRICE + (
            output_tokens / 1_000_000
        ) * _OUTPUT_PRICE

    async def health_check(self) -> bool:
        try:
            response = self._client.messages.create(
                model=self._model,
                messages=[{"role": "user", "content": "Say 'ok'."}],
                max_tokens=5,
            )
            return bool(response.content)
        except Exception as e:
            logger.warning(f"Anthropic health check failed: {e}")
            return False
