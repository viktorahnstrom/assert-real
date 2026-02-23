"""
OpenAI VLM Provider

Implements the BaseVLMProvider for OpenAI's vision models.
Default model: gpt-4o-mini (cheapest: $0.15/M input, $0.60/M output)
Also supports: gpt-4o ($2.50/M input, $10.00/M output)

Uses the new Responses API (client.responses.create) with input_image type.
"""

import base64
import logging
import time

from openai import OpenAI
from openai import APIError, APITimeoutError, RateLimitError

from app.services.vlm.base import BaseVLMProvider, DetectionContext, ProviderInfo, VLMExplanation
from app.services.vlm.config import ProviderConfig
from app.services.vlm.prompt_builder import SYSTEM_PROMPT, build_explanation_prompt, parse_explanation_response

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (as of Feb 2026)
OPENAI_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}


class OpenAIProvider(BaseVLMProvider):
    """OpenAI VLM provider for generating deepfake explanations."""

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._client = OpenAI(api_key=config.api_key, timeout=config.timeout_seconds)
        self._model = config.model or "gpt-4o-mini"

    async def generate_explanation(
        self,
        image_bytes: bytes,
        heatmap_bytes: bytes,
        detection: DetectionContext,
    ) -> VLMExplanation:
        """Generate explanation using OpenAI Vision API."""

        start_time = time.time()
        user_prompt = build_explanation_prompt(detection)

        # Encode images as base64 data URLs
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        heatmap_b64 = base64.b64encode(heatmap_bytes).decode("utf-8")

        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=SYSTEM_PROMPT,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_b64}",
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{heatmap_b64}",
                            },
                        ],
                    }
                ],
                max_output_tokens=1000,
                temperature=0.3,
            )

            raw_text = response.output_text or ""
            processing_time = int((time.time() - start_time) * 1000)

            # Extract token usage from response
            input_tokens = None
            output_tokens = None
            if response.usage:
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

            estimated_cost = self.estimate_cost(input_tokens or 0, output_tokens or 0)

            # Parse the three-tier response
            sections = parse_explanation_response(raw_text)

            return VLMExplanation(
                summary=sections["summary"],
                detailed_analysis=sections["detailed_analysis"],
                technical_notes=sections["technical_notes"] or None,
                provider="openai",
                model=self._model,
                processing_time_ms=processing_time,
                estimated_cost_usd=estimated_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_response=raw_text,
            )

        except RateLimitError as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"OpenAI rate limit exceeded: {e}")
            return VLMExplanation(
                summary="Explanation unavailable: OpenAI rate limit exceeded.",
                detailed_analysis="The explanation service is temporarily rate-limited. "
                "The detection result above is still valid. Please try again shortly.",
                technical_notes=None,
                provider="openai",
                model=self._model,
                processing_time_ms=processing_time,
                estimated_cost_usd=0.0,
                raw_response=str(e),
            )
        except APITimeoutError as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"OpenAI API timeout: {e}")
            return VLMExplanation(
                summary="Explanation unavailable: request timed out.",
                detailed_analysis="The explanation service took too long to respond. "
                "The detection result above is still valid.",
                technical_notes=None,
                provider="openai",
                model=self._model,
                processing_time_ms=processing_time,
                estimated_cost_usd=0.0,
                raw_response=str(e),
            )
        except (APIError, Exception) as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"OpenAI provider error: {e}")
            return VLMExplanation(
                summary="Explanation unavailable due to an error.",
                detailed_analysis="The explanation service encountered an error. "
                "The detection result above is still valid.",
                technical_notes=None,
                provider="openai",
                model=self._model,
                processing_time_ms=processing_time,
                estimated_cost_usd=0.0,
                raw_response=str(e),
            )

    def get_provider_info(self) -> ProviderInfo:
        pricing = OPENAI_PRICING.get(self._model, OPENAI_PRICING["gpt-4o-mini"])
        return ProviderInfo(
            id="openai",
            name=f"OpenAI {self._model}",
            model=self._model,
            available=self._config.enabled,
            latency_estimate_ms=3000,
            cost_per_1m_input_tokens=pricing["input"],
            cost_per_1m_output_tokens=pricing["output"],
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = OPENAI_PRICING.get(self._model, OPENAI_PRICING["gpt-4o-mini"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    async def health_check(self) -> bool:
        """Check if OpenAI API is reachable."""
        try:
            response = self._client.responses.create(
                model=self._model,
                input="Say 'ok'.",
                max_output_tokens=5,
            )
            return response.output_text is not None
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
