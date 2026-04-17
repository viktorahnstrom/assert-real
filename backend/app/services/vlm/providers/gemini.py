"""
Google Gemini VLM Provider
"""

import logging
import time

from google import genai
from google.genai import types
from google.genai.errors import APIError

from app.services.vlm.base import BaseVLMProvider, DetectionContext, ProviderInfo, VLMExplanation
from app.services.vlm.config import ProviderConfig
from app.services.vlm.prompt_builder import (
    SYSTEM_PROMPT_WITH_GRADCAM,
    SYSTEM_PROMPT_WITHOUT_GRADCAM,
    build_explanation_prompt,
    parse_explanation_response,
)

logger = logging.getLogger(__name__)

GEMINI_PRICING = {
    "gemini-2.5-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
}


class GeminiProvider(BaseVLMProvider):
    def __init__(self, config: ProviderConfig):
        self._config = config
        self._client = genai.Client(api_key=config.api_key)
        self._model = config.model or "gemini-2.5-flash"

    async def generate_explanation(
        self,
        image_bytes: bytes,
        heatmap_bytes: bytes,
        detection: DetectionContext,
        gradcam_available: bool = True,
    ) -> VLMExplanation:
        import asyncio

        start_time = time.time()

        system_prompt = (
            SYSTEM_PROMPT_WITH_GRADCAM if gradcam_available else SYSTEM_PROMPT_WITHOUT_GRADCAM
        )
        user_prompt = build_explanation_prompt(detection, gradcam_available=gradcam_available)

        try:
            contents = [
                user_prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ]
            if gradcam_available:
                contents.append(types.Part.from_bytes(data=heatmap_bytes, mime_type="image/png"))

            # Run the synchronous SDK call in a thread so it doesn't block the event loop
            # (important when called in parallel with other providers via asyncio.gather)
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=2000,
                    temperature=0.3,
                ),
            )

            raw_text = response.text or ""
            processing_time = int((time.time() - start_time) * 1000)

            input_tokens = None
            output_tokens = None
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count

            estimated_cost = self.estimate_cost(input_tokens or 0, output_tokens or 0)
            sections = parse_explanation_response(raw_text)

            return VLMExplanation(
                summary=sections["summary"],
                detailed_analysis=sections["detailed_analysis"],
                technical_notes=sections["technical_notes"] or None,
                region_comments=sections["region_comments"] or None,
                provider="google",
                model=self._model,
                processing_time_ms=processing_time,
                estimated_cost_usd=estimated_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_response=raw_text,
            )

        except APIError as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"Gemini API error: {e}")
            return VLMExplanation(
                summary=f"Explanation unavailable: Gemini API error ({e.code})",
                detailed_analysis="The explanation service encountered an error. "
                "The detection result above is still valid.",
                technical_notes=None,
                region_comments=None,
                provider="google",
                model=self._model,
                processing_time_ms=processing_time,
                estimated_cost_usd=0.0,
                raw_response=str(e),
            )
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            logger.error(f"Unexpected error in Gemini provider: {e}")
            return VLMExplanation(
                summary="Explanation unavailable due to an unexpected error.",
                detailed_analysis="The explanation service encountered an error. "
                "The detection result above is still valid.",
                technical_notes=None,
                region_comments=None,
                provider="google",
                model=self._model,
                processing_time_ms=processing_time,
                estimated_cost_usd=0.0,
                raw_response=str(e),
            )

    def get_provider_info(self) -> ProviderInfo:
        pricing = GEMINI_PRICING.get(self._model, GEMINI_PRICING["gemini-2.5-flash"])
        return ProviderInfo(
            id="google",
            name=f"Google {self._model}",
            model=self._model,
            available=self._config.enabled,
            latency_estimate_ms=2000,
            cost_per_1m_input_tokens=pricing["input"],
            cost_per_1m_output_tokens=pricing["output"],
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = GEMINI_PRICING.get(self._model, GEMINI_PRICING["gemini-2.5-flash"])
        return (input_tokens / 1_000_000) * pricing["input"] + (
            output_tokens / 1_000_000
        ) * pricing["output"]

    async def health_check(self) -> bool:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents="Say 'ok'.",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            return response.text is not None
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            return False
