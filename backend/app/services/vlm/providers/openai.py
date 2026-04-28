"""
OpenAI VLM Provider
"""

import base64
import logging
import time

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from app.services.vlm.base import BaseVLMProvider, DetectionContext, ProviderInfo, VLMExplanation
from app.services.vlm.config import ProviderConfig
from app.services.vlm.prompt_builder import (
    SYSTEM_PROMPT_STRUCTURED,
    SYSTEM_PROMPT_WITH_GRADCAM,
    SYSTEM_PROMPT_WITHOUT_GRADCAM,
    build_explanation_prompt,
    parse_explanation_response,
    parse_structured_response,
    structured_to_legacy,
)
from app.services.vlm.structured_schema import openai_response_format

logger = logging.getLogger(__name__)

OPENAI_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}


class OpenAIProvider(BaseVLMProvider):
    def __init__(self, config: ProviderConfig):
        self._config = config
        self._client = OpenAI(api_key=config.api_key, timeout=config.timeout_seconds)
        self._model = config.model or "gpt-4o-mini"

    async def generate_explanation(
        self,
        image_bytes: bytes,
        heatmap_bytes: bytes,
        detection: DetectionContext,
        gradcam_available: bool = True,
        region_image_bytes: list[bytes] | None = None,
        ela_bytes: bytes | None = None,
    ) -> VLMExplanation:
        start_time = time.time()
        regions = region_image_bytes or []
        ela_available = ela_bytes is not None

        system_prompt = (
            SYSTEM_PROMPT_WITH_GRADCAM if gradcam_available else SYSTEM_PROMPT_WITHOUT_GRADCAM
        )
        user_prompt = build_explanation_prompt(
            detection,
            gradcam_available=gradcam_available,
            region_count=len(regions),
            ela_available=ela_available,
        )

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        content = [
            {"type": "input_text", "text": user_prompt},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
        ]
        if gradcam_available:
            heatmap_b64 = base64.b64encode(heatmap_bytes).decode("utf-8")
            content.append(
                {"type": "input_image", "image_url": f"data:image/png;base64,{heatmap_b64}"}
            )
        if ela_available:
            ela_b64 = base64.b64encode(ela_bytes).decode("utf-8")
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{ela_b64}"})
        for region_bytes in regions:
            region_b64 = base64.b64encode(region_bytes).decode("utf-8")
            content.append(
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{region_b64}"}
            )

        try:
            # Preferred path: strict json_schema response_format. The
            # Responses API rejects malformed output before it reaches us, so
            # parse_structured_response only fails if the model legitimately
            # produced nonsense. On hard failure we fall back to the legacy
            # text path so a single bad reply never breaks user-facing output.
            response_format = openai_response_format()
            response = self._client.responses.create(
                model=self._model,
                instructions=SYSTEM_PROMPT_STRUCTURED if gradcam_available else system_prompt,
                input=[{"role": "user", "content": content}],
                max_output_tokens=2000,
                temperature=0.3,
                text=response_format,
            )

            input_tokens, output_tokens = _extract_usage(response)
            estimated_cost = self.estimate_cost(input_tokens or 0, output_tokens or 0)

            raw_text = response.output_text or ""
            parsed = parse_structured_response(raw_text)

            if parsed is None:
                logger.warning("OpenAI structured response invalid; retrying once")
                response = self._client.responses.create(
                    model=self._model,
                    instructions=SYSTEM_PROMPT_STRUCTURED if gradcam_available else system_prompt,
                    input=[
                        {"role": "user", "content": content},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "Your previous reply did not match the "
                                        "required JSON schema. Re-emit the "
                                        "entire explanation as a single JSON "
                                        "object matching the schema, with all "
                                        "required fields populated."
                                    ),
                                }
                            ],
                        },
                    ],
                    max_output_tokens=2000,
                    temperature=0.3,
                    text=response_format,
                )
                retry_in, retry_out = _extract_usage(response)
                input_tokens = (input_tokens or 0) + (retry_in or 0)
                output_tokens = (output_tokens or 0) + (retry_out or 0)
                estimated_cost = self.estimate_cost(input_tokens or 0, output_tokens or 0)
                raw_text = response.output_text or ""
                parsed = parse_structured_response(raw_text)

            processing_time = int((time.time() - start_time) * 1000)

            if parsed is not None:
                legacy = structured_to_legacy(parsed)
                return VLMExplanation(
                    summary=legacy["summary"],
                    detailed_analysis=legacy["detailed_analysis"],
                    technical_notes=legacy["technical_notes"] or None,
                    region_comments=legacy["region_comments"] or None,
                    structured_regions=parsed["regions"],
                    provider="openai",
                    model=self._model,
                    processing_time_ms=processing_time,
                    estimated_cost_usd=estimated_cost,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    raw_response=raw_text,
                )

            logger.warning(
                "OpenAI structured output failed twice; falling back to free-text parser"
            )
            sections = parse_explanation_response(raw_text)

            return VLMExplanation(
                summary=sections["summary"],
                detailed_analysis=sections["detailed_analysis"],
                technical_notes=sections["technical_notes"] or None,
                region_comments=sections["region_comments"] or None,
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
                region_comments=None,
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
                region_comments=None,
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
                region_comments=None,
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
        return (input_tokens / 1_000_000) * pricing["input"] + (
            output_tokens / 1_000_000
        ) * pricing["output"]

    async def health_check(self) -> bool:
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


def _extract_usage(response) -> tuple[int | None, int | None]:
    """Pull (input_tokens, output_tokens) out of a Responses API result."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
