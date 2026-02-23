"""
Mock VLM Provider

Returns realistic-looking hardcoded explanations for testing.
No API calls, instant response. Useful for:
- Frontend development without API keys
- CI/CD pipeline testing
- Demonstrating the explanation format
"""

import random
import time

from app.services.vlm.base import BaseVLMProvider, DetectionContext, ProviderInfo, VLMExplanation

# Realistic mock responses for different detection outcomes
_FAKE_EXPLANATIONS = [
    {
        "summary": (
            "This image is likely AI-generated, with the detection model identifying "
            "subtle inconsistencies around the eyes and hairline boundary."
        ),
        "detailed_analysis": (
            "The GradCAM heatmap highlights strong activation around the eye region and "
            "the hairline-to-forehead transition zone. In the eye area, there are subtle "
            "asymmetries in the specular reflections that are uncommon in natural photographs. "
            "The hairline boundary shows an unusually smooth transition that lacks the fine "
            "stray hairs typically present in real photos. The model is {confidence}% confident "
            "in this classification."
        ),
        "technical_notes": (
            "Strongest model activations concentrated in the periocular region (around both eyes) "
            "and the forehead-hairline boundary. The activation pattern is consistent with "
            "GAN-generated imagery where fine detail synthesis is weakest. Note that this analysis "
            "is limited to the single frame provided and does not account for potential "
            "post-processing or compression artifacts."
        ),
    },
    {
        "summary": (
            "This image appears to be a deepfake, primarily due to artifacts detected "
            "around the jawline and skin texture inconsistencies."
        ),
        "detailed_analysis": (
            "The heatmap reveals concentrated attention along the jaw and chin contour, where "
            "the model detected blending artifacts typical of face-swap deepfakes. The skin "
            "texture in the highlighted cheek area shows an unnatural smoothness compared to "
            "the surrounding regions. Additionally, there are slight color discontinuities at "
            "the face-neck boundary that suggest compositing. The detection model rates this "
            "as {confidence}% likely to be manipulated."
        ),
        "technical_notes": (
            "The activation distribution follows a pattern typical of face-swap detections, "
            "with highest activations at face boundary regions rather than central facial features. "
            "This suggests the model is detecting blending seams rather than generative artifacts. "
            "Cross-dataset validation would strengthen confidence in this classification."
        ),
    },
]

_REAL_EXPLANATIONS = [
    {
        "summary": (
            "This image appears to be an authentic photograph, with natural features "
            "consistently present across all examined regions."
        ),
        "detailed_analysis": (
            "The GradCAM heatmap shows distributed attention across the face without "
            "concentrated hotspots that would indicate manipulation artifacts. The eye regions "
            "display natural specular reflections consistent with a single light source. Skin "
            "texture varies naturally across the face, with expected pore detail and subtle "
            "imperfections. The model is {confidence}% confident this is a genuine photograph."
        ),
        "technical_notes": (
            "Model activations are broadly distributed rather than concentrated, which is "
            "the typical pattern for authentic images. No anomalous frequency-domain artifacts "
            "were implied by the activation regions. However, sophisticated deepfakes may not "
            "trigger strong localized activations, so this result should be considered alongside "
            "other verification methods."
        ),
    },
]


class MockProvider(BaseVLMProvider):
    """Mock VLM provider that returns realistic explanations without API calls."""

    async def generate_explanation(
        self,
        image_bytes: bytes,
        heatmap_bytes: bytes,
        detection: DetectionContext,
    ) -> VLMExplanation:
        """Return a mock explanation based on the detection classification."""

        start_time = time.time()

        # Pick a random explanation matching the classification
        if detection.classification == "fake":
            template = random.choice(_FAKE_EXPLANATIONS)
        else:
            template = random.choice(_REAL_EXPLANATIONS)

        confidence_pct = f"{detection.confidence * 100:.1f}"

        processing_time = int((time.time() - start_time) * 1000)

        return VLMExplanation(
            summary=template["summary"],
            detailed_analysis=template["detailed_analysis"].format(confidence=confidence_pct),
            technical_notes=template["technical_notes"],
            provider="mock",
            model="mock-v1",
            processing_time_ms=processing_time,
            estimated_cost_usd=0.0,
            input_tokens=0,
            output_tokens=0,
            raw_response=None,
        )

    def get_provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            id="mock",
            name="Mock Provider (Testing)",
            model="mock-v1",
            available=True,
            latency_estimate_ms=1,
            cost_per_1m_input_tokens=0.0,
            cost_per_1m_output_tokens=0.0,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0
