"""
Base VLM Provider

Abstract base class that all VLM providers must implement.
Uses the strategy pattern so providers can be swapped without changing the rest of the codebase.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.services.forensics.report import ForensicsReport


@dataclass
class VLMExplanation:
    """Structured explanation response from a VLM provider."""

    summary: str
    detailed_analysis: str
    technical_notes: Optional[str] = None

    # Region-level comments keyed by region label
    # e.g. {"Nose and mid-face region": "The bright red activation here..."}
    region_comments: Optional[dict] = None

    # Per-region structured records when the provider used the JSON-schema
    # path. Each entry has keys: region, observation, evidence_type
    # ("visual" | "metric" | "heatmap"), evidence_ref, confidence. Surfaced
    # alongside region_comments so the frontend can show per-claim evidence
    # tags without breaking callers that only consume region_comments.
    structured_regions: Optional[list[dict]] = None

    provider: str = ""
    model: str = ""
    processing_time_ms: int = 0

    estimated_cost_usd: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    raw_response: Optional[str] = None


@dataclass
class ProviderInfo:
    """Metadata about a VLM provider's availability and capabilities."""

    id: str
    name: str
    model: str
    available: bool = False
    latency_estimate_ms: Optional[int] = None
    cost_per_1m_input_tokens: Optional[float] = None
    cost_per_1m_output_tokens: Optional[float] = None


@dataclass
class RegionWithCategory:
    """A GradCAM region label enriched with face category metadata.

    Produced by pairing a raw region label from gradcam_service with its
    matching FaceCategory via categories.get_category_for_label().

    Attributes:
        label: Original free-text label from GradCAM (e.g. "Left eye region").
        category_id: Stable snake_case category key (e.g. "eyes_pupils").
        category_label: Human-readable display label (e.g. "Eyes & Pupils").
        common_artifacts: First N artifact descriptions from the FaceCategory,
            used as guidance hints in VLM prompts.
    """

    label: str
    category_id: str
    category_label: str
    common_artifacts: tuple[str, ...]
    activation_score: float = 0.0


@dataclass
class DetectionContext:
    """Detection results passed to VLM for grounded explanations."""

    classification: str
    confidence: float
    model_used: str
    probabilities: dict = field(default_factory=dict)
    # Labeled facial regions from GradCAM evidence crops
    # e.g. ["Nose and mid-face region", "Left eye region"]
    region_labels: list = field(default_factory=list)
    # Category-enriched versions of region_labels — populated when
    # categories.get_category_for_label() resolves a label successfully.
    # Providers and prompt_builder prefer this over plain region_labels.
    region_categories: list[RegionWithCategory] = field(default_factory=list)
    # Per-region forensic metrics (sharpness, HF energy, ELA intensity).
    # When present, prompt_builder emits a [FORENSIC EVIDENCE] block that the
    # VLM can cite by name; absence falls back to today's CAM-only behavior.
    forensics_report: Optional["ForensicsReport"] = None


class BaseVLMProvider(ABC):
    """
    Abstract base class for VLM providers.

    All providers must implement generate_explanation() which takes
    the original image, GradCAM heatmap, and detection results, then
    returns a structured VLMExplanation with three tiers of detail
    plus optional per-region comments.
    """

    @abstractmethod
    async def generate_explanation(
        self,
        image_bytes: bytes,
        heatmap_bytes: bytes,
        detection: DetectionContext,
        gradcam_available: bool = True,
        region_image_bytes: list[bytes] | None = None,
        ela_bytes: bytes | None = None,
    ) -> VLMExplanation:
        """
        Generate a structured explanation from the VLM provider.

        Args:
            image_bytes: The original image in bytes.
            heatmap_bytes: The GradCAM heatmap in bytes. Only sent to the
                           VLM when gradcam_available is True.
            detection: The detection results to ground the explanation.
                       Includes region_labels from GradCAM evidence crops and,
                       when available, a forensics_report whose metrics the
                       prompt builder turns into a [FORENSIC EVIDENCE] block.
            gradcam_available: Whether heatmap_bytes contains a real heatmap.
            region_image_bytes: Optional list of zoomed-in region crop images
                                in the same order as detection.region_categories.
                                When provided, each crop is sent to the VLM as
                                a separate image so it can inspect the artifact
                                up close rather than inferring from the full image.
            ela_bytes: Optional ELA (Error Level Analysis) overlay image in PNG
                       bytes. Sent to the VLM after the GradCAM heatmap and
                       before the region crops so the model can ground claims
                       in visible recompression residuals.

        Returns:
            VLMExplanation with summary, detailed analysis, technical notes,
            and per-region comments.
        """
        ...

    @abstractmethod
    def get_provider_info(self) -> ProviderInfo:
        """Return metadata about this provider."""
        ...

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate the cost of a request based on token usage."""
        ...

    async def health_check(self) -> bool:
        """Check if the provider is reachable and configured."""
        return True
