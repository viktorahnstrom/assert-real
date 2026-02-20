"""
Base VLM Provider

Abstract base class that all VLM providers must implement.
Uses the stratergy so providers can be swapped wihout changing the rest of the codebase.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class VLMExplanation:
    """Structured explanation response from a VLM provider."""

    # Explanation fields
    summary: str
    detailed_analysis: str
    technical_notes: Optional[str] = None

    # Provider metadata
    provider: str = ""
    model: str = ""
    procesing_time_ms: int = 0

    # Cost tracking
    estimated_cost_usd: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Raw response for debugging
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
class DetectionContext:
    """Detection results passed to VLM for grounded explanations."""

    classification: str
    confidence: float
    model_used: str
    probabilities: dict =field(fdefault_factory=dict)


class BaseVLMProvider(ABC):
    """
    Abstract base class for vlm providers.

    All providers must implement generate_explanation() which takes
    the original image, GradCAM heatmap, and detection results, then
    returns a structured VLM Explanation with three tiers of detail.
    """

    @abstractmethod
    async def generate_explanation(
        self,
        image_bytes: bytes,
        heatmap_bytes: bytes,
        detection: DetectionContext,
    ) -> VLMExplanation:
        """
        Generate a structured explanation from the VLM provider.
        Args:
            image_bytes (bytes): The original image in bytes.
            heatmap_bytes (bytes): The GradCAM heatmap in bytes.
            detection (DetectionContext): The detection results to ground the explanation.

        Returns:
            VLMExplanation: A structured explanation with multiple tiers of detail.
        """
        ...


    @abstractmethod
    def get_provider_info(self) -> ProviderInfo:
        """Return metdadata about this provider (name, model, availability, costs)"""
        ...


    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate the cost of a request based on token usage.

        Args:
            input_tokens (int): Estimated number of input tokens.
            output_tokens (int): Estimated number of output tokens.

        Returns:
            float: Estimated cost in USD.
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if the provider is reachable and configured.
        Default implementation returns True — override for API-based providers.
        """
        return True
