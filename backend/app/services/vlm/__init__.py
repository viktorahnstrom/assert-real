"""
VLM Service Package
Vision-Language Model provider abstraction for generating
human-readable explanations of deepfake detection results.
"""

from app.services.vlm.base import BaseVLMProvider, DetectionContext, ProviderInfo, VLMExplanation
from app.services.vlm.config import VLMConfig, get_vlm_config
from app.services.vlm.factory import VLMProviderFactory
from app.services.vlm.usage_tracker import UsageLimitExceeded, UsageTracker

__all__ = [
    "BaseVLMProvider",
    "DetectionContext",
    "VLMExplanation",
    "ProviderInfo",
    "VLMConfig",
    "get_vlm_config",
    "VLMProviderFactory",
    "UsageTracker",
    "UsageLimitExceeded",
]
