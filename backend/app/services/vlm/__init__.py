"""
VLM Service Package
Vision-Language Model provider abstraction for generating
human-readable explanations of deepfake detection results.
"""

from app.services.vlm.base import BaseVLMProvider, ProviderInfo, VLMExplanation
from app.services.vlm.config import VLMConfig, get_vlm_config
from app.services.vlm.usage_tracker import UsageTracker

__all__ = [
    "BaseVLMProvider",
    "VLMExplanation",
    "ProviderInfo",
    "VLMConfig",
    "get_vlm_config",
    "UsageTracker",
]
