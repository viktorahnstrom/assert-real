"""
VLM Usage Tracker

Tracks per-provider request counts and estimated costs.
Enforces configurable daily request limits and monthly cost caps
to prevent accidental overspending.
"""

import logging
from datetime import UTC, datetime
from typing import Optional

from app.services.vlm.config import VLMConfig

logger = logging.getLogger(__name__)


class UsageLimitExceeded(Exception):
    """Raised when a usage limit (daily requests or monthly cost) is exceeded."""

    def __init__(self, limit_type: str, current: float, maximum: float):
        self.limit_type = limit_type
        self.current = current
        self.maximum = maximum
        super().__init__(
            f"VLM usage limit exceeded: {limit_type} (current: {current:.4f}, max: {maximum:.4f})"
        )


class UsageTracker:
    """
    In-memory usage tracker for VLM API calls.

    Tracks daily request counts and monthly estimated costs per provider.
    Resets daily counts at midnight UTC and monthly costs on the 1st.

    For production use, this could be backed by Redis or the database.
    For the thesis scope, in-memory tracking is sufficient.
    """

    def __init__(self, config: VLMConfig):
        self._config = config
        self._daily_requests: dict[str, int] = {}  # provider -> count
        self._monthly_cost: dict[str, float] = {}  # provider -> cost in USD
        self._total_requests: dict[str, int] = {}  # provider -> all-time count
        self._last_reset_day: Optional[int] = None
        self._last_reset_month: Optional[int] = None

    def _maybe_reset(self) -> None:
        """Reset counters if the day or month has changed."""
        now = datetime.now(UTC)

        if self._last_reset_day != now.day:
            self._daily_requests.clear()
            self._last_reset_day = now.day
            logger.info("Daily VLM request counters reset")

        if self._last_reset_month != now.month:
            self._monthly_cost.clear()
            self._last_reset_month = now.month
            logger.info("Monthly VLM cost counters reset")

    def check_limits(self, provider: str) -> None:
        """
        Check if the request is within configured limits.

        Args:
            provider: Provider ID (e.g. "google", "openai")

        Raises:
            UsageLimitExceeded: If daily request or monthly cost limit is exceeded
        """
        self._maybe_reset()

        # Check daily request limit
        daily_count = self._daily_requests.get(provider, 0)
        if daily_count >= self._config.max_requests_per_day:
            raise UsageLimitExceeded(
                limit_type="daily_requests",
                current=daily_count,
                maximum=self._config.max_requests_per_day,
            )

        # Check monthly cost limit (across all providers)
        total_monthly_cost = sum(self._monthly_cost.values())
        if total_monthly_cost >= self._config.max_monthly_cost_usd:
            raise UsageLimitExceeded(
                limit_type="monthly_cost_usd",
                current=total_monthly_cost,
                maximum=self._config.max_monthly_cost_usd,
            )

    def record_usage(self, provider: str, estimated_cost_usd: float) -> None:
        """
        Record a completed VLM API call.

        Args:
            provider: Provider ID
            estimated_cost_usd: Estimated cost of this request in USD
        """
        self._maybe_reset()

        self._daily_requests[provider] = self._daily_requests.get(provider, 0) + 1
        self._monthly_cost[provider] = self._monthly_cost.get(provider, 0.0) + estimated_cost_usd
        self._total_requests[provider] = self._total_requests.get(provider, 0) + 1

        logger.debug(
            f"VLM usage recorded: provider={provider}, "
            f"cost=${estimated_cost_usd:.6f}, "
            f"daily_total={self._daily_requests[provider]}, "
            f"monthly_cost=${self._monthly_cost.get(provider, 0):.6f}"
        )

    def get_usage_summary(self) -> dict:
        """Return current usage stats for all providers."""
        self._maybe_reset()

        total_monthly = sum(self._monthly_cost.values())
        total_daily = sum(self._daily_requests.values())

        return {
            "daily_requests": dict(self._daily_requests),
            "daily_total": total_daily,
            "daily_limit": self._config.max_requests_per_day,
            "monthly_cost_usd": {k: round(v, 6) for k, v in self._monthly_cost.items()},
            "monthly_cost_total_usd": round(total_monthly, 6),
            "monthly_cost_limit_usd": self._config.max_monthly_cost_usd,
            "total_requests_all_time": dict(self._total_requests),
        }
