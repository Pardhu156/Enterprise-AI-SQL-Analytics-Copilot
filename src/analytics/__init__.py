"""Deterministic analytics, Plotly visualization, and grounded Gemini insights."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .analytics_pipeline import AnalyticsPipeline, AnalyticsResult

__all__ = ["AnalyticsPipeline", "AnalyticsResult"]


def __getattr__(name: str) -> Any:
    """Load backend orchestration only when a caller requests its public exports."""
    if name in __all__:
        from .analytics_pipeline import AnalyticsPipeline, AnalyticsResult

        return {
            "AnalyticsPipeline": AnalyticsPipeline,
            "AnalyticsResult": AnalyticsResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
