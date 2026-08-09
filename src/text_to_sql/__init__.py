"""Schema-aware, read-only PostgreSQL Text-to-SQL pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .pipeline import PipelineResult, TextToSQLPipeline

__all__ = ["PipelineResult", "TextToSQLPipeline"]


def __getattr__(name: str) -> Any:
    """Avoid importing PostgreSQL dependencies for lightweight submodules."""
    if name in __all__:
        from .pipeline import PipelineResult, TextToSQLPipeline

        return {
            "PipelineResult": PipelineResult,
            "TextToSQLPipeline": TextToSQLPipeline,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
