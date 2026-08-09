"""Phase 3 orchestration above the existing Text-to-SQL pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from plotly.graph_objects import Figure

from src.text_to_sql.pipeline import PipelineResult, TextToSQLPipeline

from .chart_selector import ChartConfig, ChartSelector
from .insight_generator import InsightGenerator
from .result_analyzer import ResultAnalysis, ResultAnalyzer
from .visualization import VisualizationEngine


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyticsResult:
    question: str
    query: PipelineResult
    analysis: ResultAnalysis | None
    chart: ChartConfig | None
    figure: Figure | None
    insight: str | None
    insight_error: str | None = None
    insight_time_ms: float | None = None


class AnalyticsPipeline:
    def __init__(
        self,
        text_to_sql: TextToSQLPipeline,
        analyzer: ResultAnalyzer,
        chart_selector: ChartSelector,
        visualization: VisualizationEngine,
        insight_generator: InsightGenerator,
    ) -> None:
        self._text_to_sql = text_to_sql
        self._analyzer = analyzer
        self._chart_selector = chart_selector
        self._visualization = visualization
        self._insight_generator = insight_generator

    def analyze(self, question: str, create_figure: bool = True) -> AnalyticsResult:
        query = self._text_to_sql.ask(question)
        if query.error:
            return AnalyticsResult(question, query, None, None, None, None)

        analysis = self._analyzer.analyze(query.columns, query.rows, query.final_sql)
        chart = self._chart_selector.select(analysis)
        figure = self._visualization.create(query, chart) if create_figure else None
        insight: str | None = None
        insight_error: str | None = None
        insight_started = time.perf_counter()
        try:
            insight = self._insight_generator.generate(
                question=question,
                sql=query.final_sql or "",
                result=query,
                analysis=analysis,
                chart=chart,
            )
        except Exception as exc:
            LOGGER.exception("Business insight generation failed")
            insight_error = str(exc)
        insight_time_ms = (time.perf_counter() - insight_started) * 1_000
        return AnalyticsResult(
            question=question,
            query=query,
            analysis=analysis,
            chart=chart,
            figure=figure,
            insight=insight,
            insight_error=insight_error,
            insight_time_ms=insight_time_ms,
        )
