"""Business-friendly deterministic chart selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .result_analyzer import ResultAnalysis, humanize_column, select_primary_metric


@dataclass(frozen=True)
class ChartConfig:
    chart_type: str
    x: str | None = None
    y: str | None = None
    title: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChartSelector:
    def select(self, analysis: ResultAnalysis) -> ChartConfig:
        if analysis.is_empty:
            return ChartConfig("none", reason="empty result")

        metric = select_primary_metric(analysis.metrics) if analysis.metrics else None
        dimension = analysis.dimensions[0] if analysis.dimensions else None

        if analysis.result_type == "scalar_kpi" and metric:
            return ChartConfig(
                "kpi",
                y=metric,
                title=humanize_column(metric),
                reason="single business metric",
            )
        if analysis.result_type == "time_series" and metric:
            time_column = analysis.datetime_columns[0]
            return ChartConfig(
                "line",
                x=time_column,
                y=metric,
                title=f"{humanize_column(metric)} over Time",
                reason="time-series result",
            )
        if analysis.result_type == "ranking" and dimension and metric:
            metric = analysis.metrics[0]
            return ChartConfig(
                "horizontal_bar",
                x=metric,
                y=dimension,
                title=f"{humanize_column(metric)} by {humanize_column(dimension)}",
                reason="ranked categorical comparison",
            )
        if analysis.result_type == "categorical_comparison" and dimension and metric:
            return ChartConfig(
                "bar",
                x=dimension,
                y=metric,
                title=f"{humanize_column(metric)} by {humanize_column(dimension)}",
                reason="categorical comparison",
            )
        if analysis.result_type == "numeric_relationship" and len(analysis.metrics) >= 2:
            first_metric, second_metric = analysis.metrics[:2]
            return ChartConfig(
                "scatter",
                x=first_metric,
                y=second_metric,
                title=(
                    f"{humanize_column(second_metric)} vs "
                    f"{humanize_column(first_metric)}"
                ),
                reason="two numeric measures",
            )
        if analysis.result_type == "distribution" and metric:
            return ChartConfig(
                "histogram",
                x=metric,
                title=f"Distribution of {humanize_column(metric)}",
                reason="single numeric distribution",
            )
        return ChartConfig("table", title="Query Results", reason="tabular or ambiguous result")
