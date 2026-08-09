"""Plotly rendering for deterministic chart configurations."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

from src.text_to_sql.pipeline import PipelineResult

from .chart_selector import ChartConfig
from .result_analyzer import humanize_column


SUPPORTED_CHARTS = {"bar", "horizontal_bar", "line", "scatter", "histogram"}


class VisualizationEngine:
    def __init__(self, max_points: int = 100) -> None:
        if max_points <= 0:
            raise ValueError("max_points must be greater than zero")
        self._max_points = max_points

    def create(self, result: PipelineResult, config: ChartConfig) -> Figure | None:
        if result.error or not result.rows or config.chart_type not in SUPPORTED_CHARTS:
            return None

        return self.create_from_data(result.columns, result.rows, config)

    def create_from_data(
        self,
        columns: tuple[str, ...] | list[str],
        rows: tuple[tuple[Any, ...], ...] | list[list[Any]],
        config: ChartConfig,
    ) -> Figure | None:
        """Render API or direct-pipeline rows from the same chart specification."""
        if not rows or config.chart_type not in SUPPORTED_CHARTS:
            return None

        frame = pd.DataFrame(rows, columns=columns)
        frame = frame.map(_plot_value)
        required = [column for column in (config.x, config.y) if column]
        if not required or any(column not in frame.columns for column in required):
            return None
        frame = frame.dropna(subset=required)
        if frame.empty:
            return None

        if config.chart_type == "line":
            converted_x = pd.to_datetime(frame[config.x], errors="coerce")
            if converted_x.notna().all():
                frame[config.x] = converted_x
            frame = frame.sort_values(config.x)
            frame = _even_sample(frame, self._max_points)
            figure = px.line(frame, x=config.x, y=config.y, markers=True, title=config.title)
        elif config.chart_type == "bar":
            frame = frame.head(self._max_points)
            figure = px.bar(frame, x=config.x, y=config.y, title=config.title)
        elif config.chart_type == "horizontal_bar":
            frame = frame.head(self._max_points).iloc[::-1]
            figure = px.bar(
                frame,
                x=config.x,
                y=config.y,
                orientation="h",
                title=config.title,
            )
        elif config.chart_type == "scatter":
            frame = _even_sample(frame, self._max_points)
            figure = px.scatter(frame, x=config.x, y=config.y, title=config.title)
        else:
            frame = _even_sample(frame, self._max_points)
            figure = px.histogram(frame, x=config.x, title=config.title)

        figure.update_layout(
            template="plotly_white",
            height=460,
            margin={"l": 30, "r": 20, "t": 70, "b": 35},
            hoverlabel={"bgcolor": "white"},
            title={"font": {"size": 20}},
        )
        if config.x:
            figure.update_xaxes(title=humanize_column(config.x), showgrid=False)
        if config.y:
            figure.update_yaxes(title=humanize_column(config.y), gridcolor="#E8EDF3")
        return figure


def _plot_value(value: Any) -> Any:
    return float(value) if isinstance(value, Decimal) else value


def _even_sample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    if max_points == 1:
        return frame.iloc[[0]]
    indexes = [round(index * (len(frame) - 1) / (max_points - 1)) for index in range(max_points)]
    return frame.iloc[indexes]
