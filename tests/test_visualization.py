from datetime import date
from decimal import Decimal

from plotly.graph_objects import Figure

from src.analytics.chart_selector import ChartConfig
from src.analytics.visualization import VisualizationEngine
from src.text_to_sql.pipeline import PipelineResult


def query_result(
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
) -> PipelineResult:
    return PipelineResult(
        question="test",
        generated_sql="SELECT 1",
        final_sql="SELECT 1",
        validation_passed=True,
        validation_reason="safe",
        was_repaired=False,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        execution_time_ms=1.0,
        truncated=False,
        error=None,
    )


def test_supported_charts_return_plotly_figures() -> None:
    engine = VisualizationEngine()
    categorical = query_result(
        ("category", "revenue"),
        (("A", Decimal("20")), ("B", Decimal("10"))),
    )
    time_series = query_result(
        ("month", "revenue"),
        ((date(2024, 1, 1), 10), (date(2024, 2, 1), 20)),
    )
    numeric = query_result(("freight", "score"), ((10, 5), (20, 3), (30, 1)))

    configs = (
        (categorical, ChartConfig("bar", "category", "revenue", "Revenue")),
        (categorical, ChartConfig("horizontal_bar", "revenue", "category", "Revenue")),
        (time_series, ChartConfig("line", "month", "revenue", "Revenue over Time")),
        (numeric, ChartConfig("scatter", "freight", "score", "Score vs Freight")),
        (numeric, ChartConfig("histogram", "freight", title="Freight Distribution")),
    )
    for result, config in configs:
        assert isinstance(engine.create(result, config), Figure)


def test_empty_and_invalid_configs_fall_back_safely() -> None:
    engine = VisualizationEngine()
    empty = query_result(("category", "revenue"), ())
    populated = query_result(("category", "revenue"), (("A", 1),))

    assert engine.create(empty, ChartConfig("bar", "category", "revenue")) is None
    assert engine.create(populated, ChartConfig("unknown", "category", "revenue")) is None
    assert engine.create(populated, ChartConfig("bar", "missing", "revenue")) is None
