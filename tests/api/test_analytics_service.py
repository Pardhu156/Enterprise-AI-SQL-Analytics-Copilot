from decimal import Decimal

import pytest

from src.analytics.analytics_pipeline import AnalyticsResult
from src.analytics.chart_selector import ChartConfig
from src.analytics.result_analyzer import ResultAnalysis
from src.api.errors import APIError
from src.api.schemas.requests import AnalyticsQueryRequest
from src.api.services.analytics_service import AnalyticsService
from src.text_to_sql.pipeline import PipelineResult


class FakePipeline:
    def __init__(self, result: AnalyticsResult) -> None:
        self.result = result
        self.calls = []

    def analyze(self, question: str, create_figure: bool = True) -> AnalyticsResult:
        self.calls.append((question, create_figure))
        return self.result


def successful_result() -> AnalyticsResult:
    query = PipelineResult(
        question="What is total revenue?",
        generated_sql="SELECT 100 AS total_revenue",
        final_sql="SELECT 100 AS total_revenue",
        validation_passed=True,
        validation_reason="safe",
        was_repaired=False,
        columns=("total_revenue",),
        rows=((Decimal("100.25"),),),
        row_count=1,
        execution_time_ms=1.5,
        truncated=False,
        error=None,
    )
    analysis = ResultAnalysis(
        result_type="scalar_kpi",
        dimensions=(),
        metrics=("total_revenue",),
        categorical_columns=(),
        numeric_columns=("total_revenue",),
        datetime_columns=(),
        identifier_columns=(),
        row_count=1,
        has_datetime=False,
        is_empty=False,
    )
    return AnalyticsResult(
        question=query.question,
        query=query,
        analysis=analysis,
        chart=ChartConfig("kpi", y="total_revenue", title="Total Revenue"),
        figure=None,
        insight="Total revenue is 100.25.",
    )


def test_service_reuses_analytics_pipeline_and_serializes_json_values() -> None:
    pipeline = FakePipeline(successful_result())
    response = AnalyticsService(pipeline).query(
        AnalyticsQueryRequest(question="What is total revenue?"),
        "request-1",
    )
    assert pipeline.calls == [("What is total revenue?", False)]
    assert response.result.rows == [[100.25]]
    assert response.answer == "Total revenue is 100.25."


def test_service_honors_optional_response_sections() -> None:
    pipeline = FakePipeline(successful_result())
    response = AnalyticsService(pipeline).query(
        AnalyticsQueryRequest(
            question="What is total revenue?",
            include_sql=False,
            include_rows=False,
            include_visualization_config=False,
        ),
        "request-2",
    )
    assert response.sql is None
    assert response.result.rows is None
    assert response.visualization is None


@pytest.mark.parametrize(
    ("message", "status", "code"),
    [
        ("SQL validation failed: unsafe", 400, "SQL_VALIDATION_FAILED"),
        ("SQL generation failed: connection refused", 503, "DATABASE_UNAVAILABLE"),
        ("SQL generation failed: 429 quota", 503, "GEMINI_UNAVAILABLE"),
        ("Repaired SQL execution failed: bad query", 500, "SQL_EXECUTION_FAILED"),
    ],
)
def test_service_maps_pipeline_failures(message, status, code) -> None:
    failed_query = PipelineResult(
        question="question",
        generated_sql=None,
        final_sql=None,
        validation_passed=False,
        validation_reason=None,
        was_repaired=False,
        columns=(),
        rows=(),
        row_count=0,
        execution_time_ms=None,
        truncated=False,
        error=message,
    )
    pipeline = FakePipeline(AnalyticsResult("question", failed_query, None, None, None, None))
    with pytest.raises(APIError) as captured:
        AnalyticsService(pipeline).query(AnalyticsQueryRequest(question="question"), "id")
    assert captured.value.status_code == status
    assert captured.value.code == code
