from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import APISettings, get_analytics_service
from src.api.main import create_app
from src.api.schemas.responses import (
    AnalysisDetails,
    AnalyticsQueryResponse,
    ExecutionDetails,
    QueryResultDetails,
    SQLDetails,
    VisualizationDetails,
)


class FakeAnalyticsService:
    def __init__(self, response: AnalyticsQueryResponse | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def query(self, request, request_id: str) -> AnalyticsQueryResponse:
        self.calls.append((request.question, request_id))
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response.model_copy(update={"request_id": request_id})


def analytics_response() -> AnalyticsQueryResponse:
    return AnalyticsQueryResponse(
        request_id="test-request",
        question="What is the total revenue?",
        answer="Total revenue is R$100.00.",
        sql=SQLDetails(
            generated_sql="SELECT 100 AS total_revenue",
            final_sql="SELECT 100 AS total_revenue",
            validation_passed=True,
            was_repaired=False,
        ),
        result=QueryResultDetails(
            columns=["total_revenue"],
            rows=[[100.0]],
            row_count=1,
            truncated=False,
        ),
        analysis=AnalysisDetails(
            result_type="scalar_kpi",
            dimensions=[],
            metrics=["total_revenue"],
            categorical_columns=[],
            numeric_columns=["total_revenue"],
            datetime_columns=[],
            identifier_columns=[],
            has_datetime=False,
            is_empty=False,
        ),
        visualization=VisualizationDetails(
            chart_type="kpi",
            y="total_revenue",
            title="Total Revenue",
            reason="single business metric",
        ),
        execution=ExecutionDetails(
            sql_execution_time_ms=1.0,
            total_request_time_ms=2.0,
        ),
    )


@pytest.fixture
def app_client() -> Iterator[tuple[TestClient, FakeAnalyticsService]]:
    app = create_app(
        APISettings(
            host="127.0.0.1",
            port=8000,
            allowed_origins=("http://localhost:8501",),
        )
    )
    service = FakeAnalyticsService(analytics_response())
    app.dependency_overrides[get_analytics_service] = lambda: service
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, service

