import httpx
import pytest

from src.frontend.api_client import (
    AnalyticsAPIClient,
    FrontendAPIConfig,
    FrontendAPIError,
)

def make_client(handler) -> AnalyticsAPIClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="http://backend.test")
    return AnalyticsAPIClient(
        FrontendAPIConfig("http://backend.test", 60),
        client=http_client,
    )


def response_payload() -> dict:
    return {
        "request_id": "request-1",
        "question": "What is the total revenue?",
        "answer": "Total revenue is 100.",
        "sql": {
            "generated_sql": "SELECT 100 AS total_revenue",
            "final_sql": "SELECT 100 AS total_revenue",
            "validation_passed": True,
            "was_repaired": False,
        },
        "result": {
            "columns": ["total_revenue"],
            "rows": [[100.0]],
            "row_count": 1,
            "truncated": False,
        },
        "analysis": {
            "result_type": "scalar_kpi",
            "dimensions": [],
            "metrics": ["total_revenue"],
            "categorical_columns": [],
            "numeric_columns": ["total_revenue"],
            "datetime_columns": [],
            "identifier_columns": [],
            "has_datetime": False,
            "is_empty": False,
        },
        "visualization": {
            "chart_type": "kpi",
            "x": None,
            "y": "total_revenue",
            "title": "Total Revenue",
            "reason": "single business metric",
        },
        "execution": {
            "sql_execution_time_ms": 1.0,
            "total_request_time_ms": 2.0,
        },
    }


def test_frontend_client_validates_success_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/analytics/query"
        return httpx.Response(
            200,
            json=response_payload(),
        )

    response = make_client(handler).query("What is the total revenue?")
    assert response.result.rows == [[100.0]]


def test_frontend_client_maps_structured_api_error() -> None:
    client = make_client(
        lambda request: httpx.Response(
            503,
            json={
                "error": {
                    "code": "GEMINI_UNAVAILABLE",
                    "message": "Retry shortly.",
                    "request_id": "abc",
                }
            },
        )
    )
    with pytest.raises(FrontendAPIError) as captured:
        client.query("question")
    assert captured.value.code == "GEMINI_UNAVAILABLE"
    assert captured.value.request_id == "abc"


def test_frontend_client_maps_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(FrontendAPIError, match="timed out") as captured:
        make_client(handler).query("question")
    assert captured.value.code == "ANALYSIS_TIMEOUT"
