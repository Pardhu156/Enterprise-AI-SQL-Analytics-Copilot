import pytest

from src.api.dependencies import get_analytics_service
from src.api.errors import APIError

from .conftest import FakeAnalyticsService


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "SQL_VALIDATION_FAILED"),
        (503, "DATABASE_UNAVAILABLE"),
        (503, "GEMINI_UNAVAILABLE"),
        (500, "SQL_EXECUTION_FAILED"),
    ],
)
def test_typed_service_errors_are_sanitized(app_client, status, code) -> None:
    client, _ = app_client
    service = FakeAnalyticsService(
        error=APIError(status, code, "Safe public message."),
    )
    client.app.dependency_overrides[get_analytics_service] = lambda: service
    response = client.post(
        "/api/v1/analytics/query",
        json={"question": "A valid question"},
    )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["message"] == "Safe public message."
    assert "traceback" not in response.text.lower()


def test_unexpected_error_is_not_exposed(app_client) -> None:
    client, _ = app_client
    service = FakeAnalyticsService(error=RuntimeError("secret internal path /tmp/private"))
    client.app.dependency_overrides[get_analytics_service] = lambda: service
    response = client.post(
        "/api/v1/analytics/query",
        json={"question": "A valid question"},
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "private" not in response.text

