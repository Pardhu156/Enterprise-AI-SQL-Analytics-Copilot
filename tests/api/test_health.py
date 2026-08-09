from src.api.dependencies import get_readiness_checker


class ReadyChecker:
    def check(self):
        return True, {"postgresql": "ok", "gemini_configuration": "ok"}


class NotReadyChecker:
    def check(self):
        return False, {"postgresql": "unavailable", "gemini_configuration": "ok"}


def test_health_returns_ok(app_client) -> None:
    client, _ = app_client
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]


def test_readiness_reports_lightweight_checks(app_client) -> None:
    client, _ = app_client
    client.app.dependency_overrides[get_readiness_checker] = lambda: ReadyChecker()
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"postgresql": "ok", "gemini_configuration": "ok"},
    }


def test_readiness_returns_503_when_a_dependency_is_unavailable(app_client) -> None:
    client, _ = app_client
    client.app.dependency_overrides[get_readiness_checker] = lambda: NotReadyChecker()
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["postgresql"] == "unavailable"


def test_openapi_documents_analytics_contract(app_client) -> None:
    client, _ = app_client
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/v1/analytics/query" in response.json()["paths"]
