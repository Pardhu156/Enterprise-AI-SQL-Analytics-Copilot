from fastapi.testclient import TestClient

from src.api.dependencies import APISettings, get_business_service
from src.api.main import create_app


def _prediction(task: str) -> dict:
    return {
        "task": task,
        "model": "test_model",
        "trained_at_utc": "2026-01-01T00:00:00+00:00",
        "data": {"cluster_count": 2, "profiles": []},
        "metrics": {},
        "limitations": [],
    }


class BusinessService:
    def executive_overview(self):
        from src.api.schemas.responses import BusinessOverviewResponse

        return BusinessOverviewResponse.model_validate(
            {
                "generated_at_utc": "2026-01-01T00:00:00+00:00",
                "observed_kpis": [
                    {
                        "name": "total_revenue",
                        "value": 1000,
                        "unit": "BRL",
                        "kind": "observed",
                        "description": "Observed revenue.",
                    }
                ],
                "monthly_revenue": [{"month": "2025-12-01", "revenue": 1000, "orders": 5}],
                "forecast": _prediction("sales_forecasting"),
                "customer_segments": _prediction("customer_segmentation"),
                "classification_summary": {"task": "late_delivery_prediction"},
                "limitations": [],
            }
        )


def test_business_overview_endpoint_returns_structured_layers() -> None:
    app = create_app(APISettings("127.0.0.1", 8000, ("http://localhost:8501",)))
    app.dependency_overrides[get_business_service] = BusinessService

    with TestClient(app) as client:
        response = client.get("/api/v1/business/overview")

    assert response.status_code == 200
    assert response.json()["observed_kpis"][0]["kind"] == "observed"
    assert response.json()["forecast"]["task"] == "sales_forecasting"
