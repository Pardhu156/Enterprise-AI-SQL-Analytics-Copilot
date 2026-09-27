from fastapi.testclient import TestClient

from src.api.dependencies import APISettings, get_copilot_service, get_ml_service
from src.api.errors import APIError
from src.api.main import create_app
from src.api.schemas.responses import (
    CopilotQueryResponse,
    MLPredictionResponse,
    RoutingDetails,
)


def prediction(task: str = "sales_forecasting") -> MLPredictionResponse:
    return MLPredictionResponse(
        task=task,
        model="test-model",
        trained_at_utc="2026-01-01T00:00:00+00:00",
        data={"horizon_weeks": 2, "total_predicted_revenue": 100.0},
        metrics={"mae": 5.0},
        limitations=["Test limitation"],
    )


class FakeMLService:
    def forecast(self, horizon: int):
        assert horizon == 2
        return prediction()

    def segment_summary(self):
        return prediction("customer_segmentation")

    def segment_customer(self, customer_id: str):
        return prediction("customer_segmentation")

    def delivery_risk(self, order_id: str):
        return prediction("late_delivery_prediction")


class FakeCopilotService:
    def query(self, request, request_id: str):
        return CopilotQueryResponse(
            request_id=request_id,
            question=request.question,
            route=RoutingDetails(intent="ml", tasks=["sales_forecasting"]),
            answer="Model result.",
            predictions=[prediction()],
        )


def test_direct_forecast_and_unified_copilot_contracts() -> None:
    app = create_app(APISettings("127.0.0.1", 8000, ("http://localhost:8501",)))
    app.dependency_overrides[get_ml_service] = lambda: FakeMLService()
    app.dependency_overrides[get_copilot_service] = lambda: FakeCopilotService()
    with TestClient(app) as client:
        forecast = client.get("/api/v1/ml/forecast?horizon_weeks=2")
        copilot = client.post("/api/v1/copilot/query", json={"question": "Predict revenue"})

    assert forecast.status_code == 200
    assert forecast.json()["task"] == "sales_forecasting"
    assert copilot.status_code == 200
    assert copilot.json()["route"]["intent"] == "ml"


def test_model_errors_remain_structured() -> None:
    class FailingML(FakeMLService):
        def forecast(self, horizon: int):
            raise APIError(503, "MODEL_UNAVAILABLE", "Train models first.")

    app = create_app(APISettings("127.0.0.1", 8000, ("http://localhost:8501",)))
    app.dependency_overrides[get_ml_service] = lambda: FailingML()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/ml/forecast?horizon_weeks=2")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"
