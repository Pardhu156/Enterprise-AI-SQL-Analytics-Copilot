import pytest

from src.api.errors import APIError
from src.api.schemas.requests import CopilotQueryRequest
from src.api.schemas.responses import MLPredictionResponse
from src.api.services.copilot_service import CopilotService
from src.routing.intent_classifier import IntentDecision

from .conftest import analytics_response


class Classifier:
    def __init__(self, decision: dict) -> None:
        self.decision = IntentDecision.model_validate(decision)

    def classify(self, question: str) -> IntentDecision:
        return self.decision


class Analytics:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def query(self, request, request_id: str):
        self.questions.append(request.question)
        return analytics_response().model_copy(update={"request_id": request_id})


class ML:
    def __init__(self) -> None:
        self.forecast_horizons: list[int] = []

    def forecast(self, horizon: int) -> MLPredictionResponse:
        self.forecast_horizons.append(horizon)
        return MLPredictionResponse(
            task="sales_forecasting",
            model="xgboost",
            trained_at_utc="2026-01-01T00:00:00+00:00",
            data={"horizon_weeks": horizon, "total_predicted_revenue": 250.0},
            metrics={},
            limitations=[],
        )

    def segment_summary(self):
        raise AssertionError("not requested")

    def segment_customer(self, customer_unique_id: str):
        raise AssertionError("not requested")

    def delivery_risk(self, order_id: str):
        raise AssertionError("not requested")


def test_hybrid_route_uses_sql_subquestion_and_model_prediction() -> None:
    analytics = Analytics()
    ml = ML()
    service = CopilotService(
        Classifier(
            {
                "intent": "hybrid",
                "tasks": ["historical_analytics", "sales_forecasting"],
                "sql_question": "What is total revenue?",
                "horizon_weeks": 2,
            }
        ),
        analytics,
        ml,
    )

    response = service.query(CopilotQueryRequest(question="Compare both"), "request-1")

    assert analytics.questions == ["What is total revenue?"]
    assert ml.forecast_horizons == [2]
    assert response.route.intent == "hybrid"
    assert "R$ 250.00" in response.answer


def test_delivery_route_requires_an_order_id() -> None:
    service = CopilotService(
        Classifier({"intent": "ml", "tasks": ["late_delivery_prediction"]}),
        Analytics(),
        ML(),
    )

    with pytest.raises(APIError) as captured:
        service.query(CopilotQueryRequest(question="Will it be late?"), "request-2")

    assert captured.value.code == "ML_INPUT_REQUIRED"


def test_unsupported_route_fails_cleanly() -> None:
    service = CopilotService(
        Classifier({"intent": "unsupported", "tasks": []}),
        Analytics(),
        ML(),
    )

    with pytest.raises(APIError) as captured:
        service.query(CopilotQueryRequest(question="Write a poem"), "request-3")

    assert captured.value.code == "UNSUPPORTED_INTENT"
