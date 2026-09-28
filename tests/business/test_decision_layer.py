from src.business.impact import BusinessImpactService
from src.business.recommendations import RecommendationService
from src.business.response_formatter import GroundedResponseFormatter
from src.business.segments import interpret_segment_profiles


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def test_forecast_impact_separates_observed_predicted_and_scenario_values() -> None:
    data = {
        "horizon_weeks": 2,
        "recent_history": [
            {"week_start": "2024-01-01", "revenue": 90},
            {"week_start": "2024-01-08", "revenue": 110},
        ],
        "forecast": [
            {"week_start": "2024-01-15", "predicted_revenue": 120},
            {"week_start": "2024-01-22", "predicted_revenue": 130},
        ],
    }

    impact = BusinessImpactService().forecast(data)
    metrics = {item["name"]: item for item in impact}

    assert metrics["recent_comparable_revenue"]["kind"] == "observed"
    assert metrics["forecast_revenue"]["kind"] == "predicted"
    assert metrics["forecast_change_vs_recent"]["kind"] == "scenario"
    assert metrics["forecast_change_rate_vs_recent"]["value"] == 0.25


def test_segment_interpretations_are_derived_from_profiles() -> None:
    profiles = [
        {
            "cluster": 0,
            "segment": "One-time customers",
            "customers": 90,
            "recency_days": 250,
            "frequency": 1.0,
            "monetary": 100,
        },
        {
            "cluster": 1,
            "segment": "High-value repeat",
            "customers": 10,
            "recency_days": 50,
            "frequency": 3.0,
            "monetary": 500,
        },
    ]

    interpretations = interpret_segment_profiles(profiles)
    high_value = next(item for item in interpretations if item["cluster"] == 1)

    assert "repeat-customer" in high_value["business_importance"]
    assert high_value["evidence"]["average_monetary_value"] == 500


def test_recommendation_uses_measured_forecast_change() -> None:
    impact = BusinessImpactService().forecast(
        {
            "horizon_weeks": 1,
            "recent_history": [{"week_start": "2024-01-01", "revenue": 100}],
            "forecast": [{"week_start": "2024-01-08", "predicted_revenue": 80}],
        }
    )

    recommendation = RecommendationService().forecast(impact)[0]

    assert recommendation["priority"] == "high"
    assert "lower" in recommendation["title"].lower()


def test_formatter_rejects_invented_numbers() -> None:
    formatter = GroundedResponseFormatter(FakeLLM("Revenue will increase by 999%."))

    answer = formatter.format(
        "What next?",
        "The forecast is available.",
        {"forecast": 100},
    )

    assert answer == "The forecast is available."


def test_formatter_accepts_grounded_prose_without_new_numbers() -> None:
    formatter = GroundedResponseFormatter(
        FakeLLM("The supplied forecast supports reviewing the current operating plan.")
    )

    answer = formatter.format("What next?", "Fallback.", {"trend": "decline"})

    assert answer.startswith("The supplied forecast")
