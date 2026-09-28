"""Executive overview orchestration over observed KPIs and deployed models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.ml.service import MLInferenceService

from .data_repository import BusinessDataRepository


class BusinessOverviewService:
    def __init__(
        self,
        repository: BusinessDataRepository | None = None,
        ml: MLInferenceService | None = None,
    ) -> None:
        self._repository = repository or BusinessDataRepository()
        self._ml = ml or MLInferenceService()

    def overview(self) -> dict[str, Any]:
        kpis = self._repository.executive_kpis()
        forecast = self._ml.forecast(4)
        segments = self._ml.segment_summary()
        delivery_artifact = self._ml.registry.get("delivery_risk")
        observed_kpis = [
            _metric("total_revenue", kpis["total_revenue"], "BRL", "Observed merchandise revenue."),
            _metric("order_volume", kpis["order_volume"], "orders", "Eligible historical orders."),
            _metric("average_order_value", kpis["average_order_value"], "BRL", "Observed merchandise value per eligible order."),
            _metric("delayed_delivery_rate", kpis["delayed_delivery_rate"], "ratio", "Observed delayed-delivery proportion among eligible delivered orders."),
        ]
        return {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "observed_kpis": observed_kpis,
            "monthly_revenue": self._repository.monthly_revenue(),
            "forecast": forecast,
            "customer_segments": segments,
            "classification_summary": {
                "task": "late_delivery_prediction",
                "model": delivery_artifact["model_name"],
                "test_metrics": delivery_artifact["test_metrics"],
                "decision_threshold": delivery_artifact["threshold"],
                "note": "This is the deployed alternative classifier; no repeat-purchase model is deployed.",
            },
            "limitations": [
                "Observed KPIs come from the historical Olist snapshot and are not current market metrics.",
                "Forecasts and classifications are model estimates; scenario metrics are not realized outcomes.",
                "The overview does not infer campaign uplift or causal business impact.",
            ],
        }


def _metric(name: str, value: float, unit: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "value": float(value),
        "unit": unit,
        "kind": "observed",
        "description": description,
        "assumptions": [],
    }
