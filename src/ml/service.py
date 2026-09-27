"""Online inference facade for persisted Stage A models."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .data_repository import MLDataRepository
from .delivery_risk import predict_delivery_risk
from .forecasting import forecast_from_artifact
from .model_registry import ModelRegistry
from .segmentation import segment_customer


class MLInputError(ValueError):
    pass


class MLInferenceService:
    def __init__(
        self,
        repository: MLDataRepository | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._repository = repository or MLDataRepository()
        self._registry = registry or ModelRegistry()

    def forecast(self, horizon_weeks: int) -> dict[str, Any]:
        artifact = self._registry.get("sales_forecast")
        forecast = forecast_from_artifact(artifact, horizon_weeks)
        history = pd.DataFrame(artifact["history"]).tail(12).copy()
        return self._response(
            "sales_forecasting",
            artifact,
            {
                "horizon_weeks": horizon_weeks,
                "training_period": artifact["training_period"],
                "recent_history": _records(history, "revenue"),
                "forecast": _records(forecast, "predicted_revenue"),
                "total_predicted_revenue": float(forecast["predicted_revenue"].sum()),
            },
            artifact["test_metrics"],
        )

    def segment_summary(self) -> dict[str, Any]:
        artifact = self._registry.get("customer_segmentation")
        return self._response(
            "customer_segmentation",
            artifact,
            {
                "cluster_count": artifact["cluster_count"],
                "profiles": artifact["profiles"],
                "data_as_of": artifact["data_as_of"],
            },
            {
                "selected_silhouette": artifact["evaluation"][
                    str(artifact["cluster_count"])
                ]["silhouette"],
                "selected_davies_bouldin": artifact["evaluation"][
                    str(artifact["cluster_count"])
                ]["davies_bouldin"],
            },
        )

    def segment_customer(self, customer_unique_id: str) -> dict[str, Any]:
        artifact = self._registry.get("customer_segmentation")
        as_of = pd.Timestamp(artifact["data_as_of"])
        row = self._repository.customer_rfm_by_id(customer_unique_id, as_of)
        if row.empty:
            raise MLInputError("No eligible purchase history was found for that customer_unique_id")
        cluster, segment = segment_customer(artifact, row)
        features = {
            name: float(row.iloc[0][name]) for name in artifact["feature_columns"]
        }
        return self._response(
            "customer_segmentation",
            artifact,
            {
                "customer_unique_id": customer_unique_id,
                "cluster": cluster,
                "segment": segment,
                "rfm": features,
                "data_as_of": artifact["data_as_of"],
            },
            {},
        )

    def delivery_risk(self, order_id: str) -> dict[str, Any]:
        artifact = self._registry.get("delivery_risk")
        row = self._repository.late_delivery_order(order_id)
        if row.empty:
            raise MLInputError(
                "No eligible delivered order was found for that order_id. "
                "Use an order with complete delivery and estimate timestamps."
            )
        prediction = predict_delivery_risk(artifact, row)
        return self._response(
            "late_delivery_prediction",
            artifact,
            {"order_id": order_id, **prediction},
            artifact["test_metrics"],
        )

    @staticmethod
    def _response(
        task: str,
        artifact: dict[str, Any],
        data: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "task": task,
            "model": artifact["model_name"],
            "trained_at_utc": artifact["trained_at_utc"],
            "data": data,
            "metrics": metrics,
            "limitations": list(artifact["limitations"]),
        }


def _records(frame: pd.DataFrame, value_column: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    date_column = "week_start"
    for _, row in frame.iterrows():
        date_value = pd.Timestamp(row[date_column]).date().isoformat()
        records.append({date_column: date_value, value_column: float(row[value_column])})
    return records
