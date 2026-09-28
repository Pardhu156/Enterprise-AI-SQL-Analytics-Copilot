"""Online inference facade for persisted Stage A models."""

from __future__ import annotations

from decimal import Decimal
from threading import Lock
from typing import Any

import pandas as pd

from src.business.impact import BusinessImpactService
from src.business.recommendations import RecommendationService
from src.business.segments import interpret_segment_profiles
from src.explainability.service import ExplainabilityService

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
        explainability: ExplainabilityService | None = None,
        impact: BusinessImpactService | None = None,
        recommendations: RecommendationService | None = None,
    ) -> None:
        self._repository = repository or MLDataRepository()
        self._registry = registry or ModelRegistry()
        self._explainability = explainability or ExplainabilityService()
        self._impact = impact or BusinessImpactService()
        self._recommendations = recommendations or RecommendationService()
        self._delivery_background: pd.DataFrame | None = None
        self._background_lock = Lock()

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def forecast(self, horizon_weeks: int) -> dict[str, Any]:
        artifact = self._registry.get("sales_forecast")
        forecast = forecast_from_artifact(artifact, horizon_weeks)
        history = pd.DataFrame(artifact["history"]).tail(12).copy()
        data = {
            "horizon_weeks": horizon_weeks,
            "training_period": artifact["training_period"],
            "recent_history": _records(history, "revenue"),
            "forecast": _records(forecast, "predicted_revenue"),
            "total_predicted_revenue": float(forecast["predicted_revenue"].sum()),
        }
        explanation = self._explainability.explain_forecast(artifact, horizon_weeks)
        impact = self._impact.forecast(data)
        return self._response(
            "sales_forecasting",
            artifact,
            data,
            artifact["test_metrics"],
            explanation,
            impact,
            self._recommendations.forecast(impact),
        )

    def segment_summary(self) -> dict[str, Any]:
        artifact = self._registry.get("customer_segmentation")
        profiles = list(artifact["profiles"])
        interpretations = interpret_segment_profiles(profiles)
        impact = self._impact.segments(profiles)
        return self._response(
            "customer_segmentation",
            artifact,
            {
                "cluster_count": artifact["cluster_count"],
                "profiles": profiles,
                "interpretations": interpretations,
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
            None,
            impact,
            self._recommendations.segments(interpretations),
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
        interpretations = interpret_segment_profiles(list(artifact["profiles"]))
        interpretation = next(
            item for item in interpretations if int(item["cluster"]) == cluster
        )
        return self._response(
            "customer_segmentation",
            artifact,
            {
                "customer_unique_id": customer_unique_id,
                "cluster": cluster,
                "segment": segment,
                "rfm": features,
                "interpretation": interpretation,
                "data_as_of": artifact["data_as_of"],
            },
            {},
            None,
            [],
            self._recommendations.segments([interpretation]),
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
        background = self._get_delivery_background()
        explanation = self._explainability.explain_delivery_risk(
            artifact,
            row,
            background,
        )
        features = {
            name: _safe_scalar(row.iloc[0][name])
            for name in artifact["feature_columns"]
        }
        impact = self._impact.delivery_risk(prediction, features)
        return self._response(
            "late_delivery_prediction",
            artifact,
            {"order_id": order_id, "order_features": features, **prediction},
            artifact["test_metrics"],
            explanation,
            impact,
            self._recommendations.delivery_risk(prediction, explanation),
        )

    def _get_delivery_background(self) -> pd.DataFrame:
        with self._background_lock:
            if self._delivery_background is None:
                self._delivery_background = (
                    self._repository.late_delivery_explanation_background()
                )
            return self._delivery_background

    @staticmethod
    def _response(
        task: str,
        artifact: dict[str, Any],
        data: dict[str, Any],
        metrics: dict[str, Any],
        explanation: dict[str, Any] | None = None,
        impact: list[dict[str, Any]] | None = None,
        recommendations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "task": task,
            "model": artifact["model_name"],
            "trained_at_utc": artifact["trained_at_utc"],
            "data": data,
            "metrics": metrics,
            "limitations": list(artifact["limitations"]),
            "explanation": explanation,
            "impact": impact or [],
            "recommendations": recommendations or [],
        }


def _records(frame: pd.DataFrame, value_column: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    date_column = "week_start"
    for _, row in frame.iterrows():
        date_value = pd.Timestamp(row[date_column]).date().isoformat()
        records.append({date_column: date_value, value_column: float(row[value_column])})
    return records


def _safe_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    return value
