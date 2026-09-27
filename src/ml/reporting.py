"""Validation and concise reporting for measured Stage A model results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import load_artifact
from .config import PRODUCTION_MODELS


class MLReportError(ValueError):
    """Raised when measured results or model artifacts are incomplete or inconsistent."""


def build_metrics_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Extract the model-selection and untouched-test metrics used for review."""
    required = (
        "repeat_purchase_feasibility",
        "sales_forecasting",
        "customer_segmentation",
        "classification",
    )
    missing = [section for section in required if section not in report]
    if missing:
        raise MLReportError(f"Metrics file is missing sections: {', '.join(missing)}")

    forecast = report["sales_forecasting"]
    forecast_name = forecast["selected_model"]
    segmentation = report["customer_segmentation"]
    cluster_key = str(segmentation["selected_clusters"])
    classification = report["classification"]
    repeat = report["repeat_purchase_feasibility"]

    try:
        return {
            "trained_at_utc": report["trained_at_utc"],
            "repeat_purchase": {
                "decision": repeat["decision"],
                "unique_customers": repeat["unique_customers"],
                "repeat_customers": repeat["repeat_customers"],
                "overall_repeat_rate": repeat["overall_repeat_rate"],
            },
            "sales_forecasting": {
                "selected_model": forecast_name,
                "validation_rmse": forecast["validation"][forecast_name]["rmse"],
                "test_mae": forecast["test"]["mae"],
                "test_rmse": forecast["test"]["rmse"],
            },
            "customer_segmentation": {
                "selected_clusters": segmentation["selected_clusters"],
                "customers": segmentation["customers"],
                "silhouette": segmentation["cluster_evaluation"][cluster_key]["silhouette"],
                "davies_bouldin": segmentation["cluster_evaluation"][cluster_key][
                    "davies_bouldin"
                ],
            },
            "classification": {
                "task": classification["task_selected"],
                "selected_model": classification["selected_model"],
                "positive_rate": classification["positive_rate"],
                **classification["test"],
            },
        }
    except (KeyError, TypeError) as exc:
        raise MLReportError(f"Metrics file has an invalid structure near {exc}") from exc


def validate_artifacts(
    report: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Ensure persisted artifacts agree with the selected models in the metrics file."""
    measured_selections = {
        "sales_forecast": report["sales_forecasting"]["selected_model"],
        "customer_segmentation": "kmeans_rfm",
        "delivery_risk": report["classification"]["selected_model"],
    }
    validated: dict[str, dict[str, Any]] = {}
    for artifact_name, specification in PRODUCTION_MODELS.items():
        model_type = str(specification["model_type"])
        model_name = str(specification["model_name"])
        if measured_selections[artifact_name] != model_name:
            raise MLReportError(
                f"Metrics selected {measured_selections[artifact_name]!r} for {artifact_name}; "
                f"production configuration approves {model_name!r}"
            )
        artifact = load_artifact(artifact_dir / f"{artifact_name}.joblib")
        for field, expected_value in specification.items():
            if artifact.get(field) != expected_value:
                raise MLReportError(
                    f"{artifact_name} has {field}={artifact.get(field)!r}; "
                    f"production configuration expects {expected_value!r}"
                )
        if artifact_name == "customer_segmentation":
            measured_k = report["customer_segmentation"]["selected_clusters"]
            if measured_k != specification["cluster_count"]:
                raise MLReportError(
                    f"Metrics selected K={measured_k}; production configuration approves "
                    f"K={specification['cluster_count']}"
                )
        validated[artifact_name] = {
            "model_type": model_type,
            "model_name": model_name,
            "trained_at_utc": artifact.get("trained_at_utc"),
        }
    return validated
