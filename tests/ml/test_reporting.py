from pathlib import Path

import pytest

from src.ml.artifacts import save_artifact
from src.ml.reporting import MLReportError, build_metrics_summary, validate_artifacts


def measured_report() -> dict:
    return {
        "trained_at_utc": "2026-01-01T00:00:00+00:00",
        "repeat_purchase_feasibility": {
            "decision": "use_late_delivery_prediction",
            "unique_customers": 100,
            "repeat_customers": 3,
            "overall_repeat_rate": 0.03,
        },
        "sales_forecasting": {
            "selected_model": "xgboost",
            "validation": {"xgboost": {"rmse": 10.0}},
            "test": {"mae": 8.0, "rmse": 11.0},
        },
        "customer_segmentation": {
            "selected_clusters": 2,
            "customers": 100,
            "cluster_evaluation": {
                "2": {"silhouette": 0.7, "davies_bouldin": 0.4}
            },
        },
        "classification": {
            "task_selected": "late_delivery_prediction",
            "selected_model": "logistic_regression",
            "positive_rate": 0.08,
            "test": {
                "precision": 0.2,
                "recall": 0.3,
                "f1": 0.24,
                "roc_auc": 0.7,
                "pr_auc": 0.15,
            },
        },
    }


def test_metrics_summary_uses_selected_validation_and_test_results() -> None:
    summary = build_metrics_summary(measured_report())

    assert summary["sales_forecasting"]["validation_rmse"] == 10.0
    assert summary["customer_segmentation"]["silhouette"] == 0.7
    assert summary["classification"]["task"] == "late_delivery_prediction"


def test_metrics_summary_rejects_missing_sections() -> None:
    with pytest.raises(MLReportError, match="missing sections"):
        build_metrics_summary({})


def test_artifacts_must_match_selected_models(tmp_path: Path) -> None:
    report = measured_report()
    payloads = {
        "sales_forecast": ("sales_forecasting", "xgboost", {}),
        "customer_segmentation": (
            "customer_segmentation",
            "kmeans_rfm",
            {"cluster_count": 2},
        ),
        "delivery_risk": ("late_delivery_prediction", "logistic_regression", {}),
    }
    for name, (model_type, model_name, extra) in payloads.items():
        save_artifact(
            tmp_path / f"{name}.joblib",
            {
                "model_type": model_type,
                "model_name": model_name,
                "trained_at_utc": "2026-01-01T00:00:00+00:00",
                **extra,
            },
        )

    validated = validate_artifacts(report, tmp_path)

    assert set(validated) == set(payloads)
