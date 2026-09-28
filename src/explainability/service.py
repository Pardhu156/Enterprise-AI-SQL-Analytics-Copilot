"""SHAP explanations computed from the deployed model artifacts and real features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from src.ml.forecasting import (
    build_forecast_features,
    build_forecast_features_for_next,
    forecast_from_artifact,
)


class ExplainabilityError(RuntimeError):
    """Raised when a persisted model cannot support the requested explanation."""


class ExplainabilityService:
    """Create local and global SHAP explanations without changing model predictions."""

    def explain_forecast(
        self,
        artifact: dict[str, Any],
        horizon_weeks: int,
        top_n: int = 8,
    ) -> dict[str, Any]:
        if artifact.get("model_name") != "xgboost" or not artifact.get("xgboost_booster"):
            raise ExplainabilityError("TreeSHAP is available only for the deployed XGBoost forecast")

        from xgboost import Booster

        booster = Booster()
        booster.load_model(bytearray(artifact["xgboost_booster"]))
        explainer = shap.TreeExplainer(booster)
        feature_columns = list(artifact["feature_columns"])
        forecast = forecast_from_artifact(artifact, horizon_weeks)
        history = pd.DataFrame(artifact["history"])
        history["week_start"] = pd.to_datetime(history["week_start"])
        training_history = history.copy()
        contribution_totals = np.zeros(len(feature_columns), dtype=float)
        feature_value_totals = np.zeros(len(feature_columns), dtype=float)
        baseline_total = 0.0
        steps: list[dict[str, Any]] = []

        for index, forecast_row in forecast.iterrows():
            next_week = history["week_start"].max() + pd.Timedelta(days=7)
            candidate = pd.concat(
                [history, pd.DataFrame([{"week_start": next_week, "revenue": np.nan}])],
                ignore_index=True,
            )
            feature_row = build_forecast_features_for_next(candidate)[feature_columns]
            explanation = explainer(feature_row)
            values = np.asarray(explanation.values[0], dtype=float)
            base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
            prediction = float(forecast_row["predicted_revenue"])
            contribution_totals += values
            feature_value_totals += feature_row.iloc[0].to_numpy(dtype=float)
            baseline_total += base_value
            steps.append(
                {
                    "week_start": pd.Timestamp(forecast_row["week_start"]).date().isoformat(),
                    "prediction": prediction,
                    "baseline": base_value,
                    "top_contributions": _rank_contributions(
                        feature_columns,
                        feature_row.iloc[0].to_numpy(dtype=float),
                        values,
                        top_n=5,
                    ),
                }
            )
            history = pd.concat(
                [history, pd.DataFrame([{"week_start": next_week, "revenue": prediction}])],
                ignore_index=True,
            )

        historical_features = build_forecast_features(training_history)[feature_columns]
        global_values = np.asarray(explainer(historical_features).values, dtype=float)
        global_importance = _rank_importance(
            feature_columns,
            np.mean(np.abs(global_values), axis=0),
            top_n,
        )
        return {
            "method": "TreeSHAP",
            "output_space": "weekly revenue",
            "prediction_value": float(forecast["predicted_revenue"].sum()),
            "baseline_value": baseline_total,
            "top_contributions": _rank_contributions(
                feature_columns,
                feature_value_totals / horizon_weeks,
                contribution_totals,
                top_n,
            ),
            "global_importance": global_importance,
            "details": {"forecast_steps": steps, "horizon_weeks": horizon_weeks},
            "limitations": [
                "TreeSHAP explains the deployed forecast model; it does not establish causal revenue drivers.",
                "Recursive forecasts use earlier predicted weeks as later lag inputs, so explanations also propagate model uncertainty.",
            ],
        }

    def explain_delivery_risk(
        self,
        artifact: dict[str, Any],
        row: pd.DataFrame,
        background: pd.DataFrame,
        top_n: int = 8,
    ) -> dict[str, Any]:
        if artifact.get("model_name") != "logistic_regression":
            raise ExplainabilityError(
                "Linear SHAP is available only for the deployed logistic-regression classifier"
            )
        feature_columns = list(artifact["feature_columns"])
        pipeline = artifact["pipeline"]
        preprocessor = pipeline.named_steps["preprocess"]
        classifier = pipeline.named_steps["classifier"]
        transformed_background = preprocessor.transform(background[feature_columns])
        transformed_row = preprocessor.transform(row[feature_columns])
        masker = shap.maskers.Independent(
            transformed_background,
            max_samples=len(background),
        )
        explainer = shap.LinearExplainer(classifier, masker)
        local = explainer(transformed_row)
        local_values = np.asarray(local.values[0], dtype=float)
        transformed_values = _dense_row(transformed_row)
        raw_transformed_names = list(preprocessor.get_feature_names_out())
        transformed_names = [
            _humanize_transformed_feature(name) for name in raw_transformed_names
        ]
        display_values = _delivery_display_values(
            raw_transformed_names,
            transformed_values,
            row,
        )
        probability = float(pipeline.predict_proba(row[feature_columns])[:, 1][0])
        background_explanations = np.asarray(explainer(transformed_background).values, dtype=float)
        return {
            "method": "Linear SHAP",
            "output_space": "late-delivery log-odds",
            "prediction_value": probability,
            "baseline_value": float(np.asarray(local.base_values).reshape(-1)[0]),
            "top_contributions": _rank_contributions(
                transformed_names,
                display_values,
                local_values,
                top_n,
            ),
            "global_importance": _rank_importance(
                transformed_names,
                np.mean(np.abs(background_explanations), axis=0),
                top_n,
            ),
            "details": {
                "background_rows": int(len(background)),
                "decision_threshold": float(artifact["threshold"]),
            },
            "limitations": [
                "Linear SHAP contributions are in log-odds while the displayed prediction is a probability.",
                "The explanation is relative to a deterministic historical Olist background sample and is associative, not causal.",
            ],
        }


def _rank_contributions(
    names: list[str],
    feature_values: np.ndarray,
    contributions: np.ndarray,
    top_n: int,
) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return [
        {
            "feature": names[index],
            "feature_value": float(feature_values[index]),
            "contribution": float(contributions[index]),
            "direction": _direction(float(contributions[index])),
        }
        for index in order
    ]


def _rank_importance(
    names: list[str],
    importance: np.ndarray,
    top_n: int,
) -> list[dict[str, Any]]:
    order = np.argsort(importance)[::-1][:top_n]
    return [
        {"feature": names[index], "importance": float(importance[index])}
        for index in order
    ]


def _dense_row(matrix: Any) -> np.ndarray:
    row = matrix[0]
    if hasattr(row, "toarray"):
        row = row.toarray()
    return np.asarray(row, dtype=float).reshape(-1)


def _humanize_transformed_feature(name: str) -> str:
    cleaned = name.split("__", 1)[-1]
    for prefix in ("customer_state_", "seller_state_", "product_category_"):
        if cleaned.startswith(prefix):
            field = prefix.removesuffix("_").replace("_", " ")
            return f"{field} = {cleaned.removeprefix(prefix)}"
    return cleaned.replace("_", " ")


def _delivery_display_values(
    transformed_names: list[str],
    transformed_values: np.ndarray,
    row: pd.DataFrame,
) -> np.ndarray:
    display = transformed_values.copy()
    for index, name in enumerate(transformed_names):
        if name.startswith("numeric__"):
            source = name.removeprefix("numeric__")
            value = row.iloc[0][source]
            if pd.notna(value):
                display[index] = float(value)
    return display


def _direction(value: float) -> str:
    if value > 0:
        return "increases"
    if value < 0:
        return "decreases"
    return "neutral"
