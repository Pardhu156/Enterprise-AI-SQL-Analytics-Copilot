"""Leakage-safe weekly revenue forecasting training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


FORECAST_FEATURES = (
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_8",
    "lag_13",
    "rolling_mean_4",
    "rolling_std_4",
    "rolling_mean_8",
    "rolling_std_8",
    "rolling_mean_13",
    "week_of_year",
    "month",
    "quarter",
    "trend",
)


@dataclass(frozen=True)
class ForecastTrainingResult:
    artifact: dict[str, Any]
    metrics: dict[str, Any]


def prepare_weekly_revenue(
    raw: pd.DataFrame,
    latest_order_timestamp: pd.Timestamp,
    start: str = "2017-01-02",
) -> pd.DataFrame:
    frame = raw.copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    frame = frame[frame["week_start"] >= pd.Timestamp(start)].sort_values("week_start")
    if frame.empty:
        raise ValueError("No weekly revenue observations are available after the start date")
    last_week = frame["week_start"].max()
    if pd.Timestamp(latest_order_timestamp) < last_week + pd.Timedelta(days=7):
        frame = frame[frame["week_start"] < last_week]
    calendar = pd.date_range(frame["week_start"].min(), frame["week_start"].max(), freq="W-MON")
    frame = frame.set_index("week_start").reindex(calendar).rename_axis("week_start").reset_index()
    for column in ("revenue", "order_count", "item_count"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def build_forecast_features(history: pd.DataFrame) -> pd.DataFrame:
    frame = history[["week_start", "revenue"]].copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    revenue = frame["revenue"].astype(float)
    for lag in (1, 2, 3, 4, 8, 13):
        frame[f"lag_{lag}"] = revenue.shift(lag)
    shifted = revenue.shift(1)
    for window in (4, 8):
        frame[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"rolling_std_{window}"] = shifted.rolling(window).std(ddof=0)
    frame["rolling_mean_13"] = shifted.rolling(13).mean()
    calendar = frame["week_start"].dt.isocalendar()
    frame["week_of_year"] = calendar.week.astype(int)
    frame["month"] = frame["week_start"].dt.month
    frame["quarter"] = frame["week_start"].dt.quarter
    frame["trend"] = np.arange(len(frame), dtype=float)
    return frame.dropna().reset_index(drop=True)


def train_forecaster(
    history: pd.DataFrame,
    horizon_weeks: int,
    random_seed: int,
) -> ForecastTrainingResult:
    supervised = build_forecast_features(history)
    validation_size = max(horizon_weeks * 2, 8)
    test_size = max(horizon_weeks * 2, 8)
    if len(supervised) <= validation_size + test_size + 20:
        raise ValueError("Insufficient weekly history for chronological train/validation/test splits")
    train_end = len(supervised) - validation_size - test_size
    validation_end = len(supervised) - test_size
    train = supervised.iloc[:train_end]
    validation = supervised.iloc[train_end:validation_end]
    test = supervised.iloc[validation_end:]
    features = list(FORECAST_FEATURES)

    candidates: dict[str, Any] = {
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=random_seed,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_seed,
            n_jobs=4,
        ),
    }
    validation_metrics: dict[str, dict[str, float]] = {}
    baseline_prediction = validation["rolling_mean_4"].to_numpy()
    validation_metrics["moving_average_4"] = _regression_metrics(
        validation["revenue"], baseline_prediction
    )
    for name, model in candidates.items():
        model.fit(train[features], train["revenue"])
        validation_metrics[name] = _regression_metrics(
            validation["revenue"], np.maximum(model.predict(validation[features]), 0.0)
        )

    selected_name = min(
        validation_metrics,
        key=lambda name: validation_metrics[name]["rmse"],
    )
    fit_data = supervised.iloc[:validation_end]
    if selected_name == "moving_average_4":
        final_model = None
        test_prediction = test["rolling_mean_4"].to_numpy()
    else:
        final_model = clone(candidates[selected_name])
        final_model.fit(fit_data[features], fit_data["revenue"])
        test_prediction = np.maximum(final_model.predict(test[features]), 0.0)

    test_metrics = _regression_metrics(test["revenue"], test_prediction)
    artifact = {
        "model_type": "sales_forecasting",
        "model_name": selected_name,
        "model": final_model if selected_name != "xgboost" else None,
        "xgboost_booster": (
            bytes(final_model.get_booster().save_raw(raw_format="ubj"))
            if selected_name == "xgboost"
            else None
        ),
        "feature_columns": features,
        "horizon_weeks": horizon_weeks,
        "history": [
            {
                "week_start": pd.Timestamp(row.week_start).date().isoformat(),
                "revenue": float(row.revenue),
            }
            for row in history[["week_start", "revenue"]].itertuples(index=False)
        ],
        "training_period": {
            "start": history["week_start"].min().date().isoformat(),
            "end": history["week_start"].max().date().isoformat(),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "limitations": [
            "The Olist history is short and ends in 2018; forecasts demonstrate methodology, not current market demand.",
            "Multi-step predictions are recursive, so uncertainty grows with the horizon.",
            "The model uses historical revenue and calendar features only; promotions and macroeconomic drivers are unavailable.",
        ],
    }
    metrics = {
        "selected_model": selected_name,
        "observations": len(supervised),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation": validation_metrics,
        "test": test_metrics,
    }
    return ForecastTrainingResult(artifact=artifact, metrics=metrics)


def forecast_from_artifact(artifact: dict[str, Any], horizon_weeks: int) -> pd.DataFrame:
    maximum = int(artifact["horizon_weeks"])
    if not 1 <= horizon_weeks <= maximum:
        raise ValueError(f"horizon_weeks must be between 1 and {maximum}")
    history = pd.DataFrame(artifact["history"])
    history["week_start"] = pd.to_datetime(history["week_start"])
    predictions: list[dict[str, Any]] = []
    booster = None
    if artifact["model_name"] == "xgboost":
        from xgboost import Booster

        booster = Booster()
        booster.load_model(bytearray(artifact["xgboost_booster"]))
    for _ in range(horizon_weeks):
        next_week = history["week_start"].max() + pd.Timedelta(days=7)
        candidate = pd.concat(
            [history, pd.DataFrame([{"week_start": next_week, "revenue": np.nan}])],
            ignore_index=True,
        )
        row = build_forecast_features_for_next(candidate)
        if artifact["model_name"] == "moving_average_4":
            prediction = float(row["rolling_mean_4"].iloc[0])
        elif artifact["model_name"] == "xgboost":
            from xgboost import DMatrix

            feature_columns = list(artifact["feature_columns"])
            prediction = float(
                booster.predict(
                    DMatrix(row[feature_columns].to_numpy(), feature_names=feature_columns)
                )[0]
            )
        else:
            prediction = float(
                artifact["model"].predict(row[list(artifact["feature_columns"])])[0]
            )
        prediction = max(prediction, 0.0)
        predictions.append({"week_start": next_week, "predicted_revenue": prediction})
        history = pd.concat(
            [history, pd.DataFrame([{"week_start": next_week, "revenue": prediction}])],
            ignore_index=True,
        )
    return pd.DataFrame(predictions)


def build_forecast_features_for_next(candidate: pd.DataFrame) -> pd.DataFrame:
    frame = candidate.copy()
    target_index = len(frame) - 1
    revenue = frame["revenue"].astype(float)
    result: dict[str, float] = {}
    for lag in (1, 2, 3, 4, 8, 13):
        result[f"lag_{lag}"] = float(revenue.iloc[target_index - lag])
    prior = revenue.iloc[:target_index]
    for window in (4, 8):
        values = prior.tail(window)
        result[f"rolling_mean_{window}"] = float(values.mean())
        result[f"rolling_std_{window}"] = float(values.std(ddof=0))
    result["rolling_mean_13"] = float(prior.tail(13).mean())
    next_week = pd.Timestamp(frame.iloc[target_index]["week_start"])
    result["week_of_year"] = float(next_week.isocalendar().week)
    result["month"] = float(next_week.month)
    result["quarter"] = float(next_week.quarter)
    result["trend"] = float(target_index)
    return pd.DataFrame([result])


def _regression_metrics(actual: Any, predicted: Any) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
    }
