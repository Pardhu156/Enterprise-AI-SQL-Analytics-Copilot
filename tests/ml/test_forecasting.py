import numpy as np
import pandas as pd

from src.ml.forecasting import (
    build_forecast_features,
    forecast_from_artifact,
    prepare_weekly_revenue,
)


class ConstantModel:
    def predict(self, frame):
        return np.full(len(frame), 125.0)


def test_lags_and_rolls_use_only_prior_weeks() -> None:
    history = pd.DataFrame(
        {
            "week_start": pd.date_range("2017-01-02", periods=20, freq="W-MON"),
            "revenue": np.arange(1.0, 21.0),
        }
    )
    features = build_forecast_features(history)
    row = features.iloc[0]

    assert row["revenue"] == 14.0
    assert row["lag_1"] == 13.0
    assert row["rolling_mean_4"] == 11.5
    assert row["rolling_mean_13"] == 7.0


def test_partial_last_week_is_excluded_and_gaps_are_filled() -> None:
    raw = pd.DataFrame(
        {
            "week_start": pd.to_datetime(["2017-01-02", "2017-01-16", "2017-01-23"]),
            "revenue": [100.0, 300.0, 400.0],
            "order_count": [1, 3, 4],
            "item_count": [1, 3, 4],
        }
    )
    prepared = prepare_weekly_revenue(raw, pd.Timestamp("2017-01-23 09:00"))

    assert prepared["week_start"].max() == pd.Timestamp("2017-01-16")
    assert prepared.loc[prepared["week_start"] == pd.Timestamp("2017-01-09"), "revenue"].item() == 0


def test_recursive_forecast_uses_saved_history() -> None:
    history = pd.DataFrame(
        {
            "week_start": pd.date_range("2017-01-02", periods=20, freq="W-MON"),
            "revenue": np.arange(100.0, 120.0),
        }
    )
    artifact = {
        "horizon_weeks": 4,
        "model_name": "constant",
        "model": ConstantModel(),
        "feature_columns": ["lag_1"],
        "history": history,
    }

    result = forecast_from_artifact(artifact, 3)

    assert result["predicted_revenue"].tolist() == [125.0, 125.0, 125.0]
    assert result["week_start"].is_monotonic_increasing
