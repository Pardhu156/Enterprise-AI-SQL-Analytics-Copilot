import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

from src.explainability.service import ExplainabilityService
from src.ml.delivery_risk import DELIVERY_FEATURES, _pipeline
from src.ml.forecasting import FORECAST_FEATURES, build_forecast_features


def test_forecast_treeshap_uses_deployed_model_and_is_additive() -> None:
    history = pd.DataFrame(
        {
            "week_start": pd.date_range("2017-01-02", periods=40, freq="W-MON"),
            "revenue": np.linspace(100.0, 300.0, 40) + np.sin(np.arange(40)) * 10,
        }
    )
    supervised = build_forecast_features(history)
    model = XGBRegressor(n_estimators=12, max_depth=2, random_state=42, n_jobs=1)
    model.fit(supervised[list(FORECAST_FEATURES)], supervised["revenue"])
    artifact = {
        "model_name": "xgboost",
        "xgboost_booster": bytes(model.get_booster().save_raw(raw_format="ubj")),
        "feature_columns": list(FORECAST_FEATURES),
        "history": history.assign(
            week_start=history["week_start"].dt.date.astype(str)
        ).to_dict(orient="records"),
        "horizon_weeks": 4,
    }

    explanation = ExplainabilityService().explain_forecast(
        artifact,
        horizon_weeks=2,
        top_n=len(FORECAST_FEATURES),
    )

    contribution_sum = sum(
        item["contribution"] for item in explanation["top_contributions"]
    )
    assert explanation["method"] == "TreeSHAP"
    assert explanation["baseline_value"] + contribution_sum == pytest.approx(
        explanation["prediction_value"], abs=0.5
    )
    assert len(explanation["details"]["forecast_steps"]) == 2


def test_delivery_linear_shap_uses_real_pipeline_features() -> None:
    rows = 60
    frame = pd.DataFrame(
        {
            "item_count": np.tile([1, 2, 3], 20),
            "seller_count": 1,
            "total_price": np.linspace(20, 400, rows),
            "total_freight": np.linspace(5, 50, rows),
            "average_item_price": np.linspace(20, 150, rows),
            "average_product_weight_g": np.linspace(100, 5000, rows),
            "average_product_length_cm": 20,
            "average_product_height_cm": 10,
            "average_product_width_cm": 15,
            "estimated_delivery_days": np.tile([5, 10, 20], 20),
            "purchase_hour": np.arange(rows) % 24,
            "purchase_day_of_week": np.arange(rows) % 7,
            "purchase_month": (np.arange(rows) % 12) + 1,
            "customer_state": np.tile(["SP", "RJ"], 30),
            "seller_state": np.tile(["SP", "MG"], 30),
            "product_category": np.tile(["books", "electronics"], 30),
        }
    )
    target = ((frame["estimated_delivery_days"] < 10) | (frame["total_freight"] > 30)).astype(int)
    pipeline = _pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipeline.fit(frame[list(DELIVERY_FEATURES)], target)
    artifact = {
        "model_name": "logistic_regression",
        "pipeline": pipeline,
        "feature_columns": list(DELIVERY_FEATURES),
        "threshold": 0.4,
    }

    explanation = ExplainabilityService().explain_delivery_risk(
        artifact,
        frame.iloc[[0]],
        frame.iloc[10:50],
        top_n=200,
    )

    assert explanation["method"] == "Linear SHAP"
    assert explanation["prediction_value"] == pytest.approx(
        pipeline.predict_proba(frame.iloc[[0]][list(DELIVERY_FEATURES)])[:, 1][0]
    )
    assert any(item["contribution"] != 0 for item in explanation["top_contributions"])
    assert explanation["global_importance"]
