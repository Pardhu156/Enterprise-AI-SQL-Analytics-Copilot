import numpy as np
import pandas as pd
import pytest

from src.ml.delivery_risk import predict_delivery_risk, train_delivery_risk


class ProbabilityModel:
    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]])


def test_delivery_prediction_uses_persisted_threshold() -> None:
    artifact = {
        "pipeline": ProbabilityModel(),
        "feature_columns": ["total_price"],
        "threshold": 0.6,
    }
    result = predict_delivery_risk(artifact, pd.DataFrame([{"total_price": 10.0}]))

    assert result["predicted_late"] is True
    assert result["risk_band"] == "high"
    assert result["late_delivery_probability"] == 0.8


def test_training_rejects_too_little_chronological_data() -> None:
    frame = pd.DataFrame(
        {
            "order_purchase_timestamp": pd.date_range("2018-01-01", periods=10),
            "order_id": [f"order-{index}" for index in range(10)],
        }
    )

    with pytest.raises(ValueError, match="At least 100"):
        train_delivery_risk(frame, random_seed=42)
