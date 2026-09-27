"""Configuration shared by offline ML training and online inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Promotion is deliberately explicit: training may compare different candidates, but online
# inference loads only artifacts matching these reviewed production selections.
PRODUCTION_MODELS: dict[str, dict[str, object]] = {
    "sales_forecast": {
        "model_type": "sales_forecasting",
        "model_name": "xgboost",
    },
    "customer_segmentation": {
        "model_type": "customer_segmentation",
        "model_name": "kmeans_rfm",
        "cluster_count": 2,
    },
    "delivery_risk": {
        "model_type": "late_delivery_prediction",
        "model_name": "logistic_regression",
    },
}


@dataclass(frozen=True)
class MLSettings:
    artifact_dir: Path
    forecast_horizon_weeks: int = 4
    random_seed: int = 42

    @classmethod
    def from_env(cls) -> "MLSettings":
        load_dotenv()
        raw_dir = os.getenv("ML_ARTIFACT_DIR", "artifacts/models").strip()
        artifact_dir = Path(raw_dir)
        if not artifact_dir.is_absolute():
            artifact_dir = PROJECT_ROOT / artifact_dir
        return cls(
            artifact_dir=artifact_dir,
            forecast_horizon_weeks=_positive_int("ML_FORECAST_HORIZON_WEEKS", 4),
            random_seed=_positive_int("ML_RANDOM_SEED", 42),
        )

    def artifact_path(self, name: str) -> Path:
        return self.artifact_dir / f"{name}.joblib"


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
