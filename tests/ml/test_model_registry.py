import pytest

from src.ml.artifacts import ModelArtifactError, save_artifact
from src.ml.config import MLSettings
from src.ml.model_registry import ModelRegistry


def test_registry_loads_only_approved_production_artifact(tmp_path) -> None:
    save_artifact(
        tmp_path / "sales_forecast.joblib",
        {"model_type": "sales_forecasting", "model_name": "xgboost"},
    )
    registry = ModelRegistry(MLSettings(artifact_dir=tmp_path))

    assert registry.get("sales_forecast")["model_name"] == "xgboost"
    with pytest.raises(ModelArtifactError, match="not an approved production model"):
        registry.get("experimental_forecast")


def test_registry_rejects_artifact_that_does_not_match_promotion(tmp_path) -> None:
    save_artifact(
        tmp_path / "sales_forecast.joblib",
        {"model_type": "sales_forecasting", "model_name": "random_forest"},
    )
    registry = ModelRegistry(MLSettings(artifact_dir=tmp_path))

    with pytest.raises(ModelArtifactError, match="expected 'xgboost'"):
        registry.get("sales_forecast")
