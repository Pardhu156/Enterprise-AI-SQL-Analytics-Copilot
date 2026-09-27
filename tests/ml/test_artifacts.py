import pytest

from src.ml.artifacts import ModelArtifactError, load_artifact, save_artifact


def test_artifact_round_trip_and_version_validation(tmp_path) -> None:
    path = tmp_path / "model.joblib"
    save_artifact(path, {"model_type": "test", "value": 7})

    assert load_artifact(path)["value"] == 7
    assert not path.with_suffix(".tmp").exists()


def test_missing_artifact_has_actionable_error(tmp_path) -> None:
    with pytest.raises(ModelArtifactError, match="training command"):
        load_artifact(tmp_path / "missing.joblib")
