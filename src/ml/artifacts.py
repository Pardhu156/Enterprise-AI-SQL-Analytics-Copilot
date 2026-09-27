"""Atomic, versioned persistence for trained Stage A model bundles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib


ARTIFACT_VERSION = 1


class ModelArtifactError(RuntimeError):
    """Raised when a trained model artifact is absent or incompatible."""


def save_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = {"artifact_version": ARTIFACT_VERSION, **payload}
    temporary = path.with_suffix(".tmp")
    joblib.dump(complete, temporary)
    os.replace(temporary, path)


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelArtifactError(
            f"Model artifact {path.name!r} is unavailable. Run the Stage A training command."
        )
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ModelArtifactError(f"Model artifact {path.name!r} could not be loaded") from exc
    if not isinstance(payload, dict) or payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ModelArtifactError(f"Model artifact {path.name!r} has an unsupported version")
    return payload
