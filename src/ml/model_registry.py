"""Lazy, process-local access to immutable trained model artifacts."""

from __future__ import annotations

from threading import Lock
from typing import Any

from .artifacts import ModelArtifactError, load_artifact
from .config import MLSettings, PRODUCTION_MODELS


class ModelRegistry:
    def __init__(self, settings: MLSettings | None = None) -> None:
        self._settings = settings or MLSettings.from_env()
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def get(self, name: str) -> dict[str, Any]:
        with self._lock:
            if name not in self._cache:
                specification = PRODUCTION_MODELS.get(name)
                if specification is None:
                    raise ModelArtifactError(f"{name!r} is not an approved production model")
                artifact = load_artifact(self._settings.artifact_path(name))
                for field, expected in specification.items():
                    if artifact.get(field) != expected:
                        raise ModelArtifactError(
                            f"Production artifact {name!r} has {field}={artifact.get(field)!r}; "
                            f"expected {expected!r}"
                        )
                self._cache[name] = artifact
            return self._cache[name]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
