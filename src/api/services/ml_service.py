"""Map ML domain inference to safe, stable API contracts."""

from __future__ import annotations

import logging

import psycopg2

from src.ml.artifacts import ModelArtifactError
from src.ml.service import MLInferenceService, MLInputError

from ..errors import APIError
from ..schemas.responses import MLPredictionResponse


LOGGER = logging.getLogger(__name__)


class MLService:
    def __init__(self, inference: MLInferenceService) -> None:
        self._inference = inference

    def forecast(self, horizon_weeks: int) -> MLPredictionResponse:
        return self._call(self._inference.forecast, horizon_weeks)

    def segment_summary(self) -> MLPredictionResponse:
        return self._call(self._inference.segment_summary)

    def segment_customer(self, customer_unique_id: str) -> MLPredictionResponse:
        return self._call(self._inference.segment_customer, customer_unique_id)

    def delivery_risk(self, order_id: str) -> MLPredictionResponse:
        return self._call(self._inference.delivery_risk, order_id)

    @staticmethod
    def _call(function, *args) -> MLPredictionResponse:
        try:
            return MLPredictionResponse.model_validate(function(*args))
        except ModelArtifactError as exc:
            LOGGER.warning("ML artifact unavailable: %s", exc)
            raise APIError(503, "MODEL_UNAVAILABLE", str(exc)) from exc
        except MLInputError as exc:
            raise APIError(422, "ML_INPUT_INVALID", str(exc)) from exc
        except psycopg2.Error as exc:
            LOGGER.warning("ML database query failed: %s", type(exc).__name__)
            raise APIError(503, "DATABASE_UNAVAILABLE", "The analytics database is unavailable.") from exc
        except ValueError as exc:
            raise APIError(422, "ML_INPUT_INVALID", str(exc)) from exc
