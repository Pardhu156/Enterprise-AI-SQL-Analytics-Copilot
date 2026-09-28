"""Expose the executive overview through a stable API response."""

from __future__ import annotations

import psycopg2

from src.business.service import BusinessOverviewService
from src.explainability.service import ExplainabilityError
from src.ml.artifacts import ModelArtifactError

from ..errors import APIError
from ..schemas.responses import BusinessOverviewResponse


class BusinessService:
    def __init__(self, overview: BusinessOverviewService) -> None:
        self._overview = overview

    def executive_overview(self) -> BusinessOverviewResponse:
        try:
            return BusinessOverviewResponse.model_validate(self._overview.overview())
        except ModelArtifactError as exc:
            raise APIError(503, "MODEL_UNAVAILABLE", str(exc)) from exc
        except ExplainabilityError as exc:
            raise APIError(503, "EXPLANATION_UNAVAILABLE", str(exc)) from exc
        except psycopg2.Error as exc:
            raise APIError(
                503,
                "DATABASE_UNAVAILABLE",
                "The analytics database is unavailable.",
            ) from exc
