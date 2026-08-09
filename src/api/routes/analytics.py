"""Versioned analytics query endpoint."""

import logging

from fastapi import APIRouter, Depends, Request

from ..dependencies import get_analytics_service
from ..schemas.requests import AnalyticsQueryRequest
from ..schemas.responses import AnalyticsQueryResponse, ErrorResponse
from ..services.analytics_service import AnalyticsService


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.post(
    "/query",
    response_model=AnalyticsQueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "SQL safety rejection"},
        422: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Analytics execution failure"},
        503: {"model": ErrorResponse, "description": "Database or Gemini unavailable"},
    },
    summary="Answer a business analytics question",
    description=(
        "Generates validated read-only PostgreSQL with Gemini, executes it, and returns "
        "real query results, a grounded business answer, and visualization metadata."
    ),
)
def query_analytics(
    payload: AnalyticsQueryRequest,
    request: Request,
    service: AnalyticsService = Depends(get_analytics_service),
) -> AnalyticsQueryResponse:
    request_id = request.state.request_id
    LOGGER.info(
        "analytics_query request_id=%s question_length=%d",
        request_id,
        len(payload.question),
    )
    return service.query(payload, request_id)
