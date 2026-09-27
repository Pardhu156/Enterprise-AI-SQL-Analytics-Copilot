"""Stage 7B statistical analysis endpoints."""

from fastapi import APIRouter, Depends

from ..dependencies import get_statistics_service
from ..schemas.requests import (
    ABTestRequest,
    ConfidenceIntervalRequest,
    HypothesisTestRequest,
    SampleSizeRequest,
)
from ..schemas.responses import ErrorResponse, StatisticalAnalysisResponse
from ..services.statistics_service import StatisticsService


router = APIRouter(prefix="/api/v1/statistics", tags=["statistical analysis"])
ERRORS = {
    422: {"model": ErrorResponse, "description": "Invalid statistical inputs"},
    503: {"model": ErrorResponse, "description": "Database unavailable"},
}


@router.post(
    "/hypothesis-test",
    response_model=StatisticalAnalysisResponse,
    responses=ERRORS,
)
def hypothesis_test(
    request: HypothesisTestRequest,
    service: StatisticsService = Depends(get_statistics_service),
) -> StatisticalAnalysisResponse:
    return service.hypothesis_test(request)


@router.post(
    "/confidence-interval",
    response_model=StatisticalAnalysisResponse,
    responses=ERRORS,
)
def confidence_interval(
    request: ConfidenceIntervalRequest,
    service: StatisticsService = Depends(get_statistics_service),
) -> StatisticalAnalysisResponse:
    return service.confidence_interval(request)


@router.post("/ab-test", response_model=StatisticalAnalysisResponse, responses=ERRORS)
def ab_test(
    request: ABTestRequest,
    service: StatisticsService = Depends(get_statistics_service),
) -> StatisticalAnalysisResponse:
    return service.ab_test(request)


@router.post(
    "/sample-size",
    response_model=StatisticalAnalysisResponse,
    responses=ERRORS,
)
def sample_size(
    request: SampleSizeRequest,
    service: StatisticsService = Depends(get_statistics_service),
) -> StatisticalAnalysisResponse:
    return service.sample_size(request)
