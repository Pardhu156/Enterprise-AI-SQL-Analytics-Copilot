"""Direct Stage A prediction endpoints."""

from fastapi import APIRouter, Depends, Path, Query

from ..dependencies import get_ml_service
from ..schemas.responses import ErrorResponse, MLPredictionResponse
from ..services.ml_service import MLService


router = APIRouter(prefix="/api/v1/ml", tags=["predictive analytics"])
ERRORS = {
    422: {"model": ErrorResponse, "description": "Invalid prediction input"},
    503: {"model": ErrorResponse, "description": "Model or database unavailable"},
}


@router.get("/forecast", response_model=MLPredictionResponse, responses=ERRORS)
def forecast(
    horizon_weeks: int = Query(default=4, ge=1, le=4),
    service: MLService = Depends(get_ml_service),
) -> MLPredictionResponse:
    return service.forecast(horizon_weeks)


@router.get("/segments", response_model=MLPredictionResponse, responses=ERRORS)
def segment_summary(service: MLService = Depends(get_ml_service)) -> MLPredictionResponse:
    return service.segment_summary()


@router.get(
    "/segments/{customer_unique_id}",
    response_model=MLPredictionResponse,
    responses=ERRORS,
)
def customer_segment(
    customer_unique_id: str = Path(pattern=r"^[0-9a-f]{32}$"),
    service: MLService = Depends(get_ml_service),
) -> MLPredictionResponse:
    return service.segment_customer(customer_unique_id)


@router.get(
    "/delivery-risk/{order_id}",
    response_model=MLPredictionResponse,
    responses=ERRORS,
)
def delivery_risk(
    order_id: str = Path(pattern=r"^[0-9a-f]{32}$"),
    service: MLService = Depends(get_ml_service),
) -> MLPredictionResponse:
    return service.delivery_risk(order_id)
