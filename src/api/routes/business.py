"""Stage 7C executive overview endpoint."""

from fastapi import APIRouter, Depends

from ..dependencies import get_business_service
from ..schemas.responses import BusinessOverviewResponse, ErrorResponse
from ..services.business_service import BusinessService


router = APIRouter(prefix="/api/v1/business", tags=["business intelligence"])


@router.get(
    "/overview",
    response_model=BusinessOverviewResponse,
    responses={503: {"model": ErrorResponse, "description": "Dependency unavailable"}},
)
def executive_overview(
    service: BusinessService = Depends(get_business_service),
) -> BusinessOverviewResponse:
    return service.executive_overview()
