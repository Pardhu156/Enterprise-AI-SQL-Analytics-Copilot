"""Unified historical, predictive, and hybrid Copilot endpoint."""

from fastapi import APIRouter, Depends, Request

from ..dependencies import get_copilot_service
from ..schemas.requests import CopilotQueryRequest
from ..schemas.responses import CopilotQueryResponse, ErrorResponse
from ..services.copilot_service import CopilotService


router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])


@router.post(
    "/query",
    response_model=CopilotQueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported request"},
        422: {"model": ErrorResponse, "description": "Missing or invalid model input"},
        503: {"model": ErrorResponse, "description": "Gemini, model, or database unavailable"},
    },
    summary="Route a historical, predictive, or hybrid analytics question",
)
def query_copilot(
    payload: CopilotQueryRequest,
    request: Request,
    service: CopilotService = Depends(get_copilot_service),
) -> CopilotQueryResponse:
    return service.query(payload, request.state.request_id)
