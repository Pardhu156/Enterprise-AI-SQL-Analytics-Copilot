"""Liveness and lightweight readiness endpoints."""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from ..dependencies import ReadinessChecker, get_readiness_checker
from ..schemas.responses import HealthResponse, ReadinessResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="API liveness")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Database and Gemini configuration readiness",
)
def readiness(
    checker: ReadinessChecker = Depends(get_readiness_checker),
) -> ReadinessResponse | JSONResponse:
    ready, checks = checker.check()
    payload = ReadinessResponse(
        status="ready" if ready else "not_ready",
        checks=checks,
    )
    if ready:
        return payload
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(mode="json"),
    )

