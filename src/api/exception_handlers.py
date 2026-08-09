"""Centralized user-safe API exception responses."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .errors import APIError


LOGGER = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        LOGGER.warning(
            "api_error request_id=%s code=%s status=%d",
            request_id,
            exc.code,
            exc.status_code,
        )
        return _error_response(exc.status_code, exc.code, exc.message, request_id)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        LOGGER.info("request_validation_failed request_id=%s", request_id)
        return _error_response(
            422,
            "REQUEST_VALIDATION_FAILED",
            "The request body is invalid. Provide a non-empty question of at most 2000 characters.",
            request_id,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        LOGGER.exception("unexpected_api_error request_id=%s", request_id)
        return _error_response(
            500,
            "INTERNAL_ERROR",
            "An unexpected internal error occurred.",
            request_id,
        )


def _error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )

