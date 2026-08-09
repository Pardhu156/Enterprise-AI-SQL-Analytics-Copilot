"""FastAPI application factory and default ASGI application."""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import APISettings, get_api_settings
from .exception_handlers import register_exception_handlers
from .routes.analytics import router as analytics_router
from .routes.health import router as health_router


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _configure_api_logging() -> None:
    api_logger = logging.getLogger("src.api")
    api_logger.setLevel(logging.INFO)
    api_logger.propagate = False
    if not api_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        api_logger.addHandler(handler)


def create_app(settings: APISettings | None = None) -> FastAPI:
    _configure_api_logging()
    resolved = settings or get_api_settings()
    application = FastAPI(
        title="Enterprise AI SQL Analytics Copilot API",
        version="1.0.0",
        description=(
            "HTTP service for safe Gemini Text-to-SQL, PostgreSQL analytics, "
            "grounded business insights, and deterministic visualization metadata."
        ),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "api_request endpoint=%s request_id=%s status=%d duration_ms=%.2f",
            request.url.path,
            request_id,
            response.status_code,
            elapsed_ms,
        )
        return response

    register_exception_handlers(application)
    application.include_router(health_router)
    application.include_router(analytics_router)
    return application


app = create_app()
