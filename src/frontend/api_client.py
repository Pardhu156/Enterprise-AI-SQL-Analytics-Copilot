"""Typed HTTP client used exclusively by the Streamlit frontend."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from src.api.schemas.responses import AnalyticsQueryResponse


@dataclass(frozen=True)
class FrontendAPIConfig:
    base_url: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "FrontendAPIConfig":
        load_dotenv()
        base_url = os.getenv("BACKEND_API_URL", "http://localhost:8000").strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("BACKEND_API_URL must be an HTTP or HTTPS URL")
        raw_timeout = os.getenv("API_REQUEST_TIMEOUT_SECONDS", "60")
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise ValueError("API_REQUEST_TIMEOUT_SECONDS must be numeric") from exc
        if timeout <= 0:
            raise ValueError("API_REQUEST_TIMEOUT_SECONDS must be greater than zero")
        return cls(base_url=base_url, timeout_seconds=timeout)


class FrontendAPIError(RuntimeError):
    def __init__(self, code: str, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id


class AnalyticsAPIClient:
    def __init__(
        self,
        config: FrontendAPIConfig | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config or FrontendAPIConfig.from_env()
        self._client = client or httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_seconds,
        )

    def query(self, question: str) -> AnalyticsQueryResponse:
        try:
            response = self._client.post(
                "/api/v1/analytics/query",
                json={"question": question},
            )
        except httpx.TimeoutException as exc:
            raise FrontendAPIError(
                "ANALYSIS_TIMEOUT",
                "Analysis timed out. Please try again.",
            ) from exc
        except httpx.RequestError as exc:
            raise FrontendAPIError(
                "BACKEND_UNAVAILABLE",
                "The analytics API is unavailable. Start FastAPI and try again.",
            ) from exc

        if response.is_error:
            raise _response_error(response)
        try:
            return AnalyticsQueryResponse.model_validate(response.json())
        except (ValueError, TypeError) as exc:
            raise FrontendAPIError(
                "INVALID_API_RESPONSE",
                "The analytics API returned an invalid response.",
            ) from exc

    def close(self) -> None:
        self._client.close()


def _response_error(response: httpx.Response) -> FrontendAPIError:
    try:
        payload = response.json().get("error", {})
    except ValueError:
        payload = {}
    return FrontendAPIError(
        code=str(payload.get("code", "API_ERROR")),
        message=str(payload.get("message", "The analytics request failed.")),
        request_id=payload.get("request_id"),
    )

