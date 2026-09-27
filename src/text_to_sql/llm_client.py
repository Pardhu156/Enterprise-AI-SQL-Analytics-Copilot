"""Testable LLM protocol and the project's Google Gemini implementation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Return model-generated text for a complete prompt."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    request_timeout_seconds: float = 45.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        missing = [
            name
            for name, value in (("LLM_MODEL", model), ("LLM_API_KEY", api_key))
            if not value
        ]
        if missing:
            raise ValueError("Missing required LLM environment variables: " + ", ".join(missing))
        raw_timeout = os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "45")
        try:
            request_timeout = float(raw_timeout)
        except ValueError as exc:
            raise ValueError("LLM_REQUEST_TIMEOUT_SECONDS must be numeric") from exc
        if request_timeout <= 0:
            raise ValueError("LLM_REQUEST_TIMEOUT_SECONDS must be greater than zero")
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            request_timeout_seconds=request_timeout,
        )


class GeminiClient:
    """Thin adapter around the official Google Gen AI Python SDK."""

    def __init__(self, model: str, api_key: str, request_timeout_seconds: float = 45.0) -> None:
        from google import genai
        from google.genai import types

        self._model = model
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(request_timeout_seconds * 1_000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._generation_config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2000,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._generation_config,
        )
        if not response.text:
            raise RuntimeError("Gemini returned no text content")
        return response.text


def create_llm_client(config: LLMConfig | None = None) -> LLMClient:
    resolved = config or LLMConfig.from_env()
    if resolved.provider == "gemini":
        return GeminiClient(
            model=resolved.model,
            api_key=resolved.api_key,
            request_timeout_seconds=resolved.request_timeout_seconds,
        )
    raise ValueError(f"Unsupported LLM_PROVIDER {resolved.provider!r}. This project uses 'gemini'.")
