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
        return cls(provider=provider, model=model, api_key=api_key)


class GeminiClient:
    """Thin adapter around the official Google Gen AI Python SDK."""

    def __init__(self, model: str, api_key: str) -> None:
        from google import genai
        from google.genai import types

        self._model = model
        self._client = genai.Client(api_key=api_key)
        self._generation_config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2000,
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
        return GeminiClient(model=resolved.model, api_key=resolved.api_key)
    raise ValueError(f"Unsupported LLM_PROVIDER {resolved.provider!r}. This project uses 'gemini'.")
