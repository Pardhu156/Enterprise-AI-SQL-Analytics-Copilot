import pytest

from src.text_to_sql.llm_client import LLMConfig


def test_llm_config_reads_positive_request_timeout(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-test")
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "30")

    assert LLMConfig.from_env().request_timeout_seconds == 30


def test_llm_config_rejects_invalid_request_timeout(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-test")
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValueError, match="greater than zero"):
        LLMConfig.from_env()
