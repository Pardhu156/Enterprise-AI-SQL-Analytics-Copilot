import pytest

from scripts.validate_environment import validate_backend, validate_frontend


def set_valid_environment(monkeypatch) -> None:
    values = {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "olist",
        "DB_USER": "reader",
        "DB_PASSWORD": "test-only",
        "LLM_PROVIDER": "gemini",
        "LLM_MODEL": "gemini-test",
        "LLM_API_KEY": "test-only",
        "API_ALLOWED_ORIGINS": "http://localhost:8501",
        "BACKEND_API_URL": "http://localhost:8000",
        "API_REQUEST_TIMEOUT_SECONDS": "60",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_runtime_configuration_validation_accepts_complete_environment(monkeypatch) -> None:
    set_valid_environment(monkeypatch)
    validate_backend()
    validate_frontend()


def test_runtime_configuration_rejects_non_gemini_provider(monkeypatch) -> None:
    set_valid_environment(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")
    with pytest.raises(ValueError, match="gemini"):
        validate_backend()
