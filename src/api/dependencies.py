"""Construction and operational dependencies for the FastAPI application."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import psycopg2
from dotenv import load_dotenv

from src.analytics.analytics_pipeline import AnalyticsPipeline
from src.analytics.chart_selector import ChartSelector
from src.analytics.insight_generator import InsightGenerator
from src.analytics.result_analyzer import ResultAnalyzer
from src.analytics.visualization import VisualizationEngine
from src.db_config import DatabaseConfig
from src.ml.config import MLSettings
from src.ml.service import MLInferenceService
from src.routing.intent_classifier import IntentClassifier
from src.text_to_sql.llm_client import LLMClient, LLMConfig, create_llm_client
from src.text_to_sql.pipeline import TextToSQLPipeline

from .services.analytics_service import AnalyticsService
from .services.copilot_service import CopilotService
from .services.ml_service import MLService


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class APISettings:
    host: str
    port: int
    allowed_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "APISettings":
        load_dotenv()
        origins = tuple(
            origin.strip().rstrip("/")
            for origin in os.getenv(
                "API_ALLOWED_ORIGINS",
                "http://localhost:8501,http://127.0.0.1:8501",
            ).split(",")
            if origin.strip()
        )
        if not origins or "*" in origins:
            raise ValueError("API_ALLOWED_ORIGINS must contain explicit origins, not '*'")
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=_positive_int_env("API_PORT", 8_000),
            allowed_origins=origins,
        )


@lru_cache(maxsize=1)
def get_api_settings() -> APISettings:
    return APISettings.from_env()


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    return create_llm_client()


@lru_cache(maxsize=1)
def get_analytics_service() -> AnalyticsService:
    llm_client = get_llm_client()
    text_to_sql = TextToSQLPipeline.from_env(llm_client=llm_client)
    pipeline = AnalyticsPipeline(
        text_to_sql=text_to_sql,
        analyzer=ResultAnalyzer(),
        chart_selector=ChartSelector(),
        visualization=VisualizationEngine(
            max_points=_positive_int_env("CHART_MAX_POINTS", 100)
        ),
        insight_generator=InsightGenerator(llm_client),
    )
    return AnalyticsService(pipeline)


@lru_cache(maxsize=1)
def get_ml_service() -> MLService:
    return MLService(MLInferenceService())


@lru_cache(maxsize=1)
def get_copilot_service() -> CopilotService:
    return CopilotService(
        classifier=IntentClassifier(get_llm_client()),
        analytics=get_analytics_service(),
        ml=get_ml_service(),
    )


class ReadinessChecker:
    def check(self) -> tuple[bool, dict[str, str]]:
        checks = {
            "postgresql": self._check_database(),
            "gemini_configuration": self._check_gemini_configuration(),
            "ml_artifacts": self._check_ml_artifacts(),
        }
        return all(value == "ok" for value in checks.values()), checks

    @staticmethod
    def _check_database() -> str:
        try:
            config = DatabaseConfig.from_env()
            with psycopg2.connect(
                **config.as_connect_kwargs(),
                connect_timeout=3,
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            return "ok"
        except Exception as exc:
            LOGGER.warning("Readiness database check failed: %s", type(exc).__name__)
            return "unavailable"

    @staticmethod
    def _check_gemini_configuration() -> str:
        try:
            config = LLMConfig.from_env()
            return "ok" if config.provider == "gemini" else "invalid"
        except Exception as exc:
            LOGGER.warning("Readiness Gemini configuration check failed: %s", type(exc).__name__)
            return "invalid"

    @staticmethod
    def _check_ml_artifacts() -> str:
        try:
            settings = MLSettings.from_env()
            names = ("sales_forecast", "customer_segmentation", "delivery_risk")
            return (
                "ok"
                if all(settings.artifact_path(name).is_file() for name in names)
                else "unavailable"
            )
        except Exception as exc:
            LOGGER.warning("Readiness ML artifact check failed: %s", type(exc).__name__)
            return "unavailable"


@lru_cache(maxsize=1)
def get_readiness_checker() -> ReadinessChecker:
    return ReadinessChecker()


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
