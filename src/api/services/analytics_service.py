"""Application service mapping Phase 2/3 results to the public API contract."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.analytics.analytics_pipeline import AnalyticsPipeline

from ..errors import APIError
from ..schemas.requests import AnalyticsQueryRequest
from ..schemas.responses import (
    AnalysisDetails,
    AnalyticsQueryResponse,
    ExecutionDetails,
    QueryResultDetails,
    SQLDetails,
    VisualizationDetails,
)


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class AnalyticsService:
    def __init__(self, pipeline: AnalyticsPipeline) -> None:
        self._pipeline = pipeline

    def query(
        self,
        request: AnalyticsQueryRequest,
        request_id: str,
    ) -> AnalyticsQueryResponse:
        started = time.perf_counter()
        result = self._pipeline.analyze(request.question, create_figure=False)
        query = result.query
        if query.error:
            raise _pipeline_error(query.error)
        if result.analysis is None or result.chart is None:
            raise APIError(
                500,
                "ANALYTICS_PROCESSING_FAILED",
                "The query succeeded but its result could not be analyzed.",
            )

        total_ms = (time.perf_counter() - started) * 1_000
        LOGGER.info(
            "analytics_complete request_id=%s sql_generated=%s validation=%s "
            "repair=%s rows=%d insight=%s total_ms=%.2f",
            request_id,
            query.generated_sql is not None,
            query.validation_passed,
            query.was_repaired,
            query.row_count,
            result.insight is not None,
            total_ms,
        )
        return AnalyticsQueryResponse(
            request_id=request_id,
            question=result.question,
            answer=result.insight,
            sql=(
                SQLDetails(
                    generated_sql=query.generated_sql,
                    final_sql=query.final_sql,
                    validation_passed=query.validation_passed,
                    was_repaired=query.was_repaired,
                )
                if request.include_sql
                else None
            ),
            result=QueryResultDetails(
                columns=list(query.columns),
                rows=(
                    [[_json_value(value) for value in row] for row in query.rows]
                    if request.include_rows
                    else None
                ),
                row_count=query.row_count,
                truncated=query.truncated,
            ),
            analysis=AnalysisDetails(
                result_type=result.analysis.result_type,
                dimensions=list(result.analysis.dimensions),
                metrics=list(result.analysis.metrics),
                categorical_columns=list(result.analysis.categorical_columns),
                numeric_columns=list(result.analysis.numeric_columns),
                datetime_columns=list(result.analysis.datetime_columns),
                identifier_columns=list(result.analysis.identifier_columns),
                has_datetime=result.analysis.has_datetime,
                is_empty=result.analysis.is_empty,
            ),
            visualization=(
                VisualizationDetails(**result.chart.to_dict())
                if request.include_visualization_config
                else None
            ),
            execution=ExecutionDetails(
                sql_execution_time_ms=query.execution_time_ms,
                total_request_time_ms=total_ms,
            ),
        )


def _pipeline_error(error: str) -> APIError:
    lowered = error.lower()
    database_markers = (
        "connection refused",
        "could not connect",
        "connection timed out",
        "server closed the connection",
        "password authentication failed",
        "database does not exist",
    )
    if any(marker in lowered for marker in database_markers):
        return APIError(
            503,
            "DATABASE_UNAVAILABLE",
            "The analytics database is currently unavailable.",
        )
    if "validation failed" in lowered:
        return APIError(
            400,
            "SQL_VALIDATION_FAILED",
            "The generated query was rejected by the SQL safety validator.",
        )
    if "generation failed" in lowered or "repair failed" in lowered:
        return APIError(
            503,
            "GEMINI_UNAVAILABLE",
            "Gemini could not complete the analytics request. Please retry shortly.",
        )
    if "execution failed" in lowered:
        return APIError(
            500,
            "SQL_EXECUTION_FAILED",
            "The validated query could not be executed successfully.",
        )
    return APIError(500, "ANALYTICS_FAILED", "The analytics request could not be completed.")


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value
