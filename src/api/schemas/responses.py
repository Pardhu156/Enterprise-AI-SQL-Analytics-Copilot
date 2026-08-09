"""Stable response models for analytics and operational endpoints."""

from pydantic import BaseModel, Field, JsonValue


class SQLDetails(BaseModel):
    generated_sql: str | None
    final_sql: str | None
    validation_passed: bool
    was_repaired: bool


class QueryResultDetails(BaseModel):
    columns: list[str]
    rows: list[list[JsonValue]] | None
    row_count: int
    truncated: bool


class AnalysisDetails(BaseModel):
    result_type: str
    dimensions: list[str]
    metrics: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    datetime_columns: list[str]
    identifier_columns: list[str]
    has_datetime: bool
    is_empty: bool


class VisualizationDetails(BaseModel):
    chart_type: str
    x: str | None = None
    y: str | None = None
    title: str
    reason: str


class ExecutionDetails(BaseModel):
    sql_execution_time_ms: float | None
    total_request_time_ms: float


class AnalyticsQueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str | None
    sql: SQLDetails | None
    result: QueryResultDetails
    analysis: AnalysisDetails
    visualization: VisualizationDetails | None
    execution: ExecutionDetails


class ErrorDetails(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetails


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, str]
