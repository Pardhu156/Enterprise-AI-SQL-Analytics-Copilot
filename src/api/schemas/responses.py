"""Stable response models for analytics and operational endpoints."""

from typing import Literal

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
    sql_generation_time_ms: float | None = None
    sql_validation_time_ms: float | None = None
    sql_execution_time_ms: float | None
    sql_repair_time_ms: float | None = None
    insight_generation_time_ms: float | None = None
    text_to_sql_total_time_ms: float | None = None
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


class FeatureContributionDetails(BaseModel):
    feature: str
    feature_value: JsonValue
    contribution: float
    direction: Literal["increases", "decreases", "neutral"]


class FeatureImportanceDetails(BaseModel):
    feature: str
    importance: float


class ModelExplanationDetails(BaseModel):
    method: str
    output_space: str
    prediction_value: float | None = None
    baseline_value: float | None = None
    top_contributions: list[FeatureContributionDetails] = Field(default_factory=list)
    global_importance: list[FeatureImportanceDetails] = Field(default_factory=list)
    details: dict[str, JsonValue] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


class BusinessImpactMetric(BaseModel):
    name: str
    value: float
    unit: str
    kind: Literal["observed", "predicted", "scenario"]
    description: str
    assumptions: list[str] = Field(default_factory=list)


class BusinessRecommendationDetails(BaseModel):
    title: str
    action: str
    rationale: str
    priority: Literal["low", "medium", "high"]
    evidence: list[str] = Field(default_factory=list)


class MLPredictionResponse(BaseModel):
    task: str
    model: str
    trained_at_utc: str
    data: dict[str, JsonValue]
    metrics: dict[str, JsonValue]
    limitations: list[str]
    explanation: ModelExplanationDetails | None = None
    impact: list[BusinessImpactMetric] = Field(default_factory=list)
    recommendations: list[BusinessRecommendationDetails] = Field(default_factory=list)


class StatisticalAnalysisResponse(BaseModel):
    task: str
    dataset: str | None = None
    source: str
    description: str
    result: dict[str, JsonValue]
    limitations: list[str] = Field(default_factory=list)
    impact: list[BusinessImpactMetric] = Field(default_factory=list)
    recommendations: list[BusinessRecommendationDetails] = Field(default_factory=list)


class RoutingDetails(BaseModel):
    intent: str
    tasks: list[str]


class CopilotQueryResponse(BaseModel):
    request_id: str
    question: str
    route: RoutingDetails
    answer: str
    historical: AnalyticsQueryResponse | None = None
    predictions: list[MLPredictionResponse] = Field(default_factory=list)
    statistical_analyses: list[StatisticalAnalysisResponse] = Field(default_factory=list)
    business_recommendations: list[BusinessRecommendationDetails] = Field(default_factory=list)


class BusinessOverviewResponse(BaseModel):
    generated_at_utc: str
    observed_kpis: list[BusinessImpactMetric]
    monthly_revenue: list[dict[str, JsonValue]]
    forecast: MLPredictionResponse
    customer_segments: MLPredictionResponse
    classification_summary: dict[str, JsonValue]
    limitations: list[str] = Field(default_factory=list)


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
