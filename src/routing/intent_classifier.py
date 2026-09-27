"""Gemini intent classification with strict Pydantic validation."""

from __future__ import annotations

import json
import re
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from src.text_to_sql.llm_client import LLMClient


class IntentClassificationError(RuntimeError):
    pass


class IntentType(StrEnum):
    SQL = "sql"
    ML = "ml"
    HYBRID = "hybrid"
    UNSUPPORTED = "unsupported"


class RoutedTask(StrEnum):
    HISTORICAL_ANALYTICS = "historical_analytics"
    SALES_FORECASTING = "sales_forecasting"
    CUSTOMER_SEGMENTATION = "customer_segmentation"
    LATE_DELIVERY_PREDICTION = "late_delivery_prediction"


class IntentDecision(BaseModel):
    intent: IntentType
    tasks: list[RoutedTask] = Field(default_factory=list, max_length=4)
    sql_question: str | None = Field(default=None, max_length=2_000)
    customer_unique_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    order_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    horizon_weeks: int = Field(default=4, ge=1, le=4)

    @model_validator(mode="after")
    def validate_route(self) -> "IntentDecision":
        task_set = set(self.tasks)
        historical = RoutedTask.HISTORICAL_ANALYTICS in task_set
        ml_tasks = task_set - {RoutedTask.HISTORICAL_ANALYTICS}
        if len(task_set) != len(self.tasks):
            raise ValueError("Routing tasks must be unique")
        if self.intent == IntentType.SQL and (not historical or ml_tasks):
            raise ValueError("SQL intent requires only historical_analytics")
        if self.intent == IntentType.ML and (historical or not ml_tasks):
            raise ValueError("ML intent requires one or more ML tasks")
        if self.intent == IntentType.HYBRID and (not historical or not ml_tasks):
            raise ValueError("Hybrid intent requires historical and ML tasks")
        if self.intent == IntentType.UNSUPPORTED and self.tasks:
            raise ValueError("Unsupported intent must not include tasks")
        return self


class IntentClassifier:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def classify(self, question: str) -> IntentDecision:
        prompt = _classification_prompt(question)
        try:
            raw = self._llm_client.generate(prompt)
            payload = _extract_json_object(raw)
            decision = IntentDecision.model_validate(payload)
            return _reconcile_explicit_predictive_intent(question, decision)
        except Exception as exc:
            raise IntentClassificationError(
                "Gemini could not return a valid supported routing decision"
            ) from exc


def _classification_prompt(question: str) -> str:
    return f"""You are an intent router for an e-commerce SQL and ML analytics application.
Return exactly one JSON object and no Markdown.

Allowed intent values: sql, ml, hybrid, unsupported.
Allowed tasks:
- historical_analytics: past/current facts answerable from PostgreSQL
- sales_forecasting: predict 1-4 future weekly revenue periods
- customer_segmentation: segment summary or a customer_unique_id lookup
- late_delivery_prediction: risk for an existing order_id using purchase-time features

Rules:
- SQL questions use intent=sql and tasks=["historical_analytics"].
- Predictive questions use intent=ml and only the required ML task(s).
- Questions combining history and prediction use intent=hybrid and include historical_analytics.
- For hybrid questions, sql_question must contain only the historical portion.
- Copy customer_unique_id or order_id exactly when explicitly supplied. Never invent identifiers.
- horizon_weeks defaults to 4 and must be 1-4.
- Use unsupported when the request is outside these capabilities.
- "Predict revenue for the next 2 weeks" is ml + sales_forecasting, never SQL.
- "Show customer segments" is ml + customer_segmentation, never SQL.
- "Predict late-delivery risk for order <id>" is ml + late_delivery_prediction.
- Repeat-purchase prediction is not deployed; route it as unsupported, not late delivery.
- Do not calculate, estimate, or invent any numerical answer.

JSON shape:
{{
  "intent": "sql|ml|hybrid|unsupported",
  "tasks": ["..."],
  "sql_question": null,
  "customer_unique_id": null,
  "order_id": null,
  "horizon_weeks": 4
}}

User question: {question}
"""


def _reconcile_explicit_predictive_intent(
    question: str,
    decision: IntentDecision,
) -> IntentDecision:
    """Fail closed or correct unmistakable ML wording when the LLM misroutes it."""
    lowered = question.lower()
    if re.search(r"\b(repeat[- ]?purchase|repurchase|buy again)\b", lowered):
        return IntentDecision(intent=IntentType.UNSUPPORTED, tasks=[])

    explicit_task: RoutedTask | None = None
    forecast_terms = re.search(r"\b(forecast|predict(?:ion|ed|ing)?)\b", lowered)
    if forecast_terms and re.search(r"\b(revenue|sales)\b", lowered):
        explicit_task = RoutedTask.SALES_FORECASTING
    elif re.search(r"\b(customer\s+segments?|segment(?:ation)?)\b", lowered):
        explicit_task = RoutedTask.CUSTOMER_SEGMENTATION
    elif (
        re.search(r"\b(late|delay(?:ed)?)\b", lowered)
        and re.search(r"\b(delivery|deliveries)\b", lowered)
        and re.search(r"\b(predict|risk|probability|likely|will)\b", lowered)
    ):
        explicit_task = RoutedTask.LATE_DELIVERY_PREDICTION

    if explicit_task is None or explicit_task in decision.tasks:
        return decision
    # Preserve a valid Gemini hybrid decision; extracting arbitrary historical subquestions
    # deterministically would be less reliable than the validated model output.
    if decision.intent == IntentType.HYBRID:
        return decision

    horizon_match = re.search(r"\b(?:next\s+)?([1-4])\s+weeks?\b", lowered)
    horizon = int(horizon_match.group(1)) if horizon_match else decision.horizon_weeks
    identifier_match = re.search(r"\b[0-9a-f]{32}\b", lowered)
    identifier = identifier_match.group(0) if identifier_match else None
    return IntentDecision(
        intent=IntentType.ML,
        tasks=[explicit_task],
        horizon_weeks=horizon,
        customer_unique_id=(
            decision.customer_unique_id
            or (identifier if explicit_task == RoutedTask.CUSTOMER_SEGMENTATION else None)
        ),
        order_id=(
            decision.order_id
            or (identifier if explicit_task == RoutedTask.LATE_DELIVERY_PREDICTION else None)
        ),
    )


def _extract_json_object(value: str) -> dict:
    start = value.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    payload, _ = json.JSONDecoder().raw_decode(value[start:])
    if not isinstance(payload, dict):
        raise ValueError("Routing output must be a JSON object")
    return payload
