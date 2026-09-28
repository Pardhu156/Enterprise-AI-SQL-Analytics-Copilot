"""Gemini intent classification with strict Pydantic validation."""

from __future__ import annotations

import json
import re
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from src.text_to_sql.llm_client import LLMClient
from src.statistical.service import StatisticalDataset


class IntentClassificationError(RuntimeError):
    pass


class IntentType(StrEnum):
    SQL = "sql"
    ML = "ml"
    STATS = "stats"
    HYBRID = "hybrid"
    UNSUPPORTED = "unsupported"


class RoutedTask(StrEnum):
    HISTORICAL_ANALYTICS = "historical_analytics"
    SALES_FORECASTING = "sales_forecasting"
    CUSTOMER_SEGMENTATION = "customer_segmentation"
    LATE_DELIVERY_PREDICTION = "late_delivery_prediction"
    HYPOTHESIS_TEST = "hypothesis_test"
    CONFIDENCE_INTERVAL = "confidence_interval"
    AB_TEST = "ab_test"
    SAMPLE_SIZE_ESTIMATION = "sample_size_estimation"


class IntentDecision(BaseModel):
    intent: IntentType
    tasks: list[RoutedTask] = Field(default_factory=list, max_length=8)
    sql_question: str | None = Field(default=None, max_length=2_000)
    customer_unique_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    order_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    horizon_weeks: int = Field(default=4, ge=1, le=4)
    statistical_dataset: StatisticalDataset | None = None
    group_a: str | None = Field(default=None, max_length=80)
    group_b: str | None = Field(default=None, max_length=80)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    alternative: str = Field(default="two-sided", pattern=r"^(two-sided|greater|less)$")
    metric_type: str | None = Field(default=None, pattern=r"^(proportion|mean)$")
    baseline: float | None = None
    minimum_detectable_effect: float | None = None
    standard_deviation: float | None = Field(default=None, gt=0)
    power: float = Field(default=0.8, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_route(self) -> "IntentDecision":
        task_set = set(self.tasks)
        historical = RoutedTask.HISTORICAL_ANALYTICS in task_set
        ml_tasks = task_set & {
            RoutedTask.SALES_FORECASTING,
            RoutedTask.CUSTOMER_SEGMENTATION,
            RoutedTask.LATE_DELIVERY_PREDICTION,
        }
        stats_tasks = task_set & {
            RoutedTask.HYPOTHESIS_TEST,
            RoutedTask.CONFIDENCE_INTERVAL,
            RoutedTask.AB_TEST,
            RoutedTask.SAMPLE_SIZE_ESTIMATION,
        }
        if len(task_set) != len(self.tasks):
            raise ValueError("Routing tasks must be unique")
        if self.intent == IntentType.SQL and task_set != {RoutedTask.HISTORICAL_ANALYTICS}:
            raise ValueError("SQL intent requires only historical_analytics")
        if self.intent == IntentType.ML and (historical or stats_tasks or not ml_tasks):
            raise ValueError("ML intent requires one or more ML tasks")
        if self.intent == IntentType.STATS and (historical or ml_tasks or not stats_tasks):
            raise ValueError("Stats intent requires one or more statistical tasks")
        if self.intent == IntentType.HYBRID and (
            not historical or not (ml_tasks or stats_tasks)
        ):
            raise ValueError("Hybrid intent requires historical plus ML or statistical tasks")
        if self.intent == IntentType.UNSUPPORTED and self.tasks:
            raise ValueError("Unsupported intent must not include tasks")
        if stats_tasks & {
            RoutedTask.HYPOTHESIS_TEST,
            RoutedTask.CONFIDENCE_INTERVAL,
            RoutedTask.AB_TEST,
        } and self.statistical_dataset is None:
            raise ValueError("This statistical task requires a supported statistical_dataset")
        if RoutedTask.SAMPLE_SIZE_ESTIMATION in stats_tasks and (
            self.metric_type is None
            or self.baseline is None
            or self.minimum_detectable_effect is None
        ):
            raise ValueError("Sample-size routing requires metric_type, baseline, and effect")
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
            return _reconcile_explicit_intent(question, decision)
        except Exception as exc:
            fallback = _deterministic_statistical_intent(question)
            if fallback is not None:
                return fallback
            raise IntentClassificationError(
                "Gemini could not return a valid supported routing decision"
            ) from exc


def _classification_prompt(question: str) -> str:
    return f"""You are an intent router for an e-commerce SQL, ML, and statistical analytics application.
Return exactly one JSON object and no Markdown.

Allowed intent values: sql, ml, stats, hybrid, unsupported.
Allowed tasks:
- historical_analytics: past/current facts answerable from PostgreSQL
- sales_forecasting: predict 1-4 future weekly revenue periods
- customer_segmentation: segment summary or a customer_unique_id lookup
- late_delivery_prediction: risk for an existing order_id using purchase-time features
- hypothesis_test: compare Olist groups using Python statistical tests
- confidence_interval: estimate an Olist mean or proportion interval
- ab_test: analyze an explicit experiment; Olist has only a synthetic demo
- sample_size_estimation: prospective power analysis from supplied assumptions

Supported statistical_dataset values:
- review_score_by_delivery_status
- order_value_by_customer_state
- delivery_time_by_customer_state
- repeat_purchase_by_customer_state
- average_order_value
- delayed_delivery_rate
- average_review_score
- synthetic_conversion_demo

Rules:
- SQL questions use intent=sql and tasks=["historical_analytics"].
- Predictive questions use intent=ml and only the required ML task(s).
- Statistical questions use intent=stats and the relevant statistical task.
- Questions combining a normal historical answer with ML or statistics use intent=hybrid.
- For hybrid questions, sql_question must contain only the historical portion.
- Copy customer_unique_id or order_id exactly when explicitly supplied. Never invent identifiers.
- horizon_weeks defaults to 4 and must be 1-4.
- Use unsupported when the request is outside these capabilities.
- "Predict revenue for the next 2 weeks" is ml + sales_forecasting, never SQL.
- "Show customer segments" is ml + customer_segmentation, never SQL.
- "Predict late-delivery risk for order <id>" is ml + late_delivery_prediction.
- Repeat-purchase prediction is not deployed; route it as unsupported, not late delivery.
- "95% confidence interval for average order value" uses confidence_interval and average_order_value.
- "Are delayed reviews significantly lower?" uses hypothesis_test and review_score_by_delivery_status.
- State comparisons require two uppercase state abbreviations in group_a and group_b.
- "If conversion improves from 8% to 10%, how many users per group?" uses sample_size_estimation, metric_type=proportion, baseline=0.08, minimum_detectable_effect=0.02.
- Olist has no randomized experiment. Generic treatment/control questions may use synthetic_conversion_demo only when clearly labeled as a demo.
- Requests asking why a forecast changes still use sales_forecasting; Python supplies SHAP explanations.
- Requests asking what to do with a segment still use customer_segmentation; Python supplies recommendations.
- "Are delayed deliveries associated with lower ratings?" uses hypothesis_test and review_score_by_delivery_status.
- Do not calculate, estimate, or invent any numerical answer.

JSON shape:
{{
  "intent": "sql|ml|stats|hybrid|unsupported",
  "tasks": ["..."],
  "sql_question": null,
  "customer_unique_id": null,
  "order_id": null,
  "horizon_weeks": 4,
  "statistical_dataset": null,
  "group_a": null,
  "group_b": null,
  "confidence_level": 0.95,
  "alpha": 0.05,
  "alternative": "two-sided",
  "metric_type": null,
  "baseline": null,
  "minimum_detectable_effect": null,
  "standard_deviation": null,
  "power": 0.8
}}

User question: {question}
"""


def _reconcile_explicit_intent(
    question: str,
    decision: IntentDecision,
) -> IntentDecision:
    """Fail closed or correct unmistakable supported wording when the LLM misroutes it."""
    lowered = question.lower()
    statistical = _deterministic_statistical_intent(question)
    if statistical is not None and not any(
        task in decision.tasks
        for task in {
            RoutedTask.HYPOTHESIS_TEST,
            RoutedTask.CONFIDENCE_INTERVAL,
            RoutedTask.AB_TEST,
            RoutedTask.SAMPLE_SIZE_ESTIMATION,
        }
    ):
        return statistical
    if re.search(r"\b(repeat[- ]?purchase|repurchase|buy again)\b", lowered):
        return IntentDecision(intent=IntentType.UNSUPPORTED, tasks=[])

    explicit_task: RoutedTask | None = None
    forecast_terms = re.search(
        r"\b(forecast|predict(?:ion|ed|ing)?|expect(?:ed|ation)?)\b",
        lowered,
    )
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


def _deterministic_statistical_intent(question: str) -> IntentDecision | None:
    lowered = question.lower()
    states = _extract_state_codes(question)
    common: dict = {"intent": IntentType.STATS, "alpha": 0.05, "power": 0.8}
    if "confidence interval" in lowered:
        dataset = None
        if "order value" in lowered:
            dataset = StatisticalDataset.AVERAGE_ORDER_VALUE
        elif "review" in lowered:
            dataset = StatisticalDataset.AVERAGE_REVIEW_SCORE
        elif "delay" in lowered or "late" in lowered:
            dataset = StatisticalDataset.DELAYED_DELIVERY_RATE
        if dataset is not None:
            confidence_match = re.search(r"\b(\d{2}(?:\.\d+)?)\s*%\s*confidence", lowered)
            confidence = float(confidence_match.group(1)) / 100 if confidence_match else 0.95
            return IntentDecision(
                **common,
                tasks=[RoutedTask.CONFIDENCE_INTERVAL],
                statistical_dataset=dataset,
                confidence_level=confidence,
            )

    sample_size_terms = "sample size" in lowered or (
        "how many" in lowered and re.search(r"\b(users?|customers?|observations?|people)\b", lowered)
    )
    if sample_size_terms:
        percentages = [float(value) / 100 for value in re.findall(r"(\d+(?:\.\d+)?)\s*%", lowered)]
        if len(percentages) >= 2:
            return IntentDecision(
                **common,
                tasks=[RoutedTask.SAMPLE_SIZE_ESTIMATION],
                metric_type="proportion",
                baseline=percentages[0],
                minimum_detectable_effect=percentages[1] - percentages[0],
            )

    hypothesis_terms = re.search(
        r"\b(significant|significantly|hypothesis|different between|difference between|associated|association)\b",
        lowered,
    )
    if hypothesis_terms:
        dataset = None
        group_a = group_b = None
        if ("review" in lowered or "rating" in lowered) and (
            "delay" in lowered or "on-time" in lowered or "on time" in lowered
        ):
            dataset = StatisticalDataset.REVIEW_SCORE_BY_DELIVERY_STATUS
        elif len(states) >= 2 and "order value" in lowered:
            dataset = StatisticalDataset.ORDER_VALUE_BY_CUSTOMER_STATE
            group_a, group_b = states[:2]
        elif len(states) >= 2 and "delivery" in lowered and "time" in lowered:
            dataset = StatisticalDataset.DELIVERY_TIME_BY_CUSTOMER_STATE
            group_a, group_b = states[:2]
        elif len(states) >= 2 and re.search(r"repeat|repurchase", lowered):
            dataset = StatisticalDataset.REPEAT_PURCHASE_BY_CUSTOMER_STATE
            group_a, group_b = states[:2]
        if dataset is not None:
            return IntentDecision(
                **common,
                tasks=[RoutedTask.HYPOTHESIS_TEST],
                statistical_dataset=dataset,
                group_a=group_a,
                group_b=group_b,
            )

    if re.search(r"\b(a/?b|treatment|control)\b", lowered) and re.search(
        r"\b(test|experiment|significant|outperform)\b", lowered
    ):
        return IntentDecision(
            **common,
            tasks=[RoutedTask.AB_TEST],
            statistical_dataset=StatisticalDataset.SYNTHETIC_CONVERSION_DEMO,
        )
    return None


def _extract_state_codes(question: str) -> list[str]:
    name_map = {
        "são paulo": "SP",
        "sao paulo": "SP",
        "rio de janeiro": "RJ",
        "minas gerais": "MG",
        "paraná": "PR",
        "parana": "PR",
        "bahia": "BA",
    }
    lowered = question.lower()
    extracted = [code for name, code in name_map.items() if name in lowered]
    valid_codes = {
        "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
        "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
        "SP", "SE", "TO",
    }
    for code in re.findall(r"\b[A-Z]{2}\b", question):
        if code in valid_codes and code not in extracted:
            extracted.append(code)
    return extracted


def _extract_json_object(value: str) -> dict:
    start = value.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    payload, _ = json.JSONDecoder().raw_decode(value[start:])
    if not isinstance(payload, dict):
        raise ValueError("Routing output must be a JSON object")
    return payload
