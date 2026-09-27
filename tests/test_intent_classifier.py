import pytest
from pydantic import ValidationError

from src.routing.intent_classifier import (
    IntentClassificationError,
    IntentClassifier,
    IntentDecision,
)


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FailingLLM:
    def generate(self, prompt: str) -> str:
        raise TimeoutError("provider timeout")


def test_classifier_validates_structured_hybrid_route() -> None:
    llm = FakeLLM(
        '```json\n{"intent":"hybrid","tasks":["historical_analytics",'
        '"sales_forecasting"],"sql_question":"What was total revenue?",'
        '"horizon_weeks":4}\n```'
    )
    decision = IntentClassifier(llm).classify("Compare history with the forecast")

    assert decision.intent == "hybrid"
    assert decision.sql_question == "What was total revenue?"
    assert "Do not calculate, estimate, or invent" in llm.prompts[0]


def test_invalid_llm_route_fails_closed() -> None:
    with pytest.raises(IntentClassificationError):
        IntentClassifier(FakeLLM('{"intent":"ml","tasks":[]}')).classify("predict")


def test_sql_intent_cannot_smuggle_ml_task() -> None:
    with pytest.raises(ValidationError):
        IntentDecision.model_validate(
            {"intent": "sql", "tasks": ["historical_analytics", "sales_forecasting"]}
        )


@pytest.mark.parametrize(
    ("task", "parameters"),
    [
        ("sales_forecasting", ',"horizon_weeks":2'),
        ("customer_segmentation", ""),
        (
            "late_delivery_prediction",
            ',"order_id":"e481f51cbdc54678b7cc49136f2d6af7"',
        ),
    ],
)
def test_classifier_maps_each_supported_ml_task(task: str, parameters: str) -> None:
    response = f'{{"intent":"ml","tasks":["{task}"]{parameters}}}'

    decision = IntentClassifier(FakeLLM(response)).classify("predictive question")

    assert decision.intent == "ml"
    assert [routed.value for routed in decision.tasks] == [task]


@pytest.mark.parametrize(
    ("question", "task"),
    [
        ("Predict revenue for the next 2 weeks", "sales_forecasting"),
        ("Show the customer segments", "customer_segmentation"),
        (
            "Predict late delivery risk for order e481f51cbdc54678b7cc49136f2d6af7",
            "late_delivery_prediction",
        ),
    ],
)
def test_explicit_predictive_wording_cannot_be_misrouted_as_sql(
    question: str,
    task: str,
) -> None:
    wrong_sql_route = '{"intent":"sql","tasks":["historical_analytics"]}'

    decision = IntentClassifier(FakeLLM(wrong_sql_route)).classify(question)

    assert decision.intent == "ml"
    assert [routed.value for routed in decision.tasks] == [task]
    if task == "sales_forecasting":
        assert decision.horizon_weeks == 2


def test_repeat_purchase_request_fails_closed_instead_of_using_delivery_model() -> None:
    wrong_delivery_route = '{"intent":"ml","tasks":["late_delivery_prediction"]}'

    decision = IntentClassifier(FakeLLM(wrong_delivery_route)).classify(
        "Predict whether this customer will make a repeat purchase"
    )

    assert decision.intent == "unsupported"
    assert decision.tasks == []


@pytest.mark.parametrize(
    ("question", "task", "dataset"),
    [
        (
            "What is the 95% confidence interval for average order value?",
            "confidence_interval",
            "average_order_value",
        ),
        (
            "Is average review score significantly different between delayed and on-time deliveries?",
            "hypothesis_test",
            "review_score_by_delivery_status",
        ),
        (
            "Did treatment outperform control significantly in the A/B experiment?",
            "ab_test",
            "synthetic_conversion_demo",
        ),
    ],
)
def test_statistical_questions_have_safe_fallback_routing(
    question: str,
    task: str,
    dataset: str,
) -> None:
    decision = IntentClassifier(FailingLLM()).classify(question)

    assert decision.intent == "stats"
    assert [routed.value for routed in decision.tasks] == [task]
    assert decision.statistical_dataset == dataset


def test_sample_size_parameters_are_extracted_without_calculating_result() -> None:
    decision = IntentClassifier(FailingLLM()).classify(
        "If conversion improves from 8% to 10%, how many users do we need per group?"
    )

    assert decision.intent == "stats"
    assert decision.tasks[0] == "sample_size_estimation"
    assert decision.baseline == pytest.approx(0.08)
    assert decision.minimum_detectable_effect == pytest.approx(0.02)


def test_state_hypothesis_extracts_supported_groups() -> None:
    decision = IntentClassifier(FailingLLM()).classify(
        "Is average order value significantly different between SP and RJ?"
    )

    assert decision.statistical_dataset == "order_value_by_customer_state"
    assert (decision.group_a, decision.group_b) == ("SP", "RJ")
