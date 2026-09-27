"""Application-independent orchestration for Olist-backed statistical tasks."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from .ab_testing import synthetic_conversion_demo
from .data_repository import StatisticalDataRepository
from .engine import (
    Alternative,
    StatisticalInputError,
    confidence_interval_mean,
    confidence_interval_proportion,
    test_continuous_groups,
    test_two_proportions,
)
from .sample_size import (
    estimate_two_mean_sample_size,
    estimate_two_proportion_sample_size,
)


class StatisticalDataset(StrEnum):
    REVIEW_SCORE_BY_DELIVERY_STATUS = "review_score_by_delivery_status"
    ORDER_VALUE_BY_CUSTOMER_STATE = "order_value_by_customer_state"
    DELIVERY_TIME_BY_CUSTOMER_STATE = "delivery_time_by_customer_state"
    REPEAT_PURCHASE_BY_CUSTOMER_STATE = "repeat_purchase_by_customer_state"
    AVERAGE_ORDER_VALUE = "average_order_value"
    DELAYED_DELIVERY_RATE = "delayed_delivery_rate"
    AVERAGE_REVIEW_SCORE = "average_review_score"
    SYNTHETIC_CONVERSION_DEMO = "synthetic_conversion_demo"


class StatisticalAnalysisService:
    def __init__(self, repository: StatisticalDataRepository | None = None) -> None:
        self._repository = repository or StatisticalDataRepository()

    def hypothesis_test(
        self,
        dataset: StatisticalDataset,
        *,
        group_a: str | None = None,
        group_b: str | None = None,
        alpha: float = 0.05,
        alternative: Alternative = "two-sided",
    ) -> dict[str, Any]:
        if dataset == StatisticalDataset.REVIEW_SCORE_BY_DELIVERY_STATUS:
            frame = self._repository.review_scores_by_delivery_status()
            result = test_continuous_groups(
                _group_values(frame, "Delayed"),
                _group_values(frame, "On time"),
                group_a_name="Delayed",
                group_b_name="On time",
                alpha=alpha,
                alternative=alternative,
                scale="ordinal",
            )
            description = "Order-level average review scores grouped by delivery timeliness."
        elif dataset in {
            StatisticalDataset.ORDER_VALUE_BY_CUSTOMER_STATE,
            StatisticalDataset.DELIVERY_TIME_BY_CUSTOMER_STATE,
        }:
            state_a, state_b = _validate_states(group_a, group_b)
            frame = (
                self._repository.order_values_by_state(state_a, state_b)
                if dataset == StatisticalDataset.ORDER_VALUE_BY_CUSTOMER_STATE
                else self._repository.delivery_times_by_state(state_a, state_b)
            )
            result = test_continuous_groups(
                _group_values(frame, state_a),
                _group_values(frame, state_b),
                group_a_name=state_a,
                group_b_name=state_b,
                alpha=alpha,
                alternative=alternative,
            )
            description = (
                "Merchandise order value by customer state."
                if dataset == StatisticalDataset.ORDER_VALUE_BY_CUSTOMER_STATE
                else "Observed delivery duration by customer state."
            )
        elif dataset == StatisticalDataset.REPEAT_PURCHASE_BY_CUSTOMER_STATE:
            state_a, state_b = _validate_states(group_a, group_b)
            frame = self._repository.repeat_purchase_counts_by_state(state_a, state_b)
            counts = frame.set_index("group_name")
            if state_a not in counts.index or state_b not in counts.index:
                raise StatisticalInputError("Both states must contain eligible customers")
            result = test_two_proportions(
                int(counts.loc[state_a, "successes"]),
                int(counts.loc[state_a, "trials"]),
                int(counts.loc[state_b, "successes"]),
                int(counts.loc[state_b, "trials"]),
                group_a_name=state_a,
                group_b_name=state_b,
                alpha=alpha,
                alternative=alternative,
            )
            description = "Historical repeat-purchase proportions by customer state."
        else:
            raise StatisticalInputError(f"Dataset {dataset.value!r} is not a hypothesis dataset")
        return _observational_payload("hypothesis_test", dataset, description, result.as_dict())

    def confidence_interval(
        self,
        dataset: StatisticalDataset,
        *,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        if dataset == StatisticalDataset.AVERAGE_ORDER_VALUE:
            interval = confidence_interval_mean(
                self._repository.order_values(),
                confidence_level,
                "average merchandise order value",
            )
            description = "Order-level merchandise value for non-canceled, available orders."
        elif dataset == StatisticalDataset.AVERAGE_REVIEW_SCORE:
            interval = confidence_interval_mean(
                self._repository.review_scores(),
                confidence_level,
                "average order review score",
            )
            description = "Order-level average review score from available reviews."
        elif dataset == StatisticalDataset.DELAYED_DELIVERY_RATE:
            successes, trials = self._repository.delayed_delivery_counts()
            interval = confidence_interval_proportion(
                successes,
                trials,
                confidence_level,
                "the delayed-delivery rate",
            )
            description = "Delivered orders with complete actual and estimated delivery dates."
        else:
            raise StatisticalInputError(
                f"Dataset {dataset.value!r} is not a confidence-interval dataset"
            )
        return {
            "task": "confidence_interval",
            "dataset": dataset.value,
            "source": "Olist PostgreSQL observational data",
            "description": description,
            "result": interval.as_dict(),
            "limitations": [
                "The interval quantifies sampling uncertainty under the stated method; it does not remove source-data bias.",
                "The Olist snapshot is historical and should not be interpreted as current market performance.",
            ],
        }

    def ab_demo(self) -> dict[str, Any]:
        result = synthetic_conversion_demo()
        return {
            "task": "ab_test",
            "dataset": StatisticalDataset.SYNTHETIC_CONVERSION_DEMO.value,
            "source": "Synthetic demonstration data",
            "description": "A labeled conversion-rate experiment used only to demonstrate the engine.",
            "result": result,
            "limitations": [
                "These counts are synthetic and are not Olist observations or real business evidence.",
                "Causal interpretation requires valid random assignment and reliable experiment instrumentation.",
            ],
        }

    def sample_size(
        self,
        *,
        metric_type: str,
        baseline: float,
        minimum_detectable_effect: float,
        standard_deviation: float | None = None,
        alpha: float = 0.05,
        power: float = 0.8,
        alternative: Alternative = "two-sided",
    ) -> dict[str, Any]:
        if metric_type == "proportion":
            result = estimate_two_proportion_sample_size(
                baseline,
                minimum_detectable_effect,
                alpha=alpha,
                power=power,
                alternative=alternative,
            )
        elif metric_type == "mean":
            if standard_deviation is None:
                raise StatisticalInputError(
                    "standard_deviation is required for a two-mean sample-size calculation"
                )
            result = estimate_two_mean_sample_size(
                baseline,
                minimum_detectable_effect,
                standard_deviation,
                alpha=alpha,
                power=power,
                alternative=alternative,
            )
        else:
            raise StatisticalInputError("metric_type must be 'proportion' or 'mean'")
        return {
            "task": "sample_size_estimation",
            "dataset": None,
            "source": "User-supplied design assumptions",
            "description": "Prospective equal-allocation two-group power analysis.",
            "result": result,
            "limitations": [
                "The estimate assumes independent observations, equal group allocation, and the supplied effect size.",
                "Increase the target for expected attrition, noncompliance, or multiple-comparison adjustments.",
            ],
        }


def _group_values(frame, group_name: str) -> list[float]:
    values = frame.loc[frame["group_name"] == group_name, "metric_value"].astype(float).tolist()
    if not values:
        raise StatisticalInputError(f"No eligible observations were found for {group_name}")
    return values


def _validate_states(group_a: str | None, group_b: str | None) -> tuple[str, str]:
    states = tuple((value or "").strip().upper() for value in (group_a, group_b))
    if any(not re.fullmatch(r"[A-Z]{2}", value) for value in states):
        raise StatisticalInputError("Two Brazilian state abbreviations such as SP and RJ are required")
    if states[0] == states[1]:
        raise StatisticalInputError("The two state groups must be different")
    return states


def _observational_payload(
    task: str,
    dataset: StatisticalDataset,
    description: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": task,
        "dataset": dataset.value,
        "source": "Olist PostgreSQL observational data",
        "description": description,
        "result": result,
        "limitations": [
            "This is an observational comparison and does not establish causation.",
            "Statistical significance does not by itself imply practical business importance.",
        ],
    }
