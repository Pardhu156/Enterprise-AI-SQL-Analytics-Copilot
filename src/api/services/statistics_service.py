"""Map Stage 7B statistical calculations to stable API responses."""

from __future__ import annotations

import psycopg2

from src.business.impact import BusinessImpactService
from src.business.recommendations import RecommendationService
from src.statistical.ab_testing import (
    analyze_average_value_experiment,
    analyze_conversion_experiment,
)
from src.statistical.engine import (
    StatisticalInputError,
    confidence_interval_difference_means,
    confidence_interval_difference_proportions,
    confidence_interval_mean,
    confidence_interval_proportion,
    test_contingency_table,
    test_continuous_groups,
    test_two_proportions,
)
from src.statistical.service import StatisticalAnalysisService, StatisticalDataset
from src.statistical.sample_size import (
    estimate_two_mean_sample_size,
    estimate_two_proportion_sample_size,
)

from ..errors import APIError
from ..schemas.requests import (
    ABTestRequest,
    ConfidenceIntervalRequest,
    HypothesisTestRequest,
    SampleSizeRequest,
)
from ..schemas.responses import StatisticalAnalysisResponse


class StatisticsService:
    def __init__(
        self,
        analysis: StatisticalAnalysisService | None = None,
        impact: BusinessImpactService | None = None,
        recommendations: RecommendationService | None = None,
    ) -> None:
        self._analysis = analysis or StatisticalAnalysisService()
        self._impact = impact or BusinessImpactService()
        self._recommendations = recommendations or RecommendationService()

    def hypothesis_test(self, request: HypothesisTestRequest) -> StatisticalAnalysisResponse:
        def calculate():
            if request.analysis_type == "continuous":
                return test_continuous_groups(
                    request.group_a_values or [],
                    request.group_b_values or [],
                    group_a_name=request.group_a_name,
                    group_b_name=request.group_b_name,
                    alpha=request.alpha,
                    alternative=request.alternative,
                    scale=request.scale,
                    method=request.method,
                ).as_dict()
            if request.analysis_type == "proportion":
                return test_two_proportions(
                    request.successes_a or 0,
                    request.trials_a or 0,
                    request.successes_b or 0,
                    request.trials_b or 0,
                    group_a_name=request.group_a_name,
                    group_b_name=request.group_b_name,
                    alpha=request.alpha,
                    alternative=request.alternative,
                ).as_dict()
            return test_contingency_table(
                request.contingency_table or [], alpha=request.alpha
            ).as_dict()

        return self._safe_user_result(
            "hypothesis_test",
            "User-supplied samples or counts",
            "Reusable two-group or categorical hypothesis test.",
            calculate,
        )

    def confidence_interval(
        self, request: ConfidenceIntervalRequest
    ) -> StatisticalAnalysisResponse:
        def calculate():
            if request.estimand == "mean":
                return confidence_interval_mean(
                    request.values or [], request.confidence_level
                ).as_dict()
            if request.estimand == "proportion":
                return confidence_interval_proportion(
                    request.successes or 0,
                    request.trials or 0,
                    request.confidence_level,
                ).as_dict()
            if request.estimand == "difference_means":
                return confidence_interval_difference_means(
                    request.group_a_values or [],
                    request.group_b_values or [],
                    request.confidence_level,
                    request.group_a_name,
                    request.group_b_name,
                ).as_dict()
            return confidence_interval_difference_proportions(
                request.successes_a or 0,
                request.trials_a or 0,
                request.successes_b or 0,
                request.trials_b or 0,
                request.confidence_level,
                request.group_a_name,
                request.group_b_name,
            ).as_dict()

        return self._safe_user_result(
            "confidence_interval",
            "User-supplied samples or counts",
            f"{request.confidence_level:.0%} confidence interval for {request.estimand}.",
            calculate,
        )

    def ab_test(self, request: ABTestRequest) -> StatisticalAnalysisResponse:
        def calculate():
            if request.metric_type == "conversion":
                return analyze_conversion_experiment(
                    request.control_successes or 0,
                    request.control_trials or 0,
                    request.treatment_successes or 0,
                    request.treatment_trials or 0,
                    alpha=request.alpha,
                    alternative=request.alternative,
                    randomized_experiment=request.randomized_experiment,
                )
            return analyze_average_value_experiment(
                request.control_values or [],
                request.treatment_values or [],
                alpha=request.alpha,
                alternative=request.alternative,
                randomized_experiment=request.randomized_experiment,
            )

        return self._safe_user_result(
            "ab_test",
            "User-supplied experiment data",
            f"A/B comparison for {request.metric_type}.",
            calculate,
            causal=request.randomized_experiment,
        )

    def sample_size(self, request: SampleSizeRequest) -> StatisticalAnalysisResponse:
        def calculate():
            if request.metric_type == "proportion":
                return estimate_two_proportion_sample_size(
                    request.baseline,
                    request.minimum_detectable_effect,
                    alpha=request.alpha,
                    power=request.power,
                    alternative=request.alternative,
                )
            return estimate_two_mean_sample_size(
                request.baseline,
                request.minimum_detectable_effect,
                request.standard_deviation or 0,
                alpha=request.alpha,
                power=request.power,
                alternative=request.alternative,
            )

        return self._safe_user_result(
            "sample_size_estimation",
            "User-supplied design assumptions",
            f"Prospective sample-size calculation for a {request.metric_type} metric.",
            calculate,
        )

    def olist_hypothesis(
        self,
        dataset: StatisticalDataset,
        **parameters,
    ) -> StatisticalAnalysisResponse:
        return self._safe_olist(self._analysis.hypothesis_test, dataset, **parameters)

    def olist_confidence_interval(
        self,
        dataset: StatisticalDataset,
        **parameters,
    ) -> StatisticalAnalysisResponse:
        return self._safe_olist(self._analysis.confidence_interval, dataset, **parameters)

    def olist_ab_demo(self) -> StatisticalAnalysisResponse:
        return self._safe_olist(self._analysis.ab_demo)

    def olist_sample_size(self, **parameters) -> StatisticalAnalysisResponse:
        return self._safe_olist(self._analysis.sample_size, **parameters)

    def _safe_user_result(self, task, source, description, calculate, causal: bool = False):
        try:
            result = calculate()
        except StatisticalInputError as exc:
            raise APIError(422, "STATISTICAL_INPUT_INVALID", str(exc)) from exc
        limitations = []
        if task == "ab_test" and not causal:
            limitations.append(
                "Random assignment was not confirmed; interpret the result as association, not causation."
            )
        return StatisticalAnalysisResponse(
            task=task,
            source=source,
            description=description,
            result=result,
            limitations=limitations,
            impact=self._impact.statistical(task, result),
            recommendations=self._recommendations.statistical(task, result),
        )

    def _safe_olist(self, function, *args, **kwargs) -> StatisticalAnalysisResponse:
        try:
            payload = function(*args, **kwargs)
            task = payload["task"]
            result = payload["result"]
            payload["impact"] = self._impact.statistical(task, result)
            payload["recommendations"] = self._recommendations.statistical(task, result)
            return StatisticalAnalysisResponse.model_validate(payload)
        except StatisticalInputError as exc:
            raise APIError(422, "STATISTICAL_INPUT_INVALID", str(exc)) from exc
        except psycopg2.Error as exc:
            raise APIError(
                503,
                "DATABASE_UNAVAILABLE",
                "The analytics database is unavailable.",
            ) from exc
