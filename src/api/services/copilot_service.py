"""Deterministic orchestration after Gemini has classified user intent."""

from __future__ import annotations

from src.routing.intent_classifier import (
    IntentClassificationError,
    IntentClassifier,
    IntentType,
    RoutedTask,
)

from ..errors import APIError
from ..schemas.requests import AnalyticsQueryRequest, CopilotQueryRequest
from ..schemas.responses import (
    CopilotQueryResponse,
    MLPredictionResponse,
    RoutingDetails,
    StatisticalAnalysisResponse,
)
from .analytics_service import AnalyticsService
from .ml_service import MLService
from .statistics_service import StatisticsService


class CopilotService:
    def __init__(
        self,
        classifier: IntentClassifier,
        analytics: AnalyticsService,
        ml: MLService,
        statistics: StatisticsService | None = None,
    ) -> None:
        self._classifier = classifier
        self._analytics = analytics
        self._ml = ml
        self._statistics = statistics or StatisticsService()

    def query(self, request: CopilotQueryRequest, request_id: str) -> CopilotQueryResponse:
        try:
            decision = self._classifier.classify(request.question)
        except IntentClassificationError as exc:
            raise APIError(
                503,
                "INTENT_CLASSIFICATION_FAILED",
                "Gemini could not classify this request. Please retry or use a direct endpoint.",
            ) from exc
        if decision.intent == IntentType.UNSUPPORTED:
            raise APIError(
                400,
                "UNSUPPORTED_INTENT",
                "This copilot supports historical analytics, predictive models, hypothesis tests, confidence intervals, A/B analysis, and sample-size estimation.",
            )

        historical = None
        predictions: list[MLPredictionResponse] = []
        statistical_analyses: list[StatisticalAnalysisResponse] = []
        if RoutedTask.HISTORICAL_ANALYTICS in decision.tasks:
            historical = self._analytics.query(
                AnalyticsQueryRequest(question=decision.sql_question or request.question),
                request_id,
            )
        for task in decision.tasks:
            if task == RoutedTask.SALES_FORECASTING:
                predictions.append(self._ml.forecast(decision.horizon_weeks))
            elif task == RoutedTask.CUSTOMER_SEGMENTATION:
                predictions.append(
                    self._ml.segment_customer(decision.customer_unique_id)
                    if decision.customer_unique_id
                    else self._ml.segment_summary()
                )
            elif task == RoutedTask.LATE_DELIVERY_PREDICTION:
                if not decision.order_id:
                    raise APIError(
                        422,
                        "ML_INPUT_REQUIRED",
                        "Provide an Olist order_id for late-delivery prediction.",
                    )
                predictions.append(self._ml.delivery_risk(decision.order_id))
            elif task == RoutedTask.HYPOTHESIS_TEST:
                if decision.statistical_dataset is None:
                    raise APIError(422, "STATISTICAL_INPUT_REQUIRED", "A supported dataset is required.")
                statistical_analyses.append(
                    self._statistics.olist_hypothesis(
                        decision.statistical_dataset,
                        group_a=decision.group_a,
                        group_b=decision.group_b,
                        alpha=decision.alpha,
                        alternative=decision.alternative,
                    )
                )
            elif task == RoutedTask.CONFIDENCE_INTERVAL:
                if decision.statistical_dataset is None:
                    raise APIError(422, "STATISTICAL_INPUT_REQUIRED", "A supported dataset is required.")
                statistical_analyses.append(
                    self._statistics.olist_confidence_interval(
                        decision.statistical_dataset,
                        confidence_level=decision.confidence_level,
                    )
                )
            elif task == RoutedTask.AB_TEST:
                statistical_analyses.append(self._statistics.olist_ab_demo())
            elif task == RoutedTask.SAMPLE_SIZE_ESTIMATION:
                statistical_analyses.append(
                    self._statistics.olist_sample_size(
                        metric_type=decision.metric_type,
                        baseline=decision.baseline,
                        minimum_detectable_effect=decision.minimum_detectable_effect,
                        standard_deviation=decision.standard_deviation,
                        alpha=decision.alpha,
                        power=decision.power,
                        alternative=decision.alternative,
                    )
                )

        return CopilotQueryResponse(
            request_id=request_id,
            question=request.question,
            route=RoutingDetails(
                intent=decision.intent.value,
                tasks=[task.value for task in decision.tasks],
            ),
            answer=_combined_answer(
                historical.answer if historical else None,
                predictions,
                statistical_analyses,
            ),
            historical=historical,
            predictions=predictions,
            statistical_analyses=statistical_analyses,
        )


def _combined_answer(
    historical_answer: str | None,
    predictions: list[MLPredictionResponse],
    statistical_analyses: list[StatisticalAnalysisResponse],
) -> str:
    parts = [historical_answer] if historical_answer else []
    for prediction in predictions:
        if prediction.task == "sales_forecasting":
            total = float(prediction.data["total_predicted_revenue"])
            horizon = int(prediction.data["horizon_weeks"])
            parts.append(
                f"The {prediction.model} model predicts R$ {total:,.2f} in revenue "
                f"across the next {horizon} weeks."
            )
        elif prediction.task == "customer_segmentation":
            if "segment" in prediction.data:
                parts.append(f"The customer belongs to the {prediction.data['segment']} segment.")
            else:
                parts.append("The RFM model identified the customer segments shown below.")
        elif prediction.task == "late_delivery_prediction":
            probability = float(prediction.data["late_delivery_probability"])
            parts.append(
                f"The model estimates a {probability:.1%} late-delivery probability "
                f"({prediction.data['risk_band']} risk)."
            )
    for analysis in statistical_analyses:
        result = analysis.result
        if analysis.task == "hypothesis_test":
            parts.append(str(result.get("interpretation", "The statistical test completed.")))
        elif analysis.task == "confidence_interval":
            parts.append(str(result.get("interpretation", "The confidence interval was calculated.")))
        elif analysis.task == "ab_test":
            parts.append(str(result.get("recommendation", "The A/B analysis completed.")))
        elif analysis.task == "sample_size_estimation":
            per_group = int(result["required_sample_size_per_group"])
            total = int(result["total_required_sample_size"])
            parts.append(
                f"The design requires approximately {per_group:,} observations per group "
                f"({total:,} total)."
            )
    return " ".join(part for part in parts if part) or "The request completed successfully."
