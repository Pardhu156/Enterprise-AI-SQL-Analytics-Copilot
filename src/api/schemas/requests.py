"""Validated request models for the public API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator


Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000),
]


class AnalyticsQueryRequest(BaseModel):
    question: Question = Field(
        description="Natural-language business question to answer from the analytics database."
    )
    include_sql: bool = Field(default=True, description="Include generated and executed SQL.")
    include_rows: bool = Field(default=True, description="Include returned query rows.")
    include_visualization_config: bool = Field(
        default=True,
        description="Include the deterministic visualization specification.",
    )


class CopilotQueryRequest(BaseModel):
    question: Question = Field(
        description=(
            "Historical, predictive, statistical, explainability, recommendation, "
            "or supported hybrid business question."
        )
    )


class HypothesisTestRequest(BaseModel):
    analysis_type: Literal["continuous", "proportion", "categorical"]
    group_a_name: str = Field(default="Group A", min_length=1, max_length=80)
    group_b_name: str = Field(default="Group B", min_length=1, max_length=80)
    group_a_values: list[float] | None = Field(default=None, min_length=2, max_length=100_000)
    group_b_values: list[float] | None = Field(default=None, min_length=2, max_length=100_000)
    successes_a: int | None = Field(default=None, ge=0)
    trials_a: int | None = Field(default=None, gt=0)
    successes_b: int | None = Field(default=None, ge=0)
    trials_b: int | None = Field(default=None, gt=0)
    contingency_table: list[list[int]] | None = None
    alpha: float = Field(default=0.05, gt=0, lt=1)
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    scale: Literal["continuous", "ordinal"] = "continuous"
    method: Literal["auto", "student_t", "welch_t", "mann_whitney"] = "auto"

    @model_validator(mode="after")
    def validate_analysis_inputs(self) -> "HypothesisTestRequest":
        if self.analysis_type == "continuous" and (
            self.group_a_values is None or self.group_b_values is None
        ):
            raise ValueError("continuous analysis requires group_a_values and group_b_values")
        counts = (self.successes_a, self.trials_a, self.successes_b, self.trials_b)
        if self.analysis_type == "proportion" and any(value is None for value in counts):
            raise ValueError("proportion analysis requires successes and trials for both groups")
        if self.analysis_type == "categorical" and self.contingency_table is None:
            raise ValueError("categorical analysis requires contingency_table")
        return self


class ConfidenceIntervalRequest(BaseModel):
    estimand: Literal["mean", "proportion", "difference_means", "difference_proportions"]
    values: list[float] | None = Field(default=None, min_length=2, max_length=100_000)
    group_a_values: list[float] | None = Field(default=None, min_length=2, max_length=100_000)
    group_b_values: list[float] | None = Field(default=None, min_length=2, max_length=100_000)
    successes: int | None = Field(default=None, ge=0)
    trials: int | None = Field(default=None, gt=0)
    successes_a: int | None = Field(default=None, ge=0)
    trials_a: int | None = Field(default=None, gt=0)
    successes_b: int | None = Field(default=None, ge=0)
    trials_b: int | None = Field(default=None, gt=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    group_a_name: str = Field(default="Group A", min_length=1, max_length=80)
    group_b_name: str = Field(default="Group B", min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_estimand_inputs(self) -> "ConfidenceIntervalRequest":
        if self.estimand == "mean" and self.values is None:
            raise ValueError("mean interval requires values")
        if self.estimand == "proportion" and (self.successes is None or self.trials is None):
            raise ValueError("proportion interval requires successes and trials")
        if self.estimand == "difference_means" and (
            self.group_a_values is None or self.group_b_values is None
        ):
            raise ValueError("difference_means requires both value samples")
        difference_counts = (self.successes_a, self.trials_a, self.successes_b, self.trials_b)
        if self.estimand == "difference_proportions" and any(
            value is None for value in difference_counts
        ):
            raise ValueError("difference_proportions requires both groups' counts")
        return self


class ABTestRequest(BaseModel):
    metric_type: Literal["conversion", "average_value"]
    control_successes: int | None = Field(default=None, ge=0)
    control_trials: int | None = Field(default=None, gt=0)
    treatment_successes: int | None = Field(default=None, ge=0)
    treatment_trials: int | None = Field(default=None, gt=0)
    control_values: list[float] | None = Field(default=None, min_length=2, max_length=100_000)
    treatment_values: list[float] | None = Field(default=None, min_length=2, max_length=100_000)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    randomized_experiment: bool = False

    @model_validator(mode="after")
    def validate_experiment_inputs(self) -> "ABTestRequest":
        counts = (
            self.control_successes,
            self.control_trials,
            self.treatment_successes,
            self.treatment_trials,
        )
        if self.metric_type == "conversion" and any(value is None for value in counts):
            raise ValueError("conversion experiments require successes and trials for both groups")
        if self.metric_type == "average_value" and (
            self.control_values is None or self.treatment_values is None
        ):
            raise ValueError("average_value experiments require both value samples")
        return self


class SampleSizeRequest(BaseModel):
    metric_type: Literal["proportion", "mean"]
    baseline: float
    minimum_detectable_effect: float
    standard_deviation: float | None = Field(default=None, gt=0)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    power: float = Field(default=0.8, gt=0, lt=1)
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"

    @model_validator(mode="after")
    def validate_sample_size_inputs(self) -> "SampleSizeRequest":
        if self.metric_type == "mean" and self.standard_deviation is None:
            raise ValueError("standard_deviation is required for mean comparisons")
        return self
