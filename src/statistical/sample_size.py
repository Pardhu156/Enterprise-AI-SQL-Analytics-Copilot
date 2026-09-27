"""Power-based sample-size estimation for future two-group experiments."""

from __future__ import annotations

from math import ceil
from typing import Any, Literal

from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

from .engine import StatisticalInputError


def estimate_two_proportion_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict[str, Any]:
    treatment_rate = baseline_rate + minimum_detectable_effect
    _validate_design(alpha, power, alternative)
    if minimum_detectable_effect == 0:
        raise StatisticalInputError("minimum_detectable_effect must be non-zero")
    if not 0 < baseline_rate < 1 or not 0 < treatment_rate < 1:
        raise StatisticalInputError(
            "baseline_rate and baseline_rate + minimum_detectable_effect must be between 0 and 1"
        )
    effect_size = abs(float(proportion_effectsize(treatment_rate, baseline_rate)))
    per_group = ceil(
        NormalIndPower().solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1,
            alternative=_power_alternative(alternative),
        )
    )
    return {
        "task": "sample_size_estimation",
        "method": "Normal approximation for two independent proportions",
        "required_sample_size_per_group": per_group,
        "total_required_sample_size": per_group * 2,
        "assumptions": {
            "baseline_rate": baseline_rate,
            "target_rate": treatment_rate,
            "minimum_detectable_effect": minimum_detectable_effect,
            "alpha": alpha,
            "power": power,
            "alternative": alternative,
            "allocation_ratio": 1.0,
        },
    }


def estimate_two_mean_sample_size(
    baseline_mean: float,
    minimum_detectable_effect: float,
    standard_deviation: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> dict[str, Any]:
    _validate_design(alpha, power, alternative)
    if standard_deviation <= 0 or minimum_detectable_effect == 0:
        raise StatisticalInputError(
            "standard_deviation must be positive and minimum_detectable_effect must be non-zero"
        )
    standardized_effect = abs(minimum_detectable_effect) / standard_deviation
    per_group = ceil(
        TTestIndPower().solve_power(
            effect_size=standardized_effect,
            alpha=alpha,
            power=power,
            ratio=1,
            alternative=_power_alternative(alternative),
        )
    )
    return {
        "task": "sample_size_estimation",
        "method": "Independent two-sample t-test power analysis",
        "required_sample_size_per_group": per_group,
        "total_required_sample_size": per_group * 2,
        "assumptions": {
            "baseline_mean": baseline_mean,
            "target_mean": baseline_mean + minimum_detectable_effect,
            "minimum_detectable_effect": minimum_detectable_effect,
            "standard_deviation": standard_deviation,
            "standardized_effect_size": standardized_effect,
            "alpha": alpha,
            "power": power,
            "alternative": alternative,
            "allocation_ratio": 1.0,
        },
    }


def _validate_design(alpha: float, power: float, alternative: str) -> None:
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise StatisticalInputError("alpha and power must be between zero and one")
    if alternative not in {"two-sided", "greater", "less"}:
        raise StatisticalInputError("alternative must be two-sided, greater, or less")


def _power_alternative(alternative: str) -> str:
    return {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}[alternative]
