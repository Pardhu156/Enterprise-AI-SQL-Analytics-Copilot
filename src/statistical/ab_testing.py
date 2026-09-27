"""Reusable A/B analysis with explicit causal-language safeguards."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .engine import Alternative, test_continuous_groups, test_two_proportions


def analyze_conversion_experiment(
    control_successes: int,
    control_trials: int,
    treatment_successes: int,
    treatment_trials: int,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    randomized_experiment: bool = False,
    synthetic_demo: bool = False,
) -> dict[str, Any]:
    test = test_two_proportions(
        treatment_successes,
        treatment_trials,
        control_successes,
        control_trials,
        group_a_name="Treatment",
        group_b_name="Control",
        alpha=alpha,
        alternative=alternative,
    )
    control = control_successes / control_trials
    treatment = treatment_successes / treatment_trials
    return _ab_payload(
        test.as_dict(),
        control,
        treatment,
        randomized_experiment,
        synthetic_demo,
    )


def analyze_average_value_experiment(
    control_values: Sequence[float],
    treatment_values: Sequence[float],
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    randomized_experiment: bool = False,
    synthetic_demo: bool = False,
) -> dict[str, Any]:
    test = test_continuous_groups(
        treatment_values,
        control_values,
        group_a_name="Treatment",
        group_b_name="Control",
        alpha=alpha,
        alternative=alternative,
        method="welch_t",
    )
    control = float(np.mean(np.asarray(control_values, dtype=float)))
    treatment = float(np.mean(np.asarray(treatment_values, dtype=float)))
    return _ab_payload(
        test.as_dict(),
        control,
        treatment,
        randomized_experiment,
        synthetic_demo,
    )


def synthetic_conversion_demo() -> dict[str, Any]:
    """Return a deterministic, prominently labeled demonstration—not Olist evidence."""
    return analyze_conversion_experiment(
        control_successes=800,
        control_trials=10_000,
        treatment_successes=900,
        treatment_trials=10_000,
        alpha=0.05,
        alternative="two-sided",
        randomized_experiment=True,
        synthetic_demo=True,
    )


def _ab_payload(
    test: dict[str, Any],
    control: float,
    treatment: float,
    randomized: bool,
    synthetic_demo: bool,
) -> dict[str, Any]:
    absolute_lift = treatment - control
    relative_lift = absolute_lift / control if control != 0 else None
    significant = bool(test["significant"])
    favorable = absolute_lift > 0
    if significant and favorable:
        recommendation = "The treatment has statistically significant positive evidence."
    elif significant:
        recommendation = "The treatment has statistically significant negative evidence."
    else:
        recommendation = "Do not declare a winner; the observed difference is inconclusive."
    if not randomized:
        recommendation += " Treat this as an association because random assignment was not confirmed."
    if synthetic_demo:
        recommendation += " This is synthetic demonstration data and not business evidence."
    return {
        "task": "ab_test",
        "control_metric": control,
        "treatment_metric": treatment,
        "absolute_lift": absolute_lift,
        "relative_lift": relative_lift,
        "statistically_significant": significant,
        "randomized_experiment": randomized,
        "synthetic_demo": synthetic_demo,
        "recommendation": recommendation,
        "test_result": test,
    }
