"""Reusable hypothesis tests and confidence intervals with assumption checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import asin, sqrt
from typing import Any, Literal, Sequence

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)


Alternative = Literal["two-sided", "greater", "less"]


class StatisticalInputError(ValueError):
    """Raised when statistical inputs cannot support the requested inference."""


@dataclass(frozen=True)
class ConfidenceInterval:
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    method: str
    interpretation: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffectSize:
    name: str
    value: float


@dataclass(frozen=True)
class HypothesisResult:
    test_used: str
    null_hypothesis: str
    alternative_hypothesis: str
    statistic: float
    p_value: float
    significance_level: float
    significant: bool
    decision: str
    estimate: float | None
    confidence_interval: ConfidenceInterval | None
    effect_size: EffectSize | None
    group_summaries: dict[str, dict[str, float | int]]
    assumptions: list[str]
    interpretation: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def confidence_interval_mean(
    values: Sequence[float],
    confidence_level: float = 0.95,
    label: str = "the population mean",
) -> ConfidenceInterval:
    sample = _clean_numeric(values, "values", minimum=2)
    alpha = _alpha_from_confidence(confidence_level)
    mean = float(np.mean(sample))
    standard_error = float(stats.sem(sample))
    margin = float(stats.t.ppf(1 - alpha / 2, len(sample) - 1) * standard_error)
    return ConfidenceInterval(
        point_estimate=mean,
        lower=mean - margin,
        upper=mean + margin,
        confidence_level=confidence_level,
        method="Student t confidence interval",
        interpretation=(
            f"The {confidence_level:.0%} confidence interval for {label} is "
            f"[{mean - margin:.4f}, {mean + margin:.4f}]."
        ),
    )


def confidence_interval_proportion(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
    label: str = "the population proportion",
) -> ConfidenceInterval:
    _validate_counts(successes, trials, "proportion")
    alpha = _alpha_from_confidence(confidence_level)
    lower, upper = proportion_confint(successes, trials, alpha=alpha, method="wilson")
    estimate = successes / trials
    return ConfidenceInterval(
        point_estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        method="Wilson score interval",
        interpretation=(
            f"The {confidence_level:.0%} confidence interval for {label} is "
            f"[{float(lower):.2%}, {float(upper):.2%}]."
        ),
    )


def confidence_interval_difference_means(
    group_a: Sequence[float],
    group_b: Sequence[float],
    confidence_level: float = 0.95,
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
) -> ConfidenceInterval:
    a = _clean_numeric(group_a, "group_a", minimum=2)
    b = _clean_numeric(group_b, "group_b", minimum=2)
    if np.std(a, ddof=1) == 0 and np.std(b, ddof=1) == 0:
        raise StatisticalInputError("At least one group must contain measurable variation")
    alpha = _alpha_from_confidence(confidence_level)
    difference = float(np.mean(a) - np.mean(b))
    variance_a = float(np.var(a, ddof=1) / len(a))
    variance_b = float(np.var(b, ddof=1) / len(b))
    standard_error = sqrt(variance_a + variance_b)
    if standard_error == 0:
        lower = upper = difference
    else:
        degrees_freedom = (variance_a + variance_b) ** 2 / (
            variance_a**2 / (len(a) - 1) + variance_b**2 / (len(b) - 1)
        )
        margin = float(stats.t.ppf(1 - alpha / 2, degrees_freedom) * standard_error)
        lower, upper = difference - margin, difference + margin
    return ConfidenceInterval(
        point_estimate=difference,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        method="Welch t confidence interval",
        interpretation=(
            f"The {confidence_level:.0%} confidence interval for the mean difference "
            f"({group_a_name} minus {group_b_name}) is [{lower:.4f}, {upper:.4f}]."
        ),
    )


def confidence_interval_difference_proportions(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    confidence_level: float = 0.95,
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
) -> ConfidenceInterval:
    _validate_counts(successes_a, trials_a, group_a_name)
    _validate_counts(successes_b, trials_b, group_b_name)
    alpha = _alpha_from_confidence(confidence_level)
    lower, upper = confint_proportions_2indep(
        successes_a,
        trials_a,
        successes_b,
        trials_b,
        method="newcomb",
        compare="diff",
        alpha=alpha,
    )
    difference = successes_a / trials_a - successes_b / trials_b
    return ConfidenceInterval(
        point_estimate=difference,
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        method="Newcombe difference-of-proportions interval",
        interpretation=(
            f"The {confidence_level:.0%} confidence interval for the proportion difference "
            f"({group_a_name} minus {group_b_name}) is [{float(lower):.2%}, "
            f"{float(upper):.2%}]."
        ),
    )


def test_continuous_groups(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    scale: Literal["continuous", "ordinal"] = "continuous",
    method: Literal["auto", "student_t", "welch_t", "mann_whitney"] = "auto",
) -> HypothesisResult:
    _validate_alpha(alpha)
    a = _clean_numeric(group_a, "group_a", minimum=2)
    b = _clean_numeric(group_b, "group_b", minimum=2)
    if np.std(a, ddof=1) == 0 and np.std(b, ddof=1) == 0:
        raise StatisticalInputError("At least one group must contain measurable variation")
    normal_a, normal_note_a = _normality_assessment(a, group_a_name, alpha)
    normal_b, normal_note_b = _normality_assessment(b, group_b_name, alpha)
    levene = stats.levene(a, b, center="median")
    equal_variance = bool(float(levene.pvalue) >= alpha)
    assumptions = [normal_note_a, normal_note_b]
    assumptions.append(
        f"Brown-Forsythe variance test p={float(levene.pvalue):.4g}; "
        + ("equal variance was not rejected." if equal_variance else "variances differ.")
    )

    selected = method
    if method == "auto":
        if scale == "ordinal" or not (normal_a and normal_b):
            selected = "mann_whitney"
        elif equal_variance:
            selected = "student_t"
        else:
            selected = "welch_t"

    estimate = float(np.mean(a) - np.mean(b))
    interval = confidence_interval_difference_means(
        a,
        b,
        confidence_level=1 - alpha,
        group_a_name=group_a_name,
        group_b_name=group_b_name,
    )
    if selected == "mann_whitney":
        result = stats.mannwhitneyu(a, b, alternative=alternative, method="auto")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect = EffectSize(
            "rank_biserial_correlation",
            float(2 * statistic / (len(a) * len(b)) - 1),
        )
        test_used = "Mann-Whitney U test"
        assumptions.append("A rank-based test was used because the outcome is ordinal or non-normal.")
    else:
        equal_var = selected == "student_t"
        result = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        effect = EffectSize("hedges_g", _hedges_g(a, b))
        test_used = "Independent Student t-test" if equal_var else "Welch independent t-test"

    significant = p_value < alpha
    direction = _alternative_text(alternative, group_a_name, group_b_name)
    return HypothesisResult(
        test_used=test_used,
        null_hypothesis=f"There is no difference between {group_a_name} and {group_b_name}.",
        alternative_hypothesis=direction,
        statistic=statistic,
        p_value=p_value,
        significance_level=alpha,
        significant=significant,
        decision=("Reject the null hypothesis." if significant else "Fail to reject the null hypothesis."),
        estimate=estimate,
        confidence_interval=interval,
        effect_size=effect,
        group_summaries={
            group_a_name: _continuous_summary(a),
            group_b_name: _continuous_summary(b),
        },
        assumptions=assumptions,
        interpretation=_test_interpretation(significant, test_used, group_a_name, group_b_name),
    )


def test_two_proportions(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    *,
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> HypothesisResult:
    _validate_alpha(alpha)
    _validate_counts(successes_a, trials_a, group_a_name)
    _validate_counts(successes_b, trials_b, group_b_name)
    total_successes = successes_a + successes_b
    total_failures = trials_a + trials_b - total_successes
    if total_successes == 0 or total_failures == 0:
        raise StatisticalInputError(
            "A proportion comparison requires at least one success and one failure overall"
        )
    p_a, p_b = successes_a / trials_a, successes_b / trials_b
    cells = np.array(
        [[successes_a, trials_a - successes_a], [successes_b, trials_b - successes_b]],
        dtype=float,
    )
    _, _, _, expected = stats.chi2_contingency(cells, correction=False)
    assumptions: list[str] = []
    if np.any(expected < 5):
        odds_ratio, p_value = stats.fisher_exact(cells, alternative=alternative)
        statistic = float(odds_ratio)
        test_used = "Fisher exact test"
        effect = EffectSize("odds_ratio", statistic)
        assumptions.append("Expected cell counts below five required Fisher's exact test.")
    else:
        statistic, p_value = proportions_ztest(
            [successes_a, successes_b],
            [trials_a, trials_b],
            alternative=_statsmodels_alternative(alternative),
        )
        statistic, p_value = float(statistic), float(p_value)
        test_used = "Two-proportion z-test"
        effect = EffectSize(
            "cohens_h",
            float(2 * asin(sqrt(p_a)) - 2 * asin(sqrt(p_b))),
        )
        assumptions.append("All expected success/failure cell counts were at least five.")
    interval = confidence_interval_difference_proportions(
        successes_a,
        trials_a,
        successes_b,
        trials_b,
        confidence_level=1 - alpha,
        group_a_name=group_a_name,
        group_b_name=group_b_name,
    )
    significant = p_value < alpha
    return HypothesisResult(
        test_used=test_used,
        null_hypothesis=f"{group_a_name} and {group_b_name} have equal proportions.",
        alternative_hypothesis=_alternative_text(alternative, group_a_name, group_b_name),
        statistic=statistic,
        p_value=p_value,
        significance_level=alpha,
        significant=significant,
        decision=("Reject the null hypothesis." if significant else "Fail to reject the null hypothesis."),
        estimate=p_a - p_b,
        confidence_interval=interval,
        effect_size=effect,
        group_summaries={
            group_a_name: {"successes": successes_a, "trials": trials_a, "proportion": p_a},
            group_b_name: {"successes": successes_b, "trials": trials_b, "proportion": p_b},
        },
        assumptions=assumptions,
        interpretation=_test_interpretation(significant, test_used, group_a_name, group_b_name),
    )


def test_contingency_table(
    table: Sequence[Sequence[int]],
    *,
    alpha: float = 0.05,
) -> HypothesisResult:
    _validate_alpha(alpha)
    observed = np.asarray(table, dtype=float)
    if observed.ndim != 2 or min(observed.shape) < 2:
        raise StatisticalInputError("contingency_table must contain at least two rows and columns")
    if not np.all(np.isfinite(observed)) or np.any(observed < 0):
        raise StatisticalInputError("contingency_table must contain finite non-negative counts")
    if np.any(observed.sum(axis=0) == 0) or np.any(observed.sum(axis=1) == 0):
        raise StatisticalInputError("contingency_table cannot contain an empty row or column")
    chi2, chi_p, _, expected = stats.chi2_contingency(observed, correction=False)
    assumptions: list[str]
    if observed.shape == (2, 2) and np.any(expected < 5):
        statistic, p_value = stats.fisher_exact(observed)
        test_used = "Fisher exact test"
        effect = EffectSize("odds_ratio", float(statistic))
        assumptions = ["Expected counts below five required Fisher's exact test."]
    elif np.any(expected < 5):
        raise StatisticalInputError(
            "Chi-square assumptions fail because expected counts are below five; "
            "combine sparse categories or collect more data"
        )
    else:
        statistic, p_value = float(chi2), float(chi_p)
        test_used = "Chi-square test of independence"
        denominator = observed.sum() * min(observed.shape[0] - 1, observed.shape[1] - 1)
        effect = EffectSize("cramers_v", float(sqrt(chi2 / denominator)))
        assumptions = ["All expected cell counts were at least five."]
    significant = float(p_value) < alpha
    return HypothesisResult(
        test_used=test_used,
        null_hypothesis="The two categorical variables are independent.",
        alternative_hypothesis="The two categorical variables are associated.",
        statistic=float(statistic),
        p_value=float(p_value),
        significance_level=alpha,
        significant=significant,
        decision=("Reject the null hypothesis." if significant else "Fail to reject the null hypothesis."),
        estimate=None,
        confidence_interval=None,
        effect_size=effect,
        group_summaries={"table": {"rows": int(observed.shape[0]), "columns": int(observed.shape[1]), "observations": int(observed.sum())}},
        assumptions=assumptions,
        interpretation=(
            "The data provide evidence of an association between the categorical variables."
            if significant
            else "The data do not provide sufficient evidence of an association."
        ),
    )


def _clean_numeric(values: Sequence[float], name: str, minimum: int) -> np.ndarray:
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1:
        raise StatisticalInputError(f"{name} must be a one-dimensional sample")
    sample = sample[np.isfinite(sample)]
    if len(sample) < minimum:
        raise StatisticalInputError(f"{name} requires at least {minimum} finite observations")
    return sample


def _normality_assessment(sample: np.ndarray, name: str, alpha: float) -> tuple[bool, str]:
    if len(sample) >= 30:
        return True, f"{name} has n={len(sample)}; mean inference uses the central limit theorem."
    if len(sample) < 3:
        return False, f"{name} is too small for a normality test."
    result = stats.shapiro(sample)
    normal = bool(float(result.pvalue) >= alpha)
    return normal, f"{name} Shapiro-Wilk p={float(result.pvalue):.4g}."


def _continuous_summary(sample: np.ndarray) -> dict[str, float | int]:
    return {
        "n": len(sample),
        "mean": float(np.mean(sample)),
        "standard_deviation": float(np.std(sample, ddof=1)),
        "median": float(np.median(sample)),
    }


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    degrees_freedom = len(a) + len(b) - 2
    pooled_variance = ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / degrees_freedom
    if pooled_variance <= 0:
        return 0.0
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    return float((np.mean(a) - np.mean(b)) / sqrt(pooled_variance) * correction)


def _validate_counts(successes: int, trials: int, label: str) -> None:
    if trials <= 0 or successes < 0 or successes > trials:
        raise StatisticalInputError(
            f"{label} requires trials > 0 and successes between zero and trials"
        )


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise StatisticalInputError("alpha must be between zero and one")


def _alpha_from_confidence(confidence_level: float) -> float:
    if not 0 < confidence_level < 1:
        raise StatisticalInputError("confidence_level must be between zero and one")
    return 1 - confidence_level


def _statsmodels_alternative(alternative: Alternative) -> str:
    return {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}[alternative]


def _alternative_text(alternative: Alternative, group_a: str, group_b: str) -> str:
    if alternative == "greater":
        return f"{group_a} is greater than {group_b}."
    if alternative == "less":
        return f"{group_a} is less than {group_b}."
    return f"{group_a} and {group_b} differ."


def _test_interpretation(
    significant: bool,
    test_used: str,
    group_a: str,
    group_b: str,
) -> str:
    if significant:
        return (
            f"Using the {test_used}, the observed difference between {group_a} and "
            f"{group_b} is statistically significant at the configured alpha."
        )
    return (
        f"Using the {test_used}, the evidence is insufficient to conclude that "
        f"{group_a} and {group_b} differ at the configured alpha."
    )
