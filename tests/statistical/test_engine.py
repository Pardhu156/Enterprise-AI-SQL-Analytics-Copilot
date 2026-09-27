import numpy as np
import pytest

from src.statistical.engine import (
    StatisticalInputError,
    confidence_interval_difference_means,
    confidence_interval_difference_proportions,
    confidence_interval_mean,
    confidence_interval_proportion,
    test_contingency_table as run_contingency_test,
    test_continuous_groups as run_continuous_test,
    test_two_proportions as run_proportion_test,
)


def test_auto_selection_uses_welch_for_large_unequal_variance_samples() -> None:
    rng = np.random.default_rng(42)
    group_a = rng.normal(10, 1, 80)
    group_b = rng.normal(8, 5, 80)

    result = run_continuous_test(group_a, group_b)

    assert result.test_used == "Welch independent t-test"
    assert result.significant
    assert result.effect_size.name == "hedges_g"
    assert result.confidence_interval.lower < result.estimate < result.confidence_interval.upper


def test_ordinal_outcome_uses_mann_whitney() -> None:
    delayed = [1, 1, 2, 2, 3, 2, 1]
    on_time = [4, 5, 4, 4, 5, 3, 4]

    result = run_continuous_test(delayed, on_time, scale="ordinal")

    assert result.test_used == "Mann-Whitney U test"
    assert result.p_value < 0.05
    assert result.effect_size.name == "rank_biserial_correlation"


def test_two_proportion_z_test_and_newcombe_interval() -> None:
    result = run_proportion_test(120, 1000, 80, 1000)

    assert result.test_used == "Two-proportion z-test"
    assert result.estimate == pytest.approx(0.04)
    assert result.confidence_interval.lower < 0.04 < result.confidence_interval.upper
    assert result.effect_size.name == "cohens_h"


def test_sparse_two_by_two_table_uses_fisher_exact() -> None:
    result = run_contingency_test([[1, 9], [8, 2]])

    assert result.test_used == "Fisher exact test"
    assert result.effect_size.name == "odds_ratio"


def test_chi_square_reports_cramers_v() -> None:
    result = run_contingency_test([[30, 20, 10], [10, 20, 30]])

    assert result.test_used == "Chi-square test of independence"
    assert result.effect_size.name == "cramers_v"
    assert result.significant


def test_confidence_interval_variants_are_finite_and_ordered() -> None:
    intervals = [
        confidence_interval_mean([8, 9, 10, 11, 12]),
        confidence_interval_proportion(80, 1000),
        confidence_interval_difference_means([10, 11, 12], [7, 8, 9]),
        confidence_interval_difference_proportions(120, 1000, 80, 1000),
    ]

    for interval in intervals:
        assert np.isfinite([interval.lower, interval.upper]).all()
        assert interval.lower <= interval.point_estimate <= interval.upper


def test_invalid_or_constant_inputs_fail_cleanly() -> None:
    with pytest.raises(StatisticalInputError, match="at least 2"):
        confidence_interval_mean([1])
    with pytest.raises(StatisticalInputError, match="successes"):
        confidence_interval_proportion(11, 10)
    with pytest.raises(StatisticalInputError, match="variation"):
        run_continuous_test([1, 1, 1], [1, 1, 1])
    with pytest.raises(StatisticalInputError, match="one success and one failure"):
        run_proportion_test(0, 10, 0, 10)
