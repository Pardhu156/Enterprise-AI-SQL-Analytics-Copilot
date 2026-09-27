import pytest

from src.statistical.ab_testing import (
    analyze_average_value_experiment,
    analyze_conversion_experiment,
    synthetic_conversion_demo,
)
from src.statistical.engine import StatisticalInputError
from src.statistical.sample_size import (
    estimate_two_mean_sample_size,
    estimate_two_proportion_sample_size,
)


def test_conversion_ab_test_reports_lift_and_causal_guard() -> None:
    result = analyze_conversion_experiment(80, 1000, 110, 1000)

    assert result["absolute_lift"] == pytest.approx(0.03)
    assert result["relative_lift"] == pytest.approx(0.375)
    assert "association" in result["recommendation"]


def test_average_value_ab_test_uses_welch() -> None:
    result = analyze_average_value_experiment(
        [90, 100, 110, 100, 95],
        [110, 115, 120, 108, 117],
        randomized_experiment=True,
    )

    assert result["test_result"]["test_used"] == "Welch independent t-test"
    assert result["treatment_metric"] > result["control_metric"]


def test_synthetic_demo_is_prominently_labeled() -> None:
    result = synthetic_conversion_demo()

    assert result["synthetic_demo"] is True
    assert "synthetic" in result["recommendation"].lower()


def test_two_proportion_sample_size_matches_known_order_of_magnitude() -> None:
    result = estimate_two_proportion_sample_size(0.08, 0.02, power=0.8)

    assert 3000 < result["required_sample_size_per_group"] < 5000
    assert result["total_required_sample_size"] == 2 * result["required_sample_size_per_group"]


def test_two_mean_sample_size_uses_standardized_effect() -> None:
    result = estimate_two_mean_sample_size(100, 5, 20, power=0.8)

    assert result["assumptions"]["standardized_effect_size"] == pytest.approx(0.25)
    assert result["required_sample_size_per_group"] > 100


def test_sample_size_rejects_impossible_proportion() -> None:
    with pytest.raises(StatisticalInputError, match="between 0 and 1"):
        estimate_two_proportion_sample_size(0.99, 0.02)


def test_sample_size_rejects_zero_effect() -> None:
    with pytest.raises(StatisticalInputError, match="non-zero"):
        estimate_two_proportion_sample_size(0.08, 0)
