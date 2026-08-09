from scripts.benchmark_api import _percentile


def test_nearest_rank_percentiles_are_deterministic() -> None:
    values = [100.0, 200.0, 300.0, 400.0]
    assert _percentile(values, 50) == 200.0
    assert _percentile(values, 95) == 400.0
    assert _percentile([], 95) is None
