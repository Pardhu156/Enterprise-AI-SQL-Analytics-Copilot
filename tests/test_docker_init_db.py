import pytest

from scripts.docker_init_db import classify_counts


def test_classify_counts_empty() -> None:
    assert classify_counts({"orders": 0, "order_items": 0}) == "empty"


def test_classify_counts_populated() -> None:
    assert classify_counts({"orders": 10, "order_items": 12}) == "populated"


def test_classify_counts_rejects_partial_database() -> None:
    with pytest.raises(RuntimeError, match="partially populated"):
        classify_counts({"orders": 10, "order_items": 0})
