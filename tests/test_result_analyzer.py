from datetime import date
from decimal import Decimal

from src.analytics.result_analyzer import ResultAnalyzer


def test_scalar_result() -> None:
    analysis = ResultAnalyzer().analyze(
        ("total_revenue",),
        ((Decimal("13494400.74"),),),
        "SELECT SUM(item_revenue) AS total_revenue FROM revenue",
    )
    assert analysis.result_type == "scalar_kpi"
    assert analysis.metrics == ("total_revenue",)


def test_categorical_result() -> None:
    analysis = ResultAnalyzer().analyze(
        ("customer_state", "order_count"),
        (("SP", 10), ("RJ", 20), ("MG", 15)),
        "SELECT customer_state, order_count FROM results",
    )
    assert analysis.result_type == "categorical_comparison"
    assert analysis.dimensions == ("customer_state",)
    assert analysis.metrics == ("order_count",)


def test_ranking_result() -> None:
    analysis = ResultAnalyzer().analyze(
        ("category_name", "item_revenue"),
        (("health", Decimal("20")), ("sports", Decimal("10"))),
        "SELECT category_name, item_revenue FROM results ORDER BY item_revenue DESC",
    )
    assert analysis.result_type == "ranking"


def test_ranking_prioritizes_the_metric_used_for_ordering() -> None:
    analysis = ResultAnalyzer().analyze(
        ("category_name", "average_review_score", "order_count", "item_revenue"),
        (("A", Decimal("4.8"), 10, Decimal("100")), ("B", Decimal("4.5"), 20, Decimal("500"))),
        (
            "SELECT category_name, average_review_score, order_count, item_revenue "
            "FROM results ORDER BY average_review_score DESC, order_count DESC"
        ),
    )
    assert analysis.result_type == "ranking"
    assert analysis.metrics[0] == "average_review_score"


def test_time_series_result() -> None:
    analysis = ResultAnalyzer().analyze(
        ("revenue_month", "order_count", "item_revenue"),
        ((date(2024, 1, 1), 10, Decimal("100")), (date(2024, 2, 1), 12, Decimal("140"))),
        "SELECT revenue_month, order_count, item_revenue FROM monthly ORDER BY revenue_month",
    )
    assert analysis.result_type == "time_series"
    assert analysis.datetime_columns == ("revenue_month",)


def test_numeric_relationship_result() -> None:
    analysis = ResultAnalyzer().analyze(
        ("freight_value", "review_score"),
        ((10.0, 5), (20.0, 3), (30.0, 1)),
    )
    assert analysis.result_type == "numeric_relationship"


def test_empty_result() -> None:
    analysis = ResultAnalyzer().analyze(("category", "revenue"), ())
    assert analysis.result_type == "empty"
    assert analysis.is_empty
