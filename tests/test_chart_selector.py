from datetime import date
from decimal import Decimal

from src.analytics.chart_selector import ChartSelector
from src.analytics.result_analyzer import ResultAnalyzer


def select(columns: tuple[str, ...], rows: tuple[tuple[object, ...], ...], sql: str = ""):
    analysis = ResultAnalyzer().analyze(columns, rows, sql)
    return ChartSelector().select(analysis)


def test_scalar_selects_kpi() -> None:
    config = select(("total_revenue",), ((Decimal("100"),),))
    assert config.chart_type == "kpi"
    assert config.y == "total_revenue"


def test_time_series_selects_line_and_prefers_revenue() -> None:
    config = select(
        ("revenue_month", "order_count", "item_revenue"),
        ((date(2024, 1, 1), 2, Decimal("20")), (date(2024, 2, 1), 3, Decimal("30"))),
    )
    assert config.chart_type == "line"
    assert config.x == "revenue_month"
    assert config.y == "item_revenue"


def test_ranking_selects_horizontal_bar() -> None:
    config = select(
        ("category", "revenue"),
        (("A", 20), ("B", 10)),
        "SELECT category, revenue FROM result ORDER BY revenue DESC",
    )
    assert config.chart_type == "horizontal_bar"


def test_review_ranking_charts_review_score_not_revenue() -> None:
    config = select(
        ("category_name", "average_review_score", "order_count", "item_revenue"),
        (("A", Decimal("4.8"), 10, Decimal("100")), ("B", Decimal("4.5"), 20, Decimal("500"))),
        (
            "SELECT category_name, average_review_score, order_count, item_revenue "
            "FROM result ORDER BY average_review_score DESC, order_count DESC"
        ),
    )
    assert config.chart_type == "horizontal_bar"
    assert config.x == "average_review_score"
    assert config.y == "category_name"


def test_table_result_selects_table() -> None:
    config = select(
        ("order_id", "customer_id", "seller_id"),
        (("o1", "c1", "s1"), ("o2", "c2", "s2")),
    )
    assert config.chart_type == "table"


def test_empty_result_selects_no_chart() -> None:
    config = select(("category", "revenue"), ())
    assert config.chart_type == "none"
