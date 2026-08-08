from evaluation.evaluate_text_to_sql import results_equivalent
from src.text_to_sql.pipeline import PipelineResult
from src.text_to_sql.sql_executor import QueryResult


def generated(columns: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> PipelineResult:
    return PipelineResult(
        question="test",
        generated_sql="SELECT 1",
        final_sql="SELECT 1",
        validation_passed=True,
        validation_reason="safe",
        was_repaired=False,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        execution_time_ms=1.0,
        truncated=False,
        error=None,
    )


def test_equivalence_reorders_columns_and_unordered_rows_with_float_tolerance() -> None:
    actual = generated(
        ("metric", "state"),
        ((2.000000001, "RJ"), (1.0, "SP")),
    )
    reference = QueryResult(
        ("state", "metric"),
        (("SP", 1.0), ("RJ", 2.0)),
        2,
        1.0,
    )

    assert results_equivalent(actual, reference, "SELECT state, metric FROM results")


def test_equivalence_respects_top_level_order_by() -> None:
    actual = generated(("state",), (("RJ",), ("SP",)))
    reference = QueryResult(("state",), (("SP",), ("RJ",)), 2, 1.0)

    assert not results_equivalent(
        actual,
        reference,
        "SELECT state FROM results ORDER BY state",
    )
