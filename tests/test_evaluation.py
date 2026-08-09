from evaluation.evaluate_text_to_sql import (
    categorize_failure,
    results_equivalent,
    resume_metrics_payload,
)
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


def test_equivalence_allows_extra_supporting_generated_columns() -> None:
    actual = generated(
        ("state", "order_count", "supporting_revenue"),
        (("SP", 10, 1000.0), ("RJ", 5, 400.0)),
    )
    reference = QueryResult(
        ("state", "order_count"),
        (("SP", 10), ("RJ", 5)),
        2,
        1.0,
    )
    assert results_equivalent(
        actual,
        reference,
        "SELECT state, order_count FROM results ORDER BY order_count DESC",
    )


def test_equivalence_rejects_missing_reference_columns() -> None:
    actual = generated(("state",), (("SP",),))
    reference = QueryResult(("state", "order_count"), (("SP", 10),), 1, 1.0)
    assert not results_equivalent(actual, reference, "SELECT state, order_count FROM results")


def test_failure_categories_distinguish_validation_and_ranking_mismatches() -> None:
    invalid = generated((), ())
    invalid = PipelineResult(
        **{
            **invalid.to_dict(),
            "validation_passed": False,
            "validation_reason": "Unknown relation: invented",
            "error": "SQL validation failed: Unknown relation: invented",
        }
    )
    assert categorize_failure(invalid, False, "ranking") == "schema_hallucination"
    assert categorize_failure(generated(("state",), (("SP",),)), False, "ranking") == (
        "ranking_mismatch"
    )


def test_resume_metrics_are_derived_from_report() -> None:
    report = {
        "metrics": {
            "total_questions": 2,
            "valid_sql_rate": 1.0,
            "execution_success_rate": 0.5,
            "execution_accuracy": 0.5,
            "repair_rate": 0.5,
            "average_end_to_end_latency_ms": 2500.0,
        }
    }
    assert resume_metrics_payload(report)["avg_end_to_end_latency_seconds"] == 2.5
