#!/usr/bin/env python3
"""Execution-based evaluation of generated SQL against Phase 1 references."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import sqlglot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_to_sql.pipeline import PipelineResult, TextToSQLPipeline  # noqa: E402
from src.text_to_sql.llm_client import LLMConfig  # noqa: E402
from src.text_to_sql.schema_manager import SchemaManager  # noqa: E402
from src.text_to_sql.sql_executor import QueryResult, SQLExecutionError, SQLExecutor  # noqa: E402
from src.text_to_sql.sql_validator import SQLValidator  # noqa: E402


DEFAULT_BENCHMARK = PROJECT_ROOT / "evaluation" / "benchmark_questions.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--csv-output", type=Path, help="Optional question-level CSV path")
    parser.add_argument("--summary-output", type=Path, help="Optional measured summary JSON path")
    parser.add_argument(
        "--resume-metrics-output",
        type=Path,
        help="Optional compact measured metrics JSON path",
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions")
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.0,
        help="Delay between Gemini questions to respect provider rate limits",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        records = records[: args.limit]
    if args.request_delay_seconds < 0:
        raise ValueError("--request-delay-seconds cannot be negative")

    schema_manager = SchemaManager()
    schema = schema_manager.get_snapshot()
    validator = SQLValidator()
    reference_executor = SQLExecutor()
    pipeline = TextToSQLPipeline.from_env(max_repair_attempts=0 if args.no_repair else 1)
    details: list[dict[str, Any]] = []

    for index, benchmark in enumerate(records, start=1):
        if index > 1 and args.request_delay_seconds:
            time.sleep(args.request_delay_seconds)
        logging.info("Evaluating %d/%d: %s", index, len(records), benchmark["question"])
        generated = pipeline.ask(benchmark["question"])
        reference_validation = validator.validate(benchmark["reference_sql"], schema)
        reference_result: QueryResult | None = None
        reference_error: str | None = None
        if reference_validation.valid:
            try:
                reference_result = reference_executor.execute(
                    benchmark["reference_sql"], reference_validation
                )
            except SQLExecutionError as exc:
                reference_error = str(exc)
        else:
            reference_error = reference_validation.reason

        equivalent = False
        if generated.error is None and reference_result is not None:
            equivalent = results_equivalent(
                generated,
                reference_result,
                reference_sql=benchmark["reference_sql"],
            )

        failure_category = categorize_failure(
            generated,
            results_match=equivalent,
            benchmark_category=benchmark["category"],
        )
        details.append(
            {
                "id": benchmark["id"],
                "category": benchmark["category"],
                "question": benchmark["question"],
                "generated_sql": generated.generated_sql,
                "final_sql": generated.final_sql,
                "validation_passed": generated.validation_passed,
                "execution_succeeded": generated.error is None,
                "repair_required": generated.was_repaired,
                "generation_time_ms": generated.generation_time_ms,
                "validation_time_ms": generated.validation_time_ms,
                "execution_time_ms": generated.execution_time_ms,
                "repair_time_ms": generated.repair_time_ms,
                "end_to_end_time_ms": generated.total_time_ms,
                "generated_result": result_payload(generated),
                "generated_error": generated.error,
                "reference_sql": benchmark["reference_sql"],
                "reference_execution_succeeded": reference_result is not None,
                "reference_result": query_result_payload(reference_result),
                "reference_error": reference_error,
                "results_equivalent": equivalent,
                "failure_category": failure_category,
            }
        )

    total = len(details)
    valid_count = sum(item["validation_passed"] for item in details)
    execution_count = sum(item["execution_succeeded"] for item in details)
    accurate_count = sum(item["results_equivalent"] for item in details)
    repair_count = sum(item["repair_required"] for item in details)
    failure_breakdown = dict(
        sorted(Counter(item["failure_category"] for item in details if item["failure_category"]).items())
    )
    metrics = {
        "total_questions": total,
        "valid_sql_rate": _rate(valid_count, total),
        "execution_success_rate": _rate(execution_count, total),
        "execution_accuracy": _rate(accurate_count, total),
        "repair_rate": _rate(repair_count, total),
        "average_sql_generation_latency_ms": _average(details, "generation_time_ms"),
        "average_sql_validation_latency_ms": _average(details, "validation_time_ms"),
        "average_sql_execution_latency_ms": _average(details, "execution_time_ms"),
        "average_repair_latency_ms": _average(details, "repair_time_ms"),
        "average_end_to_end_latency_ms": _average(details, "end_to_end_time_ms"),
        "failure_count": sum(failure_breakdown.values()),
        "failure_breakdown": failure_breakdown,
    }
    llm_config = LLMConfig.from_env()
    return {
        "metadata": {
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "llm_provider": llm_config.provider,
            "llm_model": llm_config.model,
            "comparison_method": "PostgreSQL result equivalence",
        },
        "metrics": metrics,
        "questions": details,
    }


def categorize_failure(
    generated: PipelineResult,
    results_match: bool,
    benchmark_category: str,
) -> str | None:
    """Return a deterministic, non-LLM failure category for benchmark analysis."""
    if results_match:
        return None

    error = (generated.error or "").lower()
    reason = (generated.validation_reason or "").lower()
    if error:
        if "timeout" in error or "timed out" in error:
            return "timeout"
        if "generation failed" in error:
            if any(marker in error for marker in ("gemini", "quota", "429", "api")):
                return "gemini_api_error"
            return "generation_failure"
        if "validation failed" in error:
            if any(
                marker in reason
                for marker in ("unknown relation", "unknown table", "unknown view", "unknown column")
            ):
                return "schema_hallucination"
            return "invalid_generated_sql"
        if "repair" in error:
            return "repair_failure"
        if "execution failed" in error:
            return "execution_failure"
        return "pipeline_failure"

    mismatch_categories = {
        "ranking": "ranking_mismatch",
        "time_series": "date_time_mismatch",
        "aggregation": "aggregation_mismatch",
        "correlation": "relationship_mismatch",
    }
    return mismatch_categories.get(benchmark_category, "result_mismatch")


def results_equivalent(
    generated: PipelineResult,
    reference: QueryResult,
    reference_sql: str,
) -> bool:
    if generated.truncated or reference.truncated:
        return False
    if len(generated.columns) < len(reference.columns):
        return False

    generated_rows = [list(row) for row in generated.rows]
    reference_rows = [list(row) for row in reference.rows]
    generated_names = [name.lower() for name in generated.columns]
    reference_names = [name.lower() for name in reference.columns]
    if (
        len(set(generated_names)) == len(generated_names)
        and len(set(reference_names)) == len(reference_names)
        and set(reference_names).issubset(generated_names)
    ):
        positions = [generated_names.index(name) for name in reference_names]
        generated_rows = [[row[position] for position in positions] for row in generated_rows]
    elif len(generated.columns) != len(reference.columns):
        return False

    generated_normalized = [tuple(_normalize_cell(value) for value in row) for row in generated_rows]
    reference_normalized = [tuple(_normalize_cell(value) for value in row) for row in reference_rows]
    if not _has_order_by(reference_sql):
        generated_normalized.sort(key=repr)
        reference_normalized.sort(key=repr)
    if len(generated_normalized) != len(reference_normalized):
        return False
    return all(
        all(_cells_equal(left, right) for left, right in zip(generated_row, reference_row))
        for generated_row, reference_row in zip(generated_normalized, reference_normalized)
    )


def result_payload(result: PipelineResult) -> dict[str, Any] | None:
    if result.error:
        return None
    return {
        "columns": list(result.columns),
        "rows": [list(row) for row in result.rows],
        "row_count": result.row_count,
        "truncated": result.truncated,
    }


def query_result_payload(result: QueryResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "columns": list(result.columns),
        "rows": [list(row) for row in result.rows],
        "row_count": result.row_count,
        "truncated": result.truncated,
        "execution_time_ms": result.execution_time_ms,
    }


def _normalize_cell(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _cells_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-7)
    return left == right


def _has_order_by(sql: str) -> bool:
    try:
        statement = sqlglot.parse_one(sql, read="postgres")
    except sqlglot.errors.ParseError:
        return True
    return statement.args.get("order") is not None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _average(details: list[dict[str, Any]], key: str) -> float | None:
    values = [item[key] for item in details if item[key] is not None]
    return sum(values) / len(values) if values else None


def write_csv(path: Path, questions: list[dict[str, Any]]) -> None:
    fields = (
        "id",
        "category",
        "question",
        "validation_passed",
        "execution_succeeded",
        "results_equivalent",
        "repair_required",
        "failure_category",
        "generation_time_ms",
        "validation_time_ms",
        "execution_time_ms",
        "repair_time_ms",
        "end_to_end_time_ms",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: question.get(field) for field in fields} for question in questions)


def summary_payload(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    return {
        "benchmark": "Olist Text-to-SQL execution-equivalence benchmark",
        "metadata": report.get("metadata", {}),
        "metrics": metrics,
    }


def resume_metrics_payload(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    return {
        "benchmark_questions": metrics["total_questions"],
        "sql_validation_pass_rate": metrics["valid_sql_rate"],
        "execution_success_rate": metrics["execution_success_rate"],
        "execution_accuracy": metrics["execution_accuracy"],
        "repair_rate": metrics["repair_rate"],
        "avg_end_to_end_latency_seconds": (
            metrics["average_end_to_end_latency_ms"] / 1_000
            if metrics["average_end_to_end_latency_ms"] is not None
            else None
        ),
    }


def _json_default(value: Any) -> str:
    if isinstance(value, (Decimal, datetime, date)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = evaluate(args)
    output = args.output
    if output is None:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = DEFAULT_RESULTS_DIR / "latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    csv_output = args.csv_output or output.with_suffix(".csv")
    write_csv(csv_output, report["questions"])
    for destination, payload in (
        (args.summary_output, summary_payload(report)),
        (args.resume_metrics_output, resume_metrics_payload(report)),
    ):
        if destination is None:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Detailed results: {output}")
    print(f"Question metrics: {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
