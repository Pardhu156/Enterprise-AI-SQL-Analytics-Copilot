#!/usr/bin/env python3
"""Execution-based evaluation of generated SQL against Phase 1 references."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import sqlglot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_to_sql.pipeline import PipelineResult, TextToSQLPipeline  # noqa: E402
from src.text_to_sql.schema_manager import SchemaManager  # noqa: E402
from src.text_to_sql.sql_executor import QueryResult, SQLExecutionError, SQLExecutor  # noqa: E402
from src.text_to_sql.sql_validator import SQLValidator  # noqa: E402


DEFAULT_BENCHMARK = PROJECT_ROOT / "evaluation" / "benchmark_questions.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions")
    parser.add_argument("--no-repair", action="store_true")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        records = records[: args.limit]

    schema_manager = SchemaManager()
    schema = schema_manager.get_snapshot()
    validator = SQLValidator()
    reference_executor = SQLExecutor()
    pipeline = TextToSQLPipeline.from_env(max_repair_attempts=0 if args.no_repair else 1)
    details: list[dict[str, Any]] = []

    for index, benchmark in enumerate(records, start=1):
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
                "execution_time_ms": generated.execution_time_ms,
                "generated_result": result_payload(generated),
                "generated_error": generated.error,
                "reference_sql": benchmark["reference_sql"],
                "reference_execution_succeeded": reference_result is not None,
                "reference_result": query_result_payload(reference_result),
                "reference_error": reference_error,
                "results_equivalent": equivalent,
            }
        )

    total = len(details)
    valid_count = sum(item["validation_passed"] for item in details)
    execution_count = sum(item["execution_succeeded"] for item in details)
    accurate_count = sum(item["results_equivalent"] for item in details)
    repair_count = sum(item["repair_required"] for item in details)
    latencies = [
        item["execution_time_ms"]
        for item in details
        if item["execution_time_ms"] is not None
    ]
    metrics = {
        "total_questions": total,
        "valid_sql_rate": _rate(valid_count, total),
        "execution_success_rate": _rate(execution_count, total),
        "execution_accuracy": _rate(accurate_count, total),
        "repair_rate": _rate(repair_count, total),
        "average_query_latency_ms": (
            sum(latencies) / len(latencies) if latencies else None
        ),
    }
    return {"metrics": metrics, "questions": details}


def results_equivalent(
    generated: PipelineResult,
    reference: QueryResult,
    reference_sql: str,
) -> bool:
    if generated.truncated or reference.truncated:
        return False
    if len(generated.columns) != len(reference.columns):
        return False

    generated_rows = [list(row) for row in generated.rows]
    reference_rows = [list(row) for row in reference.rows]
    generated_names = [name.lower() for name in generated.columns]
    reference_names = [name.lower() for name in reference.columns]
    if len(set(generated_names)) == len(generated_names) and set(generated_names) == set(reference_names):
        positions = [generated_names.index(name) for name in reference_names]
        generated_rows = [[row[position] for position in positions] for row in generated_rows]

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
    print(json.dumps(report["metrics"], indent=2))
    print(f"Detailed results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
