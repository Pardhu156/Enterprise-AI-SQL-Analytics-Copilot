#!/usr/bin/env python3
"""Interactive and one-shot CLI for the Phase 2 Text-to-SQL pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_to_sql.pipeline import PipelineResult, TextToSQLPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="Business question to ask")
    parser.add_argument("--no-repair", action="store_true", help="Disable the one repair attempt")
    return parser.parse_args()


def print_result(result: PipelineResult) -> None:
    print("\nGenerated SQL:")
    print(result.generated_sql or "(none)")
    if result.was_repaired:
        print("\nRepaired SQL:")
        print(result.final_sql or "(none)")
    print("\nValidation:")
    print("PASSED" if result.validation_passed else "FAILED", "-", result.validation_reason or "not run")
    if result.error:
        print("\nError:")
        print(result.error)
        return
    print("\nResults:")
    _print_table(result.columns, result.rows)
    suffix = " (truncated)" if result.truncated else ""
    print(f"\nRows returned: {result.row_count}{suffix}")
    print(f"Execution time: {result.execution_time_ms:.2f} ms")


def _print_table(columns: tuple[str, ...], rows: tuple[tuple[Any, ...], ...]) -> None:
    rendered = [[_cell(value) for value in row] for row in rows]
    widths = [len(column) for column in columns]
    for row in rendered:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    widths = [min(width, 40) for width in widths]

    def line(values: list[str]) -> str:
        return " | ".join(value[:width].ljust(width) for value, width in zip(values, widths))

    print(line(list(columns)))
    print("-+-".join("-" * width for width in widths))
    for row in rendered:
        print(line(row))


def _cell(value: Any) -> str:
    if value is None:
        return "NULL"
    return str(value).replace("\n", " ")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Enterprise AI SQL Analytics Copilot")
    question = " ".join(args.question).strip()
    if not question:
        question = input("\nAsk a business question:\n> ").strip()
    try:
        pipeline = TextToSQLPipeline.from_env(max_repair_attempts=0 if args.no_repair else 1)
        result = pipeline.ask(question)
    except Exception as exc:
        print(f"\nConfiguration error: {exc}", file=sys.stderr)
        return 2
    print_result(result)
    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
