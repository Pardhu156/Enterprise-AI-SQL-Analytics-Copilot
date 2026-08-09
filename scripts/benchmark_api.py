#!/usr/bin/env python3
"""Run a small, quota-conscious latency benchmark against a live FastAPI service."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


QUESTIONS = (
    "What is the total revenue?",
    "Show the monthly revenue trend.",
    "Which 10 product categories generated the most revenue?",
    "Who are the top 10 sellers by revenue?",
    "Are delayed deliveries associated with lower review scores?",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--repeat", type=int, default=1, choices=range(1, 4))
    parser.add_argument("--limit", type=int, choices=range(1, len(QUESTIONS) + 1))
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.0,
        help="Delay between API requests for Gemini quota control",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_benchmark(
    base_url: str,
    timeout: float,
    questions: tuple[str, ...],
    repeat: int,
    request_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        request_number = 0
        for _ in range(repeat):
            for question in questions:
                if request_number and request_delay_seconds:
                    time.sleep(request_delay_seconds)
                request_number += 1
                started = time.perf_counter()
                try:
                    response = client.post(
                        "/api/v1/analytics/query",
                        json={"question": question, "include_rows": False},
                    )
                    wall_time_ms = (time.perf_counter() - started) * 1_000
                    payload = response.json()
                    sql_details = payload.get("sql") or {}
                    visualization = payload.get("visualization") or {}
                    result = payload.get("result") or {}
                    has_answer = bool(payload.get("answer"))
                    has_sql = bool(sql_details.get("final_sql"))
                    has_result = "row_count" in result
                    has_visualization = bool(visualization.get("chart_type"))
                    success = (
                        response.is_success
                        and has_answer
                        and has_sql
                        and has_result
                        and has_visualization
                    )
                    records.append(
                        {
                            "question": question,
                            "status_code": response.status_code,
                            "success": success,
                            "wall_time_ms": wall_time_ms,
                            "api_total_time_ms": payload.get("execution", {}).get(
                                "total_request_time_ms"
                            ),
                            "has_answer": has_answer,
                            "has_sql": has_sql,
                            "has_result": has_result,
                            "row_count": result.get("row_count"),
                            "chart_type": visualization.get("chart_type"),
                            "request_id": payload.get("request_id"),
                            "error_code": payload.get("error", {}).get("code"),
                        }
                    )
                except (httpx.RequestError, ValueError) as exc:
                    records.append(
                        {
                            "question": question,
                            "status_code": None,
                            "success": False,
                            "wall_time_ms": (time.perf_counter() - started) * 1_000,
                            "api_total_time_ms": None,
                            "has_answer": False,
                            "has_sql": False,
                            "has_result": False,
                            "row_count": None,
                            "chart_type": None,
                            "request_id": None,
                            "error_code": type(exc).__name__,
                        }
                    )

    latencies = [record["wall_time_ms"] for record in records]
    successes = sum(record["success"] for record in records)
    return {
        "summary": {
            "requests": len(records),
            "successful_requests": successes,
            "success_rate": successes / len(records) if records else None,
            "average_latency_ms": statistics.fmean(latencies) if latencies else None,
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
        },
        "requests": records,
    }


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]


def main() -> int:
    args = parse_args()
    if args.request_delay_seconds < 0:
        raise ValueError("--request-delay-seconds cannot be negative")
    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    report = run_benchmark(
        args.url,
        args.timeout,
        questions,
        args.repeat,
        args.request_delay_seconds,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["summary"]["success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
