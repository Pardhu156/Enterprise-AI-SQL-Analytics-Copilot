#!/usr/bin/env python3
"""Optional live smoke test for a running Phase 4 FastAPI service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--question",
        default="What are the top 5 product categories by revenue?",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        response = httpx.post(
            f"{args.url.rstrip('/')}/api/v1/analytics/query",
            json={"question": args.question},
            timeout=args.timeout,
        )
    except httpx.RequestError as exc:
        print(f"API request failed: {exc}", file=sys.stderr)
        return 2

    if response.is_error:
        print(json.dumps(response.json(), indent=2), file=sys.stderr)
        return 1

    payload = response.json()
    print(f"Question: {payload['question']}")
    print(f"Answer: {payload['answer'] or '(insight unavailable)'}")
    print("SQL:")
    print(payload.get("sql", {}).get("final_sql") or "(not included)")
    print(f"Rows returned: {payload['result']['row_count']}")
    print(f"Chart: {payload.get('visualization', {}).get('chart_type', 'none')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

