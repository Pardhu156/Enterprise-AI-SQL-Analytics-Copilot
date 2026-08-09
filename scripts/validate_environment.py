#!/usr/bin/env python3
"""Validate runtime configuration without printing secret values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.dependencies import APISettings  # noqa: E402
from src.db_config import DatabaseConfig  # noqa: E402
from src.frontend.api_client import FrontendAPIConfig  # noqa: E402
from src.text_to_sql.llm_client import LLMConfig  # noqa: E402


def validate_backend() -> None:
    DatabaseConfig.from_env()
    llm = LLMConfig.from_env()
    if llm.provider != "gemini":
        raise ValueError("LLM_PROVIDER must be 'gemini'")
    APISettings.from_env()


def validate_frontend() -> None:
    FrontendAPIConfig.from_env()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("backend", "frontend", "all"))
    args = parser.parse_args()
    try:
        if args.component in ("backend", "all"):
            validate_backend()
        if args.component in ("frontend", "all"):
            validate_frontend()
    except ValueError as exc:
        print(f"Configuration invalid: {exc}")
        return 1
    print(f"Configuration valid for: {args.component}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
