#!/usr/bin/env python3
"""Review measured Stage A metrics and validate saved model artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.config import MLSettings  # noqa: E402
from src.ml.reporting import build_metrics_summary, validate_artifacts  # noqa: E402


DEFAULT_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "ml_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Metrics JSON path (defaults to ML_RESULTS_PATH or evaluation/ml_results.json).",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Print metrics without requiring local joblib artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configured = args.results or Path(os.getenv("ML_RESULTS_PATH", str(DEFAULT_RESULTS_PATH)))
    results_path = configured if configured.is_absolute() else PROJECT_ROOT / configured
    report = json.loads(results_path.read_text(encoding="utf-8"))
    output = {
        "results_path": str(results_path),
        "metrics": build_metrics_summary(report),
    }
    if not args.metrics_only:
        settings = MLSettings.from_env()
        output["artifacts"] = validate_artifacts(report, settings.artifact_dir)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
