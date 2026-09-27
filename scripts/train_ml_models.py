#!/usr/bin/env python3
"""Train all Stage A models from PostgreSQL and persist reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.artifacts import save_artifact  # noqa: E402
from src.ml.config import MLSettings  # noqa: E402
from src.ml.data_repository import MLDataRepository  # noqa: E402
from src.ml.delivery_risk import train_delivery_risk  # noqa: E402
from src.ml.forecasting import prepare_weekly_revenue, train_forecaster  # noqa: E402
from src.ml.segmentation import train_segmentation  # noqa: E402


DEFAULT_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "ml_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("all", "forecast", "segmentation", "delivery"),
        default="all",
        help="Train one model family or all Stage A models.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Exit successfully when every requested artifact already exists.",
    )
    return parser.parse_args()


def repeat_purchase_feasibility(purchases: pd.DataFrame) -> dict[str, Any]:
    frame = purchases.copy()
    frame["order_purchase_timestamp"] = pd.to_datetime(frame["order_purchase_timestamp"])
    frequency = frame.groupby("customer_unique_id")["order_id"].nunique()
    overall_rate = float((frequency > 1).mean())
    end = frame["order_purchase_timestamp"].max()
    rolling: list[dict[str, Any]] = []
    for cutoff in pd.to_datetime(["2017-09-01", "2017-12-01", "2018-03-01", "2018-06-01"]):
        eligible = set(
            frame.loc[
                (frame["order_purchase_timestamp"] >= cutoff - pd.Timedelta(days=90))
                & (frame["order_purchase_timestamp"] < cutoff),
                "customer_unique_id",
            ]
        )
        future = set(
            frame.loc[
                (frame["order_purchase_timestamp"] >= cutoff)
                & (frame["order_purchase_timestamp"] < cutoff + pd.Timedelta(days=90)),
                "customer_unique_id",
            ]
        )
        positives = len(eligible & future)
        rolling.append(
            {
                "cutoff": cutoff.date().isoformat(),
                "eligible_customers": len(eligible),
                "positive_customers": positives,
                "positive_rate": positives / len(eligible) if eligible else None,
            }
        )
    return {
        "dataset_end": end.isoformat(),
        "unique_customers": int(frequency.size),
        "repeat_customers": int((frequency > 1).sum()),
        "overall_repeat_rate": overall_rate,
        "rolling_90_day_active_customer_cohorts": rolling,
        "decision": "use_late_delivery_prediction",
        "reason": (
            "Only about 3% of customers repeat anywhere in the snapshot, while leakage-safe "
            "90-day cohort targets remain around 1% or lower. The abrupt dataset end also "
            "right-censors future purchases, making repeat-purchase deployment unreliable."
        ),
    }


def main() -> int:
    args = parse_args()
    settings = MLSettings.from_env()
    results_path = Path(os.getenv("ML_RESULTS_PATH", str(DEFAULT_RESULTS_PATH)))
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path
    requested = {
        "all": ("sales_forecast", "customer_segmentation", "delivery_risk"),
        "forecast": ("sales_forecast",),
        "segmentation": ("customer_segmentation",),
        "delivery": ("delivery_risk",),
    }[args.model]
    if args.skip_existing and all(settings.artifact_path(name).is_file() for name in requested):
        print("All requested ML artifacts already exist; skipping training")
        return 0
    repository = MLDataRepository()
    trained_at = datetime.now(UTC).isoformat()
    report: dict[str, Any] = {}
    if args.model != "all" and results_path.is_file():
        report = json.loads(results_path.read_text(encoding="utf-8"))
    report.update(
        {
            "trained_at_utc": trained_at,
            "random_seed": settings.random_seed,
        }
    )

    purchases = repository.customer_purchase_history()
    report["repeat_purchase_feasibility"] = repeat_purchase_feasibility(purchases)

    if args.model in ("all", "forecast"):
        latest = repository.latest_valid_order_timestamp()
        history = prepare_weekly_revenue(repository.weekly_revenue(), latest)
        result = train_forecaster(
            history,
            horizon_weeks=settings.forecast_horizon_weeks,
            random_seed=settings.random_seed,
        )
        artifact = {
            **result.artifact,
            "trained_at_utc": trained_at,
            "data_as_of": latest.isoformat(),
        }
        save_artifact(settings.artifact_path("sales_forecast"), artifact)
        report["sales_forecasting"] = result.metrics
        print(f"sales_forecasting selected={result.metrics['selected_model']}")

    if args.model in ("all", "segmentation"):
        latest = repository.latest_valid_order_timestamp()
        as_of = latest + pd.Timedelta(days=1)
        result = train_segmentation(repository.customer_rfm(as_of), settings.random_seed)
        artifact = {
            **result.artifact,
            "trained_at_utc": trained_at,
            "data_as_of": as_of.isoformat(),
        }
        save_artifact(settings.artifact_path("customer_segmentation"), artifact)
        report["customer_segmentation"] = result.metrics
        print(f"customer_segmentation clusters={result.metrics['selected_clusters']}")

    if args.model in ("all", "delivery"):
        result = train_delivery_risk(
            repository.late_delivery_training_data(), settings.random_seed
        )
        artifact = {
            **result.artifact,
            "trained_at_utc": trained_at,
        }
        save_artifact(settings.artifact_path("delivery_risk"), artifact)
        report["classification"] = result.metrics
        print(f"classification selected={result.metrics['selected_model']}")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"metrics={results_path}")
    print(f"artifacts={settings.artifact_dir}")
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
