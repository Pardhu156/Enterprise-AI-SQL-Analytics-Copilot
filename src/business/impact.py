"""Deterministic business-impact calculations over verified analytical outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


class BusinessImpactService:
    def forecast(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        forecast = pd.DataFrame(data["forecast"])
        history = pd.DataFrame(data["recent_history"])
        horizon = int(data["horizon_weeks"])
        predicted_total = float(forecast["predicted_revenue"].sum())
        comparison = history.tail(horizon)
        historical_total = float(comparison["revenue"].sum())
        change = predicted_total - historical_total
        change_rate = change / historical_total if historical_total else 0.0
        assumption = (
            f"The comparison baseline is the most recent {len(comparison)} complete historical weeks."
        )
        return [
            _metric(
                "forecast_revenue",
                predicted_total,
                "BRL",
                "predicted",
                f"Model forecast across the next {horizon} weeks.",
            ),
            _metric(
                "recent_comparable_revenue",
                historical_total,
                "BRL",
                "observed",
                "Observed revenue across the equally sized recent comparison window.",
            ),
            _metric(
                "forecast_change_vs_recent",
                change,
                "BRL",
                "scenario",
                "Forecast minus the recent comparison window; this is not realized revenue.",
                [assumption],
            ),
            _metric(
                "forecast_change_rate_vs_recent",
                change_rate,
                "ratio",
                "scenario",
                "Relative forecast change against the recent comparison window.",
                [assumption],
            ),
        ]

    def segments(self, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total_value = sum(
            float(profile["customers"]) * float(profile["monetary"])
            for profile in profiles
        )
        impacts: list[dict[str, Any]] = []
        for profile in profiles:
            segment_value = float(profile["customers"]) * float(profile["monetary"])
            impacts.extend(
                [
                    _metric(
                        f"{profile['segment']}_historical_customer_value",
                        segment_value,
                        "BRL",
                        "observed",
                        "Sum-equivalent historical customer monetary value represented by the segment profile.",
                    ),
                    _metric(
                        f"{profile['segment']}_value_contribution",
                        segment_value / total_value if total_value else 0.0,
                        "ratio",
                        "observed",
                        "Share of historical customer monetary value represented by this segment.",
                    ),
                ]
            )
        return impacts

    def delivery_risk(
        self,
        prediction: dict[str, Any],
        order_features: dict[str, Any],
    ) -> list[dict[str, Any]]:
        probability = float(prediction["late_delivery_probability"])
        order_value = float(order_features.get("total_price") or 0.0)
        return [
            _metric(
                "order_value_exposed_to_delivery_risk",
                order_value,
                "BRL",
                "observed",
                "Observed merchandise value of the scored order.",
            ),
            _metric(
                "probability_weighted_order_value",
                order_value * probability,
                "BRL",
                "scenario",
                "Order value multiplied by predicted late-delivery probability; this is exposure, not expected revenue loss.",
                [
                    "The calculation assumes order value is a useful exposure proxy.",
                    "No cancellation, refund, or margin effect is inferred.",
                ],
            ),
        ]

    def statistical(self, task: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        if task == "ab_test":
            return [
                _metric(
                    "absolute_lift",
                    float(result["absolute_lift"]),
                    "metric units",
                    "observed",
                    "Treatment metric minus control metric in the supplied experiment data.",
                ),
                _metric(
                    "relative_lift",
                    float(result["relative_lift"] or 0.0),
                    "ratio",
                    "observed",
                    "Absolute lift divided by the control metric.",
                ),
            ]
        if task == "hypothesis_test" and result.get("estimate") is not None:
            return [
                _metric(
                    "observed_group_difference",
                    float(result["estimate"]),
                    "metric units",
                    "observed",
                    "Observed Group A minus Group B difference; interpretation depends on the selected metric.",
                )
            ]
        return []


def _metric(
    name: str,
    value: float,
    unit: str,
    kind: str,
    description: str,
    assumptions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "kind": kind,
        "description": description,
        "assumptions": assumptions or [],
    }
