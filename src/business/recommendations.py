"""Evidence-bounded recommendation rules for ML and statistical results."""

from __future__ import annotations

from typing import Any


class RecommendationService:
    def historical(
        self,
        result_type: str,
        row_count: int,
        truncated: bool,
    ) -> list[dict[str, Any]]:
        if row_count == 0:
            return []
        if result_type == "time_series":
            action = "Monitor the latest periods and investigate operational drivers before changing the plan."
        elif result_type in {"ranking", "categorical_comparison"}:
            action = "Prioritize follow-up analysis on the leading and trailing groups before reallocating resources."
        elif result_type == "scalar_kpi":
            action = "Use the verified KPI as a baseline and compare it with a defined target or prior period."
        else:
            action = "Use the returned records to define a narrower decision question before taking action."
        limitations = " The result was row-limited." if truncated else ""
        return [
            {
                "title": "Historical analytics follow-up",
                "action": action,
                "rationale": (
                    "The recommendation uses only the verified result shape and does not infer an unsupported cause."
                    + limitations
                ),
                "priority": "low",
                "evidence": [f"Result type: {result_type}", f"Rows returned: {row_count}"],
            }
        ]

    def forecast(self, impact: list[dict[str, Any]]) -> list[dict[str, Any]]:
        metrics = {item["name"]: item for item in impact}
        rate = float(metrics["forecast_change_rate_vs_recent"]["value"])
        forecast = float(metrics["forecast_revenue"]["value"])
        baseline = float(metrics["recent_comparable_revenue"]["value"])
        if rate <= -0.05:
            title = "Prepare for lower near-term demand"
            action = "Review inventory commitments and acquisition spend against the lower forecast scenario."
            priority = "high"
        elif rate >= 0.05:
            title = "Plan capacity for forecast growth"
            action = "Validate inventory, seller capacity, and fulfillment readiness before increasing commitments."
            priority = "medium"
        else:
            title = "Operate near the recent revenue baseline"
            action = "Maintain the current plan and monitor weekly actuals for divergence from the forecast."
            priority = "low"
        return [
            {
                "title": title,
                "action": action,
                "rationale": "The forecast was compared with an equally sized recent historical window.",
                "priority": priority,
                "evidence": [
                    f"Predicted revenue: BRL {forecast:.2f}",
                    f"Recent comparable revenue: BRL {baseline:.2f}",
                    f"Relative change: {rate:.4f}",
                ],
            }
        ]

    def segments(self, interpretations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "title": f"{item['segment']} strategy",
                "action": item["recommended_consideration"],
                "rationale": item["business_importance"],
                "priority": "high" if "protecting" in item["business_importance"] else "medium",
                "evidence": [
                    f"Customers: {item['evidence']['customers']}",
                    f"Average frequency: {item['evidence']['average_frequency']:.4f}",
                    f"Average monetary value: BRL {item['evidence']['average_monetary_value']:.2f}",
                ],
            }
            for item in interpretations
        ]

    def delivery_risk(
        self,
        prediction: dict[str, Any],
        explanation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        band = str(prediction["risk_band"])
        drivers = [
            item["feature"]
            for item in explanation["top_contributions"]
            if item["direction"] == "increases"
        ][:3]
        if band == "high":
            action = "Prioritize fulfillment review and proactive delivery communication for this order."
            priority = "high"
        elif band == "medium":
            action = "Monitor fulfillment milestones and escalate only if operational signals deteriorate."
            priority = "medium"
        else:
            action = "Use the standard fulfillment workflow while retaining normal tracking controls."
            priority = "low"
        return [
            {
                "title": f"{band.title()} late-delivery risk",
                "action": action,
                "rationale": "The action tier follows the deployed model's probability and saved decision threshold.",
                "priority": priority,
                "evidence": [
                    f"Predicted probability: {float(prediction['late_delivery_probability']):.6f}",
                    *[f"Increasing model contribution: {driver}" for driver in drivers],
                ],
            }
        ]

    def statistical(self, task: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        if task == "hypothesis_test":
            significant = bool(result["significant"])
            return [
                {
                    "title": "Investigate the measured group difference" if significant else "Do not act on significance alone",
                    "action": (
                        "Review the effect size and operational context, then test targeted changes before rollout."
                        if significant
                        else "Collect more evidence or define a practically meaningful effect before changing operations."
                    ),
                    "rationale": str(result["interpretation"]),
                    "priority": "medium" if significant else "low",
                    "evidence": [
                        f"Test: {result['test_used']}",
                        f"p-value: {float(result['p_value']):.6g}",
                    ],
                }
            ]
        if task == "ab_test":
            significant = bool(result["statistically_significant"])
            positive = float(result["absolute_lift"]) > 0
            randomized = bool(result["randomized_experiment"])
            supported = significant and positive and randomized and not result.get("synthetic_demo")
            return [
                {
                    "title": "Treatment evidence review",
                    "action": (
                        "Consider a guarded rollout with continued measurement."
                        if supported
                        else "Do not claim a treatment win from this result."
                    ),
                    "rationale": str(result["recommendation"]),
                    "priority": "medium" if supported else "low",
                    "evidence": [
                        f"Absolute lift: {float(result['absolute_lift']):.6g}",
                        f"Relative lift: {float(result['relative_lift'] or 0):.6g}",
                    ],
                }
            ]
        return []
