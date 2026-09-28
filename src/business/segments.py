"""Data-derived interpretations for persisted RFM cluster profiles."""

from __future__ import annotations

from typing import Any


def interpret_segment_profiles(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not profiles:
        return []
    average_recency = _weighted_average(profiles, "recency_days")
    average_frequency = _weighted_average(profiles, "frequency")
    average_monetary = _weighted_average(profiles, "monetary")
    interpretations: list[dict[str, Any]] = []
    for profile in profiles:
        recent = float(profile["recency_days"]) <= average_recency
        repeat = float(profile["frequency"]) > max(1.05, average_frequency)
        valuable = float(profile["monetary"]) >= average_monetary
        characteristics = [
            "more recent than the portfolio average" if recent else "less recent than the portfolio average",
            "repeat purchasing behavior" if repeat else "predominantly one-time purchasing behavior",
            "above-average historical value" if valuable else "below-average historical value",
        ]
        if repeat and valuable:
            importance = "A comparatively valuable repeat-customer segment worth protecting."
            action = "Prioritize retention, loyalty recognition, and relevant cross-sell tests."
        elif recent and valuable:
            importance = "A recent, valuable segment with potential to deepen engagement."
            action = "Use timely post-purchase engagement and carefully measured second-order offers."
        elif not recent:
            importance = "A less-recent segment where broad spending may have low efficiency."
            action = "Test low-cost reactivation and suppress outreach if incremental response is weak."
        else:
            importance = "A broad lower-engagement segment that needs evidence before heavier investment."
            action = "Use onboarding and repeat-purchase experiments with explicit holdouts."
        interpretations.append(
            {
                "cluster": int(profile["cluster"]),
                "segment": str(profile["segment"]),
                "characteristics": characteristics,
                "business_importance": importance,
                "recommended_consideration": action,
                "evidence": {
                    "customers": int(profile["customers"]),
                    "average_recency_days": float(profile["recency_days"]),
                    "average_frequency": float(profile["frequency"]),
                    "average_monetary_value": float(profile["monetary"]),
                },
            }
        )
    return interpretations


def _weighted_average(profiles: list[dict[str, Any]], field: str) -> float:
    customers = sum(int(profile["customers"]) for profile in profiles)
    return (
        sum(float(profile[field]) * int(profile["customers"]) for profile in profiles)
        / customers
    )
