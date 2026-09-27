"""RFM-based customer segmentation with persisted preprocessing and K-Means."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


RFM_FEATURES = ("recency_days", "frequency", "monetary")


@dataclass(frozen=True)
class SegmentationTrainingResult:
    artifact: dict[str, Any]
    metrics: dict[str, Any]


def train_segmentation(rfm: pd.DataFrame, random_seed: int) -> SegmentationTrainingResult:
    clean = rfm.loc[(rfm["monetary"] > 0) & (rfm["frequency"] > 0)].copy()
    features = list(RFM_FEATURES)
    if len(clean) < 20:
        raise ValueError("At least 20 eligible customers are required for segmentation")
    if clean[features].isna().any().any():
        raise ValueError("RFM features must not contain missing values")
    evaluations: dict[str, dict[str, float | int]] = {}
    candidates: dict[int, Pipeline] = {}
    for clusters in range(2, 7):
        pipeline = Pipeline(
            [
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
                (
                    "cluster",
                    KMeans(n_clusters=clusters, n_init=20, random_state=random_seed),
                ),
            ]
        )
        labels = pipeline.fit_predict(clean[features])
        transformed = pipeline[:-1].transform(clean[features])
        counts = pd.Series(labels).value_counts()
        evaluations[str(clusters)] = {
            "silhouette": float(
                silhouette_score(
                    transformed,
                    labels,
                    sample_size=min(10_000, len(clean)),
                    random_state=random_seed,
                )
            ),
            "davies_bouldin": float(davies_bouldin_score(transformed, labels)),
            "minimum_cluster_size": int(counts.min()),
        }
        candidates[clusters] = pipeline

    eligible = [
        clusters
        for clusters in candidates
        if evaluations[str(clusters)]["minimum_cluster_size"] >= len(clean) * 0.01
    ]
    chosen_clusters = max(
        eligible or list(candidates),
        key=lambda value: evaluations[str(value)]["silhouette"],
    )
    pipeline = candidates[chosen_clusters]
    clean["cluster"] = pipeline.predict(clean[features])
    profiles = (
        clean.groupby("cluster")
        .agg(
            customers=("customer_unique_id", "nunique"),
            recency_days=("recency_days", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
            average_order_value=("average_order_value", "mean"),
        )
        .reset_index()
    )
    label_map = _segment_labels(profiles)
    profiles["segment"] = profiles["cluster"].map(label_map)
    artifact = {
        "model_type": "customer_segmentation",
        "model_name": "kmeans_rfm",
        "pipeline": pipeline,
        "feature_columns": features,
        "cluster_count": chosen_clusters,
        "label_map": label_map,
        "profiles": profiles.to_dict(orient="records"),
        "evaluation": evaluations,
        "limitations": [
            "More than 96% of customers purchased once, so frequency contributes less separation than recency and monetary value.",
            "Segments describe behavior in the historical Olist snapshot and are not causal customer personas.",
        ],
    }
    metrics = {
        "customers": len(clean),
        "selected_clusters": chosen_clusters,
        "cluster_evaluation": evaluations,
        "profiles": artifact["profiles"],
    }
    return SegmentationTrainingResult(artifact=artifact, metrics=metrics)


def segment_customer(artifact: dict[str, Any], rfm_row: pd.DataFrame) -> tuple[int, str]:
    columns = list(artifact["feature_columns"])
    cluster = int(artifact["pipeline"].predict(rfm_row[columns])[0])
    return cluster, str(artifact["label_map"][cluster])


def _segment_labels(profiles: pd.DataFrame) -> dict[int, str]:
    normalized = profiles.copy()
    for column in ("recency_days", "frequency", "monetary"):
        spread = normalized[column].std(ddof=0) or 1.0
        normalized[f"z_{column}"] = (normalized[column] - normalized[column].mean()) / spread
    normalized["value_score"] = (
        -normalized["z_recency_days"]
        + normalized["z_frequency"]
        + normalized["z_monetary"]
    )
    ordered = normalized.sort_values("value_score", ascending=False)["cluster"].astype(int).tolist()
    if len(ordered) == 2:
        return {ordered[0]: "High-value repeat", ordered[1]: "One-time customers"}
    names = [
        "High-value active",
        "Loyal",
        "Promising",
        "Occasional",
        "At risk",
        "Dormant",
    ]
    return {cluster: names[min(index, len(names) - 1)] for index, cluster in enumerate(ordered)}
