"""Leakage-safe late-delivery classification used instead of repeat purchase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


NUMERIC_FEATURES = (
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "average_item_price",
    "average_product_weight_g",
    "average_product_length_cm",
    "average_product_height_cm",
    "average_product_width_cm",
    "estimated_delivery_days",
    "purchase_hour",
    "purchase_day_of_week",
    "purchase_month",
)
CATEGORICAL_FEATURES = ("customer_state", "seller_state", "product_category")
DELIVERY_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class DeliveryTrainingResult:
    artifact: dict[str, Any]
    metrics: dict[str, Any]


def train_delivery_risk(data: pd.DataFrame, random_seed: int) -> DeliveryTrainingResult:
    frame = data.sort_values(["order_purchase_timestamp", "order_id"]).reset_index(drop=True)
    if len(frame) < 100:
        raise ValueError("At least 100 chronological orders are required for classification")
    train_end = int(len(frame) * 0.70)
    validation_end = int(len(frame) * 0.85)
    train = frame.iloc[:train_end]
    validation = frame.iloc[train_end:validation_end]
    test = frame.iloc[validation_end:]
    for split_name, split in (("training", train), ("validation", validation), ("test", test)):
        if split["late_delivery"].nunique() < 2:
            raise ValueError(f"The {split_name} split must contain both delivery target classes")
    features = list(DELIVERY_FEATURES)
    train_ratio = max((train["late_delivery"] == 0).sum() / (train["late_delivery"] == 1).sum(), 1)
    candidates: dict[str, Pipeline] = {
        "logistic_regression": _pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1_000,
                random_state=random_seed,
            )
        ),
        "random_forest": _pipeline(
            RandomForestClassifier(
                n_estimators=350,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=random_seed,
                n_jobs=-1,
            ),
            scale_numeric=False,
        ),
        "xgboost": _pipeline(
            XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                scale_pos_weight=float(train_ratio),
                random_state=random_seed,
                n_jobs=4,
            ),
            scale_numeric=False,
        ),
    }
    validation_metrics: dict[str, dict[str, float]] = {}
    thresholds: dict[str, float] = {}
    for name, model in candidates.items():
        model.fit(train[features], train["late_delivery"])
        probability = model.predict_proba(validation[features])[:, 1]
        threshold = _best_f1_threshold(validation["late_delivery"].to_numpy(), probability)
        thresholds[name] = threshold
        validation_metrics[name] = _classification_metrics(
            validation["late_delivery"], probability, threshold
        )
    best_name = max(validation_metrics, key=lambda name: validation_metrics[name]["pr_auc"])
    best_pr_auc = validation_metrics[best_name]["pr_auc"]
    selected_name = (
        "logistic_regression"
        if validation_metrics["logistic_regression"]["pr_auc"] >= best_pr_auc * 0.98
        else best_name
    )
    threshold = thresholds[selected_name]
    final_model = clone(candidates[selected_name])
    fit = frame.iloc[:validation_end]
    final_model.fit(fit[features], fit["late_delivery"])
    test_probability = final_model.predict_proba(test[features])[:, 1]
    test_metrics = _classification_metrics(test["late_delivery"], test_probability, threshold)
    artifact = {
        "model_type": "late_delivery_prediction",
        "model_name": selected_name,
        "pipeline": final_model,
        "feature_columns": features,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "training_period": {
            "start": pd.Timestamp(frame["order_purchase_timestamp"].min()).isoformat(),
            "end": pd.Timestamp(fit["order_purchase_timestamp"].max()).isoformat(),
        },
        "limitations": [
            "This model estimates late-delivery risk for historical Olist order patterns; it does not predict repeat purchase.",
            "The source contains no live carrier capacity, weather, traffic, or operational backlog features.",
            "Risk is associative and should not be interpreted as a causal explanation.",
        ],
    }
    metrics = {
        "task_selected": "late_delivery_prediction",
        "reason": (
            "Repeat purchase was not deployed because only about 3% of customers repeat overall "
            "and leakage-safe 90-day targets are below roughly 1%. Late delivery has a usable "
            f"positive rate of {frame['late_delivery'].mean():.4f}."
        ),
        "rows": len(frame),
        "positive_rate": float(frame["late_delivery"].mean()),
        "selected_model": selected_name,
        "selection_rule": (
            "Highest validation PR-AUC, preferring Logistic Regression when it is within "
            "2% of the best score to reduce overfitting and artifact size."
        ),
        "selected_threshold": threshold,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    return DeliveryTrainingResult(artifact=artifact, metrics=metrics)


def predict_delivery_risk(artifact: dict[str, Any], row: pd.DataFrame) -> dict[str, Any]:
    probability = float(
        artifact["pipeline"].predict_proba(row[list(artifact["feature_columns"])])[:, 1][0]
    )
    threshold = float(artifact["threshold"])
    if probability >= max(threshold, 0.5):
        band = "high"
    elif probability >= threshold * 0.6:
        band = "medium"
    else:
        band = "low"
    return {
        "late_delivery_probability": probability,
        "decision_threshold": threshold,
        "predicted_late": probability >= threshold,
        "risk_band": band,
    }


def _pipeline(classifier: Any, scale_numeric: bool = True) -> Pipeline:
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def _best_f1_threshold(actual: np.ndarray, probability: np.ndarray) -> float:
    choices = np.linspace(0.05, 0.75, 71)
    scores = [f1_score(actual, probability >= threshold, zero_division=0) for threshold in choices]
    return float(choices[int(np.argmax(scores))])


def _classification_metrics(actual: Any, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probability >= threshold
    return {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "pr_auc": float(average_precision_score(actual, probability)),
    }
