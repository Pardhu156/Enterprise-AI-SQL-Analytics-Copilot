import numpy as np
import pandas as pd

from src.ml.segmentation import segment_customer, train_segmentation


def test_segmentation_compares_candidate_k_values_and_persists_pipeline() -> None:
    rng = np.random.default_rng(42)
    customers = 120
    high_value = np.arange(customers) >= customers // 2
    rfm = pd.DataFrame(
        {
            "customer_unique_id": [f"customer-{index}" for index in range(customers)],
            "recency_days": np.where(high_value, 25, 250) + rng.normal(0, 3, customers),
            "frequency": np.where(high_value, 4, 1),
            "monetary": np.where(high_value, 600, 80) + rng.normal(0, 10, customers),
            "average_order_value": np.where(high_value, 150, 80),
        }
    )

    result = train_segmentation(rfm, random_seed=42)
    cluster, label = segment_customer(result.artifact, rfm.iloc[[0]])

    assert set(result.metrics["cluster_evaluation"]) == {"2", "3", "4", "5", "6"}
    assert result.artifact["cluster_count"] in range(2, 7)
    assert isinstance(cluster, int)
    assert label in result.artifact["label_map"].values()
