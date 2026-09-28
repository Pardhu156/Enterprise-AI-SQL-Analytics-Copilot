"""Inference-service serialization boundary tests."""

from decimal import Decimal

import numpy as np

from src.ml.service import _safe_scalar


def test_safe_scalar_returns_json_compatible_database_and_numpy_values() -> None:
    assert _safe_scalar(Decimal("15.625")) == 15.625
    assert _safe_scalar(np.float64(2.5)) == 2.5
    assert _safe_scalar(np.nan) is None
    assert _safe_scalar("SP") == "SP"
