"""Curated, parameterized, read-only PostgreSQL datasets for statistical analysis."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import pandas as pd
import psycopg2
from psycopg2.extensions import connection as Connection

from src.db_config import DatabaseConfig


ORDER_VALUE_BY_STATE_SQL = """
WITH order_values AS (
    SELECT
        o.order_id,
        c.customer_state,
        SUM(oi.price)::double precision AS metric_value
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
      AND c.customer_state IN (%s, %s)
    GROUP BY o.order_id, c.customer_state
)
SELECT customer_state AS group_name, metric_value
FROM order_values
ORDER BY customer_state, order_id;
"""

DELIVERY_TIME_BY_STATE_SQL = """
SELECT
    c.customer_state AS group_name,
    EXTRACT(EPOCH FROM (o.order_delivered_customer_date - o.order_purchase_timestamp))
        / 86400.0 AS metric_value
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_delivered_customer_date >= o.order_purchase_timestamp
  AND c.customer_state IN (%s, %s)
ORDER BY c.customer_state, o.order_id;
"""

REVIEW_BY_DELIVERY_STATUS_SQL = """
WITH order_reviews AS (
    SELECT order_id, AVG(review_score)::double precision AS metric_value
    FROM reviews
    WHERE review_score IS NOT NULL
    GROUP BY order_id
)
SELECT
    CASE
        WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 'Delayed'
        ELSE 'On time'
    END AS group_name,
    r.metric_value
FROM orders AS o
JOIN order_reviews AS r ON r.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
ORDER BY group_name, o.order_id;
"""

REPEAT_PURCHASE_BY_STATE_SQL = """
WITH customer_state_frequency AS (
    SELECT
        c.customer_unique_id,
        c.customer_state,
        COUNT(DISTINCT o.order_id)::integer AS order_count
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
      AND c.customer_state IN (%s, %s)
    GROUP BY c.customer_unique_id, c.customer_state
)
SELECT
    customer_state AS group_name,
    COUNT(*)::integer AS trials,
    COUNT(*) FILTER (WHERE order_count > 1)::integer AS successes
FROM customer_state_frequency
GROUP BY customer_state
ORDER BY customer_state;
"""

ORDER_VALUES_SQL = """
SELECT SUM(oi.price)::double precision AS metric_value
FROM orders AS o
JOIN order_items AS oi ON oi.order_id = o.order_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY o.order_id
ORDER BY o.order_id;
"""

REVIEW_SCORES_SQL = """
SELECT AVG(review_score)::double precision AS metric_value
FROM reviews
WHERE review_score IS NOT NULL
GROUP BY order_id
ORDER BY order_id;
"""

DELAYED_DELIVERY_COUNTS_SQL = """
SELECT
    COUNT(*)::integer AS trials,
    COUNT(*) FILTER (
        WHERE order_delivered_customer_date > order_estimated_delivery_date
    )::integer AS successes
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL;
"""


class StatisticalDataRepository:
    """Load reviewed statistical datasets without LLM-generated SQL."""

    def __init__(
        self,
        config: DatabaseConfig | None = None,
        statement_timeout_ms: int = 60_000,
    ) -> None:
        self._config = config or DatabaseConfig.from_env()
        self._statement_timeout_ms = statement_timeout_ms

    def order_values_by_state(self, state_a: str, state_b: str) -> pd.DataFrame:
        return self._query(ORDER_VALUE_BY_STATE_SQL, (state_a, state_b))

    def delivery_times_by_state(self, state_a: str, state_b: str) -> pd.DataFrame:
        return self._query(DELIVERY_TIME_BY_STATE_SQL, (state_a, state_b))

    def review_scores_by_delivery_status(self) -> pd.DataFrame:
        return self._query(REVIEW_BY_DELIVERY_STATUS_SQL)

    def repeat_purchase_counts_by_state(self, state_a: str, state_b: str) -> pd.DataFrame:
        return self._query(REPEAT_PURCHASE_BY_STATE_SQL, (state_a, state_b))

    def order_values(self) -> list[float]:
        return self._query(ORDER_VALUES_SQL)["metric_value"].astype(float).tolist()

    def review_scores(self) -> list[float]:
        return self._query(REVIEW_SCORES_SQL)["metric_value"].astype(float).tolist()

    def delayed_delivery_counts(self) -> tuple[int, int]:
        row = self._query(DELAYED_DELIVERY_COUNTS_SQL).iloc[0]
        return int(row["successes"]), int(row["trials"])

    def _query(self, query: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout = %s", (self._statement_timeout_ms,))
                cursor.execute(query, params)
                columns = [description.name for description in cursor.description]
                rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=columns)

    @contextmanager
    def _connection(self) -> Iterator[Connection]:
        connection = psycopg2.connect(**self._config.as_connect_kwargs())
        try:
            connection.set_session(readonly=True, autocommit=False)
            yield connection
            connection.rollback()
        finally:
            connection.close()
