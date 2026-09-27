"""Fixed, read-only PostgreSQL queries used by ML training and inference."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Sequence

import pandas as pd
import psycopg2
from psycopg2.extensions import connection as Connection

from src.db_config import DatabaseConfig


WEEKLY_REVENUE_SQL = """
SELECT
    DATE_TRUNC('week', o.order_purchase_timestamp)::date AS week_start,
    SUM(oi.price)::double precision AS revenue,
    COUNT(DISTINCT o.order_id)::integer AS order_count,
    COUNT(*)::integer AS item_count
FROM orders AS o
JOIN order_items AS oi ON oi.order_id = o.order_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1
ORDER BY 1;
"""

LATEST_VALID_ORDER_SQL = """
SELECT MAX(order_purchase_timestamp)
FROM orders
WHERE order_status NOT IN ('canceled', 'unavailable');
"""

RFM_SQL = """
WITH order_values AS (
    SELECT
        o.order_id,
        o.customer_id,
        o.order_purchase_timestamp,
        SUM(oi.price)::double precision AS order_revenue,
        COUNT(*)::integer AS item_count
    FROM orders AS o
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
      AND o.order_purchase_timestamp < %s
    GROUP BY o.order_id, o.customer_id, o.order_purchase_timestamp
)
SELECT
    c.customer_unique_id,
    (%s::timestamp - MAX(v.order_purchase_timestamp))::interval AS recency_interval,
    COUNT(DISTINCT v.order_id)::integer AS frequency,
    SUM(v.order_revenue)::double precision AS monetary,
    AVG(v.order_revenue)::double precision AS average_order_value,
    SUM(v.item_count)::integer AS item_count
FROM order_values AS v
JOIN customers AS c ON c.customer_id = v.customer_id
GROUP BY c.customer_unique_id
ORDER BY c.customer_unique_id;
"""

RFM_CUSTOMER_SQL = RFM_SQL.replace(
    "GROUP BY c.customer_unique_id\nORDER BY c.customer_unique_id;",
    "WHERE c.customer_unique_id = %s\nGROUP BY c.customer_unique_id\nORDER BY c.customer_unique_id;",
)

LATE_DELIVERY_SQL = """
WITH item_features AS (
    SELECT
        oi.order_id,
        COUNT(*)::integer AS item_count,
        COUNT(DISTINCT oi.seller_id)::integer AS seller_count,
        SUM(oi.price)::double precision AS total_price,
        SUM(oi.freight_value)::double precision AS total_freight,
        AVG(oi.price)::double precision AS average_item_price,
        AVG(p.product_weight_g)::double precision AS average_product_weight_g,
        AVG(p.product_length_cm)::double precision AS average_product_length_cm,
        AVG(p.product_height_cm)::double precision AS average_product_height_cm,
        AVG(p.product_width_cm)::double precision AS average_product_width_cm,
        MIN(s.seller_state)::text AS seller_state,
        MIN(COALESCE(t.product_category_name_english, p.product_category_name, 'unknown'))
            AS product_category
    FROM order_items AS oi
    JOIN products AS p ON p.product_id = oi.product_id
    JOIN sellers AS s ON s.seller_id = oi.seller_id
    LEFT JOIN product_category_translation AS t
        ON t.product_category_name = p.product_category_name
    GROUP BY oi.order_id
)
SELECT
    o.order_id,
    o.order_purchase_timestamp,
    c.customer_state::text AS customer_state,
    f.seller_state,
    f.product_category,
    f.item_count,
    f.seller_count,
    f.total_price,
    f.total_freight,
    f.average_item_price,
    f.average_product_weight_g,
    f.average_product_length_cm,
    f.average_product_height_cm,
    f.average_product_width_cm,
    EXTRACT(EPOCH FROM (o.order_estimated_delivery_date - o.order_purchase_timestamp))
        / 86400.0 AS estimated_delivery_days,
    EXTRACT(HOUR FROM o.order_purchase_timestamp)::integer AS purchase_hour,
    EXTRACT(DOW FROM o.order_purchase_timestamp)::integer AS purchase_day_of_week,
    EXTRACT(MONTH FROM o.order_purchase_timestamp)::integer AS purchase_month,
    CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
        THEN 1 ELSE 0 END::integer AS late_delivery
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN item_features AS f ON f.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
ORDER BY o.order_purchase_timestamp, o.order_id;
"""

LATE_DELIVERY_ORDER_SQL = LATE_DELIVERY_SQL.replace(
    "ORDER BY o.order_purchase_timestamp, o.order_id;",
    "AND o.order_id = %s\nORDER BY o.order_purchase_timestamp, o.order_id;",
)

CUSTOMER_PURCHASE_HISTORY_SQL = """
SELECT
    c.customer_unique_id,
    o.order_id,
    o.order_purchase_timestamp
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
ORDER BY o.order_purchase_timestamp, o.order_id;
"""


class MLDataRepository:
    """Extract model features with reviewed SQL and a read-only transaction."""

    def __init__(
        self,
        config: DatabaseConfig | None = None,
        statement_timeout_ms: int = 60_000,
    ) -> None:
        self._config = config or DatabaseConfig.from_env()
        self._statement_timeout_ms = statement_timeout_ms

    def weekly_revenue(self) -> pd.DataFrame:
        return self._query(WEEKLY_REVENUE_SQL)

    def latest_valid_order_timestamp(self) -> pd.Timestamp:
        frame = self._query(LATEST_VALID_ORDER_SQL)
        value = frame.iloc[0, 0]
        if value is None:
            raise ValueError("No valid orders are available for ML feature extraction")
        return pd.Timestamp(value)

    def customer_rfm(self, as_of: datetime | pd.Timestamp) -> pd.DataFrame:
        frame = self._query(RFM_SQL, (as_of, as_of))
        frame["recency_days"] = pd.to_timedelta(
            frame.pop("recency_interval")
        ).dt.total_seconds() / 86_400
        return frame

    def customer_rfm_by_id(
        self,
        customer_unique_id: str,
        as_of: datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        frame = self._query(RFM_CUSTOMER_SQL, (as_of, as_of, customer_unique_id))
        if not frame.empty:
            frame["recency_days"] = pd.to_timedelta(
                frame.pop("recency_interval")
            ).dt.total_seconds() / 86_400
        return frame

    def late_delivery_training_data(self) -> pd.DataFrame:
        return self._query(LATE_DELIVERY_SQL)

    def late_delivery_order(self, order_id: str) -> pd.DataFrame:
        return self._query(LATE_DELIVERY_ORDER_SQL, (order_id,))

    def customer_purchase_history(self) -> pd.DataFrame:
        return self._query(CUSTOMER_PURCHASE_HISTORY_SQL)

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
