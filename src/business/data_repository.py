"""Reviewed read-only SQL for executive business KPIs."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pandas as pd
import psycopg2
from psycopg2.extensions import connection as Connection

from src.db_config import DatabaseConfig


EXECUTIVE_KPI_SQL = """
WITH order_values AS (
    SELECT o.order_id, SUM(oi.price)::double precision AS order_value
    FROM orders AS o
    JOIN order_items AS oi ON oi.order_id = o.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY o.order_id
), delivery AS (
    SELECT
        COUNT(*)::integer AS delivered_orders,
        COUNT(*) FILTER (
            WHERE order_delivered_customer_date > order_estimated_delivery_date
        )::integer AS delayed_orders
    FROM orders
    WHERE order_status = 'delivered'
      AND order_delivered_customer_date IS NOT NULL
      AND order_estimated_delivery_date IS NOT NULL
)
SELECT
    SUM(v.order_value)::double precision AS total_revenue,
    COUNT(*)::integer AS order_volume,
    AVG(v.order_value)::double precision AS average_order_value,
    d.delivered_orders,
    d.delayed_orders,
    d.delayed_orders::double precision / NULLIF(d.delivered_orders, 0) AS delayed_delivery_rate
FROM order_values AS v
CROSS JOIN delivery AS d
GROUP BY d.delivered_orders, d.delayed_orders;
"""

MONTHLY_REVENUE_SQL = """
SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::date AS month,
    SUM(oi.price)::double precision AS revenue,
    COUNT(DISTINCT o.order_id)::integer AS orders
FROM orders AS o
JOIN order_items AS oi ON oi.order_id = o.order_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1
ORDER BY 1;
"""


class BusinessDataRepository:
    def __init__(
        self,
        config: DatabaseConfig | None = None,
        statement_timeout_ms: int = 60_000,
    ) -> None:
        self._config = config or DatabaseConfig.from_env()
        self._statement_timeout_ms = statement_timeout_ms

    def executive_kpis(self) -> dict[str, float | int]:
        frame = self._query(EXECUTIVE_KPI_SQL)
        if frame.empty:
            raise ValueError("No eligible orders are available for the executive overview")
        row = frame.iloc[0]
        return {
            "total_revenue": float(row["total_revenue"]),
            "order_volume": int(row["order_volume"]),
            "average_order_value": float(row["average_order_value"]),
            "delivered_orders": int(row["delivered_orders"]),
            "delayed_orders": int(row["delayed_orders"]),
            "delayed_delivery_rate": float(row["delayed_delivery_rate"]),
        }

    def monthly_revenue(self) -> list[dict[str, float | int | str]]:
        frame = self._query(MONTHLY_REVENUE_SQL)
        return [
            {
                "month": pd.Timestamp(row.month).date().isoformat(),
                "revenue": float(row.revenue),
                "orders": int(row.orders),
            }
            for row in frame.itertuples(index=False)
        ]

    def _query(self, query: str) -> pd.DataFrame:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout = %s", (self._statement_timeout_ms,))
                cursor.execute(query)
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
