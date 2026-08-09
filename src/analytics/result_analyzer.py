"""Deterministic classification of PostgreSQL query result shapes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from numbers import Number
from typing import Any


METRIC_TERMS = (
    "amount",
    "average",
    "avg",
    "cost",
    "count",
    "days",
    "duration",
    "freight",
    "metric",
    "payment",
    "percentage",
    "price",
    "qty",
    "quantity",
    "rate",
    "revenue",
    "score",
    "total",
    "value",
    "volume",
)
DATE_TERMS = ("date", "day", "month", "quarter", "timestamp", "week", "year")
IDENTIFIER_TERMS = ("_id", "id_", "identifier", "sequential", "zip_code")


@dataclass(frozen=True)
class ResultAnalysis:
    result_type: str
    dimensions: tuple[str, ...]
    metrics: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    row_count: int
    has_datetime: bool
    is_empty: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResultAnalyzer:
    def analyze(
        self,
        columns: tuple[str, ...],
        rows: tuple[tuple[Any, ...], ...],
        sql: str | None = None,
    ) -> ResultAnalysis:
        row_count = len(rows)
        if not columns or row_count == 0:
            return ResultAnalysis(
                result_type="empty",
                dimensions=(),
                metrics=(),
                categorical_columns=(),
                numeric_columns=(),
                datetime_columns=(),
                identifier_columns=(),
                row_count=row_count,
                has_datetime=False,
                is_empty=True,
            )

        values_by_column = {
            column: tuple(row[index] for row in rows if index < len(row))
            for index, column in enumerate(columns)
        }
        numeric: list[str] = []
        datetimes: list[str] = []
        categorical: list[str] = []
        identifiers: list[str] = []

        for column, values in values_by_column.items():
            non_null = [value for value in values if value is not None]
            normalized_name = column.lower()
            if _is_identifier(normalized_name):
                identifiers.append(column)
            if non_null and all(_is_datetime(value) for value in non_null):
                datetimes.append(column)
            elif non_null and all(_is_numeric(value) for value in non_null):
                numeric.append(column)
            else:
                categorical.append(column)

        metrics = [
            column
            for column in numeric
            if column not in identifiers and _is_likely_metric(column)
        ]
        if not metrics:
            metrics = [column for column in numeric if column not in identifiers]
        metrics = _prioritize_ordered_metrics(metrics, sql)

        dimensions = [
            column
            for column in (*datetimes, *categorical)
            if column not in identifiers
        ]
        if not dimensions:
            dimensions = [column for column in identifiers if column not in metrics]

        result_type = _classify_result(
            columns=columns,
            rows=rows,
            metrics=metrics,
            dimensions=dimensions,
            datetime_columns=datetimes,
            numeric_columns=numeric,
            identifier_columns=identifiers,
            sql=sql,
        )
        return ResultAnalysis(
            result_type=result_type,
            dimensions=tuple(dimensions),
            metrics=tuple(metrics),
            categorical_columns=tuple(categorical),
            numeric_columns=tuple(numeric),
            datetime_columns=tuple(datetimes),
            identifier_columns=tuple(identifiers),
            row_count=row_count,
            has_datetime=bool(datetimes),
            is_empty=False,
        )


def _classify_result(
    columns: tuple[str, ...],
    rows: tuple[tuple[Any, ...], ...],
    metrics: list[str],
    dimensions: list[str],
    datetime_columns: list[str],
    numeric_columns: list[str],
    identifier_columns: list[str],
    sql: str | None,
) -> str:
    if len(rows) == 1 and len(columns) == 1 and metrics:
        return "scalar_kpi"
    if len(rows) > 1 and datetime_columns and metrics:
        return "time_series"
    if len(rows) > 1 and dimensions and metrics:
        if _looks_ranked(columns, rows, metrics, sql):
            return "ranking"
        return "categorical_comparison"
    non_identifier_numeric = [
        column for column in numeric_columns if column not in identifier_columns
    ]
    if len(rows) > 2 and len(non_identifier_numeric) >= 2:
        return "numeric_relationship"
    if len(rows) > 5 and len(non_identifier_numeric) == 1 and len(columns) == 1:
        return "distribution"
    if len(columns) >= 2 or identifier_columns:
        return "table"
    return "unsupported"


def _looks_ranked(
    columns: tuple[str, ...],
    rows: tuple[tuple[Any, ...], ...],
    metrics: list[str],
    sql: str | None,
) -> bool:
    if not sql or "order by" not in sql.lower():
        return False
    order_clause = sql.lower().rsplit("order by", maxsplit=1)[-1]
    for metric in metrics:
        if metric.lower() not in order_clause:
            continue
        try:
            index = columns.index(metric)
        except ValueError:
            continue
        values = [
            row[index]
            for row in rows
            if index < len(row) and _is_numeric(row[index])
        ]
        if len(values) < 2:
            continue
        if all(left >= right for left, right in zip(values, values[1:])) or all(
            left <= right for left, right in zip(values, values[1:])
        ):
            return True
    return False


def _prioritize_ordered_metrics(metrics: list[str], sql: str | None) -> list[str]:
    """Put an ORDER BY metric first so ranking charts use the ranked measure."""
    if not sql or "order by" not in sql.lower():
        return metrics
    order_clause = sql.lower().rsplit("order by", maxsplit=1)[-1]
    ordered = [metric for metric in metrics if metric.lower() in order_clause]
    return ordered + [metric for metric in metrics if metric not in ordered]


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (Number, Decimal)) and not isinstance(value, bool)


def _is_datetime(value: Any) -> bool:
    return isinstance(value, (datetime, date))


def _is_identifier(name: str) -> bool:
    return name == "id" or any(term in name for term in IDENTIFIER_TERMS)


def _is_likely_metric(name: str) -> bool:
    normalized = name.lower()
    return any(term in normalized for term in METRIC_TERMS)


def humanize_column(name: str) -> str:
    return name.replace("_", " ").strip().title()


def select_primary_metric(metrics: list[str] | tuple[str, ...]) -> str:
    priorities = (
        "revenue",
        "percentage",
        "rate",
        "score",
        "value",
        "freight",
        "cost",
        "price",
        "payment",
        "average",
        "avg",
        "days",
        "duration",
        "count",
        "qty",
        "quantity",
        "volume",
    )
    return min(
        metrics,
        key=lambda metric: next(
            (index for index, term in enumerate(priorities) if term in metric.lower()),
            len(priorities),
        ),
    )
