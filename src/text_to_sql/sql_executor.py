"""Read-only, bounded PostgreSQL execution for validated generated SQL."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import psycopg2

from src.db_config import DatabaseConfig

from .sql_validator import ValidationResult


@dataclass(frozen=True)
class ExecutorConfig:
    statement_timeout_ms: int = 15_000
    max_rows: int = 1_000

    @classmethod
    def from_env(cls) -> "ExecutorConfig":
        return cls(
            statement_timeout_ms=_positive_int_env("SQL_STATEMENT_TIMEOUT_MS", 15_000),
            max_rows=_positive_int_env("SQL_MAX_ROWS", 1_000),
        )


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    execution_time_ms: float
    truncated: bool = False


class SQLExecutionError(RuntimeError):
    """A sanitized, recoverable database execution error."""


ConnectionFactory = Callable[..., "psycopg2.extensions.connection"]


class SQLExecutor:
    def __init__(
        self,
        database_config: DatabaseConfig | None = None,
        executor_config: ExecutorConfig | None = None,
        connection_factory: ConnectionFactory = psycopg2.connect,
    ) -> None:
        self._database_config = database_config
        self._executor_config = executor_config or ExecutorConfig.from_env()
        self._connection_factory = connection_factory

    def execute(self, sql: str, validation: ValidationResult) -> QueryResult:
        if not validation.valid or not validation.normalized_sql:
            raise ValueError("SQLExecutor requires a successful ValidationResult")

        config = self._database_config or DatabaseConfig.from_env()
        started = time.perf_counter()
        try:
            with self._connection_factory(**config.as_connect_kwargs()) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute(
                        "SET LOCAL statement_timeout = %s",
                        (self._executor_config.statement_timeout_ms,),
                    )
                    cursor.execute(validation.normalized_sql)
                    if cursor.description is None:
                        raise SQLExecutionError("Validated query did not return a result set")
                    columns = tuple(description.name for description in cursor.description)
                    fetched = cursor.fetchmany(self._executor_config.max_rows + 1)
                    truncated = len(fetched) > self._executor_config.max_rows
                    rows = tuple(tuple(row) for row in fetched[: self._executor_config.max_rows])
            elapsed = (time.perf_counter() - started) * 1000
            return QueryResult(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=elapsed,
                truncated=truncated,
            )
        except SQLExecutionError:
            raise
        except psycopg2.Error as exc:
            message = exc.diag.message_primary if exc.diag and exc.diag.message_primary else str(exc)
            raise SQLExecutionError(message.strip()) from exc


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
