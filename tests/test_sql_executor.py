from types import SimpleNamespace

import psycopg2
import pytest

from src.db_config import DatabaseConfig
from src.text_to_sql.sql_executor import ExecutorConfig, SQLExecutionError, SQLExecutor
from src.text_to_sql.sql_validator import ValidationResult


DATABASE = DatabaseConfig("db", 5432, "olist", "reader", "secret")
VALIDATION = ValidationResult(True, "safe", "SELECT 1 AS value")


class FakeCursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows
        self.description = None
        self.executed: list[tuple[str, object | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, statement: str, parameters=None) -> None:
        self.executed.append((statement, parameters))
        if statement.startswith("SELECT"):
            self.description = [SimpleNamespace(name="value")]

    def fetchmany(self, size: int):
        return self.rows[:size]


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self) -> FakeCursor:
        return self._cursor


def test_executor_enforces_read_only_timeout_and_row_limit() -> None:
    cursor = FakeCursor([(1,), (2,), (3,)])
    executor = SQLExecutor(
        database_config=DATABASE,
        executor_config=ExecutorConfig(statement_timeout_ms=250, max_rows=2),
        connection_factory=lambda **kwargs: FakeConnection(cursor),
    )

    result = executor.execute("ignored", VALIDATION)

    assert result.rows == ((1,), (2,))
    assert result.truncated is True
    assert cursor.executed == [
        ("SET TRANSACTION READ ONLY", None),
        ("SET LOCAL statement_timeout = %s", (250,)),
        ("SELECT 1 AS value", None),
    ]


def test_executor_rejects_unvalidated_sql() -> None:
    executor = SQLExecutor(database_config=DATABASE)
    with pytest.raises(ValueError, match="successful ValidationResult"):
        executor.execute("DROP TABLE orders", ValidationResult(False, "unsafe"))


def test_executor_sanitizes_database_driver_errors() -> None:
    def unavailable(**kwargs):
        raise psycopg2.OperationalError("connection unavailable")

    executor = SQLExecutor(database_config=DATABASE, connection_factory=unavailable)
    with pytest.raises(SQLExecutionError, match="connection unavailable"):
        executor.execute("ignored", VALIDATION)
