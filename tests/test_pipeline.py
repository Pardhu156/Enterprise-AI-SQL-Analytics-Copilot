from collections import deque

from src.text_to_sql.pipeline import TextToSQLPipeline
from src.text_to_sql.schema_manager import ColumnInfo, RelationInfo, SchemaSnapshot
from src.text_to_sql.sql_executor import QueryResult, SQLExecutionError
from src.text_to_sql.sql_validator import SQLValidator


SCHEMA = SchemaSnapshot(
    relations=(
        RelationInfo(
            "customers",
            "TABLE",
            (ColumnInfo("customer_id", "text", False, True),),
        ),
    )
)


class FakeSchemaManager:
    def get_snapshot(self) -> SchemaSnapshot:
        return SCHEMA


class FakeGenerator:
    def __init__(self, sql: str) -> None:
        self.sql = sql

    def generate(self, question: str) -> str:
        return self.sql


class FakeExecutor:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = deque(outcomes)
        self.calls = 0

    def execute(self, sql: str, validation: object) -> QueryResult:
        self.calls += 1
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, QueryResult)
        return outcome


class FakeRepairer:
    def __init__(self, repaired_sql: str) -> None:
        self.repaired_sql = repaired_sql
        self.calls = 0

    def repair(self, **kwargs: object) -> str:
        self.calls += 1
        return self.repaired_sql


def build_pipeline(executor: FakeExecutor, repairer: FakeRepairer, attempts: int = 1) -> TextToSQLPipeline:
    return TextToSQLPipeline(
        schema_manager=FakeSchemaManager(),  # type: ignore[arg-type]
        generator=FakeGenerator("SELECT c.customer_id FROM customers AS c"),  # type: ignore[arg-type]
        validator=SQLValidator(),
        executor=executor,  # type: ignore[arg-type]
        repairer=repairer,  # type: ignore[arg-type]
        max_repair_attempts=attempts,
    )


def test_pipeline_success_without_repair() -> None:
    query_result = QueryResult(("customer_id",), (("abc",),), 1, 3.5)
    executor = FakeExecutor([query_result])
    repairer = FakeRepairer("SELECT c.customer_id FROM customers AS c")

    result = build_pipeline(executor, repairer).ask("List customers")

    assert result.error is None
    assert result.rows == (("abc",),)
    assert not result.was_repaired
    assert repairer.calls == 0
    assert result.generation_time_ms is not None
    assert result.validation_time_ms is not None
    assert result.total_time_ms is not None


def test_pipeline_repairs_database_error_once() -> None:
    query_result = QueryResult(("customer_id",), (("abc",),), 1, 4.0)
    executor = FakeExecutor([SQLExecutionError("bad column"), query_result])
    repairer = FakeRepairer("SELECT c.customer_id FROM customers AS c")

    result = build_pipeline(executor, repairer).ask("List customers")

    assert result.error is None
    assert result.was_repaired
    assert repairer.calls == 1
    assert executor.calls == 2
    assert result.repair_time_ms is not None


def test_pipeline_does_not_retry_after_repaired_query_fails() -> None:
    executor = FakeExecutor(
        [SQLExecutionError("first failure"), SQLExecutionError("second failure")]
    )
    repairer = FakeRepairer("SELECT c.customer_id FROM customers AS c")

    result = build_pipeline(executor, repairer).ask("List customers")

    assert result.error == "Repaired SQL execution failed: second failure"
    assert repairer.calls == 1
    assert executor.calls == 2


def test_unvalidated_repair_is_never_executed() -> None:
    executor = FakeExecutor([SQLExecutionError("first failure")])
    repairer = FakeRepairer("DROP TABLE customers")

    result = build_pipeline(executor, repairer).ask("List customers")

    assert result.error is not None
    assert "validation failed" in result.error.lower()
    assert executor.calls == 1
    assert repairer.calls == 1
