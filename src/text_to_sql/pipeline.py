"""End-to-end schema-aware Text-to-SQL orchestration."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from .llm_client import LLMClient, create_llm_client
from .schema_manager import SchemaManager
from .sql_executor import QueryResult, SQLExecutionError, SQLExecutor
from .sql_generator import SQLGenerator
from .sql_repair import SQLRepairer
from .sql_validator import SQLValidator


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    question: str
    generated_sql: str | None
    final_sql: str | None
    validation_passed: bool
    validation_reason: str | None
    was_repaired: bool
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    execution_time_ms: float | None
    truncated: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextToSQLPipeline:
    def __init__(
        self,
        schema_manager: SchemaManager,
        generator: SQLGenerator,
        validator: SQLValidator,
        executor: SQLExecutor,
        repairer: SQLRepairer,
        max_repair_attempts: int = 1,
    ) -> None:
        if max_repair_attempts not in (0, 1):
            raise ValueError("Phase 2 supports zero or one SQL repair attempt")
        self._schema_manager = schema_manager
        self._generator = generator
        self._validator = validator
        self._executor = executor
        self._repairer = repairer
        self._max_repair_attempts = max_repair_attempts

    @classmethod
    def from_env(
        cls,
        max_repair_attempts: int = 1,
        llm_client: LLMClient | None = None,
    ) -> "TextToSQLPipeline":
        schema_manager = SchemaManager()
        resolved_llm_client = llm_client or create_llm_client()
        return cls(
            schema_manager=schema_manager,
            generator=SQLGenerator(schema_manager, resolved_llm_client),
            validator=SQLValidator(),
            executor=SQLExecutor(),
            repairer=SQLRepairer(resolved_llm_client),
            max_repair_attempts=max_repair_attempts,
        )

    def ask(self, question: str) -> PipelineResult:
        question = question.strip()
        LOGGER.info("Question received: %s", question)
        if not question:
            return self._failure(question, error="Question must not be empty")

        try:
            schema = self._schema_manager.get_snapshot()
            generated_sql = self._generator.generate(question)
        except Exception as exc:
            LOGGER.error("SQL generation failed: %s", exc)
            return self._failure(question, error=f"SQL generation failed: {exc}")

        validation = self._validator.validate(generated_sql, schema)
        LOGGER.info("SQL validation: %s (%s)", "PASSED" if validation.valid else "FAILED", validation.reason)
        if not validation.valid:
            return self._failure(
                question,
                generated_sql=generated_sql,
                final_sql=generated_sql,
                validation_passed=False,
                validation_reason=validation.reason,
                error=f"SQL validation failed: {validation.reason}",
            )

        try:
            result = self._executor.execute(generated_sql, validation)
            LOGGER.info("Query executed in %.2f ms", result.execution_time_ms)
            return self._success(question, generated_sql, generated_sql, False, validation.reason, result)
        except SQLExecutionError as first_error:
            LOGGER.warning("Generated SQL execution failed: %s", first_error)
            if self._max_repair_attempts == 0:
                return self._failure(
                    question,
                    generated_sql=generated_sql,
                    final_sql=generated_sql,
                    validation_passed=True,
                    validation_reason=validation.reason,
                    error=f"SQL execution failed: {first_error}",
                )

            try:
                repaired_sql = self._repairer.repair(
                    question=question,
                    original_sql=generated_sql,
                    database_error=str(first_error),
                    schema_context=schema.to_prompt(),
                )
            except Exception as repair_error:
                LOGGER.error("SQL repair generation failed: %s", repair_error)
                return self._failure(
                    question,
                    generated_sql=generated_sql,
                    final_sql=generated_sql,
                    validation_passed=True,
                    validation_reason=validation.reason,
                    was_repaired=True,
                    error=f"SQL repair failed: {repair_error}",
                )

            repaired_validation = self._validator.validate(repaired_sql, schema)
            LOGGER.info(
                "Repaired SQL validation: %s (%s)",
                "PASSED" if repaired_validation.valid else "FAILED",
                repaired_validation.reason,
            )
            if not repaired_validation.valid:
                return self._failure(
                    question,
                    generated_sql=generated_sql,
                    final_sql=repaired_sql,
                    validation_passed=False,
                    validation_reason=repaired_validation.reason,
                    was_repaired=True,
                    error=f"Repaired SQL validation failed: {repaired_validation.reason}",
                )

            try:
                result = self._executor.execute(repaired_sql, repaired_validation)
                LOGGER.info("Repaired query executed in %.2f ms", result.execution_time_ms)
                return self._success(
                    question,
                    generated_sql,
                    repaired_sql,
                    True,
                    repaired_validation.reason,
                    result,
                )
            except SQLExecutionError as second_error:
                LOGGER.error("Repaired SQL execution failed: %s", second_error)
                return self._failure(
                    question,
                    generated_sql=generated_sql,
                    final_sql=repaired_sql,
                    validation_passed=True,
                    validation_reason=repaired_validation.reason,
                    was_repaired=True,
                    error=f"Repaired SQL execution failed: {second_error}",
                )

    @staticmethod
    def _success(
        question: str,
        generated_sql: str,
        final_sql: str,
        was_repaired: bool,
        validation_reason: str,
        result: QueryResult,
    ) -> PipelineResult:
        return PipelineResult(
            question=question,
            generated_sql=generated_sql,
            final_sql=final_sql,
            validation_passed=True,
            validation_reason=validation_reason,
            was_repaired=was_repaired,
            columns=result.columns,
            rows=result.rows,
            row_count=result.row_count,
            execution_time_ms=result.execution_time_ms,
            truncated=result.truncated,
            error=None,
        )

    @staticmethod
    def _failure(
        question: str,
        generated_sql: str | None = None,
        final_sql: str | None = None,
        validation_passed: bool = False,
        validation_reason: str | None = None,
        was_repaired: bool = False,
        error: str | None = None,
    ) -> PipelineResult:
        return PipelineResult(
            question=question,
            generated_sql=generated_sql,
            final_sql=final_sql,
            validation_passed=validation_passed,
            validation_reason=validation_reason,
            was_repaired=was_repaired,
            columns=(),
            rows=(),
            row_count=0,
            execution_time_ms=None,
            truncated=False,
            error=error,
        )
