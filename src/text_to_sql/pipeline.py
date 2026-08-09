"""End-to-end schema-aware Text-to-SQL orchestration."""

from __future__ import annotations

import logging
import time
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
    generation_time_ms: float | None = None
    validation_time_ms: float | None = None
    repair_time_ms: float | None = None
    total_time_ms: float | None = None

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
        started = time.perf_counter()
        timings = _StageTimings()
        question = question.strip()
        LOGGER.info("Question received: length=%d", len(question))

        def failure(**kwargs: Any) -> PipelineResult:
            return self._failure(
                question,
                timings=timings,
                total_time_ms=_elapsed_ms(started),
                **kwargs,
            )

        if not question:
            return failure(error="Question must not be empty")

        try:
            schema = self._schema_manager.get_snapshot()
            generation_started = time.perf_counter()
            generated_sql = self._generator.generate(question)
        except Exception as exc:
            if "generation_started" in locals():
                timings.generation_time_ms = _elapsed_ms(generation_started)
            LOGGER.error("SQL generation failed: %s", exc)
            return failure(error=f"SQL generation failed: {exc}")
        timings.generation_time_ms = _elapsed_ms(generation_started)

        validation_started = time.perf_counter()
        validation = self._validator.validate(generated_sql, schema)
        timings.validation_time_ms = _elapsed_ms(validation_started)
        LOGGER.info(
            "SQL validation: %s (%s) duration_ms=%.2f",
            "PASSED" if validation.valid else "FAILED",
            validation.reason,
            timings.validation_time_ms,
        )
        if not validation.valid:
            return failure(
                generated_sql=generated_sql,
                final_sql=generated_sql,
                validation_passed=False,
                validation_reason=validation.reason,
                error=f"SQL validation failed: {validation.reason}",
            )

        try:
            result = self._executor.execute(generated_sql, validation)
            LOGGER.info("Query executed in %.2f ms", result.execution_time_ms)
            return self._success(
                question,
                generated_sql,
                generated_sql,
                False,
                validation.reason,
                result,
                timings,
                _elapsed_ms(started),
            )
        except SQLExecutionError as first_error:
            LOGGER.warning("Generated SQL execution failed: %s", first_error)
            if self._max_repair_attempts == 0:
                return failure(
                    generated_sql=generated_sql,
                    final_sql=generated_sql,
                    validation_passed=True,
                    validation_reason=validation.reason,
                    error=f"SQL execution failed: {first_error}",
                )

            try:
                repair_started = time.perf_counter()
                repaired_sql = self._repairer.repair(
                    question=question,
                    original_sql=generated_sql,
                    database_error=str(first_error),
                    schema_context=schema.to_prompt(),
                )
            except Exception as repair_error:
                timings.repair_time_ms = _elapsed_ms(repair_started)
                LOGGER.error("SQL repair generation failed: %s", repair_error)
                return failure(
                    generated_sql=generated_sql,
                    final_sql=generated_sql,
                    validation_passed=True,
                    validation_reason=validation.reason,
                    was_repaired=True,
                    error=f"SQL repair failed: {repair_error}",
                )
            timings.repair_time_ms = _elapsed_ms(repair_started)

            repaired_validation_started = time.perf_counter()
            repaired_validation = self._validator.validate(repaired_sql, schema)
            repaired_validation_ms = _elapsed_ms(repaired_validation_started)
            timings.validation_time_ms = (
                (timings.validation_time_ms or 0.0) + repaired_validation_ms
            )
            LOGGER.info(
                "Repaired SQL validation: %s (%s) duration_ms=%.2f",
                "PASSED" if repaired_validation.valid else "FAILED",
                repaired_validation.reason,
                repaired_validation_ms,
            )
            if not repaired_validation.valid:
                return failure(
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
                    timings,
                    _elapsed_ms(started),
                )
            except SQLExecutionError as second_error:
                LOGGER.error("Repaired SQL execution failed: %s", second_error)
                return failure(
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
        timings: "_StageTimings | None" = None,
        total_time_ms: float | None = None,
    ) -> PipelineResult:
        timings = timings or _StageTimings()
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
            generation_time_ms=timings.generation_time_ms,
            validation_time_ms=timings.validation_time_ms,
            repair_time_ms=timings.repair_time_ms,
            total_time_ms=total_time_ms,
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
        timings: "_StageTimings | None" = None,
        total_time_ms: float | None = None,
    ) -> PipelineResult:
        timings = timings or _StageTimings()
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
            generation_time_ms=timings.generation_time_ms,
            validation_time_ms=timings.validation_time_ms,
            repair_time_ms=timings.repair_time_ms,
            total_time_ms=total_time_ms,
        )


@dataclass
class _StageTimings:
    generation_time_ms: float | None = None
    validation_time_ms: float | None = None
    repair_time_ms: float | None = None


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1_000
