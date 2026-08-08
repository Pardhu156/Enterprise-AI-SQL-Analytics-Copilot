"""One-shot LLM repair of PostgreSQL execution errors."""

from __future__ import annotations

import logging

from .llm_client import LLMClient
from .prompt_builder import build_repair_prompt
from .sql_generator import extract_sql


LOGGER = logging.getLogger(__name__)


class SQLRepairer:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def repair(
        self,
        question: str,
        original_sql: str,
        database_error: str,
        schema_context: str,
    ) -> str:
        LOGGER.info("Attempting one SQL repair")
        prompt = build_repair_prompt(
            question=question,
            original_sql=original_sql,
            database_error=database_error,
            schema_context=schema_context,
        )
        return extract_sql(self._llm_client.generate(prompt))
