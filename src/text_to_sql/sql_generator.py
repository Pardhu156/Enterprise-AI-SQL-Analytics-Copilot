"""Schema-aware SQL generation and model-response extraction."""

from __future__ import annotations

import logging
import re

from .llm_client import LLMClient
from .prompt_builder import build_generation_prompt
from .schema_manager import SchemaManager


LOGGER = logging.getLogger(__name__)
FENCED_SQL = re.compile(r"```(?:sql|postgresql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class SQLGenerationError(ValueError):
    """Raised when a model response does not contain usable SQL."""


def extract_sql(model_response: str) -> str:
    """Extract a single SQL-shaped response without interpreting or executing it."""
    if not model_response or not model_response.strip():
        raise SQLGenerationError("The LLM returned an empty response")

    text = model_response.strip()
    fenced = FENCED_SQL.findall(text)
    if fenced:
        text = fenced[0].strip()
    else:
        # Models occasionally prefix an otherwise valid query with "SQL:".
        text = re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE)

    match = re.search(r"\b(SELECT|WITH)\b", text, flags=re.IGNORECASE)
    if not match:
        raise SQLGenerationError("The LLM response did not contain a SELECT or WITH query")
    text = text[match.start() :].strip()
    if not text:
        raise SQLGenerationError("The extracted SQL is empty")
    return text


class SQLGenerator:
    def __init__(self, schema_manager: SchemaManager, llm_client: LLMClient) -> None:
        self._schema_manager = schema_manager
        self._llm_client = llm_client

    def generate(self, question: str) -> str:
        schema_context = self._schema_manager.get_schema_context()
        prompt = build_generation_prompt(question, schema_context)
        response = self._llm_client.generate(prompt)
        generated_sql = extract_sql(response)
        LOGGER.info("SQL generated for question")
        LOGGER.debug("Generated SQL: %s", generated_sql)
        return generated_sql
