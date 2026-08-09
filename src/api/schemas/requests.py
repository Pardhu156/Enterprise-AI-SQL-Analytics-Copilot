"""Validated request models for the public API."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000),
]


class AnalyticsQueryRequest(BaseModel):
    question: Question = Field(
        description="Natural-language business question to answer from the analytics database."
    )
    include_sql: bool = Field(default=True, description="Include generated and executed SQL.")
    include_rows: bool = Field(default=True, description="Include returned query rows.")
    include_visualization_config: bool = Field(
        default=True,
        description="Include the deterministic visualization specification.",
    )

