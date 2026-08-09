import pytest

from src.analytics.chart_selector import ChartSelector
from src.analytics.insight_generator import InsightConfig, InsightGenerator
from src.analytics.result_analyzer import ResultAnalyzer
from src.text_to_sql.pipeline import PipelineResult


class FakeGemini:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Category A leads the returned result with revenue of 30."


def result() -> PipelineResult:
    return PipelineResult(
        question="Which categories lead revenue?",
        generated_sql="SELECT category, revenue FROM results",
        final_sql="SELECT category, revenue FROM results",
        validation_passed=True,
        validation_reason="safe",
        was_repaired=False,
        columns=("category", "revenue"),
        rows=(("Category A", 30), ("Category B", 20), ("do-not-include", 10)),
        row_count=3,
        execution_time_ms=1.0,
        truncated=False,
        error=None,
    )


def test_insight_prompt_is_grounded_and_row_limited() -> None:
    fake = FakeGemini()
    query = result()
    analysis = ResultAnalyzer().analyze(query.columns, query.rows, query.final_sql)
    chart = ChartSelector().select(analysis)
    generator = InsightGenerator(fake, InsightConfig(max_rows=2))

    insight = generator.generate(query.question, query.final_sql or "", query, analysis, chart)

    assert insight.startswith("Category A")
    prompt = fake.prompts[0]
    assert "use ONLY the QUERY RESULT DATA" in prompt
    assert "Do not make causal claims" in prompt
    assert "Category A" in prompt and "Category B" in prompt
    assert "do-not-include" not in prompt
    assert '"result_was_context_limited": true' in prompt


def test_insight_max_rows_must_be_positive() -> None:
    fake = FakeGemini()
    try:
        InsightGenerator(fake, InsightConfig(max_rows=0))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a positive max_rows validation error")


def test_incomplete_gemini_insight_is_rejected() -> None:
    fake = FakeGemini()
    fake.generate = lambda prompt: "Category A leads but this response was cut"
    query = result()
    analysis = ResultAnalyzer().analyze(query.columns, query.rows, query.final_sql)

    with pytest.raises(RuntimeError, match="incomplete business insight"):
        InsightGenerator(fake).generate(
            query.question,
            query.final_sql or "",
            query,
            analysis,
            ChartSelector().select(analysis),
        )
