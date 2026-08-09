"""Streamlit interface for the Enterprise AI SQL Analytics Copilot."""

from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import Any

import pandas as pd
import streamlit as st

from src.analytics.analytics_pipeline import AnalyticsPipeline, AnalyticsResult
from src.analytics.chart_selector import ChartSelector
from src.analytics.insight_generator import InsightGenerator
from src.analytics.result_analyzer import ResultAnalyzer
from src.analytics.visualization import VisualizationEngine
from src.text_to_sql.llm_client import create_llm_client
from src.text_to_sql.pipeline import TextToSQLPipeline


LOGGER = logging.getLogger(__name__)
SAMPLE_QUESTIONS = (
    "What is the total revenue?",
    "Show the monthly revenue trend.",
    "Which 10 product categories generated the most revenue?",
    "Which states generated the most orders?",
    "Who are the top 10 sellers by revenue?",
    "Which categories have the highest average review score?",
    "What is the average freight cost by state?",
    "Are delayed deliveries associated with lower review scores?",
)


st.set_page_config(
    page_title="Enterprise AI SQL Analytics Copilot",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
    [data-testid="stAppViewContainer"] {background: #f6f8fb;}
    .hero {
        padding: 2rem 2.25rem; border-radius: 18px;
        background: linear-gradient(125deg, #10243e 0%, #173f5f 55%, #16697a 100%);
        color: white; margin-bottom: 1.4rem; box-shadow: 0 14px 35px rgba(16,36,62,.16);
    }
    .hero h1 {font-size: 2.25rem; margin: 0 0 .45rem 0; letter-spacing: -.02em;}
    .hero p {font-size: 1rem; color: #d9e8f1; margin: 0; max-width: 760px;}
    .section-label {font-size: .78rem; font-weight: 700; letter-spacing: .08em;
        color: #557086; text-transform: uppercase; margin: 1.5rem 0 .5rem;}
    div[data-testid="stMetric"] {background: white; border: 1px solid #e3e9ef;
        padding: 1.1rem 1.3rem; border-radius: 14px; box-shadow: 0 4px 16px rgba(20,45,70,.05);}
    div[data-testid="stDataFrame"] {border: 1px solid #e3e9ef; border-radius: 12px; overflow: hidden;}
    .stButton > button {border-radius: 10px; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def build_analytics_pipeline() -> AnalyticsPipeline:
    """Construct one shared Gemini client and all stateless Phase 2/3 services."""
    llm_client = create_llm_client()
    text_to_sql = TextToSQLPipeline.from_env(llm_client=llm_client)
    max_points = _positive_env_int("CHART_MAX_POINTS", 100)
    return AnalyticsPipeline(
        text_to_sql=text_to_sql,
        analyzer=ResultAnalyzer(),
        chart_selector=ChartSelector(),
        visualization=VisualizationEngine(max_points=max_points),
        insight_generator=InsightGenerator(llm_client),
    )


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def render_result(result: AnalyticsResult) -> None:
    query = result.query
    if query.error:
        st.error(_friendly_query_error(query.error))
        if query.generated_sql:
            with st.expander("View Generated SQL"):
                st.code(query.final_sql or query.generated_sql, language="sql")
        return

    st.markdown('<div class="section-label">AI Business Answer</div>', unsafe_allow_html=True)
    if result.insight:
        st.success(result.insight, icon="💡")
    else:
        st.warning(
            "The query succeeded, but Gemini could not generate a business explanation. "
            "The verified results are still shown below."
        )

    if result.chart and result.chart.chart_type == "kpi":
        metric = result.chart.y
        if metric and query.rows:
            index = query.columns.index(metric)
            st.markdown('<div class="section-label">Business Metric</div>', unsafe_allow_html=True)
            st.metric(result.chart.title, _format_metric(metric, query.rows[0][index]))
    elif result.figure is not None:
        st.markdown('<div class="section-label">Visualization</div>', unsafe_allow_html=True)
        st.plotly_chart(result.figure, width="stretch", config={"displaylogo": False})

    st.markdown('<div class="section-label">Query Results</div>', unsafe_allow_html=True)
    if not query.rows:
        st.info("The query completed successfully but returned no rows.")
    else:
        st.dataframe(_display_frame(query), width="stretch", hide_index=True)
        if query.truncated:
            st.caption("Results were limited by SQL_MAX_ROWS for safe display.")

    with st.expander("View Generated SQL"):
        st.code(query.final_sql or query.generated_sql or "", language="sql")
        if query.was_repaired:
            st.caption("Gemini repaired the initial SQL once; the validated repaired query is shown.")

    st.markdown('<div class="section-label">Execution Details</div>', unsafe_allow_html=True)
    metadata_columns = st.columns(4)
    metadata_columns[0].metric("Rows", f"{query.row_count:,}")
    metadata_columns[1].metric(
        "Execution",
        f"{query.execution_time_ms:.1f} ms" if query.execution_time_ms is not None else "—",
    )
    metadata_columns[2].metric("SQL Validation", "Passed" if query.validation_passed else "Failed")
    metadata_columns[3].metric("SQL Repair", "Used" if query.was_repaired else "Not needed")


def _display_frame(result: Any) -> pd.DataFrame:
    frame = pd.DataFrame(result.rows, columns=result.columns)
    return frame.map(lambda value: float(value) if isinstance(value, Decimal) else value)


def _format_metric(name: str, value: Any) -> str:
    if value is None:
        return "No value"
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        number = float(value)
        lowered = name.lower()
        if any(term in lowered for term in ("revenue", "price", "freight", "payment", "cost", "value")):
            return f"R$ {_compact_number(number)}"
        if "percentage" in lowered or "rate" in lowered or lowered.endswith("_pct"):
            return f"{number:,.2f}%"
        if "count" in lowered or lowered.startswith("total_"):
            return f"{number:,.0f}"
        return f"{number:,.2f}"
    return str(value)


def _compact_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:,.2f}K"
    return f"{value:,.2f}"


def _friendly_query_error(error: str) -> str:
    lowered = error.lower()
    if "generation failed" in lowered:
        return "Gemini could not generate SQL. Check the API key, model access, and free-tier quota, then retry."
    if "validation failed" in lowered:
        return "The generated SQL was blocked by the safety validator. Try rephrasing the question."
    if "execution failed" in lowered:
        return "PostgreSQL could not execute the validated query. Check the database connection or retry."
    if "question must not be empty" in lowered:
        return "Enter a business question before running the analysis."
    return "The analysis could not be completed. Check the application logs for technical details."


st.markdown(
    """
    <div class="hero">
      <h1>Enterprise AI SQL Analytics Copilot</h1>
      <p>Ask a business question in plain language. Gemini generates safe PostgreSQL,
      the database returns verified results, and the copilot builds a grounded insight and visualization.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Try a sample question</div>', unsafe_allow_html=True)
sample_columns = st.columns(4)
sample_clicked: str | None = None
for index, sample in enumerate(SAMPLE_QUESTIONS):
    short_label = sample.rstrip(".?")
    if len(short_label) > 38:
        short_label = short_label[:37] + "…"
    if sample_columns[index % 4].button(short_label, key=f"sample_{index}", width="stretch"):
        sample_clicked = sample

if sample_clicked:
    st.session_state["question_input"] = sample_clicked

question = st.text_input(
    "Ask your business data",
    key="question_input",
    placeholder="e.g. Which 10 product categories generated the most revenue?",
)
analyze_clicked = st.button("Analyze", type="primary", width="stretch")

if analyze_clicked or sample_clicked:
    submitted_question = (sample_clicked or question).strip()
    if not submitted_question:
        st.warning("Enter a business question before running the analysis.")
    else:
        try:
            with st.spinner("Generating safe SQL, querying PostgreSQL, and analyzing the result…"):
                st.session_state["analytics_result"] = build_analytics_pipeline().analyze(
                    submitted_question
                )
        except Exception:
            LOGGER.exception("Analytics application configuration or execution failed")
            st.error(
                "The analytics service could not start. Verify Gemini and PostgreSQL settings "
                "in .env, then check the application logs."
            )

if "analytics_result" in st.session_state:
    render_result(st.session_state["analytics_result"])

st.divider()
st.caption(
    "Read-only analytics · PostgreSQL safety validation · Result-limited Gemini insights · No data is fabricated"
)
