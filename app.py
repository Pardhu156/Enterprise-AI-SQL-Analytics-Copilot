"""Streamlit interface for the Enterprise AI SQL Analytics Copilot."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import pandas as pd
import streamlit as st

from src.analytics.chart_selector import ChartConfig
from src.analytics.visualization import VisualizationEngine
from src.api.schemas.responses import AnalyticsQueryResponse
from src.frontend.api_client import AnalyticsAPIClient, FrontendAPIError


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
def build_api_client() -> AnalyticsAPIClient:
    """Create one persistent HTTP client for the FastAPI backend."""
    return AnalyticsAPIClient()


def render_result(response: AnalyticsQueryResponse) -> None:
    st.markdown('<div class="section-label">AI Business Answer</div>', unsafe_allow_html=True)
    if response.answer:
        st.success(response.answer, icon="💡")
    else:
        st.warning(
            "The query succeeded, but Gemini could not generate a business explanation. "
            "The verified results are still shown below."
        )

    visualization = response.visualization
    rows = response.result.rows or []
    if visualization and visualization.chart_type == "kpi":
        metric = visualization.y
        if metric and rows:
            index = response.result.columns.index(metric)
            st.markdown('<div class="section-label">Business Metric</div>', unsafe_allow_html=True)
            st.metric(visualization.title, _format_metric(metric, rows[0][index]))
    elif visualization:
        chart = ChartConfig(**visualization.model_dump())
        figure = VisualizationEngine().create_from_data(
            response.result.columns,
            rows,
            chart,
        )
        if figure is not None:
            st.markdown('<div class="section-label">Visualization</div>', unsafe_allow_html=True)
            st.plotly_chart(figure, width="stretch", config={"displaylogo": False})

    st.markdown('<div class="section-label">Query Results</div>', unsafe_allow_html=True)
    if not rows:
        st.info("The query completed successfully but returned no rows.")
    else:
        st.dataframe(
            _display_frame(response.result.columns, rows),
            width="stretch",
            hide_index=True,
        )
        if response.result.truncated:
            st.caption("Results were limited by SQL_MAX_ROWS for safe display.")

    if response.sql:
        with st.expander("View Generated SQL"):
            st.code(response.sql.final_sql or response.sql.generated_sql or "", language="sql")
            if response.sql.was_repaired:
                st.caption(
                    "Gemini repaired the initial SQL once; the validated repaired query is shown."
                )

    st.markdown('<div class="section-label">Execution Details</div>', unsafe_allow_html=True)
    metadata_columns = st.columns(5)
    metadata_columns[0].metric("Rows", f"{response.result.row_count:,}")
    metadata_columns[1].metric(
        "SQL execution",
        (
            f"{response.execution.sql_execution_time_ms:.1f} ms"
            if response.execution.sql_execution_time_ms is not None
            else "—"
        ),
    )
    metadata_columns[2].metric(
        "Total request",
        f"{response.execution.total_request_time_ms:.1f} ms",
    )
    metadata_columns[3].metric(
        "SQL validation",
        "Passed" if response.sql and response.sql.validation_passed else "Not included",
    )
    metadata_columns[4].metric(
        "SQL repair",
        "Used" if response.sql and response.sql.was_repaired else "Not needed",
    )
    st.caption(f"Request ID: {response.request_id}")


def _display_frame(columns: list[str], rows: list[list[Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
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
                st.session_state["api_analytics_result"] = build_api_client().query(
                    submitted_question
                )
        except FrontendAPIError as exc:
            LOGGER.warning("Frontend API request failed: code=%s", exc.code)
            st.error(exc.message)
            if exc.request_id:
                st.caption(f"Request ID: {exc.request_id}")
        except Exception:
            LOGGER.exception("Frontend configuration failed")
            st.error("The frontend could not connect to the analytics API. Check .env and logs.")

if "api_analytics_result" in st.session_state:
    render_result(st.session_state["api_analytics_result"])

st.divider()
st.caption(
    "Read-only analytics · PostgreSQL safety validation · Result-limited Gemini insights · No data is fabricated"
)
