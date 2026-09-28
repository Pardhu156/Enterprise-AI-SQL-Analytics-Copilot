"""Streamlit interface for the Enterprise AI SQL & Predictive Analytics Copilot."""

from __future__ import annotations

import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import pandas as pd
import streamlit as st

from src.analytics.chart_selector import ChartConfig
from src.analytics.visualization import VisualizationEngine
from src.api.schemas.responses import (
    AnalyticsQueryResponse,
    BusinessImpactMetric,
    BusinessOverviewResponse,
    BusinessRecommendationDetails,
    CopilotQueryResponse,
    MLPredictionResponse,
    StatisticalAnalysisResponse,
)
from src.frontend.api_client import AnalyticsAPIClient, FrontendAPIError


LOGGER = logging.getLogger(__name__)
SAMPLE_QUESTIONS = (
    "What is total revenue?",
    "Show the monthly revenue trend.",
    "Which 10 product categories generated the most revenue?",
    "Who are the top 10 sellers by revenue?",
    "Which states generated the most orders?",
    "What is the average order value?",
    "Which categories have the highest average review score?",
    "Are delayed deliveries associated with lower review scores?",
    "Predict revenue for the next 4 weeks.",
    "Show the customer segments.",
    "Compare total historical revenue with the next 4 weeks forecast.",
    "What is the 95% confidence interval for average order value?",
    "Is the average review score significantly different between delayed and on-time deliveries?",
    "If conversion improves from 8% to 10%, how many users do we need per group?",
    "Why is revenue expected to change over the next 4 weeks?",
    "Which customer segment generates the most value and what should we consider doing?",
)


st.set_page_config(
    page_title="Enterprise AI SQL & Predictive Analytics Copilot",
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


def render_result(response: AnalyticsQueryResponse, show_answer: bool = True) -> None:
    if show_answer:
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
    stage_times = (
        ("SQL generation", response.execution.sql_generation_time_ms),
        ("SQL validation", response.execution.sql_validation_time_ms),
        ("Gemini insight", response.execution.insight_generation_time_ms),
    )
    rendered_stage_times = " · ".join(
        f"{label}: {value:.1f} ms" if value is not None else f"{label}: —"
        for label, value in stage_times
    )
    st.caption(rendered_stage_times)
    st.caption(f"Request ID: {response.request_id}")


def render_copilot_result(response: CopilotQueryResponse) -> None:
    route_label = response.route.intent.upper()
    st.caption(f"Response type: {route_label} · Tasks: {', '.join(response.route.tasks)}")
    st.success(response.answer, icon=":material/analytics:")
    st.caption(f"Unified request ID: {response.request_id}")


def render_prediction(prediction: MLPredictionResponse) -> None:
    st.markdown("### Predictive analytics")
    st.caption(
        f"Task: {prediction.task} · Model: {prediction.model} · "
        f"Trained: {prediction.trained_at_utc}"
    )
    if prediction.task == "sales_forecasting":
        forecast = pd.DataFrame(prediction.data["forecast"])
        history = pd.DataFrame(prediction.data["recent_history"])
        history["week_start"] = pd.to_datetime(history["week_start"])
        forecast["week_start"] = pd.to_datetime(forecast["week_start"])
        chart = history.rename(columns={"revenue": "Historical revenue"}).merge(
            forecast.rename(columns={"predicted_revenue": "Forecast revenue"}),
            on="week_start",
            how="outer",
        )
        with st.container(horizontal=True):
            st.metric(
                "Forecast revenue",
                f"R$ {float(prediction.data['total_predicted_revenue']):,.2f}",
                border=True,
            )
            st.metric(
                "Prediction horizon",
                f"{int(prediction.data['horizon_weeks'])} weeks",
                border=True,
            )
        st.line_chart(
            chart,
            x="week_start",
            y=["Historical revenue", "Forecast revenue"],
            x_label="Week",
            y_label="Revenue (R$)",
        )
        st.dataframe(forecast, hide_index=True)
    elif prediction.task == "customer_segmentation":
        if "profiles" in prediction.data:
            profiles = pd.DataFrame(prediction.data["profiles"])
            st.bar_chart(profiles, x="segment", y="customers", x_label="Segment")
            st.dataframe(
                profiles,
                hide_index=True,
                column_config={
                    "monetary": st.column_config.NumberColumn(format="R$ %.2f"),
                    "average_order_value": st.column_config.NumberColumn(format="R$ %.2f"),
                },
            )
        else:
            with st.container(horizontal=True):
                st.metric("Segment", str(prediction.data["segment"]), border=True)
                st.metric("Cluster", str(prediction.data["cluster"]), border=True)
            st.json(prediction.data["rfm"])
    elif prediction.task == "late_delivery_prediction":
        probability = float(prediction.data["late_delivery_probability"])
        with st.container(horizontal=True):
            st.metric("Late-delivery probability", f"{probability:.1%}", border=True)
            st.metric("Risk band", str(prediction.data["risk_band"]).title(), border=True)
    with st.expander("Model metrics and limitations"):
        if prediction.metrics:
            st.json(prediction.metrics)
        for limitation in prediction.limitations:
            st.caption(f"• {limitation}")


def render_statistical_analysis(analysis: StatisticalAnalysisResponse) -> None:
    st.markdown("### Statistical analysis")
    st.caption(f"Task: {analysis.task} · Source: {analysis.source}")
    st.write(analysis.description)
    result = analysis.result
    if analysis.task == "hypothesis_test":
        effect = result.get("effect_size") or {}
        interval = result.get("confidence_interval") or {}
        with st.container(horizontal=True):
            st.metric("Test", str(result.get("test_used", "—")), border=True)
            st.metric("p-value", _format_probability(result.get("p_value")), border=True)
            st.metric(
                "Decision",
                "Significant" if result.get("significant") else "Not significant",
                border=True,
            )
            st.metric(
                str(effect.get("name", "Effect size")).replace("_", " ").title(),
                _format_number(effect.get("value")),
                border=True,
            )
        st.info(str(result.get("interpretation", "")))
        if interval:
            st.caption(
                f"{float(interval['confidence_level']):.0%} CI: "
                f"[{float(interval['lower']):.4f}, {float(interval['upper']):.4f}]"
            )
        summaries = result.get("group_summaries")
        if summaries:
            st.dataframe(pd.DataFrame(summaries).T.reset_index(names="Group"), hide_index=True)
    elif analysis.task == "confidence_interval":
        with st.container(horizontal=True):
            st.metric("Point estimate", _format_number(result.get("point_estimate")), border=True)
            st.metric("Lower bound", _format_number(result.get("lower")), border=True)
            st.metric("Upper bound", _format_number(result.get("upper")), border=True)
            st.metric(
                "Confidence",
                f"{float(result.get('confidence_level', 0)):.0%}",
                border=True,
            )
        st.info(str(result.get("interpretation", "")))
    elif analysis.task == "ab_test":
        test_result = result.get("test_result") or {}
        effect = test_result.get("effect_size") or {}
        interval = test_result.get("confidence_interval") or {}
        with st.container(horizontal=True):
            st.metric("Control", _format_number(result.get("control_metric")), border=True)
            st.metric("Treatment", _format_number(result.get("treatment_metric")), border=True)
            st.metric("Absolute lift", _format_number(result.get("absolute_lift")), border=True)
            relative = result.get("relative_lift")
            st.metric(
                "Relative lift",
                f"{float(relative):.2%}" if relative is not None else "—",
                border=True,
            )
        with st.container(horizontal=True):
            st.metric("Test", str(test_result.get("test_used", "—")), border=True)
            st.metric("p-value", _format_probability(test_result.get("p_value")), border=True)
            st.metric(
                "Decision",
                "Significant" if result.get("statistically_significant") else "Not significant",
                border=True,
            )
            st.metric(
                str(effect.get("name", "Effect size")).replace("_", " ").title(),
                _format_number(effect.get("value")),
                border=True,
            )
        if interval:
            st.caption(
                f"{float(interval['confidence_level']):.0%} CI for treatment minus control: "
                f"[{float(interval['lower']):.4f}, {float(interval['upper']):.4f}]"
            )
        chart = pd.DataFrame(
            {
                "Group": ["Control", "Treatment"],
                "Metric": [result.get("control_metric"), result.get("treatment_metric")],
            }
        )
        st.bar_chart(chart, x="Group", y="Metric")
        if result.get("synthetic_demo"):
            st.warning("Synthetic demonstration only — this is not Olist business evidence.")
        st.info(str(result.get("recommendation", "")))
    elif analysis.task == "sample_size_estimation":
        with st.container(horizontal=True):
            st.metric(
                "Required per group",
                f"{int(result['required_sample_size_per_group']):,}",
                border=True,
            )
            st.metric(
                "Total required",
                f"{int(result['total_required_sample_size']):,}",
                border=True,
            )
        st.caption(str(result.get("method", "")))
        st.json(result.get("assumptions", {}))
    if analysis.limitations:
        with st.expander("Assumptions and limitations"):
            for limitation in analysis.limitations:
                st.caption(f"• {limitation}")


def render_model_explanation(prediction: MLPredictionResponse) -> None:
    explanation = prediction.explanation
    if explanation is None:
        st.info(f"{prediction.task.replace('_', ' ').title()} has no SHAP explanation.")
        return
    st.subheader(prediction.task.replace("_", " ").title())
    st.caption(f"Method: {explanation.method} · Output: {explanation.output_space}")
    contributions = pd.DataFrame(
        [item.model_dump() for item in explanation.top_contributions]
    )
    if not contributions.empty:
        contributions = contributions.sort_values("contribution")
        st.bar_chart(
            contributions,
            x="feature",
            y="contribution",
            color="direction",
            horizontal=True,
            x_label="Model contribution",
            y_label="Feature",
        )
        st.dataframe(
            contributions,
            hide_index=True,
            column_config={
                "feature_value": st.column_config.NumberColumn(format="%.4f"),
                "contribution": st.column_config.NumberColumn(format="%.4f"),
            },
        )
    importance = pd.DataFrame(
        [item.model_dump() for item in explanation.global_importance]
    )
    if not importance.empty:
        with st.expander("Global model importance"):
            st.bar_chart(
                importance.sort_values("importance"),
                x="feature",
                y="importance",
                horizontal=True,
                x_label="Mean absolute SHAP value",
                y_label="Feature",
            )
    for limitation in explanation.limitations:
        st.caption(f"• {limitation}")


def render_impacts(impacts: list[BusinessImpactMetric]) -> None:
    if not impacts:
        st.info("No separate business-impact metric was calculated for this result.")
        return
    frame = pd.DataFrame([impact.model_dump() for impact in impacts])
    st.dataframe(
        frame,
        hide_index=True,
        column_config={
            "value": st.column_config.NumberColumn(format="%.4f"),
            "assumptions": st.column_config.ListColumn(),
        },
    )


def render_recommendations(
    recommendations: list[BusinessRecommendationDetails],
) -> None:
    if not recommendations:
        st.info("Run a predictive or statistical analysis to generate evidence-based considerations.")
        return
    for recommendation in recommendations:
        with st.container(border=True):
            st.markdown(f"**{recommendation.title}** · `{recommendation.priority}` priority")
            st.write(recommendation.action)
            st.caption(recommendation.rationale)
            for evidence in recommendation.evidence:
                st.caption(f"• {evidence}")


def render_business_overview(overview: BusinessOverviewResponse) -> None:
    metrics = {metric.name: metric for metric in overview.observed_kpis}
    forecast_impact = {metric.name: metric for metric in overview.forecast.impact}
    with st.container(horizontal=True):
        st.metric(
            "Total revenue",
            f"R$ {metrics['total_revenue'].value:,.2f}",
            border=True,
        )
        st.metric(
            "Order volume",
            f"{metrics['order_volume'].value:,.0f}",
            border=True,
        )
        st.metric(
            "Average order value",
            f"R$ {metrics['average_order_value'].value:,.2f}",
            border=True,
        )
        st.metric(
            "Delayed delivery rate",
            f"{metrics['delayed_delivery_rate'].value:.2%}",
            border=True,
        )
    with st.container(horizontal=True):
        st.metric(
            "Four-week forecast",
            f"R$ {forecast_impact['forecast_revenue'].value:,.2f}",
            delta=f"{forecast_impact['forecast_change_rate_vs_recent'].value:.2%} vs recent",
            border=True,
        )
        st.metric(
            "Customer segments",
            str(overview.customer_segments.data["cluster_count"]),
            border=True,
        )
        st.metric(
            "Delivery-risk test PR-AUC",
            f"{float(overview.classification_summary['test_metrics']['pr_auc']):.4f}",
            border=True,
        )
    monthly = pd.DataFrame(overview.monthly_revenue)
    monthly["month"] = pd.to_datetime(monthly["month"])
    with st.container(border=True):
        st.subheader("Historical monthly revenue")
        st.line_chart(monthly, x="month", y="revenue", x_label="Month", y_label="Revenue (R$)")
    profiles = pd.DataFrame(overview.customer_segments.data["profiles"])
    with st.container(border=True):
        st.subheader("Customer segment value")
        st.bar_chart(profiles, x="segment", y="monetary", x_label="Segment", y_label="Average monetary value (R$)")
    for limitation in overview.limitations:
        st.caption(f"• {limitation}")


def _format_probability(value: Any) -> str:
    if value is None:
        return "—"
    number = float(value)
    return f"{number:.3e}" if number < 0.001 else f"{number:.4f}"


def _format_number(value: Any) -> str:
    return "—" if value is None else f"{float(value):,.4f}"


def _parse_number_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parsed) < 2:
        raise ValueError("Enter at least two comma-separated numbers per group.")
    return parsed


def submit_statistical_request(
    method_name: str,
    payload: dict | Callable[[], dict],
) -> None:
    st.session_state.pop("statistical_result", None)
    try:
        request_payload = payload() if callable(payload) else payload
        method = getattr(build_api_client(), method_name)
        with st.spinner("Running the statistical calculation in Python…"):
            st.session_state["statistical_result"] = method(request_payload)
    except FrontendAPIError as exc:
        st.error(exc.message)
    except ValueError as exc:
        st.error(str(exc))


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
      <h1>Enterprise AI SQL &amp; Predictive Analytics Copilot</h1>
      <p>Ask historical, predictive, or statistical business questions in plain language. Gemini routes
      the request while PostgreSQL, trained ML models, and Python statistics produce the results.</p>
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
    "Ask a historical, predictive, or statistical business question",
    key="question_input",
    placeholder="e.g. Compare total historical revenue with the next 4 weeks forecast.",
)
analyze_clicked = st.button("Analyze", type="primary", width="stretch")

if analyze_clicked or sample_clicked:
    submitted_question = (sample_clicked or question).strip()
    if not submitted_question:
        st.warning("Enter a business question before running the analysis.")
    else:
        # Never leave a previous answer visible when a new analysis fails.
        st.session_state.pop("api_analytics_result", None)
        try:
            with st.spinner("Routing the request and running verified SQL or trained models…"):
                st.session_state["api_analytics_result"] = build_api_client().copilot_query(
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
    render_copilot_result(st.session_state["api_analytics_result"])

current_response: CopilotQueryResponse | None = st.session_state.get("api_analytics_result")
(
    overview_tab,
    sql_tab,
    predictive_tab,
    segmentation_tab,
    statistics_tab,
    explainability_tab,
    recommendations_tab,
) = st.tabs(
    [
        "Executive overview",
        "SQL analytics",
        "Predictive analytics",
        "Customer segmentation",
        "Statistical analysis",
        "Explainable AI",
        "Business recommendations",
    ]
)

with overview_tab:
    st.caption(
        "Observed Olist KPIs are kept separate from forecasts and scenario comparisons."
    )
    if st.button(
        "Load executive overview",
        icon=":material/dashboard:",
        key="load_executive_overview",
    ):
        try:
            with st.spinner("Loading observed KPIs and deployed-model summaries…"):
                st.session_state["business_overview"] = (
                    build_api_client().business_overview()
                )
        except FrontendAPIError as exc:
            st.error(exc.message)
    if "business_overview" in st.session_state:
        render_business_overview(st.session_state["business_overview"])
    else:
        st.info("Load the overview to retrieve current PostgreSQL and model outputs.")

with sql_tab:
    if current_response and current_response.historical:
        render_result(current_response.historical, show_answer=False)
    else:
        st.info("Ask a historical or hybrid question to view validated SQL and real rows here.")

with predictive_tab:
    predictions = (
        [
            item
            for item in current_response.predictions
            if item.task != "customer_segmentation"
        ]
        if current_response
        else []
    )
    if predictions:
        for prediction in predictions:
            render_prediction(prediction)
    else:
        st.info("Ask for a revenue forecast or late-delivery risk prediction.")

with segmentation_tab:
    segment_predictions = (
        [
            item
            for item in current_response.predictions
            if item.task == "customer_segmentation"
        ]
        if current_response
        else []
    )
    if segment_predictions:
        for prediction in segment_predictions:
            render_prediction(prediction)
            interpretations = prediction.data.get("interpretations")
            if interpretations:
                st.dataframe(pd.DataFrame(interpretations), hide_index=True)
    else:
        st.info("Ask to show customer segments or provide a customer_unique_id.")

with statistics_tab:
    if current_response and current_response.statistical_analyses:
        for analysis in current_response.statistical_analyses:
            render_statistical_analysis(analysis)
    else:
        st.info("Ask a statistical question or use the explicit calculators below.")

with explainability_tab:
    explained = (
        [item for item in current_response.predictions if item.explanation]
        if current_response
        else []
    )
    if explained:
        for prediction in explained:
            render_model_explanation(prediction)
    else:
        st.info("Run a forecast or late-delivery prediction to view actual SHAP values.")

with recommendations_tab:
    if current_response:
        render_recommendations(current_response.business_recommendations)
        for result in [
            *current_response.predictions,
            *current_response.statistical_analyses,
        ]:
            if result.impact:
                st.subheader(f"{result.task.replace('_', ' ').title()} impact")
                render_impacts(result.impact)
    else:
        st.info("Run an analysis to generate evidence-bounded business considerations.")

st.markdown('<div class="section-label">Statistical analysis tools</div>', unsafe_allow_html=True)
st.caption(
    "Use explicit samples or experiment assumptions below. All calculations run in Python; "
    "Gemini does not create p-values, intervals, or sample sizes."
)
hypothesis_tab, interval_tab, ab_tab, sample_tab = st.tabs(
    ["Hypothesis test", "Confidence interval", "A/B test", "Sample size"]
)

with hypothesis_tab:
    hypothesis_type = st.segmented_control(
        "Metric type",
        ["Continuous", "Proportion"],
        default="Continuous",
        key="hypothesis_type",
    )
    with st.form("hypothesis_form", border=True):
        if hypothesis_type == "Continuous":
            group_a_values = st.text_input("Group A values", value="10, 12, 11, 13, 12")
            group_b_values = st.text_input("Group B values", value="8, 9, 10, 9, 8")
        else:
            count_row = st.columns(4)
            successes_a = count_row[0].number_input("A successes", min_value=0, value=80)
            trials_a = count_row[1].number_input("A trials", min_value=1, value=1000)
            successes_b = count_row[2].number_input("B successes", min_value=0, value=100)
            trials_b = count_row[3].number_input("B trials", min_value=1, value=1000)
        hypothesis_alpha = st.number_input(
            "Significance level", min_value=0.001, max_value=0.2, value=0.05, format="%.3f"
        )
        hypothesis_submitted = st.form_submit_button(
            "Run hypothesis test", type="primary", icon=":material/science:"
        )
    if hypothesis_submitted:
        if hypothesis_type == "Continuous":
            submit_statistical_request(
                "hypothesis_test",
                lambda: {
                    "analysis_type": "continuous",
                    "group_a_values": _parse_number_list(group_a_values),
                    "group_b_values": _parse_number_list(group_b_values),
                    "alpha": hypothesis_alpha,
                },
            )
        else:
            submit_statistical_request(
                "hypothesis_test",
                {
                    "analysis_type": "proportion",
                    "successes_a": successes_a,
                    "trials_a": trials_a,
                    "successes_b": successes_b,
                    "trials_b": trials_b,
                    "alpha": hypothesis_alpha,
                },
            )

with interval_tab:
    interval_type = st.segmented_control(
        "Interval type",
        ["Mean", "Proportion"],
        default="Mean",
        key="interval_type",
    )
    with st.form("interval_form", border=True):
        if interval_type == "Mean":
            interval_values = st.text_input(
                "Observed values", value="100, 120, 110, 130, 115, 125"
            )
        else:
            interval_counts = st.columns(2)
            interval_successes = interval_counts[0].number_input(
                "Successes", min_value=0, value=80
            )
            interval_trials = interval_counts[1].number_input(
                "Trials", min_value=1, value=1000
            )
        confidence_level = st.number_input(
            "Confidence level", min_value=0.5, max_value=0.999, value=0.95, format="%.3f"
        )
        interval_submitted = st.form_submit_button(
            "Calculate interval", type="primary", icon=":material/straighten:"
        )
    if interval_submitted:
        if interval_type == "Mean":
            submit_statistical_request(
                "confidence_interval",
                lambda: {
                    "confidence_level": confidence_level,
                    "estimand": "mean",
                    "values": _parse_number_list(interval_values),
                },
            )
        else:
            submit_statistical_request(
                "confidence_interval",
                {
                    "confidence_level": confidence_level,
                    "estimand": "proportion",
                    "successes": interval_successes,
                    "trials": interval_trials,
                },
            )

with ab_tab:
    ab_metric = st.segmented_control(
        "Experiment metric",
        ["Conversion", "Average value"],
        default="Conversion",
        key="ab_metric",
    )
    with st.form("ab_form", border=True):
        if ab_metric == "Conversion":
            ab_counts = st.columns(4)
            control_successes = ab_counts[0].number_input(
                "Control successes", min_value=0, value=800
            )
            control_trials = ab_counts[1].number_input(
                "Control trials", min_value=1, value=10000
            )
            treatment_successes = ab_counts[2].number_input(
                "Treatment successes", min_value=0, value=900
            )
            treatment_trials = ab_counts[3].number_input(
                "Treatment trials", min_value=1, value=10000
            )
        else:
            control_values = st.text_input("Control values", value="100, 105, 98, 110, 102")
            treatment_values = st.text_input(
                "Treatment values", value="108, 112, 105, 115, 110"
            )
        randomized = st.checkbox("This was a randomized experiment", value=False)
        ab_submitted = st.form_submit_button(
            "Analyze experiment", type="primary", icon=":material/experiment:"
        )
    if ab_submitted:
        if ab_metric == "Conversion":
            submit_statistical_request(
                "ab_test",
                {
                    "randomized_experiment": randomized,
                    "metric_type": "conversion",
                    "control_successes": control_successes,
                    "control_trials": control_trials,
                    "treatment_successes": treatment_successes,
                    "treatment_trials": treatment_trials,
                },
            )
        else:
            submit_statistical_request(
                "ab_test",
                lambda: {
                    "randomized_experiment": randomized,
                    "metric_type": "average_value",
                    "control_values": _parse_number_list(control_values),
                    "treatment_values": _parse_number_list(treatment_values),
                },
            )

with sample_tab:
    sample_metric = st.segmented_control(
        "Planning metric",
        ["Proportion", "Mean"],
        default="Proportion",
        key="sample_metric",
    )
    with st.form("sample_size_form", border=True):
        sample_fields = st.columns(3)
        baseline = sample_fields[0].number_input("Baseline", value=0.08, format="%.4f")
        detectable_effect = sample_fields[1].number_input(
            "Minimum detectable effect", value=0.02, format="%.4f"
        )
        if sample_metric == "Mean":
            standard_deviation = sample_fields[2].number_input(
                "Standard deviation", min_value=0.0001, value=1.0, format="%.4f"
            )
        else:
            power = sample_fields[2].number_input(
                "Statistical power", min_value=0.5, max_value=0.99, value=0.8, format="%.2f"
            )
        if sample_metric == "Mean":
            power = st.number_input(
                "Statistical power", min_value=0.5, max_value=0.99, value=0.8, format="%.2f"
            )
        sample_submitted = st.form_submit_button(
            "Estimate sample size", type="primary", icon=":material/groups:"
        )
    if sample_submitted:
        payload = {
            "metric_type": sample_metric.lower(),
            "baseline": baseline,
            "minimum_detectable_effect": detectable_effect,
            "power": power,
        }
        if sample_metric == "Mean":
            payload["standard_deviation"] = standard_deviation
        submit_statistical_request("sample_size", payload)

if "statistical_result" in st.session_state:
    render_statistical_analysis(st.session_state["statistical_result"])
    render_recommendations(st.session_state["statistical_result"].recommendations)
    render_impacts(st.session_state["statistical_result"].impact)

st.divider()
st.caption(
    "Read-only analytics · Validated PostgreSQL · Persisted ML models · Python statistical inference · Gemini routes but never invents results"
)
