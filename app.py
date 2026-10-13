from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from staysignal.engine import (  # noqa: E402
    IDENTIFIER,
    NUMERIC_COLUMNS,
    TARGET,
    audit_slices,
    calibration_bins,
    explain_customer,
    experiment_assignment,
    load_data,
    model_signals,
    statistical_tests,
    train_models,
)


st.set_page_config(
    page_title="StaySignal · Churn workbench",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root { --ink:#202A24; --muted:#718078; --green:#2D6B52; --deep:#173B2E; --mint:#E6F1EA; --line:#E5EAE5; --paper:#F6F7F3; }
    html, body, [class*="css"] { font-family:Inter, Aptos, 'Segoe UI', sans-serif; color:var(--ink); }
    .stApp { background:var(--paper); }
    header[data-testid="stHeader"] { display:none !important; }
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }
    [data-testid="stToolbar"] { display:none !important; }
    .block-container { padding:1.25rem clamp(1rem, 3.5vw, 3rem) 3rem; max-width:1480px; }
    h1,h2,h3 { font-family:Inter, Aptos, 'Segoe UI', sans-serif !important; letter-spacing:-.035em; color:var(--ink); }
    h1 { font-size:2.15rem !important; font-weight:750 !important; }
    h2 { font-size:1.42rem !important; font-weight:720 !important; }
    h3 { font-size:1.06rem !important; font-weight:700 !important; }
    [data-testid="stMetric"] { padding:1rem 1.2rem; min-height:112px; background:#fff; border:1px solid var(--line); border-radius:16px; box-shadow:0 3px 12px rgba(23,59,46,.035); }
    [data-testid="stMetricLabel"] { color:#718078; font-size:.86rem; font-weight:650; white-space:normal; }
    [data-testid="stMetricValue"] { color:var(--ink); font-family:Inter, Aptos, 'Segoe UI', sans-serif; font-size:clamp(1.55rem, 2.5vw, 2.1rem); font-weight:750; line-height:1.15; }
    [data-testid="stMetricDelta"] { font-size:.76rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line); border-radius:16px; background:#fff; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    .eyebrow { color:var(--green); font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .hero-title { font-family:Inter, Aptos, 'Segoe UI', sans-serif; font-weight:800; font-size:2.15rem; letter-spacing:-.045em; margin:.18rem 0 .35rem; color:var(--ink); }
    .hero-copy { color:#65746B; font-size:1rem; max-width:760px; line-height:1.55; }
    .source-chip { display:inline-block; background:var(--mint); color:var(--green); border-radius:999px; padding:.38rem .72rem; font-size:.73rem; font-weight:700; letter-spacing:.02em; }
    .section-kicker { color:#75847A; font-size:.77rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; margin-bottom:.25rem; }
    .note { color:#66766C; font-size:.85rem; line-height:1.5; }
    .dark-callout { background:var(--deep); border-radius:16px; color:#EDF6EE; padding:1rem 1.15rem; margin:.5rem 0 1rem; }
    .dark-callout strong { color:#B8E2C8; }
    .brand-row { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.15rem 0 .85rem; }
    .brand-name { color:var(--deep); font-size:1.42rem; font-weight:800; letter-spacing:-.04em; }
    .brand-mark { color:#fff; background:var(--green); display:inline-grid; place-items:center; width:2rem; height:2rem; border-radius:.7rem; margin-right:.45rem; font-size:1.1rem; }
    .brand-subtitle { color:var(--muted); font-size:.78rem; margin-top:.12rem; }
    .local-badge { color:var(--green); background:var(--mint); border:1px solid #D5E6D9; padding:.5rem .8rem; border-radius:999px; font-size:.72rem; font-weight:700; letter-spacing:.04em; white-space:nowrap; }
    [data-testid="stSegmentedControl"] { background:#E9EEE9; border-radius:13px; padding:.22rem; }
    @media (max-width: 980px) {
      .st-key-dashboard-kpis [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; }
      .st-key-dashboard-kpis [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 calc(50% - .5rem) !important; min-width:calc(50% - .5rem) !important; }
    }
    .stButton button, .stDownloadButton button { border-radius:10px; font-weight:650; }
    .stButton button[kind="primary"] { background:var(--green); border-color:var(--green); }
    .stTabs [data-baseweb="tab-list"] { gap:.35rem; }
    .stTabs [data-baseweb="tab"] { border-radius:10px; padding:.45rem .9rem; }
    a { color:var(--green); }
    footer { visibility:hidden; }
    @media (max-width: 720px) {
      .block-container { padding:1rem .8rem 2rem; }
      [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; gap:.7rem !important; }
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 100% !important; min-width:100% !important; }
      .st-key-dashboard-kpis [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 100% !important; min-width:100% !important; }
      .local-badge { display:none; }
      .hero-title { font-size:1.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_bundle():
    return train_models(load_data())


def money(value: float) -> str:
    return f"${value:,.2f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def risk_color(band: str) -> str:
    return {"Low": "#4A8C69", "Watch": "#C49B45", "High": "#D98249", "Very high": "#B64B45"}.get(band, "#718078")


def build_risk_gauge(probability: float, band: str):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 46, "color": "#202A24", "family": "Inter"}},
            title={"text": f"<span style='font-size:14px;color:#718078'>Probability of sample Churn label · {band} risk</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "#D7DED8", "tickfont": {"color": "#718078"}},
                "bar": {"color": risk_color(band), "thickness": .23},
                "bgcolor": "#FFFFFF",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#E8F0E9"},
                    {"range": [25, 50], "color": "#F3F1E7"},
                    {"range": [50, 75], "color": "#F8EEE4"},
                    {"range": [75, 100], "color": "#F5E7E5"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=25, r=25, t=55, b=8), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif", color="#536258"))
    return fig


def style_chart(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=5, r=8, t=20, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#536258", size=13),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#E5EAE5"),
        yaxis=dict(showgrid=True, gridcolor="#EDF0EC", zeroline=False, linecolor="#E5EAE5"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def header(title: str, description: str, eyebrow: str = "STAYSIGNAL"):
    st.markdown(f"<div class='eyebrow'>{eyebrow}</div><div class='hero-title'>{title}</div><div class='hero-copy'>{description}</div>", unsafe_allow_html=True)
    st.write("")


def score_export(raw: pd.DataFrame, bundle):
    required = bundle.feature_columns
    missing = [column for column in required if column not in raw.columns]
    if missing:
        return None, None, missing, {}
    if raw.empty:
        return None, None, ["The uploaded CSV has no rows."], {}
    features = raw[required].copy()
    for column in features.select_dtypes(include="object").columns:
        features[column] = features[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        features[column] = features[column].replace("", np.nan)
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    monthly = features["MonthlyCharges"].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(monthly).all() or (monthly < 0).any():
        return None, None, ["MonthlyCharges must be a valid, non-negative number on every row for queue prioritization."], {}
    probabilities = bundle.model.predict_proba(features)[:, 1]
    if "customerID" in raw.columns:
        customer_ids = raw["customerID"].astype("string").str.strip()
        duplicate_ids = customer_ids.notna() & customer_ids.ne("") & customer_ids.duplicated(keep=False)
        if duplicate_ids.any():
            duplicate_count = int(customer_ids[duplicate_ids].nunique())
            return None, None, [f"customerID contains {duplicate_count} duplicate account IDs; keep one row per account."], {}
        generated_ids = pd.Series([f"Account {i + 1:04d}" for i in range(len(raw))], index=raw.index)
        customer_ids = customer_ids.mask(customer_ids.isna() | customer_ids.eq(""), generated_ids).astype(str)
    else:
        customer_ids = pd.Series([f"Account {i + 1:04d}" for i in range(len(raw))], index=raw.index)
    result = pd.DataFrame({
        "customerID": customer_ids.to_numpy(),
        "risk_probability": probabilities,
        "Contract": raw["Contract"].astype(str).to_numpy(),
        "tenure": features["tenure"].to_numpy(),
        "MonthlyCharges": features["MonthlyCharges"].to_numpy(),
        "InternetService": raw["InternetService"].astype(str).to_numpy(),
    })
    result["risk_band"] = pd.cut(
        result["risk_probability"], bins=[-0.001, .25, .50, .75, 1.001],
        labels=["Low", "Watch", "High", "Very high"], ordered=True,
    ).astype(str)
    result["risk_weighted_charges"] = result["risk_probability"] * result["MonthlyCharges"]
    result["actual_churn"] = False
    unknown = {}
    categorical = [column for column in required if column not in NUMERIC_COLUMNS]
    for column in categorical:
        known = set(bundle.data[column].dropna().astype(str).unique())
        unseen = set(raw[column].dropna().astype(str).unique()) - known
        if unseen:
            unknown[column] = sorted(unseen)
    return result, features, [], unknown


try:
    with st.spinner("Preparing the sample models and test-set diagnostics…"):
        bundle = get_bundle()
except Exception as exc:
    st.error(f"StaySignal could not prepare its bundled sample. {exc}")
    st.stop()

template = bundle.data.loc[bundle.data[TARGET].eq("No"), ["customerID", *bundle.feature_columns]].head(5)
brand_left, brand_right = st.columns([2.2, 1], vertical_alignment="center")
with brand_left:
    st.markdown(
        "<div class='brand-row'><div><div class='brand-name'><span class='brand-mark'>◉</span>StaySignal</div><div class='brand-subtitle'>Customer retention workbench</div></div></div>",
        unsafe_allow_html=True,
    )
with brand_right:
    st.markdown("<div style='text-align:right'><span class='local-badge'>LOCAL WORKSPACE · DATA STAYS HERE</span></div>", unsafe_allow_html=True)
st.divider()
page = st.segmented_control(
    "Workspace",
    ["Command center", "Retention queue", "Customer explorer", "Model & fairness"],
    default="Command center",
    label_visibility="collapsed",
)
with st.expander("Score active accounts from a CSV", expanded=False):
    uploaded_file = st.file_uploader("Customer CSV", type=["csv"], help="Use IBM Telco field names. Download a sample template below.")
    st.caption("Uploaded rows are assumed to be active; any Churn column is ignored. Scoring and model fitting run locally on this machine.")
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="staysignal_upload_template.csv",
        mime="text/csv",
        help="Replace these example rows with current active accounts before operational scoring.",
    )


scores = bundle.scores.copy()
uploaded_features = None
is_uploaded = False
upload_error = None
upload_warning = None
if uploaded_file is not None:
    try:
        raw_upload = pd.read_csv(uploaded_file.getvalue())
        uploaded_scores, uploaded_features, upload_errors, unseen_categories = score_export(raw_upload, bundle)
        if upload_errors:
            upload_error = "CSV needs attention: " + ", ".join(upload_errors)
            uploaded_features = None
        else:
            scores = uploaded_scores
            is_uploaded = True
            if unseen_categories:
                unseen_summary = "; ".join(f"{name}: {', '.join(values)}" for name, values in unseen_categories.items())
                upload_warning = "Categories not seen in training will be treated as unknown: " + unseen_summary
    except Exception as exc:
        upload_error = f"Could not read this CSV: {exc}"
        uploaded_features = None

if upload_error:
    st.error(upload_error)
if upload_warning:
    st.warning(upload_warning)

if not is_uploaded:
    scores = scores.loc[~scores["actual_churn"]].copy()
portfolio_weighted_total = float(scores["risk_weighted_charges"].sum())
high_risk_count = int(scores["risk_probability"].ge(.50).sum())

if page == "Command center":
    header(
        "Keep the next conversation focused",
        "See where churn risk is concentrated, review the signals behind the score, and choose who your team can actually reach.",
        "RETENTION SIGNALS  /  01",
    )
    source_label = "CURRENT CUSTOMER CSV · SCORED LOCALLY" if is_uploaded else "IBM TELCO SAMPLE · HELD-OUT RETROSPECTIVE ACCOUNTS"
    st.markdown(f"<span class='source-chip'>{source_label}</span>", unsafe_allow_html=True)
    st.write("")

    with st.container(key="dashboard-kpis"):
        kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="small")
        kpi1.metric("Accounts scored", f"{len(scores):,}", help="Uploaded active accounts, or retained accounts from the IBM sample.")
        kpi2.metric("Mean predicted score" if is_uploaded else "Sample churn rate", pct(scores["risk_probability"].mean()) if is_uploaded else pct(bundle.data[TARGET].eq("Yes").mean()), help="Probability of the sample Churn label; it has no defined future prediction window." if is_uploaded else "Historical labels in the IBM sample, not a current forecast.")
        kpi3.metric("Accounts at 50%+ risk", f"{high_risk_count:,}", help="Accounts with a model score of at least 50%.")
        kpi4.metric("Risk-weighted charges", money(portfolio_weighted_total), help="Sum of probability of the sample Churn label × monthly charges. This is a prioritization proxy, not expected loss, savings, or margin.")

    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown("<div class='section-kicker'>RISK DISTRIBUTION</div>", unsafe_allow_html=True)
        bands_order = ["Low", "Watch", "High", "Very high"]
        band_counts = scores["risk_band"].value_counts().reindex(bands_order, fill_value=0).rename_axis("Risk band").reset_index(name="Accounts")
        fig = px.bar(
            band_counts,
            x="Risk band",
            y="Accounts",
            color="Risk band",
            color_discrete_map={"Low": "#6E9B7D", "Watch": "#C7AA65", "High": "#D98752", "Very high": "#B64B45"},
            text_auto=",.0f",
        )
        fig.update_traces(marker_line_width=0, textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, bargap=.42)
        fig.update_xaxes(tickangle=0)
        st.plotly_chart(style_chart(fig, 330), width="stretch", config={"displayModeBar": False})
        st.caption("Risk bands are score ranges: low <25%, watch 25–50%, high 50–75%, very high ≥75%.")

    with right:
        st.markdown("<div class='section-kicker'>MODEL SIGNALS</div>", unsafe_allow_html=True)
        signals = model_signals(bundle).sort_values("weight", ascending=True)
        fig = px.bar(signals, x="weight", y="feature", orientation="h", color_discrete_sequence=["#2D6B52"])
        fig.update_traces(marker_line_width=0)
        fig.update_layout(xaxis_title="Relative signal strength", yaxis_title="", showlegend=False)
        st.plotly_chart(style_chart(fig, 350), width="stretch", config={"displayModeBar": False})
        st.caption("Association in the fitted model; these signals do not establish why a customer leaves.")

    st.markdown(
        "<div class='dark-callout'><strong>Use risk to choose who to review.</strong> It cannot tell you whether a discount or call will change an outcome. Test retention actions with a randomized holdout before scaling them.</div>",
        unsafe_allow_html=True,
    )
    if is_uploaded:
        st.markdown("<div class='section-kicker'>AVERAGE PREDICTED RISK BY CONTRACT</div>", unsafe_allow_html=True)
        contract_summary = scores.groupby("Contract", observed=True)["risk_probability"].mean().rename("Mean predicted risk").reset_index().sort_values("Mean predicted risk", ascending=False)
        fig = px.bar(contract_summary, x="Contract", y="Mean predicted risk", color="Mean predicted risk", color_continuous_scale=["#B7D7C1", "#2D6B52"], text=contract_summary["Mean predicted risk"].map(pct))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%", yaxis_title="Mean model score", xaxis_title="")
        st.plotly_chart(style_chart(fig, 285), width="stretch", config={"displayModeBar": False})
        st.caption("Model estimates for the uploaded accounts. This grouping does not show an effect of contract type.")
    else:
        st.markdown("<div class='section-kicker'>OBSERVED CHURN BY CONTRACT</div>", unsafe_allow_html=True)
        contract_summary = (
            bundle.data.groupby("Contract", observed=True)[TARGET]
            .apply(lambda values: (values == "Yes").mean())
            .rename("Observed churn rate")
            .reset_index()
            .sort_values("Observed churn rate", ascending=False)
        )
        fig = px.bar(contract_summary, x="Contract", y="Observed churn rate", color="Observed churn rate", color_continuous_scale=["#B7D7C1", "#2D6B52"], text=contract_summary["Observed churn rate"].map(pct))
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%", yaxis_title="Share with churn label", xaxis_title="")
        st.plotly_chart(style_chart(fig, 285), width="stretch", config={"displayModeBar": False})
        st.caption("Descriptive pattern in the historical snapshot. Contract type is not a proven cause.")


elif page == "Retention queue":
    header(
        "A queue built around team capacity",
        "Set a score floor and a review limit. The shortlist is ranked by the score multiplied by monthly charges, not by an assumed offer success rate.",
        "RETENTION OPERATIONS  /  02",
    )
    controls, table_col = st.columns([.82, 2.18], gap="large")
    with controls:
        st.markdown("<div class='section-kicker'>QUEUE SETTINGS</div>", unsafe_allow_html=True)
        minimum_risk = st.slider("Minimum model score", 0, 90, 25, 5, format="%d%%") / 100
        capacity_upper = max(1, min(1500, len(scores)))
        max_customers = st.slider("Review capacity", 1, capacity_upper, min(250, capacity_upper), 1, help="How many customers the team can review in this queue.")
        selected_bands = st.multiselect("Risk bands", ["Low", "Watch", "High", "Very high"], default=["Watch", "High", "Very high"])
        contracts = sorted(scores["Contract"].dropna().unique())
        selected_contracts = st.multiselect("Contract types", contracts, default=contracts)
        st.markdown("<div class='note'>Priority proxy = probability of the sample Churn label × monthly charges. It has no defined future time window; use it as an ordering aid, not a loss or offer estimate.</div>", unsafe_allow_html=True)

    filtered = scores[
        scores["risk_probability"].ge(minimum_risk)
        & scores["risk_band"].isin(selected_bands)
        & scores["Contract"].isin(selected_contracts)
    ].copy()
    filtered.sort_values("risk_weighted_charges", ascending=False, inplace=True)
    queue = filtered.head(max_customers).copy()

    with table_col:
        st.markdown("<div class='section-kicker'>PRIORITIZED REVIEW LIST</div>", unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        q1.metric("Eligible", f"{len(filtered):,}")
        q2.metric("In queue", f"{len(queue):,}")
        q3.metric("Risk-weighted charges", money(queue["risk_weighted_charges"].sum()))
        display = queue[[IDENTIFIER, "risk_probability", "risk_band", "Contract", "tenure", "MonthlyCharges", "risk_weighted_charges"]].rename(
            columns={IDENTIFIER: "Customer", "risk_probability": "Churn probability", "risk_band": "Risk", "Contract": "Contract", "tenure": "Tenure (mo)", "MonthlyCharges": "Monthly charges", "risk_weighted_charges": "Risk-weighted charges"}
        )
        display["Churn probability"] = display["Churn probability"] * 100
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            height=440,
            column_config={
                "Churn probability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=100, format="%.0f%%"),
                "Monthly charges": st.column_config.NumberColumn(format="$%.2f"),
                "Risk-weighted charges": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        csv_display = display.copy()
        csv_display["Churn probability"] = csv_display["Churn probability"] / 100
        st.download_button(
            "Download review queue",
            data=csv_display.to_csv(index=False).encode("utf-8"),
            file_name="staysignal_review_queue.csv",
            mime="text/csv",
            type="primary",
            disabled=queue.empty,
        )

    st.divider()
    st.markdown("## Plan a retention test")
    st.markdown("Randomly assign the prioritized list to a retention action and business-as-usual holdout. The churn score does not predict who will respond; the test measures that.")
    planner_left, planner_right = st.columns([1, 1.55], gap="large")
    with planner_left:
        holdout_share = st.slider("Holdout share", 10, 50, 20, 5, format="%d%%") / 100
        seed = st.number_input("Randomization seed", min_value=0, max_value=999999, value=24, step=1)
        st.caption("Holdout means no new retention action during the agreed test window; normal service continues.")
    assignments = experiment_assignment(queue, holdout_share, int(seed))
    treatment_count = int(assignments["experiment_arm"].eq("Treatment").sum()) if not assignments.empty else 0
    control_count = int(assignments["experiment_arm"].eq("Holdout").sum()) if not assignments.empty else 0
    if not assignments.empty:
        assignment_export = assignments[[IDENTIFIER, "risk_probability", "risk_band", "Contract", "tenure", "MonthlyCharges", "risk_weighted_charges", "experiment_arm"]].copy()
        assignment_export["randomization_seed"] = int(seed)
        assignment_export["holdout_share"] = holdout_share
        assignment_csv = assignment_export.to_csv(index=False).encode("utf-8")
    else:
        assignment_csv = b""
    with planner_right:
        e1, e2 = st.columns(2)
        e1.metric("Treatment group", f"{treatment_count:,}")
        e2.metric("Holdout group", f"{control_count:,}")
        if treatment_count and control_count:
            baseline = float(queue["risk_probability"].mean())
            standard_error = np.sqrt(max(baseline * (1 - baseline) * (1 / treatment_count + 1 / control_count), 0))
            mde = min(1.0, float(norm.ppf(.975) + norm.ppf(.80)) * standard_error)
            st.metric("Approx. detectable difference", f"{mde:.1%}", help="Normal approximation at 5% two-sided significance and 80% power, using mean predicted risk as a planning baseline. Small-sample results are rough.")
            st.caption(f"Planning baseline: {baseline:.1%} mean predicted risk among selected accounts. This is not an observed control rate or treatment effect.")
        st.download_button(
            "Download randomized assignments",
            data=assignment_csv,
            file_name="staysignal_retention_test_assignments.csv",
            mime="text/csv",
            disabled=assignments.empty,
        )
    st.markdown(
        "<div class='dark-callout'><strong>Before launch:</strong> define one primary outcome and a fixed observation window, keep assignment randomized, and track both churn and offer cost. Compare groups by original assignment after the window closes.</div>",
        unsafe_allow_html=True,
    )


elif page == "Customer explorer":
    header(
        "Understand one customer's score",
        "Inspect an uploaded account, a sample record, or an entered profile. The score is a model estimate from this snapshot, with the strongest account signals shown beside it.",
        "CUSTOMER REVIEW  /  03",
    )
    mode = st.radio("Input", ["Sample account", "Enter a profile"], horizontal=True, label_visibility="collapsed")
    if mode == "Sample account":
        ids = scores[IDENTIFIER].astype(str).tolist()
        selected_id = st.selectbox("Sample customer", ids, index=int(np.argmax(scores["risk_probability"].to_numpy())))
        selected_position = ids.index(selected_id)
        if is_uploaded:
            row = uploaded_features.iloc[[selected_position]].copy()
        else:
            chosen = bundle.data.loc[bundle.data[IDENTIFIER].astype(str).eq(selected_id)].iloc[[0]].copy()
            row = chosen[bundle.feature_columns]
        customer_label = selected_id
    else:
        reference = bundle.data.iloc[0]
        fields_left, fields_mid, fields_right = st.columns(3, gap="large")
        with fields_left:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=int(reference["tenure"]))
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], index=0)
            online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"], index=1)
            tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"], index=1)
        with fields_mid:
            monthly = st.number_input("Monthly charges", min_value=0.0, max_value=500.0, value=float(reference["MonthlyCharges"]), step=1.0)
            total = st.number_input("Total charges", min_value=0.0, max_value=100000.0, value=float(reference["TotalCharges"]), step=10.0)
            phone = st.selectbox("Phone service", ["Yes", "No"], index=0)
            multiple = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"], index=1)
            backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"], index=0)
        with fields_right:
            device = st.selectbox("Device protection", ["Yes", "No", "No internet service"], index=1)
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=1)
            streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"], index=1)
            paperless = st.selectbox("Paperless billing", ["Yes", "No"], index=0)
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
        record = {
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple if phone == "Yes" else "No phone service",
            "InternetService": internet,
            "OnlineSecurity": online_security if internet != "No" else "No internet service",
            "OnlineBackup": backup if internet != "No" else "No internet service",
            "DeviceProtection": device if internet != "No" else "No internet service",
            "TechSupport": tech_support if internet != "No" else "No internet service",
            "StreamingTV": streaming_tv if internet != "No" else "No internet service",
            "StreamingMovies": streaming_movies if internet != "No" else "No internet service",
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }
        row = pd.DataFrame([record])[bundle.feature_columns]
        customer_label = "Entered profile"
        st.caption("Gender, senior-citizen, partner, and dependent fields are not requested or used for risk scoring.")

    probability = float(bundle.model.predict_proba(row)[0, 1])
    band = "Low" if probability < .25 else "Watch" if probability < .50 else "High" if probability < .75 else "Very high"
    monthly_charges = float(row.iloc[0]["MonthlyCharges"])
    weighted_charges = probability * monthly_charges
    explanation = explain_customer(bundle, row)
    positive = explanation[explanation["contribution"].gt(0)].sort_values("contribution", ascending=True).tail(4)
    negative = explanation[explanation["contribution"].lt(0)].sort_values("contribution", ascending=False).head(4)

    st.divider()
    score_col, explain_col = st.columns([.9, 1.35], gap="large")
    with score_col:
        st.markdown(f"<div class='section-kicker'>{escape(str(customer_label).upper())}</div>", unsafe_allow_html=True)
        st.plotly_chart(build_risk_gauge(probability, band), width="stretch", config={"displayModeBar": False})
        m1, m2 = st.columns(2)
        m1.metric("Monthly charges", money(monthly_charges))
        m2.metric("Risk-weighted charges", money(weighted_charges), help="Probability of the sample Churn label × monthly charges. A prioritization proxy, not expected loss or savings.")
        st.markdown("<div class='note'>The model estimates the historical Churn label and has no defined future time window. Use this score to decide whether someone should review the account; it does not diagnose a cause or recommend an offer.</div>", unsafe_allow_html=True)
    with explain_col:
        st.markdown("<div class='section-kicker'>WHAT MOVED THE SCORE</div>", unsafe_allow_html=True)
        explanation_plot = pd.concat([positive.assign(direction="Higher risk"), negative.assign(direction="Lower risk")])
        if explanation_plot.empty:
            st.info("This profile is close to the training-set average on the available model signals.")
        else:
            explanation_plot["contribution"] = explanation_plot["contribution"].abs()
            fig = px.bar(
                explanation_plot,
                x="contribution",
                y="feature",
                color="direction",
                orientation="h",
                color_discrete_map={"Higher risk": "#C76C48", "Lower risk": "#4B8B68"},
                labels={"contribution": "Relative log-odds contribution", "feature": "Account field"},
            )
            fig.update_layout(barmode="group", yaxis_title="", legend_title="", xaxis_title="Relative contribution vs. training mix")
            st.plotly_chart(style_chart(fig, 350), width="stretch", config={"displayModeBar": False})
        st.caption("Local logistic contributions are relative to the training-set mix. They describe model behavior, not causal effects.")
        st.markdown("**Review prompts**")
        profile = row.iloc[0]
        prompts = []
        if profile.get("Contract") == "Month-to-month":
            prompts.append("Ask whether the current plan still fits the customer's needs.")
        if profile.get("tenure", 0) < 6:
            prompts.append("Check whether onboarding or setup left anything unresolved.")
        if profile.get("TechSupport") == "No":
            prompts.append("Ask whether support needs have been met recently.")
        if not prompts:
            prompts.append("Review recent service history and ask what would make the account easier to keep.")
        for prompt in prompts[:3]:
            st.markdown(f"- {prompt}")
        st.caption("These are conversation starters based on account fields, not evidence that a field caused churn.")


else:
    header(
        "Check how the model behaves",
        "Compare the probability model with the deck's Naive Bayes baseline, inspect calibration, and review selected demographic slices on an untouched test set.",
        "MODEL REVIEW  /  04",
    )
    tab_stats, tab_score, tab_cal, tab_fair, tab_method = st.tabs(["Statistical evidence", "Model scorecard", "Calibration", "Fairness review", "Methods & limits"])
    with tab_stats:
        st.markdown("### Which sample features are associated with churn?")
        st.markdown("For numeric fields, the null hypothesis is equal means for retained and churned customers; Welch's two-sample t-test reports a 95% confidence interval for the mean difference. For categorical fields, the null is independence from the churn label; chi-square reports Cramér's V. Benjamini–Hochberg q-values adjust for screening multiple fields.")
        evidence = statistical_tests(bundle.data)
        significant_count = int(evidence["BH q-value"].lt(.05).sum()) if not evidence.empty else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Fields screened", f"{len(evidence)}")
        c2.metric("Below q < 0.05", f"{significant_count}")
        c3.metric("Adjustment", "Benjamini–Hochberg")
        if not evidence.empty:
            table = evidence[["Feature", "Test", "Effect detail", "95% CI", "Effect size", "p-value", "BH q-value", "Evidence"]].copy()
            table["Effect size"] = table["Effect size"].map(lambda value: f"{value:.3f}")
            table["p-value"] = table["p-value"].map(lambda value: f"{value:.2e}")
            table["BH q-value"] = table["BH q-value"].map(lambda value: f"{value:.2e}")
            st.dataframe(table, width="stretch", hide_index=True)
        st.caption("This is a univariate screen on the full historical sample. Statistical association does not imply causation; statistical significance does not indicate practical importance. Effect sizes across the two test families are not directly comparable.")

    with tab_score:
        score_col, metric_col = st.columns([.75, 1.25], gap="large")
        logistic_row = bundle.metrics.loc[bundle.metrics["Model"].eq("Calibrated logistic regression")].iloc[0]
        with score_col:
            st.metric("ROC AUC", f"{logistic_row['ROC AUC']:.3f}", help="Ranking quality on the held-out test set.")
            st.metric("Average precision", f"{logistic_row['Average precision']:.3f}", help="Precision-recall summary; useful when churn is the minority class.")
            st.metric("Brier score", f"{logistic_row['Brier score ↓']:.3f}", help="Mean squared probability error; lower is better.")
            st.metric("Churners in top 20%", pct(logistic_row["Recall at 20% capacity"]), help="Share of all test-set churners captured in the highest-risk fifth of the test set.")
        with metric_col:
            st.markdown("### Test-set model comparison")
            styled = bundle.metrics.copy()
            for column in ["ROC AUC", "Average precision", "Recall at 20% capacity", "Precision @ 0.50"]:
                styled[column] = styled[column].map(lambda value: f"{value:.3f}")
            styled["Brier score ↓"] = styled["Brier score ↓"].map(lambda value: f"{value:.3f}")
            st.dataframe(styled, width="stretch", hide_index=True)
            st.caption("Metrics use the same fixed stratified test split. The scorecard reports ranking, probability error, and a capacity-based measure.")

        signals = model_signals(bundle).sort_values("weight", ascending=True)
        st.markdown("### Strongest fitted signals")
        fig = px.bar(signals, x="weight", y="feature", orientation="h", color_discrete_sequence=["#2D6B52"])
        fig.update_layout(xaxis_title="Relative signal strength", yaxis_title="")
        st.plotly_chart(style_chart(fig, 360), width="stretch", config={"displayModeBar": False})
        st.caption("A coefficient shows association within this fitted model. It does not measure intervention impact or prove causality.")

    with tab_cal:
        st.markdown("### Predicted risk vs. observed frequency")
        calibration = calibration_bins(bundle)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#AEB9B1", dash="dash"), name="Perfect calibration"))
        fig.add_trace(go.Scatter(x=calibration["predicted"], y=calibration["observed"], mode="lines+markers", marker=dict(size=9, color="#2D6B52"), line=dict(color="#2D6B52", width=3), name="Test set"))
        fig.update_layout(xaxis_title="Mean predicted churn probability", yaxis_title="Observed churn frequency", xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis=dict(range=[0, 1], tickformat=".0%"), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(style_chart(fig, 390), width="stretch", config={"displayModeBar": False})
        st.caption("Test predictions are grouped into fixed probability bands. The sample size in each band varies.")
        if not calibration.empty:
            details = calibration.copy()
            details["Mean predicted"] = details.pop("predicted").map(pct)
            details["Observed frequency"] = details.pop("observed").map(pct)
            details.rename(columns={"customers": "Test customers"}, inplace=True)
            st.dataframe(details, width="stretch", hide_index=True)

    with tab_fair:
        st.markdown("### Demographic review slices")
        st.markdown("These attributes were excluded from prediction and used only to compare aggregate test-set error patterns. This small sample review is not a fairness certification.")
        audit = audit_slices(bundle)
        if not audit.empty:
            formatted = audit.copy()
            for column in ["Observed churn", "Recall @ 0.50", "False positive rate"]:
                formatted[column] = formatted[column].map(pct)
            formatted["Brier score"] = formatted["Brier score"].map(lambda value: f"{value:.3f}")
            st.dataframe(formatted, width="stretch", hide_index=True)
            fig = px.bar(audit, x="Review slice", y="Recall @ 0.50", color="Review slice", color_discrete_sequence=["#2D6B52", "#77A98B", "#C7AA65", "#D98752"], text=audit["Recall @ 0.50"].map(pct))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_title="Churn recall at 50% threshold", xaxis_title="")
            st.plotly_chart(style_chart(fig, 320), width="stretch", config={"displayModeBar": False})
        st.info("Group comparisons can be unstable with limited samples, reflect historical labels, and miss intersectional or proxy effects. Review the population and use current, approved data before any operational decision.")

    with tab_method:
        method_left, method_right = st.columns(2, gap="large")
        with method_left:
            st.markdown("### Modeling flow")
            st.markdown("1. Start with the IBM telco sample and repair blank total-charge entries with training-fold median imputation.\n2. Reserve a stratified 20% test split.\n3. Fit logistic regression with cross-validated sigmoid calibration on the training 80%.\n4. Compare with Gaussian Naive Bayes on the same held-out rows.\n5. Report ROC AUC, average precision, Brier score, recall among the top 20%, and demographic slices.")
            st.markdown("### Fields excluded from prediction")
            st.markdown("Customer ID, gender, senior-citizen status, partner status, and dependent status. Service, contract, tenure, and billing fields remain in the model and may still carry proxy information.")
        with method_right:
            st.markdown("### What this dataset cannot answer")
            st.markdown("- It is a single historical snapshot, not an account-by-month history or a forward-looking churn window.\n- It does not record whether a retention offer was made or whether it worked.\n- Monthly charges are used as a revenue proxy; costs, margin, and customer lifetime value are absent.\n- Model signals and individual contributions are associations, not causal explanations.\n- The calibration curve and subgroup metrics describe this test split only.")
            st.markdown("### Evidence needed before rollout")
            st.markdown("Use current data, define a future prediction window, check drift and subgroup outcomes, and run a randomized holdout to estimate incremental retention and net value.")
        st.link_button("IBM dataset source", "https://github.com/IBM/telco-customer-churn-on-icp4d")

