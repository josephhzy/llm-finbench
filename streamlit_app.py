"""
LLM Financial Stability Bench — Results Dashboard
Interactive visualisation of benchmark results. No API keys required.
All data is pre-computed from the evaluation run (20260325_024418_284976).
"""

import os
import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Financial Stability Bench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #f8f9fb; }

  section[data-testid="stMain"] h1,
  section[data-testid="stMain"] h2,
  section[data-testid="stMain"] h3,
  section[data-testid="stMain"] p,
  section[data-testid="stMain"] span,
  section[data-testid="stMain"] li,
  section[data-testid="stMain"] label,
  section[data-testid="stMain"] div[data-testid="stMarkdownContainer"] * {
    color: #1a1a1a !important;
  }

  section[data-testid="stSidebar"] {
    background-color: #040b14 !important;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] li,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] * {
    color: #f0f0f0 !important;
  }
  section[data-testid="stSidebar"] hr { border-color: #2a3a4a !important; }

  /* Fix code blocks — dark theme makes them black-on-dark */
  section[data-testid="stMain"] pre,
  section[data-testid="stMain"] pre *,
  section[data-testid="stMain"] code {
    color: #f0f0f0 !important;
    background-color: #1e2a3a !important;
  }
  section[data-testid="stSidebar"] code {
    color: #f0f0f0 !important;
    background-color: #1e2a3a !important;
  }

  .stat-box {
    background: #ffffff;
    border-radius: 10px;
    padding: 20px 24px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    text-align: center;
  }
  .stat-num   { font-size: 2rem; font-weight: 700; color: #173b6c; }
  .stat-label { font-size: 0.85rem; color: #555; margin-top: 4px; }

  .finding-card {
    background: #ffffff;
    border-left: 4px solid #149ddd;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 10px 0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
  }
  .finding-num { font-size: 0.75rem; font-weight: 700; color: #149ddd;
                 text-transform: uppercase; letter-spacing: 1px; }
  .finding-title { font-size: 1rem; font-weight: 700; color: #173b6c; margin: 4px 0 6px; }
  .finding-body  { font-size: 0.88rem; color: #444; line-height: 1.6; }

  .flag-green  { color: #2e7d32; font-weight: 700; }
  .flag-yellow { color: #e65100; font-weight: 700; }
  .flag-red    { color: #c62828; font-weight: 700; }

  /* Formula box — must override the !important wildcard above */
  .formula-box, .formula-box * {
    color: #1a1a1a !important;
    background-color: #eef2f7 !important;
  }

  /* Tab labels — dark theme makes them white on light bg */
  button[data-baseweb="tab"] p,
  button[data-baseweb="tab"] span,
  div[data-baseweb="tab-list"] button span {
    color: #1a1a1a !important;
  }

  #MainMenu { visibility: hidden; }
  footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Embedded fallback data ───────────────────────────────────────────────────
# Pre-computed from evaluation run 20260325_024418_284976

_BY_TEMPLATE_CSV = """template,semantic_score,factual_score,hallucination_rate,composite_stability
comparative,0.9152,0.3993,0.9127,0.4604
contextual_extraction,0.9886,0.9976,0.4848,0.8502
direct_extraction,0.8441,0.3725,0.9121,0.4286
qualitative,0.9096,0.4689,0.9055,0.4888"""

_BY_COMPANY_CSV = """company,semantic_score,factual_score,hallucination_rate,composite_stability
CapitaLand,0.9098,0.5265,0.9467,0.4995
DBS,0.9078,0.5948,0.7500,0.5853
OCBC,0.9339,0.5641,0.6571,0.6087
Singtel,0.9007,0.5339,0.9133,0.5098
UOB,0.9169,0.5701,0.7879,0.5668"""

_BY_TEMPERATURE_CSV = """temperature,semantic_score,factual_score,hallucination_rate,composite_stability
0.0,0.9947,0.8076,0.7977,0.6821
0.3,0.9309,0.5582,0.7985,0.5630
0.5,0.8994,0.5258,0.8076,0.5379
0.7,0.8848,0.4565,0.7985,0.5085
1.0,0.8619,0.4497,0.8167,0.4935"""

_BY_CATEGORY_CSV = """category,semantic_score,factual_score,hallucination_rate,composite_stability
asset_quality,0.9277,0.5206,0.7067,0.5746
aum,0.9350,0.4483,1.0000,0.4598
balance_sheet,0.8778,0.6062,1.0000,0.5058
capital,0.9544,0.5572,0.6233,0.6222
efficiency,0.9231,0.5248,0.6650,0.5874
income,0.8903,0.6204,1.0000,0.5152
leverage,0.8798,0.5050,0.7350,0.5454
liquidity,0.8755,0.5589,0.5650,0.6167
operational,0.8912,0.6156,1.0000,0.5136
profitability,0.9381,0.5353,0.6031,0.6146
revenue,0.8955,0.5483,1.0000,0.4880
shareholder,0.9193,0.5045,0.7450,0.5541"""


# ── Data loaders ─────────────────────────────────────────────────────────────
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


@st.cache_data
def load_by_template() -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, "by_template.csv")
    if os.path.exists(path):
        return pd.read_csv(path, comment="#")
    return pd.read_csv(io.StringIO(_BY_TEMPLATE_CSV))


@st.cache_data
def load_by_company() -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, "by_company.csv")
    if os.path.exists(path):
        return pd.read_csv(path, comment="#")
    return pd.read_csv(io.StringIO(_BY_COMPANY_CSV))


@st.cache_data
def load_by_temperature() -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, "by_temperature.csv")
    if os.path.exists(path):
        return pd.read_csv(path, comment="#")
    return pd.read_csv(io.StringIO(_BY_TEMPERATURE_CSV))


@st.cache_data
def load_by_category() -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, "by_metric_type.csv")
    if os.path.exists(path):
        return pd.read_csv(path, comment="#")
    return pd.read_csv(io.StringIO(_BY_CATEGORY_CSV))


@st.cache_data
def load_per_fact() -> pd.DataFrame | None:
    path = os.path.join(REPORTS_DIR, "per_fact_report.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, comment="#")
    return (
        df.groupby(["fact_id", "company", "metric", "category"])
        .agg(
            composite_stability=("composite_stability", "mean"),
            hallucination_rate=("hallucination_rate", "mean"),
            factual_score=("factual_score", "mean"),
            n_groups=("composite_stability", "count"),
        )
        .reset_index()
        .sort_values("composite_stability", ascending=False)
    )


# ── Colour helpers ────────────────────────────────────────────────────────────
def stability_color(val: float) -> str:
    if val >= 0.75:
        return "#2e7d32"
    if val >= 0.5:
        return "#e65100"
    return "#c62828"


def flag_emoji(val: float) -> str:
    if val >= 0.75:
        return "🟢"
    if val >= 0.5:
        return "🟡"
    return "🔴"


# ── Load data ─────────────────────────────────────────────────────────────────
df_template    = load_by_template()
df_company     = load_by_company()
df_temperature = load_by_temperature()
df_category    = load_by_category()
df_per_fact    = load_per_fact()

OVERALL_STABILITY = 0.5570
TOTAL_API_CALLS   = 6600
N_FACTS           = 33
N_COMPANIES       = 5
HALLUCINATION_RATE = 0.8038
GREEN_PCT  = 14.7
YELLOW_PCT = 35.5
RED_PCT    = 49.8

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 LLM Financial Stability Bench")
    st.markdown("**Benchmarking LLM consistency on SGX-listed company financials**")
    st.divider()

    st.markdown("### About This Project")
    st.markdown("""
How reliably does an LLM reproduce verified numerical facts from corporate
financial reports — and does that reliability hold under **different phrasings
and temperatures**?

This benchmark measured:
- **Semantic consistency** — are responses saying the same thing?
- **Factual accuracy** — do extracted numbers match ground truth?
- **Hallucination rate** — how often does the model fabricate values?

Each fact was tested across **4 prompt templates** × **5 temperature levels**
× **10 runs = 200 API calls per fact**, totalling **6,600 calls** across
33 verified financial facts.
""")
    st.divider()

    st.markdown("### Evaluation Setup")
    st.markdown("""
| Parameter | Value |
|---|---|
| Model | GPT-5.4-nano |
| Companies | DBS, OCBC, UOB, Singtel, CapitaLand |
| Source | FY2024 Annual Reports |
| Facts | 33 verified financial metrics |
| Templates | 4 prompt variants |
| Temperatures | 0.0, 0.3, 0.5, 0.7, 1.0 |
| Runs each | 10 |
| Embeddings | all-MiniLM-L6-v2 |
""")
    st.divider()

    st.markdown("### Stability Scale")
    st.markdown("""
<span style='color:#2e7d32;font-weight:700'>🟢 GREEN ≥ 0.75</span> — Reliable
<span style='color:#e65100;font-weight:700'>🟡 YELLOW ≥ 0.50</span> — Moderate
<span style='color:#c62828;font-weight:700'>🔴 RED < 0.50</span> — Unreliable
""", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Links")
    st.markdown("""
- [GitHub Repository](https://github.com/josephhzy/llm-finbench)
- [Portfolio](https://josephhzy.github.io)
""")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 📊 LLM Financial Stability Bench")
st.markdown(
    "Measuring LLM consistency and hallucination on verified financial facts "
    "from FY2024 annual reports of five SGX-listed companies."
)

# ── Hero stats ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, num, lbl in [
    (c1, f"{TOTAL_API_CALLS:,}", "API Calls Made"),
    (c2, str(N_FACTS),           "Facts Tested"),
    (c3, str(N_COMPANIES),       "Companies"),
    (c4, f"{OVERALL_STABILITY:.3f}", "Mean Stability (🟡 Yellow)"),
    (c5, f"{HALLUCINATION_RATE:.0%}", "Hallucination Rate"),
]:
    col.markdown(
        f'<div class="stat-box">'
        f'<div class="stat-num">{num}</div>'
        f'<div class="stat-label">{lbl}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Flag distribution ─────────────────────────────────────────────────────────
st.markdown("#### Overall Flag Distribution — 660 Evaluation Groups")
st.markdown(f"""
<div style="display:flex;gap:12px;margin:10px 0 4px;">
  <div style="flex:1;background:#e8f5e9;border-left:5px solid #2e7d32;border-radius:8px;padding:16px 20px;">
    <div style="font-size:1.6rem;font-weight:700;color:#2e7d32;">🟢 {GREEN_PCT}%</div>
    <div style="font-weight:700;color:#2e7d32;font-size:0.95rem;margin:4px 0 2px;">GREEN — Stable</div>
    <div style="color:#555;font-size:0.82rem;">Composite stability ≥ 0.75 &nbsp;·&nbsp; 97 of 660 groups</div>
  </div>
  <div style="flex:1;background:#fff8e1;border-left:5px solid #e65100;border-radius:8px;padding:16px 20px;">
    <div style="font-size:1.6rem;font-weight:700;color:#e65100;">🟡 {YELLOW_PCT}%</div>
    <div style="font-weight:700;color:#e65100;font-size:0.95rem;margin:4px 0 2px;">YELLOW — Moderate</div>
    <div style="color:#555;font-size:0.82rem;">Composite stability ≥ 0.50 &nbsp;·&nbsp; 234 of 660 groups</div>
  </div>
  <div style="flex:1;background:#fce4ec;border-left:5px solid #c62828;border-radius:8px;padding:16px 20px;">
    <div style="font-size:1.6rem;font-weight:700;color:#c62828;">🔴 {RED_PCT}%</div>
    <div style="font-weight:700;color:#c62828;font-size:0.95rem;margin:4px 0 2px;">RED — Unreliable</div>
    <div style="color:#555;font-size:0.82rem;">Composite stability &lt; 0.50 &nbsp;·&nbsp; 329 of 660 groups</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Key Findings ──────────────────────────────────────────────────────────────
st.markdown("## Key Findings")

findings = [
    (
        "01",
        "Prompt wording matters more than temperature",
        "The <b>contextual_extraction</b> template scored <b>0.85</b>. "
        "The <b>direct_extraction</b> template scored <b>0.43</b>. "
        "That's nearly double the stability from the same model on the same facts, "
        "purely from prompt design — before touching temperature at all.",
    ),
    (
        "02",
        "LLMs handle percentages better than large numbers",
        "Net Interest Margin and Return on Equity were the most stable facts across all companies. "
        "CapitaLand's AUM and Singtel's revenue — both large absolute figures in the hundreds of billions — "
        "were the least stable. The model consistently struggles with large-scale numbers, "
        "which is precisely the data type that matters most in financial analysis.",
    ),
    (
        "03",
        "Lower temperature helps, but doesn't solve it",
        "T=0.0 scored <b>0.68</b> vs T=1.0 at <b>0.49</b>. "
        "Setting temperature to zero makes the model more deterministic — "
        "but it still hallucinates. It just hallucinates the <em>same wrong answer</em> consistently, "
        "which is a different problem than random variance, but still a problem.",
    ),
    (
        "04",
        "Unit normalisation is a first-class evaluation concern",
        "80% of responses were flagged as hallucinations. But many flags came from unit mismatches, "
        "not model errors. A model answering 'S$3 billion' is factually correct — but if ground truth "
        "is stored as 3000 (SGD millions), a naive comparator flags it. "
        "Evaluation integrity requires unit normalisation, not just string matching.",
    ),
]

for num, title, body in findings:
    st.markdown(
        f'<div class="finding-card">'
        f'<div class="finding-num">Finding {num}</div>'
        f'<div class="finding-title">{title}</div>'
        f'<div class="finding-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("## Results by Dimension")
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 By Template",
    "🏢 By Company",
    "🌡️ By Temperature",
    "📂 By Metric Category",
])

CHART_H = 380
TICK_FONT = dict(size=12, color="#1a1a1a")
AXIS_STYLE = dict(showgrid=True, gridcolor="#eeeeee", tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a"))
CHART_FONT = dict(color="#1a1a1a")


def _chart_colors(df: pd.DataFrame, col: str = "composite_stability") -> list[str]:
    return [stability_color(v) for v in df[col]]


with tab1:
    st.markdown("##### Composite Stability Score by Prompt Template")
    st.caption(
        "Four templates were tested: direct question, contextual (with surrounding context), "
        "comparative (vs benchmark), and qualitative (descriptive framing)."
    )
    df_t = df_template.sort_values("composite_stability")
    fig = go.Figure(go.Bar(
        x=df_t["composite_stability"],
        y=df_t["template"],
        orientation="h",
        marker_color=_chart_colors(df_t),
        text=[f"{v:.3f}" for v in df_t["composite_stability"]],
        textposition="outside",
        width=0.5,
    ))
    fig.add_vline(x=0.75, line_dash="dash", line_color="#2e7d32", annotation_text="Green threshold",
                  annotation_position="top right")
    fig.add_vline(x=0.50, line_dash="dash", line_color="#e65100", annotation_text="Yellow threshold",
                  annotation_position="top right")
    fig.update_layout(
        height=CHART_H, xaxis=dict(range=[0, 1.05], **AXIS_STYLE), yaxis=AXIS_STYLE,
        plot_bgcolor="#f8f9fb", paper_bgcolor="#f8f9fb",
        margin=dict(l=0, r=60, t=20, b=20), showlegend=False,
        xaxis_title="Composite Stability", yaxis_title=None,
        font=CHART_FONT,
    )
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Detailed scores by template**")
        display_t = df_template[["template", "composite_stability", "factual_score",
                                  "semantic_score", "hallucination_rate"]].copy()
        display_t.columns = ["Template", "Stability", "Factual Score",
                              "Semantic Score", "Hallucination Rate"]
        display_t["Flag"] = display_t["Stability"].apply(flag_emoji)
        st.dataframe(display_t.set_index("Template").round(3), use_container_width=True)
    with col_b:
        st.markdown("**What the templates look like**")
        st.markdown("""
| Template | Prompt style |
|---|---|
| `contextual_extraction` | Gives the model surrounding context before asking for a number |
| `qualitative` | Asks for a descriptive / framing answer |
| `comparative` | Asks the model to compare vs a peer or benchmark |
| `direct_extraction` | Bare question, no context, no framing |
""")


with tab2:
    st.markdown("##### Composite Stability Score by Company")
    st.caption(
        "Stability is averaged across all facts, templates, and temperatures for each company."
    )
    df_c = df_company.sort_values("composite_stability")
    fig = go.Figure(go.Bar(
        x=df_c["composite_stability"],
        y=df_c["company"],
        orientation="h",
        marker_color=_chart_colors(df_c),
        text=[f"{v:.3f}" for v in df_c["composite_stability"]],
        textposition="outside",
        width=0.5,
    ))
    fig.add_vline(x=0.50, line_dash="dash", line_color="#e65100")
    fig.update_layout(
        height=CHART_H, xaxis=dict(range=[0, 0.75], **AXIS_STYLE), yaxis=AXIS_STYLE,
        plot_bgcolor="#f8f9fb", paper_bgcolor="#f8f9fb",
        margin=dict(l=0, r=60, t=20, b=20), showlegend=False,
        xaxis_title="Composite Stability", yaxis_title=None,
        font=CHART_FONT,
    )
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

    col_a, col_b = st.columns(2)
    with col_a:
        display_c = df_company[["company", "composite_stability", "factual_score",
                                 "hallucination_rate"]].copy()
        display_c.columns = ["Company", "Stability", "Factual Score", "Hallucination Rate"]
        display_c["Flag"] = display_c["Stability"].apply(flag_emoji)
        st.dataframe(display_c.set_index("Company").round(3), use_container_width=True)
    with col_b:
        st.info(
            "**Spread: 0.11** between highest (OCBC 0.609) and lowest (CapitaLand 0.500). "
            "CapitaLand's low score is partly driven by large-scale AUM / FUM metrics "
            "that the model consistently struggles with."
        )


with tab3:
    st.markdown("##### Composite Stability Score vs Temperature")
    st.caption(
        "Each data point is the mean stability across all 33 facts and all 4 templates at that temperature."
    )
    df_temp = df_temperature.copy()
    df_temp["temperature"] = df_temp["temperature"].astype(float)
    df_temp = df_temp.sort_values("temperature")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_temp["temperature"], y=df_temp["composite_stability"],
        mode="lines+markers+text",
        line=dict(color="#149ddd", width=2),
        marker=dict(size=10, color=[stability_color(v) for v in df_temp["composite_stability"]]),
        text=[f"{v:.3f}" for v in df_temp["composite_stability"]],
        textposition="top center",
        name="Composite Stability",
    ))
    fig.add_trace(go.Scatter(
        x=df_temp["temperature"], y=df_temp["factual_score"],
        mode="lines+markers",
        line=dict(color="#9c27b0", width=2, dash="dot"),
        marker=dict(size=8, color="#9c27b0"),
        name="Factual Score",
    ))
    fig.add_trace(go.Scatter(
        x=df_temp["temperature"], y=df_temp["semantic_score"],
        mode="lines+markers",
        line=dict(color="#4caf50", width=2, dash="dash"),
        marker=dict(size=8, color="#4caf50"),
        name="Semantic Consistency",
    ))
    fig.add_hrect(y0=0.75, y1=1.0, fillcolor="#2e7d32", opacity=0.05, line_width=0,
                  annotation_text="Green zone", annotation_position="right")
    fig.add_hrect(y0=0.5, y1=0.75, fillcolor="#e65100", opacity=0.05, line_width=0,
                  annotation_text="Yellow zone", annotation_position="right")
    fig.add_hrect(y0=0, y1=0.5, fillcolor="#c62828", opacity=0.05, line_width=0,
                  annotation_text="Red zone", annotation_position="right")
    fig.update_layout(
        height=CHART_H, xaxis=dict(title="Temperature", **AXIS_STYLE),
        yaxis=dict(title="Score", range=[0.3, 1.05], **AXIS_STYLE),
        plot_bgcolor="#f8f9fb", paper_bgcolor="#f8f9fb",
        margin=dict(l=0, r=80, t=20, b=20),
        legend=dict(x=0.75, y=1.0, font=dict(color="#1a1a1a")),
        font=CHART_FONT,
    )
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

    col_a, col_b = st.columns(2)
    with col_a:
        display_temp = df_temp[["temperature", "composite_stability", "factual_score",
                                 "semantic_score", "hallucination_rate"]].copy()
        display_temp.columns = ["Temperature", "Stability", "Factual Score",
                                 "Semantic Score", "Hallucination Rate"]
        st.dataframe(display_temp.set_index("Temperature").round(3), use_container_width=True)
    with col_b:
        st.info(
            "**Temperature spread: 0.19** between T=0.0 (0.682) and T=1.0 (0.493). "
            "Semantic consistency drops sharply with temperature; "
            "factual accuracy degrades even faster. "
            "But even at T=0.0 the model hallucinates 80% of the time — "
            "just *consistently*."
        )


with tab4:
    st.markdown("##### Composite Stability Score by Metric Category")
    st.caption(
        "Facts are grouped by financial metric type. AUM, revenue, and balance sheet metrics "
        "show the lowest stability, while capital ratios and liquidity metrics are most reliable."
    )
    df_cat = df_category.sort_values("composite_stability")
    fig = go.Figure(go.Bar(
        x=df_cat["composite_stability"],
        y=df_cat["category"].str.replace("_", " ").str.title(),
        orientation="h",
        marker_color=_chart_colors(df_cat),
        text=[f"{v:.3f}" for v in df_cat["composite_stability"]],
        textposition="outside",
        width=0.55,
    ))
    fig.add_vline(x=0.75, line_dash="dash", line_color="#2e7d32")
    fig.add_vline(x=0.50, line_dash="dash", line_color="#e65100")
    fig.update_layout(
        height=480, xaxis=dict(range=[0, 0.85], **AXIS_STYLE), yaxis=AXIS_STYLE,
        plot_bgcolor="#f8f9fb", paper_bgcolor="#f8f9fb",
        margin=dict(l=0, r=60, t=20, b=20), showlegend=False,
        xaxis_title="Composite Stability", yaxis_title=None,
        font=CHART_FONT,
    )
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})

# ── Per-Fact Explorer ─────────────────────────────────────────────────────────
st.divider()
st.markdown("## Per-Fact Explorer")

if df_per_fact is not None:
    st.caption(
        f"{len(df_per_fact)} unique facts — each score is the mean across all 20 template × temperature combinations."
    )

    col_filter1, col_filter2, _ = st.columns([2, 2, 4])
    with col_filter1:
        company_filter = st.multiselect(
            "Filter by company",
            options=sorted(df_per_fact["company"].unique()),
            default=[],
            placeholder="All companies",
        )
    with col_filter2:
        cat_filter = st.multiselect(
            "Filter by category",
            options=sorted(df_per_fact["category"].unique()),
            default=[],
            placeholder="All categories",
        )

    filtered = df_per_fact.copy()
    if company_filter:
        filtered = filtered[filtered["company"].isin(company_filter)]
    if cat_filter:
        filtered = filtered[filtered["category"].isin(cat_filter)]

    filtered["flag"] = filtered["composite_stability"].apply(flag_emoji)
    display_pf = filtered[[
        "flag", "company", "metric", "category",
        "composite_stability", "factual_score", "hallucination_rate",
    ]].rename(columns={
        "flag": "",
        "company": "Company",
        "metric": "Metric",
        "category": "Category",
        "composite_stability": "Stability",
        "factual_score": "Factual Score",
        "hallucination_rate": "Hallucination Rate",
    })
    st.dataframe(
        display_pf.set_index("").style.format({
            "Stability": "{:.3f}",
            "Factual Score": "{:.3f}",
            "Hallucination Rate": "{:.1%}",
        }),
        use_container_width=True,
        height=420,
    )
else:
    st.info(
        "Per-fact detail data (`reports/per_fact_report.csv`) is not included in this deployment. "
        "Clone the repository and run the evaluation to generate it, "
        "or view the full results on [GitHub](https://github.com/josephhzy/llm-finbench)."
    )
    st.markdown("""
**Summary of top/bottom facts from the evaluation run:**

| Rank | Fact | Company | Metric | Stability |
|------|------|---------|--------|-----------|
| 🟡 1 | ocbc_fy2024_roe | OCBC | Return on Equity | 0.7235 |
| 🟡 2 | ocbc_fy2024_nim | OCBC | Net Interest Margin | 0.7088 |
| 🟡 3 | uob_fy2024_nim | UOB | Net Interest Margin | 0.6896 |
| 🟡 4 | dbs_fy2024_cet1 | DBS | Common Equity Tier 1 | 0.6483 |
| 🟡 5 | dbs_fy2024_nim | DBS | Net Interest Margin | 0.6220 |
| ... | *23 more facts* | ... | ... | ... |
| 🔴 29 | capitaland_fy2024_total_assets | CapitaLand | Total Assets | 0.4906 |
| 🔴 30 | singtel_fy2024_revenue | Singtel | Revenue | 0.4880 |
| 🔴 31 | capitaland_fy2024_fum | CapitaLand | Funds Under Management | 0.4721 |
| 🔴 32 | singtel_fy2024_net_profit | Singtel | Net Profit | 0.4593 |
| 🔴 33 | capitaland_fy2024_aum | CapitaLand | Assets Under Management | 0.4476 |
""")

# ── Methodology ───────────────────────────────────────────────────────────────
st.divider()
st.markdown("## Scoring Methodology")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown("**Composite Stability Formula**")
    st.markdown(
        '<div class="formula-box" style="border-radius:6px;padding:12px 16px;font-family:monospace;'
        'font-size:13px;line-height:1.7;margin:8px 0 12px;border:1px solid #d0d7e3;">'
        'stability = 0.3 × semantic_score<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0.4 × factual_score<br>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ 0.3 × (1 − hallucination_rate)'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("Factual score is weighted highest because exact number reproduction is the core task.")
with col_m2:
    st.markdown("""
**Factual Accuracy**
Uses an independent regex-based extraction
stage — the model under test never scores
its own outputs, eliminating evaluator bias.

Modal value across 10 runs is compared to
ground truth with a ±5% relative tolerance.
""")
with col_m3:
    st.markdown("""
**Semantic Consistency**
Cosine similarity between sentence-transformer
embeddings (all-MiniLM-L6-v2) of the 10 runs.

High semantic score + low factual score =
*confidently wrong* answers — the most
dangerous failure mode in production.
""")
