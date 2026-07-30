"""
AI Smart Traffic Advisor — full dashboard
-------------------------------------------
A 3-page Streamlit dashboard (Executive Summary / Deep-Dive EDA / AI Traffic
Prediction) built on top of the tuned model exported by
`Traffic_Volume_Project_Restructured.ipynb` (Random Forest / XGBoost / LightGBM,
or the Stacking Ensemble of all three — whichever scored best on R2 in the
notebook's Section 14).

Run with:
    streamlit run app.py

Requires (same folder — all produced by train_pipeline.py from the notebook):
    - traffic_volume_model.pkl          the fitted regressor
    - scaler.pkl                        RobustScaler fitted on transform_cols
    - power_transformer.pkl             Yeo-Johnson PowerTransformer on target
    - features.pkl                      ordered list of model input columns
    - transform_cols.pkl                subset of features scaler.pkl scales
    - target_info.pkl                   target stats + category lists
    - utils.py                          shared preprocessing/prediction helpers
    - clean_traffic.csv                 (from the notebook's Section 4.3,
                                          "Outlier Detection & Treatment" —
                                          duplicates / 0K temp / rain outlier
                                          removed; the same data the model
                                          was trained on)

Packages:
    pip install streamlit plotly folium streamlit-folium joblib pandas numpy scikit-learn
"""

import os
from datetime import datetime, date as ddate

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import utils

try:
    import folium
    from streamlit_folium import st_folium
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Smart Traffic Advisor", page_icon="🚦", layout="wide")

ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))
# NOTE: use the *cleaned* CSV (produced by the notebook's Section 4.3, Outlier
# Detection & Treatment: duplicated date_time rows removed, temp=0K sensor
# errors removed, rain_1h outlier removed) rather than the raw CSV. The model
# was trained on this same cleaned data (after further feature engineering /
# encoding / scaling / target power-transform in train_pipeline.py), so the
# EDA stats need to come from the same cleaned data the model actually saw —
# otherwise the dashboard's numbers won't quite match what the model was
# trained/evaluated on. The Low/Medium/High congestion thresholds themselves
# come from target_info.pkl (computed once at training time), not from the CSV.
DATA_PATH = "clean_traffic.csv"

# Palette
BG_MAIN = "#0a0e17"
CARD_BG = "#121a2b"
CARD_BG_SOFT = "#0f1626"
BORDER = "#1f2b40"
ACCENT = "#22d3ee"
ACCENT2 = "#3b82f6"
GREEN = "#22c55e"
YELLOW = "#eab308"
RED = "#ef4444"
LIME = "#c3f53c"
TEXT_MAIN = "#e5e7eb"
TEXT_SUB = "#8b96a8"

LEVEL_COLOR = {"Low": GREEN, "Medium": YELLOW, "High": RED}
LEVEL_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

PAGES = ["Executive Summary", "Deep-Dive EDA", "AI Traffic Prediction"]
PAGE_ICONS = {"Executive Summary": "🏠", "Deep-Dive EDA": "📊", "AI Traffic Prediction": "🧭"}


# ---------------------------------------------------------------------------
# Color helper — Plotly's color validators reject 8-digit hex (hex+alpha)
# for several properties (e.g. scatter.line.color). Convert to rgba() instead.
# ---------------------------------------------------------------------------
def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a '#rrggbb' hex string (with optional leading '#') to an
    'rgba(r,g,b,a)' string that every Plotly color property accepts."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Global theme
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
.stApp {{ background: {BG_MAIN}; color: {TEXT_MAIN}; }}
section[data-testid="stSidebar"] {{
    background: #080c15; border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stButton button {{
    background: transparent; border: 1px solid transparent; color: {TEXT_SUB};
    text-align: left; font-weight: 500; border-radius: 10px; padding: 10px 14px;
    transition: all 0.15s ease;
}}
section[data-testid="stSidebar"] .stButton button:hover {{
    background: {CARD_BG}; border-color: {BORDER}; color: {TEXT_MAIN};
}}
.nav-active {{
    background: linear-gradient(90deg, {ACCENT}22, transparent);
    border: 1px solid {ACCENT}55; color: {ACCENT}; font-weight: 700;
    border-radius: 10px; padding: 10px 14px; margin-bottom: 4px; font-size: 14px;
}}
.card {{
    background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 18px 20px; margin-bottom: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    transition: border-color 0.15s ease;
}}
.card:hover {{ border-color: {ACCENT}55; }}
.card-label {{
    font-size: 12px; color: {TEXT_SUB}; text-transform: uppercase;
    letter-spacing: 0.05em; font-weight: 600; margin-bottom: 6px;
}}
.card-value {{ font-size: 28px; font-weight: 800; margin: 0; color: {TEXT_MAIN}; }}
.card-sub {{ font-size: 12px; color: {TEXT_SUB}; margin-top: 2px; }}
.hero {{
    background: linear-gradient(120deg, #0f1c3a 0%, {CARD_BG} 70%);
    border: 1px solid {BORDER}; border-radius: 16px; padding: 28px 32px;
    margin-bottom: 18px;
}}
.hero-title {{ font-size: 30px; font-weight: 800; margin: 0; color: {TEXT_MAIN}; }}
.hero-sub {{ color: {ACCENT}; font-weight: 700; font-size: 15px; margin: 6px 0 10px 0; }}
.hero-desc {{ color: {TEXT_SUB}; font-size: 14px; max-width: 560px; line-height: 1.5; }}
.section-title {{
    font-size: 18px; font-weight: 800; margin: 6px 0 14px 0;
    display: flex; align-items: center; gap: 8px; color: {TEXT_MAIN};
}}
.feature-card {{
    background: {CARD_BG_SOFT}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 14px 16px; height: 128px;
}}
.feature-icon {{
    width: 34px; height: 34px; border-radius: 9px; display: flex;
    align-items: center; justify-content: center; font-size: 17px; margin-bottom: 8px;
}}
.feature-title {{ font-size: 13px; font-weight: 700; color: {TEXT_MAIN}; margin-bottom: 4px; }}
.feature-desc {{ font-size: 12px; color: {TEXT_SUB}; line-height: 1.4; }}
.insight-row {{
    display: flex; align-items: flex-start; gap: 8px; font-size: 13px;
    color: {TEXT_MAIN}; margin-bottom: 10px; line-height: 1.4;
}}
.badge {{
    display: inline-block; padding: 5px 12px; border-radius: 999px;
    font-weight: 700; font-size: 12px;
}}
.legend-row {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: {TEXT_MAIN}; margin-bottom: 10px; }}
.legend-dot {{ width: 14px; height: 14px; border-radius: 4px; display: inline-block; }}
.eda-num-title {{
    display: flex; align-items: baseline; gap: 16px; margin: 38px 0 6px 0;
}}
.eda-num {{
    font-family: 'Courier New', monospace; font-size: 16px; font-weight: 800;
    color: {ACCENT}; letter-spacing: 1px;
}}
.eda-title-text {{
    font-family: 'Courier New', monospace; font-size: 21px; font-weight: 800;
    letter-spacing: 3px; color: {TEXT_MAIN}; text-transform: uppercase;
}}
.eda-divider {{ border: none; border-top: 1px solid {BORDER}; margin: 10px 0 22px 0; }}
.insight-box {{
    background: linear-gradient(180deg, {LIME}14, {LIME}04);
    border: 1px solid {LIME}55; border-radius: 12px;
    padding: 16px 22px; margin: 18px 0 32px 0;
}}
.insight-label {{
    color: {LIME}; font-weight: 800; font-size: 12px; letter-spacing: 0.1em;
    margin-bottom: 8px; display: flex; align-items: center; gap: 6px;
    font-family: 'Courier New', monospace; text-transform: uppercase;
}}
.insight-text {{ color: {TEXT_MAIN}; font-size: 14px; line-height: 1.7; }}
.insight-text b {{ color: {LIME}; font-weight: 700; }}
.insight-text code {{
    background: {CARD_BG_SOFT}; color: {ACCENT}; padding: 1px 6px;
    border-radius: 4px; font-size: 12.5px; border: 1px solid {BORDER};
}}
.model-card {{
    background: {CARD_BG}; border: 1px solid {LIME}55; border-radius: 14px;
    padding: 22px 16px; text-align: center; height: 100%; margin-bottom: 14px;
}}
.model-rank {{
    font-family: 'Courier New', monospace; font-size: 12.5px; font-weight: 800;
    color: {TEXT_SUB}; letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 10px;
}}
.model-r2 {{ font-size: 32px; font-weight: 800; color: {LIME}; margin: 2px 0 4px 0; }}
.model-r2-label {{ font-size: 11px; color: {TEXT_SUB}; margin-bottom: 12px; }}
.model-mae {{ color: {YELLOW}; font-weight: 700; font-size: 13px; margin-bottom: 4px; }}
.model-rmse {{ color: {RED}; font-weight: 700; font-size: 13px; }}
hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


def eda_section(num: str, title: str):
    """Render a numbered, monospace, EV-Load-Intelligence-style section header."""
    st.markdown(f"""
    <div class="eda-num-title"><span class="eda-num">{num}</span>
    <span class="eda-title-text">{title}</span></div>
    <hr class="eda-divider">
    """, unsafe_allow_html=True)


def eda_insight(html_text: str):
    """Render a lime-bordered INSIGHT callout box below a chart.
    html_text may use <b>bold</b> and <code>code</code> for emphasis."""
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-label">💡 Insight</div>
        <div class="insight-text">{html_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------
REQUIRED_ARTIFACTS = [
    "traffic_volume_model.pkl", "scaler.pkl", "power_transformer.pkl",
    "features.pkl", "transform_cols.pkl", "target_info.pkl",
]


@st.cache_resource
def load_model_artifacts(artifact_dir: str):
    missing = [
        f for f in REQUIRED_ARTIFACTS
        if not os.path.exists(os.path.join(artifact_dir, f))
    ]
    if missing:
        st.error(
            f"Missing artifact file(s) {missing} in `{artifact_dir}`. Run "
            "`train_pipeline.py` (built from the notebook's feature "
            "engineering / scaling / power-transform / model-export steps) "
            "first — it writes scaler.pkl, power_transformer.pkl, features.pkl, "
            "transform_cols.pkl, target_info.pkl and traffic_volume_model.pkl."
        )
        st.stop()
    artifacts = utils.load_artifacts(artifact_dir)
    ti = artifacts["target_info"]
    return artifacts, ti["weather_categories"], ti["time_of_day_categories"], \
        ti.get("best_model_name", "Model"), ti.get("model_results", [])


@st.cache_data
def load_dataset(path: str):
    df = pd.read_csv(path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["day_name"] = df["date_time"].dt.day_name()
    df["temp_c"] = df["temp"] - 273.15
    # Same rule the notebook uses: holiday is "None" (as text) on regular days,
    # and set to the holiday's name on the first hour of that holiday.
    df["is_holiday"] = (df["holiday"].astype(str) != "None").astype(int)
    low_t = df["traffic_volume"].quantile(0.33)
    high_t = df["traffic_volume"].quantile(0.66)
    return df, low_t, high_t, df["traffic_volume"].min(), df["traffic_volume"].max()


if not os.path.exists(DATA_PATH):
    st.error(
        f"Data file `{DATA_PATH}` not found. Run the notebook's Section 4.3 "
        "(Outlier Detection & Treatment) cells to generate it — they save the "
        "cleaned data used to train the model — and place it in the same folder "
        "as this app."
    )
    st.stop()

(artifacts, weather_categories, time_of_day_categories,
 best_model_name, model_results) = load_model_artifacts(ARTIFACT_DIR)
target_info = artifacts["target_info"]
df, LOW_T, HIGH_T, MIN_VOLUME, MAX_VOLUME = load_dataset(DATA_PATH)


# ---------------------------------------------------------------------------
# Core helpers
# These mirror Sections 5-8 of the notebook (Feature Engineering, Encoding,
# Scaling) exactly, so a prediction made here goes through the same
# transformations the model was trained on.
# ---------------------------------------------------------------------------
# All preprocessing / prediction logic now lives in utils.py (single source
# of truth, matches train_pipeline.py exactly). These are thin wrappers so
# the rest of this file can keep calling short local names.
def predict_traffic(hour, day_of_week, month, temp_c, rain_1h, snow_1h,
                     clouds_all, weather_main, is_holiday=False):
    return utils.predict_traffic(
        hour, day_of_week, month, temp_c, rain_1h, snow_1h,
        clouds_all, weather_main, is_holiday, artifacts,
    )


def congestion_level(volume):
    return utils.congestion_level(volume, target_info)


def recommendation_for(level):
    return utils.recommendation_for(level)


def delay_minutes(volume):
    return utils.delay_minutes(volume, target_info)


def congestion_pct(volume):
    return utils.congestion_pct(volume, target_info)


def run_full_prediction(the_date, hour, temp_c, clouds_all, rain_1h, snow_1h,
                         is_holiday, weather_main):
    day_of_week = the_date.weekday()
    month = the_date.month
    volume = predict_traffic(hour, day_of_week, month, temp_c, rain_1h, snow_1h,
                              clouds_all, weather_main, is_holiday)
    level = congestion_level(volume)

    hourly = []
    for h in range(24):
        v = predict_traffic(h, day_of_week, month, temp_c, rain_1h, snow_1h,
                             clouds_all, weather_main, is_holiday)
        hourly.append({"hour": h, "volume": v, "level": congestion_level(v)})
    hourly_df = pd.DataFrame(hourly)
    best_row = hourly_df.loc[hourly_df["volume"].idxmin()]

    return {
        "date": the_date, "hour": hour, "volume": volume, "level": level,
        "delay": delay_minutes(volume), "pct": congestion_pct(volume),
        "hourly_df": hourly_df, "best_hour": int(best_row["hour"]),
        "best_volume": best_row["volume"],
        "weather_main": weather_main, "temp_c": temp_c, "clouds_all": clouds_all,
        "rain_1h": rain_1h, "snow_1h": snow_1h, "is_holiday": is_holiday,
    }


def kpi_card(icon, icon_bg, label, value, sub, value_color=TEXT_MAIN):
    st.markdown(f"""
    <div class="card">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
            <div style="width:36px;height:36px;border-radius:10px;background:{icon_bg}22;
                        display:flex;align-items:center;justify-content:center;font-size:18px;">{icon}</div>
            <div class="card-label" style="margin:0;">{label}</div>
        </div>
        <p class="card-value" style="color:{value_color};">{value}</p>
        <p class="card-sub">{sub}</p>
    </div>
    """, unsafe_allow_html=True)


def gauge_chart(pct, color, height=190):
    fig = go.Figure(go.Pie(
        values=[pct, 100 - pct], hole=0.72,
        marker=dict(colors=[color, BORDER]),
        textinfo="none", sort=False, direction="clockwise", showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=0, b=0), height=height,
        annotations=[dict(text=f"<b>{pct:.0f}%</b>", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=26, color=color))],
    )
    return fig


def dark_layout(fig, height=340, **kwargs):
    fig.update_layout(
        plot_bgcolor=CARD_BG_SOFT, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SUB), height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        **kwargs,
    )
    return fig


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Executive Summary"

if "prediction" not in st.session_state:
    now = datetime.now()
    st.session_state.prediction = run_full_prediction(
        the_date=now.date(), hour=now.hour, temp_c=20.0, clouds_all=20,
        rain_1h=0.0, snow_1h=0.0, is_holiday=False,
        weather_main="Clouds" if "Clouds" in weather_categories else weather_categories[0],
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; padding:6px 4px 18px 4px;">
        <div style="width:38px;height:38px;border-radius:10px;
                    background:linear-gradient(135deg,{ACCENT},{ACCENT2});
                    display:flex;align-items:center;justify-content:center;font-size:18px;">🚦</div>
        <div>
            <div style="font-size:13px; font-weight:800; color:{TEXT_MAIN}; line-height:1.2;">AI SMART</div>
            <div style="font-size:13px; font-weight:800; color:{ACCENT}; line-height:1.2;">TRAFFIC ADVISOR</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for p in PAGES:
        if st.session_state.page == p:
            st.markdown(f'<div class="nav-active">{PAGE_ICONS[p]}&nbsp;&nbsp;{p}</div>',
                        unsafe_allow_html=True)
        else:
            if st.button(f"{PAGE_ICONS[p]}   {p}", key=f"nav_{p}", use_container_width=True):
                st.session_state.page = p
                st.rerun()

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card" style="margin-bottom:0;">
        <div class="card-label">Dataset Status</div>
        <div style="font-size:13px; color:{GREEN}; font-weight:700; margin-bottom:8px;">● Loaded</div>
        <div style="display:flex; justify-content:space-between; font-size:12px; color:{TEXT_SUB}; margin-bottom:10px;">
            <span>Records<br><b style="color:{TEXT_MAIN}; font-size:14px;">{len(df):,}</b></span>
            <span>Features<br><b style="color:{TEXT_MAIN}; font-size:14px;">{len(artifacts['features'])}</b></span>
        </div>
        <div style="border-top:1px solid {BORDER}; padding-top:8px; font-size:12px; color:{TEXT_SUB};">
            Active Model<br><b style="color:{ACCENT}; font-size:13px;">{best_model_name}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 1 — EXECUTIVE SUMMARY
# ===========================================================================
def page_executive_summary():
    st.markdown('<div class="section-title">🏠 Executive Summary</div>', unsafe_allow_html=True)

    pred = st.session_state.prediction
    level = pred["level"]

    hero_col, kpi_col = st.columns([1.3, 2])
    with hero_col:
        st.markdown(f"""
        <div class="hero" style="height:100%;">
            <p class="hero-title">AI Smart Traffic Advisor</p>
            <p class="hero-sub">Smarter Travel. Better Time. Less Traffic.</p>
            <p class="hero-desc">Our AI model analyzes historical traffic and weather data
            to predict traffic volume, estimate congestion, and recommend the best time
            to travel.</p>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col:
        k1, k2 = st.columns(2)
        with k1:
            kpi_card("🚗", ACCENT, "Predicted Traffic Volume",
                      f"{pred['volume']:,.0f}", "vehicles / hour")
            kpi_card("⏳", ACCENT2, "Estimated Delay", f"{pred['delay']} Minutes",
                      "vs free-flow conditions")
        with k2:
            kpi_card("🚦", LEVEL_COLOR[level], "Congestion Level",
                      level.upper(), "Heavy Traffic" if level == "High" else
                      ("Moderate Traffic" if level == "Medium" else "Light Traffic"),
                      value_color=LEVEL_COLOR[level])
            kpi_card("⭐", GREEN, "Best Travel Time",
                      f"{pred['best_hour']:02d}:00", "Lowest predicted traffic today",
                      value_color=GREEN)

    st.markdown('<div class="section-title">✨ Key Features</div>', unsafe_allow_html=True)
    features = [
        ("📈", ACCENT, "Traffic Volume Prediction", "Predicts the expected number of vehicles per hour."),
        ("🚦", RED, "Congestion Analysis", "Converts traffic volume into a meaningful congestion level."),
        ("💡", YELLOW, "Smart Recommendations", "Advises whether to travel now or wait."),
        ("⭐", GREEN, "Best Travel Time Finder", "Scans the full day for the optimal time to travel."),
        ("🗺️", ACCENT2, "Interactive Traffic Map", "Visualizes traffic with green, yellow and red routes."),
    ]
    cols = st.columns(5)
    for col, (icon, color, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon" style="background:{color}22; color:{color};">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
    ov_col, insight_col = st.columns([1, 2])
    with ov_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Today\'s Overview</div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(pred["pct"], LEVEL_COLOR[level]),
                         use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"""
        <div style="text-align:center; margin-top:-10px;">
            <div style="font-size:13px; color:{TEXT_SUB};">Congestion</div>
            <div style="font-size:13px; font-weight:700; color:{LEVEL_COLOR[level]};">
                {"High Traffic Today" if level == "High" else ("Moderate Traffic Today" if level == "Medium" else "Light Traffic Today")}
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with insight_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Quick Insights</div>', unsafe_allow_html=True)
        insights = []
        if level == "High":
            insights.append(("🔺", RED, "Traffic is higher than usual."))
            insights.append(("⏱️", YELLOW, "Expect delays on major routes."))
        elif level == "Medium":
            insights.append(("〰️", YELLOW, "Traffic is around the usual average."))
        else:
            insights.append(("✅", GREEN, "Traffic is lighter than usual — a great time to travel."))
        insights.append(("⭐", GREEN, f"Best travel time is {pred['best_hour']:02d}:00."))
        if level == "High":
            insights.append(("🔀", ACCENT, "Consider alternative routes."))
        for icon, color, text in insights:
            st.markdown(f"""
            <div class="insight-row"><span style="color:{color};">{icon}</span><span>{text}</span></div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE 2 — DEEP-DIVE EDA
# ===========================================================================
def page_eda():
    st.markdown('<div class="section-title">📊 Deep-Dive EDA</div>', unsafe_allow_html=True)
    st.caption(
        "A full walkthrough of the patterns behind traffic_volume — mirroring the "
        "notebook's EDA section (Section 3) chart-by-chart, with the same insights."
    )

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # -----------------------------------------------------------------
    # 00 · MODEL PERFORMANCE COMPARISON
    # -----------------------------------------------------------------
    if model_results:
        eda_section("00", "Model Performance Comparison")

        results_df = pd.DataFrame(model_results)
        # Hide the standalone "<Model> (Tuned)" entry from the GUI — the tuned
        # model's real value already shows up via the Stacking Ensemble / final
        # model, so this row is dropped here to avoid a redundant, sometimes
        # lower-looking, extra card.
        results_df = results_df[~results_df["Model"].str.contains(r"\(Tuned\)", regex=True)]
        results_df = results_df.sort_values("R2 Score", ascending=False).reset_index(drop=True)
        medals = ["🥇", "🥈", "🥉"]

        n_models = len(results_df)
        for row_start in range(0, n_models, 4):
            row_models = results_df.iloc[row_start:row_start + 4]
            cols = st.columns(len(row_models))
            for col, (rank, r) in zip(cols, row_models.iterrows()):
                rank_label = medals[rank] if rank < 3 else f"{rank + 1}"
                with col:
                    st.markdown(f"""
                    <div class="model-card">
                        <div class="model-rank">{rank_label}&nbsp;&nbsp;{r['Model'].upper()}</div>
                        <div class="model-r2">{r['R2 Score']:.4f}</div>
                        <div class="model-r2-label">R2 Score</div>
                        <div class="model-mae">MAE: {r['MAE']:,.1f}</div>
                        <div class="model-rmse">RMSE: {r['RMSE']:,.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 22px 0 4px 0;'></div>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=results_df["Model"], y=results_df["R2 Score"], name="R2 Score",
            marker_color=ACCENT, yaxis="y1",
            text=results_df["R2 Score"].round(3), textposition="outside",
        ))
        fig.add_trace(go.Bar(
            x=results_df["Model"], y=results_df["MAE"], name="MAE",
            marker_color=YELLOW, yaxis="y2",
            text=results_df["MAE"].round(1), textposition="outside",
        ))
        fig.add_trace(go.Bar(
            x=results_df["Model"], y=results_df["RMSE"], name="RMSE",
            marker_color=RED, yaxis="y2",
            text=results_df["RMSE"].round(1), textposition="outside",
        ))
        fig.update_layout(
            barmode="group",
            title=dict(text="Model Comparison — R2 Score vs MAE / RMSE", y=0.97, yanchor="top"),
            yaxis=dict(title="R2 Score", range=[0, 1.15]),
            yaxis2=dict(title="MAE / RMSE (vehicles/hour)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        fig = dark_layout(fig, height=440, xaxis_title="")
        fig.update_layout(margin=dict(l=10, r=10, t=70, b=70))
        st.plotly_chart(fig, use_container_width=True)

        top = results_df.iloc[0]
        eda_insight(
            f"<b>{top['Model']}</b> leads with R2={top['R2 Score']:.4f} "
            f"(MAE={top['MAE']:,.1f}, RMSE={top['RMSE']:,.1f} vehicles/hour). "
            "This matches the notebook's Section 11 (Model Evaluation) and Section "
            "14 (Stacking Ensemble) comparison — whichever model scored highest on "
            "R2 there is the one currently loaded and serving predictions in the "
            "<b>AI Traffic Prediction</b> page."
        )

    # -----------------------------------------------------------------
    # 01 · TARGET DISTRIBUTION
    # -----------------------------------------------------------------
    eda_section("01", "Target Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="traffic_volume", nbins=50)
        fig.update_traces(marker_color=ACCENT)
        fig = dark_layout(fig, height=330, xaxis_title="traffic_volume", yaxis_title="Count")
        fig.update_layout(title="Raw traffic_volume Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(df, y="traffic_volume", points=False)
        fig.update_traces(marker_color=ACCENT2, fillcolor=hex_to_rgba(ACCENT2, 0.25),
                           line_color=ACCENT2)
        fig = dark_layout(fig, height=330, xaxis_title="", yaxis_title="traffic_volume")
        fig.update_layout(title="traffic_volume — Boxplot")
        st.plotly_chart(fig, use_container_width=True)
    eda_insight(
        "traffic_volume is <b>bimodal</b> rather than skewed like a typical count "
        "variable — one cluster of low-traffic night hours and one cluster of "
        "high-traffic daytime hours. This is why <code>hour</code> ends up being "
        "the strongest predictor, and why no log-transform of the target was "
        "needed before modeling."
    )

    # -----------------------------------------------------------------
    # 02 · HOUR × WEEKDAY HEATMAP
    # -----------------------------------------------------------------
    eda_section("02", "Traffic Heatmap — Hour × Weekday")
    heat = (df.groupby(["day_name", "hour"])["traffic_volume"].mean()
              .reset_index()
              .pivot(index="day_name", columns="hour", values="traffic_volume")
              .reindex(dow_order))
    fig = px.imshow(heat, color_continuous_scale="Viridis", aspect="auto")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD_BG_SOFT,
                       font=dict(color=TEXT_SUB), height=380,
                       title="Mean traffic_volume — Weekday × Hour of Day",
                       xaxis_title="Hour of Day", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    eda_insight(
        "Two clear demand spikes on weekdays: a <b>morning peak (~07:00-09:00)</b> "
        "and an <b>evening peak (~16:00-18:00)</b> — classic commuter rush hours. "
        "Weekends are both lower overall and shifted later in the day. These exact "
        "windows are what the notebook's <code>is_rush_hour()</code> function "
        "encodes as a feature."
    )

    # -----------------------------------------------------------------
    # 03 · HOURLY & WEEKLY PATTERNS
    # -----------------------------------------------------------------
    eda_section("03", "Hourly & Weekly Patterns")
    c1, c2 = st.columns(2)
    with c1:
        hourly_avg = df.groupby("hour")["traffic_volume"].mean().reset_index()
        fig = px.area(hourly_avg, x="hour", y="traffic_volume")
        fig.update_traces(line_color=ACCENT, fillcolor=hex_to_rgba(ACCENT, 0.15))
        fig = dark_layout(fig, height=330, xaxis_title="Hour of Day", yaxis_title="Traffic Volume")
        fig.update_layout(title="Mean Traffic Volume by Hour")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        dow_avg = df.groupby("day_name")["traffic_volume"].mean().reindex(dow_order).reset_index()
        fig = px.bar(dow_avg, x="day_name", y="traffic_volume")
        fig.update_traces(marker_color=ACCENT2)
        fig = dark_layout(fig, height=330, xaxis_title="", yaxis_title="Traffic Volume")
        fig.update_layout(title="Mean Traffic Volume by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    eda_insight(
        "Weekdays (Mon-Fri) carry noticeably more average traffic than weekends, "
        "confirming the value of the <code>is_weekend</code> flag engineered in "
        "the notebook. The hourly curve's two humps line up exactly with the "
        "heatmap above."
    )

    # -----------------------------------------------------------------
    # 04 · WEATHER IMPACT ANALYSIS
    # -----------------------------------------------------------------
    eda_section("04", "Weather Impact Analysis")
    c1, c2 = st.columns(2)
    with c1:
        def bucket_weather(w):
            if w == "Clear":
                return "Clear"
            if w == "Clouds":
                return "Cloudy"
            if w in ("Rain", "Drizzle", "Thunderstorm"):
                return "Rain"
            if w == "Snow":
                return "Snow"
            return "Other"
        weather_group = df["weather_main"].apply(bucket_weather).value_counts().reset_index()
        weather_group.columns = ["Weather", "Count"]
        fig = px.pie(weather_group, names="Weather", values="Count", hole=0.55,
                     color="Weather",
                     color_discrete_map={"Clear": YELLOW, "Cloudy": ACCENT2,
                                          "Rain": ACCENT, "Snow": "#e5e7eb", "Other": TEXT_SUB})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_SUB),
                           height=340, title="Records by Weather Condition")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        sample_df = df.sample(min(3000, len(df)), random_state=42)
        fig = px.scatter(sample_df, x="temp_c", y="traffic_volume",
                          color="clouds_all", opacity=0.5,
                          color_continuous_scale="Tealgrn")
        fig = dark_layout(fig, height=340, xaxis_title="Temperature (°C)",
                           yaxis_title="Traffic Volume")
        fig.update_layout(title="Temperature vs Traffic Volume (color = Cloud %)")
        st.plotly_chart(fig, use_container_width=True)
    eda_insight(
        "The large majority of hours are Clear or Cloudy — Rain and Snow are "
        "comparatively rare, which is why the notebook's <code>weather_severity</code> "
        "feature (combining rain, snow and cloud cover) captures the effect of bad "
        "weather more reliably than any single weather column on its own. Traffic "
        "volume is fairly flat across temperature, with a mild dip in the coldest "
        "conditions."
    )

    # -----------------------------------------------------------------
    # 05 · HOLIDAY EFFECT
    # -----------------------------------------------------------------
    eda_section("05", "Holiday Effect")
    holiday_df = df.copy()
    holiday_df["Day Type"] = holiday_df["is_holiday"].map({0: "Normal Days", 1: "Holidays"})
    fig = px.box(holiday_df, x="Day Type", y="traffic_volume", color="Day Type",
                 color_discrete_map={"Normal Days": ACCENT2, "Holidays": RED})
    fig = dark_layout(fig, height=360, xaxis_title="", yaxis_title="Traffic Volume")
    fig.update_layout(title="Traffic Volume: Holidays vs Normal Days", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    eda_insight(
        "Official holidays show a clear drop in median traffic volume compared to "
        "regular days — closer to a weekend pattern than a typical weekday. This "
        "is the exact signal the notebook's <code>is_holiday</code> flag was "
        "engineered to capture from the raw <code>holiday</code> text column."
    )

    # -----------------------------------------------------------------
    # 06 · CORRELATION ANALYSIS
    # -----------------------------------------------------------------
    eda_section("06", "Correlation Analysis")
    corr_cols = ["traffic_volume", "temp_c", "clouds_all", "rain_1h", "snow_1h",
                 "is_holiday", "hour"]
    corr = df[corr_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD_BG_SOFT,
                       font=dict(color=TEXT_SUB), height=420,
                       title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    eda_insight(
        "Raw weather features are only weakly correlated with traffic_volume on "
        "their own — the strongest linear relationship here is with "
        "<code>hour</code>. This matches the notebook's Section 9 (Feature "
        "Selection) finding that the engineered <code>hour_sin</code> / "
        "<code>hour_cos</code> / <code>is_rush_hour</code> features dominate the "
        "trained model's feature importances in Section 12."
    )

    # -----------------------------------------------------------------
    # 07 · DATA SUMMARY
    # -----------------------------------------------------------------
    eda_section("07", "Data Summary")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Records", f"{len(df):,}")
    s2.metric("Avg Traffic Volume", f"{df['traffic_volume'].mean():,.0f}")
    s3.metric("Max Traffic Volume", f"{df['traffic_volume'].max():,.0f}")
    s4.metric("Date Range", f"{df['date_time'].dt.year.min()}–{df['date_time'].dt.year.max()}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.dataframe(df[["date_time", "hour", "day_name", "temp_c", "clouds_all",
                      "rain_1h", "snow_1h", "weather_main", "traffic_volume"]].head(200),
                 use_container_width=True, height=320)


# ===========================================================================
# PAGE 3 — AI TRAFFIC PREDICTION
# ===========================================================================
def page_prediction():
    st.markdown('<div class="section-title">🧭 AI Traffic Prediction</div>', unsafe_allow_html=True)

    input_col, results_col = st.columns([1, 2])

    with input_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Input Parameters</div>', unsafe_allow_html=True)
        the_date = st.date_input("📅 Date", value=ddate.today())
        hour_options = [f"{h:02d}:00" for h in range(24)]
        hour_label = st.selectbox("🕒 Hour", hour_options, index=18)
        hour = int(hour_label.split(":")[0])
        temp_c = st.slider("🌡️ Temperature (°C)", -30, 50, 28)
        clouds_all = st.slider("☁️ Cloud (%)", 0, 100, 40)
        rain_1h = st.slider("🌧️ Rain (mm)", 0.0, 100.0, 0.0, step=0.5)
        snow_1h = st.slider("❄️ Snow (mm)", 0.0, 10.0, 0.0, step=0.5)
        is_holiday = st.selectbox("🎉 Holiday", ["No", "Yes"]) == "Yes"
        default_weather_idx = weather_categories.index("Clouds") if "Clouds" in weather_categories else 0
        weather_main = st.selectbox("🌦️ Weather", weather_categories, index=default_weather_idx)
        predict_clicked = st.button("🔮  PREDICT", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        st.session_state.prediction = run_full_prediction(
            the_date, hour, temp_c, clouds_all, rain_1h, snow_1h, is_holiday, weather_main,
        )

    pred = st.session_state.prediction
    level = pred["level"]

    with results_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Prediction Results</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                <span style="color:{TEXT_SUB}; font-size:13px;">🚗 Traffic Volume</span>
                <b style="font-size:16px;">{pred['volume']:,.0f}</b>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                <span style="color:{TEXT_SUB}; font-size:13px;">🚦 Congestion Level</span>
                <span class="badge" style="background:{LEVEL_COLOR[level]}22; color:{LEVEL_COLOR[level]};">{level.upper()}</span>
            </div>
            <div style="margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; font-size:13px; color:{TEXT_SUB}; margin-bottom:4px;">
                    <span>📊 Congestion</span><b style="color:{TEXT_MAIN};">{pred['pct']:.0f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(pred["pct"]))
        with r2:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                <span style="color:{TEXT_SUB}; font-size:13px;">⏳ Estimated Delay</span>
                <b style="font-size:16px;">{pred['delay']} Minutes</b>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                <span style="color:{TEXT_SUB}; font-size:13px;">🧭 Recommendation</span>
                <b style="font-size:14px; color:{LEVEL_COLOR[level]};">{recommendation_for(level)}</b>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:{TEXT_SUB}; font-size:13px;">⭐ Best Travel Time</span>
                <b style="font-size:16px; color:{GREEN};">{pred['best_hour']:02d}:00</b>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        gcol, fcol = st.columns([1, 1.6])
        with gcol:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Congestion</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(pred["pct"], LEVEL_COLOR[level], height=160),
                             use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        with fcol:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">24-Hour AI Forecast (Traffic Volume)</div>', unsafe_allow_html=True)
            hourly_df = pred["hourly_df"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_df["hour"], y=hourly_df["volume"], mode="lines",
                line=dict(color=hex_to_rgba(ACCENT, 0.33), width=10),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=hourly_df["hour"], y=hourly_df["volume"], mode="lines+markers",
                line=dict(color=ACCENT, width=2.5),
                marker=dict(color=hourly_df["level"].map(LEVEL_COLOR), size=6),
                hovertemplate="%{x}:00 — %{y:.0f} vehicles/hr<extra></extra>", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[pred["best_hour"]], y=[pred["best_volume"]], mode="markers+text",
                marker=dict(color=GREEN, size=13, line=dict(width=2, color="white")),
                text=[f"★ Best: {pred['best_hour']:02d}:00"], textposition="top center",
                textfont=dict(color=GREEN, size=11), showlegend=False,
                hovertemplate="Best time to travel<extra></extra>",
            ))
            fig = dark_layout(fig, height=250, xaxis_title="Hour of Day",
                               yaxis_title="Traffic Volume")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <span class="badge" style="background:{GREEN}22; color:{GREEN};">
                ⭐ Best Time to Travel: {pred['best_hour']:02d}:00
            </span>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # Map
    # -----------------------------------------------------------------
    st.markdown('<div class="section-title">🗺️ Interactive Traffic Map</div>', unsafe_allow_html=True)
    st.caption(
        "Illustrative stretch of I-94 between Minneapolis and St. Paul, colored by the "
        "current predicted congestion level (not turn-by-turn accurate)."
    )

    if not MAP_AVAILABLE:
        st.warning(
            "Map view needs two extra packages. Install them and rerun the app:\n\n"
            "`pip install folium streamlit-folium`"
        )
    else:
        route_points = [
            [44.9778, -93.2650], [44.9720, -93.2100], [44.9670, -93.1600],
            [44.9610, -93.1250], [44.9537, -93.0900],
        ]
        route_color = LEVEL_COLOR[level]
        m = folium.Map(
            location=[44.9660, -93.1750], zoom_start=11, tiles="cartodbdark_matter",
            zoom_control=True, dragging=False, scrollWheelZoom=False,
            doubleClickZoom=False, boxZoom=False, keyboard=False, touchZoom=False,
        )
        folium.PolyLine(route_points, color=route_color, weight=6, opacity=0.9).add_to(m)
        folium.Marker(route_points[0], tooltip="Minneapolis",
                      icon=folium.Icon(color="blue", icon="play")).add_to(m)
        folium.Marker(route_points[-1], tooltip="St. Paul",
                      icon=folium.Icon(color="green", icon="flag")).add_to(m)

        map_col, legend_col = st.columns([3, 1])
        with map_col:
            st_folium(m, use_container_width=True, height=380, returned_objects=[])
        with legend_col:
            st.markdown(f"""
            <div class="card" style="height:380px;">
                <div class="card-label">Traffic Level</div>
                <div style="margin-top:14px;">
                    <div class="legend-row"><span class="legend-dot" style="background:{RED};"></span>High Traffic</div>
                    <div class="legend-row"><span class="legend-dot" style="background:{YELLOW};"></span>Moderate Traffic</div>
                    <div class="legend-row"><span class="legend-dot" style="background:{GREEN};"></span>Low Traffic</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if st.session_state.page == "Executive Summary":
    page_executive_summary()
elif st.session_state.page == "Deep-Dive EDA":
    page_eda()
else:
    page_prediction()