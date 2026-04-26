import sys
import os
import streamlit as st
import requests
import pandas as pd

# Resolve project root (one level up from frontend/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH     = os.path.join(PROJECT_ROOT, "data", "patents_clean.csv")
from components import (
    create_novelty_gauge, create_whitespace_quadrant,
    create_technology_distribution, create_field_innovation_strength,
    create_rd_trend_chart, create_applicant_landscape_chart,
    create_filing_trend_chart, create_problem_identification_chart,
    create_forecast_chart, create_trajectory_chart, create_priority_matrix_chart,
    create_topic_clusters_chart
)

st.set_page_config(
    page_title ="Patent Intel | Semantic Analytics",
    layout     ="wide",
    page_icon  ="🧬",
    initial_sidebar_state="expanded"
)

# ─── Premium CSS ──────────────────────────────────────────────────────────────
# Use st.html() — NOT st.markdown() — for large style blocks.
# st.markdown() with huge <style> content renders the CSS as visible text in newer Streamlit.
# st.html() injects HTML directly into the page shadow DOM, bypassing the markdown parser.
_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>

/* ── Global ───────────────────────────────────────────── */
.stApp {
    background: radial-gradient(ellipse at top right, #1e1b4b 0%, #0B0F19 55%),
                radial-gradient(ellipse at bottom left, #0f172a 0%, #0B0F19 60%);
    color: #E2E8F0;
    font-family: 'Inter', sans-serif;
}

/* ── Smooth page fade-in ──────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.main .block-container {
    animation: fadeUp 0.45s ease-out both;
    padding-top: 1.5rem !important;
}

/* ── Typography ───────────────────────────────────────── */
h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    font-size: 2rem !important;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #00F0FF 0%, #B026FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem !important;
}
h2, h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #E2E8F0 !important;
}
p { color: #CBD5E1; }

/* ── Glass cards ──────────────────────────────────────── */
.glass-card {
    background: rgba(15, 23, 42, 0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(0, 240, 255, 0.14);
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 4px 32px 0 rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.04);
    transition: border-color 0.3s ease;
}
.glass-card:hover { border-color: rgba(0, 240, 255, 0.28); }

/* ── Info / error banners ─────────────────────────────── */
.info-banner {
    background: rgba(0, 240, 255, 0.06);
    border: 1px solid rgba(0, 240, 255, 0.25);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;
    color: #A5F3FC; font-size: 0.9rem;
}
.error-banner {
    background: rgba(255, 46, 99, 0.08);
    border: 1px solid rgba(255, 46, 99, 0.35);
    border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;
    color: #FDA4AF; font-size: 0.9rem;
}

/* ── Sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0B0F1E 100%) !important;
    border-right: 1px solid rgba(0, 240, 255, 0.08);
    box-shadow: 10px 0 40px rgba(0,0,0,0.6);
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
    font-size: 0.93rem !important;
}
section[data-testid="stSidebar"] .st-emotion-cache-10trblm {
    color: #FFFFFF !important;
}

/* ── Sidebar radio nav ────────────────────────────────── */
div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 10px !important;
    padding: 9px 14px !important;
    margin-bottom: 5px !important;
    border: 1px solid transparent !important;
    transition: all 0.25s ease !important;
    color: #CBD5E1 !important;
}
div[role="radiogroup"] label:hover {
    background: rgba(0, 240, 255, 0.07) !important;
    border-color: rgba(0, 240, 255, 0.18) !important;
    color: #E2E8F0 !important;
}
div[role="radiogroup"] label[data-selected="true"],
div[role="radiogroup"] label[aria-checked="true"] {
    background: rgba(0, 240, 255, 0.12) !important;
    border-color: rgba(0, 240, 255, 0.40) !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.15) !important;
}

/* ── Metrics ──────────────────────────────────────────── */
div[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.75rem !important;
}
div[data-testid="stMetricValue"] {
    color: #00F0FF !important;
    font-weight: 800 !important;
    font-size: 1.75rem !important;
}
div[data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00F0FF 0%, #B026FF 100%);
    color: #050810 !important;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    letter-spacing: 0.04em;
    font-size: 0.87rem;
    transition: all 0.25s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 24px rgba(0, 240, 255, 0.50);
    opacity: 0.93;
}
.stButton > button:active { transform: translateY(0); }

/* ── Inputs ───────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(0, 0, 0, 0.35) !important;
    border: 1px solid rgba(0, 240, 255, 0.20) !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(0, 240, 255, 0.55) !important;
    box-shadow: 0 0 0 2px rgba(0, 240, 255, 0.12) !important;
}
label[data-testid="stWidgetLabel"] {
    color: #CBD5E1 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px !important;
}

/* ── Expanders ────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}
.streamlit-expanderContent {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    background: rgba(0,0,0,0.2) !important;
}

/* ── Dataframes ───────────────────────────────────────── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Divider ──────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Spinner ──────────────────────────────────────────── */
.stSpinner > div { border-top-color: #00F0FF !important; }

/* ── Checkbox ─────────────────────────────────────────── */
div[data-testid="stCheckbox"] label p { color: #CBD5E1 !important; font-weight: 500 !important; }

/* ── Caption ──────────────────────────────────────────── */
.stCaption, small { color: #64748B !important; font-size: 0.78rem !important; }

/* ── Section subheading helper ────────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #00F0FF;
    margin-bottom: 10px;
}

/* ── Priority badge helper ────────────────────────────── */
.badge-HIGH   { background:#00F0FF; color:#050810; }
.badge-MEDIUM { background:#FFD700; color:#050810; }
.badge-LOW    { background:#475569; color:#E2E8F0; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.06em;
}

</style>
"""
st.html(_CSS)

API_URL = "http://localhost:8000/api"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _err(msg: str):
    st.markdown(f'<div class="error-banner">⚠️ {msg}</div>', unsafe_allow_html=True)

def _info(msg: str):
    st.markdown(f'<div class="info-banner">ℹ️ {msg}</div>', unsafe_allow_html=True)

def _section(label: str):
    st.markdown(f'<p class="section-label">{label}</p>', unsafe_allow_html=True)

def _card_open(style=""):
    pass

def _card_close():
    pass

def _rec_card(rec: dict):
    colors = {"HIGH": "#00F0FF", "MEDIUM": "#FFD700", "LOW": "#475569"}
    c = colors.get(rec.get('priority', 'LOW'), "#475569")
    st.markdown(f"""
    <div style="background:rgba(15,23,42,0.55);border:1px solid {c}33;
                border-left: 3px solid {c};
                border-radius:12px;padding:16px 20px;margin-bottom:10px;">
        <span class="badge badge-{rec.get('priority','LOW')}">{rec.get('priority','?')} PRIORITY</span>
        <span style="color:#64748B;font-size:12px;margin-left:10px;">
            {rec.get('field', rec.get('ipc_cpc',''))} &nbsp;|&nbsp; {rec.get('trajectory','')}
        </span>
        <p style="color:#E2E8F0;font-size:14px;font-weight:600;margin:10px 0 6px;">
            {rec.get('recommendation','')}
        </p>
        <p style="color:#94A3B8;font-size:12px;margin:0;">{rec.get('reason','')}</p>
    </div>
    """, unsafe_allow_html=True)


# ─── Cached API calls ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_white_space():
    r = requests.get(f"{API_URL}/white_space", timeout=120)
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_topic_clusters():
    r = requests.get(f"{API_URL}/topic_clusters", timeout=600)
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rd_signals():
    r = requests.get(f"{API_URL}/market_intelligence/rd_signals", timeout=120)
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_applicants():
    r = requests.get(f"{API_URL}/market_intelligence/applicants", timeout=60)
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_filing_trends():
    r = requests.get(f"{API_URL}/tech_trends/filing_trends", timeout=60)
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_problems():
    r = requests.get(f"{API_URL}/tech_trends/problems", timeout=120)
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_forecast():
    r = requests.get(f"{API_URL}/future_trends/forecast", timeout=60)
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trajectory():
    r = requests.get(f"{API_URL}/future_trends/trajectory", timeout=60)
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_recommendations():
    r = requests.get(f"{API_URL}/future_trends/recommendations", timeout=60)
    return r.json()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dashboard():
    r = requests.get(f"{API_URL}/policy_dashboard", timeout=180)
    return r.json()

@st.cache_data(ttl=300, show_spinner=False)
def search_patents(query: str, top_k: int = 20):
    r = requests.post(f"{API_URL}/search_patents", json={"idea": query}, timeout=30)
    return r.json()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
        <div style="padding: 6px 0 16px;">
            <div style="font-size:22px;font-weight:800;letter-spacing:-0.03em;
                        background:linear-gradient(90deg,#00F0FF,#B026FF);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                🧬 INTEL SYSTEM
            </div>
            <div style="font-size:11px;color:#475569;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;">
                AI-Powered IP Analytics
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("Intelligence Modules", [
        "🎯 Policy Dashboard",
        "💡 Idea Validator",
        "📈 Market Intelligence",
        "📡 Tech Trends",
        "🔮 Future Trends",
        "🚀 White-Space",
        "📊 Patent Explorer",
        "🌐 IP Landscape",
        "🧩 Topic Clusters",
    ])

    st.divider()
    st.markdown("""
        <div style="font-size:11px;color:#334155;text-align:center;padding:4px 0;">
            v2.1 · Data refreshes every hour
        </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh All Caches"):
        st.cache_data.clear()
        try:
            r = requests.post(f"{API_URL}/reload_system", timeout=60)
            if r.status_code == 200:
                st.success("System caches cleared and dataset fully reloaded.")
            else:
                st.error("Failed to reload backend system caches.")
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Pages
# ══════════════════════════════════════════════════════════════════════════════

# ── 🎯 Policy Dashboard ───────────────────────────────────────────────────────
if page == "🎯 Policy Dashboard":
    st.markdown("<h1>🎯 Policy Dashboard</h1>", unsafe_allow_html=True)
    st.info("A single-screen executive summary pulling the most important metrics from every other page. It combines your CSV data with live arXiv signals to tell you exactly where to prioritize your R&D investment.")

    with st.spinner("Loading policy dashboard..."):
        try:
            dash = fetch_dashboard()
            if isinstance(dash, dict) and "error" in dash:
                _err(dash["error"])
            else:
                _section("Executive KPIs")
                _card_open()
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Fastest Growing Field", dash['fastest_growing_field'],
                          delta=f"+{dash['fastest_slope']} filings/yr")
                k2.metric("High-Growth Fields", dash['high_growth_count'])
                k3.metric("Declining Fields", dash['declining_count'])
                k4.metric("Top White-Space", dash['top_whitespace_opportunity'],
                          delta=f"Score: {dash['top_whitespace_score']}")
                _card_close()

                _section("Investment Priority Matrix")
                st.caption("Bubble size ∝ total filings · Top-right = highest priority")
                _card_open("padding:10px;")
                st.plotly_chart(create_priority_matrix_chart(pd.DataFrame(dash['priority_matrix'])), width='stretch')
                _card_close()

                _section("Top R&D Recommendations")
                top_recs = dash.get('top_recommendations', [])
                if not top_recs:
                    _info("No HIGH priority recommendations found for the current dataset.")
                else:
                    for rec in top_recs:
                        _rec_card(rec)

                _section("Export Data")
                _card_open()
                ex1, ex2, ex3, ex4 = st.columns(4)
                exports = [
                    (ex1, "📥 Recommendations", fetch_recommendations, "rd_recommendations.csv"),
                    (ex2, "📥 White-Space",      fetch_white_space,     "white_space.csv"),
                    (ex3, "📥 Forecast",          fetch_forecast,         "filing_forecast.csv"),
                    (ex4, "📥 R&D Signals",       fetch_rd_signals,       "rd_signals.csv"),
                ]
                for col, label, fn, fname in exports:
                    with col:
                        try:
                            df_e = pd.DataFrame(fn())
                            col.download_button(label=label, 
                                                data=df_e.to_csv(index=False).encode('utf-8-sig'),
                                                file_name=fname, mime="text/csv",
                                                width='stretch')
                        except Exception:
                            col.warning("Unavailable")
                _card_close()
        except Exception as e:
            _err(f"Failed to load Policy Dashboard — is the FastAPI backend running on port 8000?")


# ── 💡 Idea Validator ─────────────────────────────────────────────────────────
elif page == "💡 Idea Validator":
    st.markdown("<h1>💡 Semantic Validation Engine</h1>", unsafe_allow_html=True)
    st.info("Type a project or research idea in plain English. The system converts it into mathematical numbers and compares it against all patent abstracts to tell you how novel it is and identify similar prior art.")

    _card_open()
    idea_input = st.text_area("DESCRIBE YOUR RESEARCH OR PROJECT IDEA", height=110,
                              placeholder="e.g., A decentralised AI system for optimising ultra-low latency edge computing...")
    if st.button("🚀 RUN VALIDATION"):
        if not idea_input.strip():
            st.warning("Please enter a description before validating.")
        else:
            with st.spinner("Scanning 9,483 patents for semantic overlap..."):
                try:
                    data = requests.post(f"{API_URL}/validate_idea", json={"idea": idea_input}, timeout=30).json()
                    if "error" in data:
                        _err(data["error"])
                    else:
                        st.divider()
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.plotly_chart(create_novelty_gauge(data['novelty_score']), width='stretch')
                            risk_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                            icon = risk_colors.get(data['risk_level'], "⚪")
                            st.metric("IP Risk Level", f"{icon} {data['risk_level']}")
                            st.caption("Based on highest cosine-similarity match in the FAISS index.")
                        with col2:
                            st.markdown("### 🔍 Top Similar Prior Art")
                            for match in data['top_matches']:
                                with st.expander(f"**{match['title']}** — {match['similarity']*100:.1f}% match"):
                                    st.markdown(f"**Year:** `{match.get('year','?')}` | **Applicant:** `{match.get('applicant','?')}`")
                                    st.write(match.get('abstract',''))
                except requests.exceptions.ConnectionError:
                    _err("Backend connection failed. Ensure FastAPI is running on port 8000.")
                except Exception as e:
                    _err(f"Validation failed: {e}")
    _card_close()


# ── 🚀 White-Space ────────────────────────────────────────────────────────────
elif page == "🚀 White-Space":
    st.markdown("<h1>🚀 White-Space Discovery</h1>", unsafe_allow_html=True)
    st.info("Finds 'untapped opportunities' (Goldmines). It checks the arXiv API to see where academic research is growing fast, but commercial patents are still rare.")

    with st.spinner("Fetching arXiv + RSS signals and computing white-space scores..."):
        try:
            ws_data = fetch_white_space()
            if isinstance(ws_data, dict) and "error" in ws_data:
                _err(ws_data["error"])
            else:
                df_ws = pd.DataFrame(ws_data)
                _section("Top Opportunities")
                _card_open()
                k1, k2, k3 = st.columns(3)
                k1.metric("Top Opportunity",   df_ws.iloc[0]['tech_keyword'], delta="Highest WS Score")
                k2.metric("Growth Signal (YoY)", f"{df_ws.iloc[0]['external_signal_velocity']*100:.1f}%")
                k3.metric("Market Density",     f"{df_ws.iloc[0]['patent_density']*100:.2f}%",
                           delta="of global patent db")
                _card_close()

                _section("Opportunity Matrix")
                _card_open("padding:10px;")
                st.plotly_chart(create_whitespace_quadrant(df_ws), width='stretch')
                _card_close()

                if st.checkbox("📋 Show detailed data table"):
                    st.dataframe(
                        df_ws[['tech_keyword','external_signal_velocity','patent_density','white_space_score','quadrant']],
                        width='stretch', hide_index=True
                    )
        except Exception as e:
            _err(f"Failed to fetch white-space analytics: {e}")


# ── 📊 Patent Explorer ────────────────────────────────────────────────────────
elif page == "📊 Patent Explorer":
    st.markdown("<h1>📊 Semantic Patent Explorer</h1>", unsafe_allow_html=True)
    st.info("A smart search engine. Type any topic, and the system uses AI vector search to find and rank the top 20 most semantically relevant patents from your database.")

    _card_open()
    search_query = st.text_input("SEARCH TECHNOLOGY FIELD OR TOPIC",
                                 placeholder="e.g., Neuromorphic Chips, Carbon-Aware Computing, Federated Learning...")
    if search_query:
        with st.spinner(f"Scanning vector space for '{search_query}'..."):
            try:
                results = search_patents(search_query)
                if not results:
                    _info("No matching patents found. Try a different topic.")
                else:
                    st.markdown(f"### 🌐 Top 20 results for `{search_query}`")
                    df_r = pd.DataFrame(results)
                    cols = [c for c in ['title', 'similarity', 'year', 'applicant', 'abstract'] if c in df_r.columns]
                    st.dataframe(df_r[cols], width='stretch', hide_index=True)

                    st.divider()
                    st.markdown("### 📄 Detailed View (top 6)")
                    for match in results[:6]:
                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.08);
                                    border-left:3px solid #00F0FF;border-radius:12px;padding:16px 18px;margin-bottom:10px;">
                            <p style="font-size:15px;font-weight:700;color:#00F0FF;margin:0 0 6px;">
                                {match.get('title','')}
                            </p>
                            <p style="font-size:12px;color:#B026FF;margin:0 0 8px;font-weight:600;">
                                SIMILARITY: {match.get('similarity',0)*100:.1f}%
                                &nbsp;·&nbsp; YEAR: {match.get('year','')}
                                &nbsp;·&nbsp; {match.get('applicant','')}
                            </p>
                            <p style="font-size:13px;color:#CBD5E1;margin:0;">{match.get('abstract','')}</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception:
                _err("Backend connection failed.")
    else:
        _info("Enter a topic above to explore semantically related patents.")
        if st.checkbox("View raw dataset (Recent 3 Years)"):
            try:
                df = pd.read_csv(CSV_PATH)
                df.columns = [c.lower().strip() for c in df.columns]
                df.rename(columns={"publication year":"year","applicants":"applicant"}, inplace=True)
                
                if 'year' in df.columns:
                    df['year'] = pd.to_numeric(df['year'], errors='coerce')
                    valid_years = df['year'].dropna().unique()
                    if len(valid_years) > 0:
                        recent_years = sorted(valid_years)[-3:]
                        df = df[df['year'].isin(recent_years)]
                        
                st.dataframe(df, width='stretch')
            except FileNotFoundError:
                _err("Dataset not found in 'data/' directory.")
    _card_close()


# ── 🌐 IP Landscape ───────────────────────────────────────────────────────────
elif page == "🌐 IP Landscape":
    st.markdown("<h1>🌐 Global IP Landscape</h1>", unsafe_allow_html=True)
    st.info("A bird's-eye overview of your entire patent database. It simply reads your CSV file to show the full distribution of patents across all technology sectors.")

    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.lower().strip() for c in df.columns]
        df.rename(columns={"publication year":"year","applicants":"applicant"}, inplace=True)
        if 'cpc classifications' in df.columns:
            df['ipc_cpc'] = df['cpc classifications'].apply(
                lambda x: str(x).split(';;')[0][:4] if pd.notna(x) and str(x) != "" else "UNKNOWN"
            )
        else:
            df['ipc_cpc'] = "UNKNOWN"

        _section("Dataset Overview")
        _card_open()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total IP Assets",      f"{len(df):,}",         delta="Indexed in Vector Store")
        m2.metric("Tech Sectors",         df['ipc_cpc'].nunique(), delta="Unique IPC/CPC classes")
        m3.metric("Intelligence Horizon", df['year'].max(),        delta="Latest patent activity")
        _card_close()

        _section("Technology Distribution")
        _card_open("padding:10px;")
        st.plotly_chart(create_technology_distribution(df), width='stretch')
        _card_close()

        _section("Field Innovation Strength")
        _card_open("padding:10px;")
        st.plotly_chart(create_field_innovation_strength(df), width='stretch')
        _card_close()

        if st.checkbox("Show raw dataset"):
            st.dataframe(df, width='stretch')

    except FileNotFoundError:
        _err("Dataset 'data/patents_clean.csv' not found.")
    except Exception as e:
        _err(f"Error loading landscape: {e}")


# ── 📈 Market Intelligence ────────────────────────────────────────────────────
elif page == "📈 Market Intelligence":
    st.markdown("<h1>📈 Market Intelligence</h1>", unsafe_allow_html=True)
    st.info("Shows where the R&D world is investing right now. It pulls live academic paper counts from the arXiv API and combines it with the top companies filing patents in your database.")

    _section("R&D Activity by Technology")
    with st.spinner("Fetching arXiv R&D signals..."):
        try:
            rd_data = fetch_rd_signals()
            if isinstance(rd_data, dict) and "error" in rd_data:
                _err(rd_data["error"])
            else:
                df_rd = pd.DataFrame(rd_data)
                _card_open()
                k1, k2, k3 = st.columns(3)
                k1.metric("Most Active R&D Field", df_rd.iloc[0]['keyword'],
                           delta="Top arXiv 2025 mentions")
                max_r = df_rd.loc[df_rd['yoy_growth_pct'].idxmax()]
                k2.metric("Fastest Growing Field", max_r['keyword'],
                           delta=f"+{max_r['yoy_growth_pct']}% YoY")
                k3.metric("Live News Signals", int(df_rd['live_rss_signal'].sum()),
                           delta="Total RSS keyword mentions")
                _card_close()
                _card_open("padding:10px;")
                st.plotly_chart(create_rd_trend_chart(df_rd), width='stretch')
                _card_close()
                if st.checkbox("Show R&D raw data"):
                    st.dataframe(df_rd, width='stretch', hide_index=True)
        except Exception as e:
            _err(f"Failed to fetch R&D signals: {e}")

    _section("Competitive Applicant Landscape")
    with st.spinner("Loading applicant breakdown..."):
        try:
            ap_data = fetch_applicants()
            if isinstance(ap_data, dict) and "error" in ap_data:
                _err(ap_data["error"])
            else:
                df_ap = pd.DataFrame(ap_data)
                _card_open("padding:10px;")
                st.plotly_chart(create_applicant_landscape_chart(df_ap), width='stretch')
                _card_close()
                if st.checkbox("Show applicant raw data"):
                    st.dataframe(df_ap, width='stretch', hide_index=True)
        except Exception as e:
            _err(f"Failed to fetch applicant data: {e}")


# ── 📡 Tech Trends ────────────────────────────────────────────────────────────
elif page == "📡 Tech Trends":
    st.markdown("<h1>📡 Technological Trends & Problem Identification</h1>", unsafe_allow_html=True)
    st.info("A historical view of your data. It graphs which technology fields are filing more patents over time, and analyzes patent abstracts to identify the most common problems inventors are trying to solve.")

    _section("Patent Filing Trends by Technology Field")
    with st.spinner("Computing filing trends..."):
        try:
            trend_data = fetch_filing_trends()
            if isinstance(trend_data, dict) and "error" in trend_data:
                _err(trend_data["error"])
            else:
                df_t = pd.DataFrame(trend_data)
                if not df_t.empty:
                    _card_open()
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Years Tracked",        df_t['year'].nunique())
                    k2.metric("Most Active Field",    df_t.groupby('ipc_cpc')['filing_count'].sum().idxmax())
                    k3.metric("Total Filings Tracked",f"{int(df_t['filing_count'].sum()):,}")
                    _card_close()
                _card_open("padding:10px;")
                st.plotly_chart(create_filing_trend_chart(df_t), width='stretch')
                _card_close()
        except Exception as e:
            _err(f"Failed to fetch filing trends: {e}")

    _section("Problem Identification from Patent Abstracts")
    with st.spinner("Scanning abstracts for problem indicators..."):
        try:
            prob_data = fetch_problems()
            if isinstance(prob_data, dict) and "error" in prob_data:
                _err(prob_data["error"])
            else:
                df_p = pd.DataFrame(prob_data)
                _card_open("padding:10px;")
                event = st.plotly_chart(
                    create_problem_identification_chart(df_p), 
                    width='stretch', 
                    on_select="rerun"
                )
                _card_close()
                
                selected_phrase = None
                if event and "selection" in event and "points" in event["selection"] and len(event["selection"]["points"]) > 0:
                    selected_phrase = event["selection"]["points"][0].get("y", None)
                
                if selected_phrase:
                    st.markdown(f"### 🔍 Detailed Context for **'{selected_phrase}'**")
                    filtered_df = df_p[df_p['problem_phrase'] == selected_phrase]
                    cols = [c for c in ['problem_phrase','context','ipc_cpc','title'] if c in filtered_df.columns]
                    st.dataframe(filtered_df[cols], width='stretch', hide_index=True)
                    st.caption("Click whitespace in the chart to clear the selection.")
                
                if st.checkbox("Show all detailed problem statements"):
                    if not df_p.empty:
                        cols = [c for c in ['problem_phrase','context','ipc_cpc','title'] if c in df_p.columns]
                        st.dataframe(df_p[cols], width='stretch', hide_index=True)
        except Exception as e:
            _err(f"Failed to fetch problem statements: {e}")


# ── 🔮 Future Trends ──────────────────────────────────────────────────────────
elif page == "🔮 Future Trends":
    st.markdown("<h1>🔮 Future Trends & R&D Policy</h1>", unsafe_allow_html=True)
    st.info("Predicts the next 3 years of patent filings using linear regression on your historical data, and generates actionable AI recommendations on whether to increase or decrease investment in specific tech fields.")

    _section("3-Year Filing Forecast")
    with st.spinner("Running linear regression forecast..."):
        try:
            fc_data = fetch_forecast()
            if isinstance(fc_data, dict) and "error" in fc_data:
                _err(fc_data["error"])
            else:
                df_fc = pd.DataFrame(fc_data)
                _card_open("padding:10px;")
                st.plotly_chart(create_forecast_chart(df_fc), width='stretch')
                _card_close()
                if st.checkbox("Show forecast raw data"):
                    st.dataframe(df_fc, width='stretch', hide_index=True)
        except Exception as e:
            _err(f"Failed to fetch forecast: {e}")

    _section("Technology Field Trajectories")
    with st.spinner("Classifying trajectories..."):
        try:
            traj_data = fetch_trajectory()
            if isinstance(traj_data, dict) and "error" in traj_data:
                _err(traj_data["error"])
            else:
                df_tr = pd.DataFrame(traj_data)
                _card_open()
                k1, k2, k3 = st.columns(3)
                k1.metric("High Growth Fields",  int((df_tr['trajectory'] == "🚀 High Growth").sum()),
                           delta="Recommended for investment")
                k2.metric("Declining Fields",    int((df_tr['trajectory'] == "📉 Declining").sum()),
                           delta="Consider reducing spend")
                top_slope = df_tr.loc[df_tr['slope'].idxmax(), 'ipc_cpc'] if not df_tr.empty else "N/A"
                k3.metric("Fastest Rising Field", top_slope, delta="Highest growth slope")
                _card_close()
                _card_open("padding:10px;")
                st.plotly_chart(create_trajectory_chart(df_tr), width='stretch')
                _card_close()
        except Exception as e:
            _err(f"Failed to fetch trajectory data: {e}")

    _section("R&D Policy Recommendations")
    with st.spinner("Generating recommendations..."):
        try:
            rec_data = fetch_recommendations()
            if isinstance(rec_data, dict) and "error" in rec_data:
                _err(rec_data["error"])
            else:
                for rec in pd.DataFrame(rec_data).to_dict(orient="records"):
                    _rec_card(rec)
        except Exception as e:
            _err(f"Failed to fetch recommendations: {e}")


# ── 🧩 Topic Clusters ─────────────────────────────────────────────────────────
elif page == "🧩 Topic Clusters":
    st.markdown("<h1>🧩 Technology Topic Clusters</h1>", unsafe_allow_html=True)
    st.info("A Topic Cluster is a grouping of patents that talk about the exact same specific technology or concept, even if they don't use the exact same words. The AI read all your patents and grouped them into these hidden themes automatically.")
    _info("Results are loaded from disk cache for speed. To rebuild clusters, run: `python -m backend.topic_modeling --force`")

    with st.spinner("Loading topic clusters (fast if precomputed, slow on first run)..."):
        try:
            data = fetch_topic_clusters()
            if isinstance(data, dict) and "error" in data:
                _err(data["error"])
            else:
                df_topics  = pd.DataFrame(data["topic_summary"])
                df_patents = pd.DataFrame(data["patents_with_topics"])

                _section("Discovered Topic Summary")
                _card_open("padding:10px;")
                st.plotly_chart(create_topic_clusters_chart(df_topics), width='stretch')
                _card_close()

                if st.checkbox("Show raw topic summary table"):
                    _card_open()
                    st.dataframe(df_topics, width='stretch', hide_index=True)
                    _card_close()

                if st.checkbox("Show patents mapped to topics"):
                    _card_open()
                    st.dataframe(df_patents, width='stretch', hide_index=True)
                    _card_close()
        except Exception as e:
            _err(f"Backend connection failed: {e}")