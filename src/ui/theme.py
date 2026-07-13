from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --brand:#5B5CE2; --brand-dark:#4445C5; --brand-soft:#F0F0FF;
          --text:#172033; --text-2:#344054; --muted:#667085; --subtle:#98A2B3;
          --line:#E7EAF0; --bg:#F6F8FC; --card:#FFFFFF;
          --success:#039855; --success-soft:#ECFDF3;
          --warning:#DC6803; --warning-soft:#FFFAEB;
          --danger:#D92D20; --danger-soft:#FEF3F2;
          --blue:#1570EF; --blue-soft:#EFF8FF;
          --shadow:0 10px 28px rgba(31,44,78,.055); --shadow-sm:0 2px 7px rgba(31,44,78,.06);
        }
        .stApp {
          background:
            radial-gradient(circle at 11% -9%, rgba(113,113,238,.13), transparent 28rem),
            radial-gradient(circle at 96% 5%, rgba(113,184,255,.10), transparent 24rem),
            var(--bg);
          color:var(--text);
        }
        .main .block-container { max-width: 1400px; padding-top:2rem; padding-bottom:4.5rem; }
        h1, h2, h3 { letter-spacing:-.025em; color:var(--text); }
        h2 { font-size:1.18rem !important; margin-top:1.7rem !important; }
        div[data-testid="stHeader"] { background:rgba(246,248,252,.76); backdrop-filter:blur(14px); }
        div[data-testid="stToolbar"] { right:1.25rem; }
        div[data-testid="stSidebar"] { border-right:1px solid var(--line); }
        div[data-testid="stSidebar"] > div:first-child { background:#fff; }
        div[data-testid="stSidebarNav"] { padding-top:1rem; }
        div[data-testid="stSidebarNav"] a { border-radius:9px; margin:.12rem .65rem; }
        div[data-testid="stSidebarNav"] a[aria-current="page"] { background:var(--brand-soft); color:var(--brand); font-weight:700; }
        div[data-testid="stNavigation"] {
          background:rgba(255,255,255,.88); border:1px solid rgba(231,234,240,.92);
          box-shadow:var(--shadow-sm); border-radius:14px; padding:5px 8px; margin:.2rem auto 1.8rem;
          max-width:1400px;
        }
        div[data-testid="stNavigation"] a { border-radius:9px; font-weight:650; color:var(--text-2); }
        div[data-testid="stNavigation"] a[aria-current="page"] { background:var(--brand-soft); color:var(--brand); }
        div[data-testid="stMetric"] {
          background:linear-gradient(145deg,#FFF,#FCFCFF); border:1px solid var(--line); border-radius:15px;
          padding:18px 19px 15px; box-shadow:var(--shadow); min-height:126px;
          transition:transform .16s ease, box-shadow .16s ease;
        }
        div[data-testid="stMetric"]:hover { transform:translateY(-2px); box-shadow:0 14px 30px rgba(31,44,78,.09); }
        div[data-testid="stMetric"] label { color: var(--muted); }
        div[data-testid="stMetricValue"] { color:var(--text); font-weight:750; letter-spacing:-.03em; }
        div[data-testid="stMetricDelta"] { font-size:.76rem; }
        div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; box-shadow:var(--shadow-sm); }
        div[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--line) !important; border-radius:15px !important; box-shadow:var(--shadow-sm); background:#fff; }
        .fl-card {
          background:var(--card); border:1px solid var(--line); border-radius:15px;
          padding:21px 23px; box-shadow:var(--shadow); margin-bottom:18px;
        }
        .fl-conclusion {
          background:linear-gradient(90deg,#FAFAFF 0%,#FFF 68%);
          border:1px solid #D9D6FE; border-radius:16px; padding:22px 24px;
          display:flex; gap:16px; align-items:flex-start; margin-bottom:20px; box-shadow:var(--shadow-sm);
        }
        .fl-icon {
          width:44px; height:44px; border-radius:13px; background:var(--brand-soft);
          color:var(--brand); display:grid; place-items:center; font-weight:800; flex:0 0 auto;
        }
        .fl-eyebrow { color:var(--brand); font-size:12px; font-weight:700; margin-bottom:5px; }
        .fl-title { color:var(--text); font-size:18px; font-weight:750; line-height:1.35; }
        .fl-desc { color:var(--muted); margin-top:6px; line-height:1.55; }
        .fl-badge {
          display:inline-flex; align-items:center; gap:6px; border-radius:999px;
          padding:4px 9px; font-size:12px; font-weight:700;
        }
        .fl-badge-success { color:#027A48; background:var(--success-soft); }
        .fl-badge-warning { color:#B54708; background:var(--warning-soft); }
        .fl-badge-danger { color:#B42318; background:var(--danger-soft); }
        .fl-badge-info { color:#175CD3; background:var(--blue-soft); }
        .fl-section-title { font-size:18px; font-weight:750; margin:0 0 5px; color:var(--text); }
        .fl-section-desc { color:var(--muted); font-size:12px; margin-bottom:12px; }
        .fl-risk {
          border:1px solid #EEF0F5; border-radius:12px; padding:14px; background:#FFF; margin-bottom:10px;
        }
        .fl-risk strong { color:var(--text); }
        .fl-risk p { color:var(--muted); margin:.25rem 0 0; font-size:12px; line-height:1.5; }
        .fl-empty {
          border:1px dashed #C9D0E0; border-radius:16px; padding:56px 20px; text-align:center;
          background:linear-gradient(145deg,#FFF,#FAFBFF); color:var(--muted); box-shadow:var(--shadow-sm);
        }
        .fl-kv { display:flex; justify-content:space-between; gap:20px; padding:8px 0; color:var(--muted); }
        .fl-kv strong { color:var(--text-2); }
        .stButton > button, .stDownloadButton > button {
          min-height:2.45rem; border-radius:10px; font-weight:700; border-color:var(--line); box-shadow:var(--shadow-sm);
          transition:all .16s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); border-color:#C8CBEF; }
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
          background:linear-gradient(135deg,#6968E9,#4F4FD8); border-color:var(--brand); box-shadow:0 5px 13px rgba(79,79,216,.22);
        }
        .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover { background:var(--brand-dark); }
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
          border-color:var(--line); border-radius:10px; background:#fff;
        }
        div[data-baseweb="input"] > div:focus-within, div[data-baseweb="select"] > div:focus-within { border-color:#A6A7F2; box-shadow:0 0 0 3px rgba(91,92,226,.12); }
        div[data-testid="stTabs"] button { color:var(--muted); font-weight:650; padding:.55rem .85rem; }
        div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--brand); }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--brand); height:3px; border-radius:3px; }
        div[data-testid="stAlert"] { border-radius:12px; border-color:var(--line); }
        div[data-testid="stFileUploader"] { border:1px dashed #BFC7DE; border-radius:14px; padding:.45rem; background:rgba(255,255,255,.75); }
        .fl-page-header {
          display:flex; justify-content:space-between; gap:24px; align-items:flex-start; margin:0 0 1.7rem;
          padding:4px 0 0;
        }
        .fl-page-kicker { color:var(--brand); font-size:11px; font-weight:800; letter-spacing:.10em; text-transform:uppercase; margin-bottom:7px; }
        .fl-page-title { font-size:31px; font-weight:780; line-height:1.18; letter-spacing:-.045em; margin:0; color:var(--text); }
        .fl-page-subtitle { color:var(--muted); margin-top:9px; line-height:1.65; max-width:760px; }
        footer { visibility:hidden; }
        @media (max-width: 900px) {
          .main .block-container { padding-left:1rem; padding-right:1rem; }
          .fl-conclusion { flex-direction:column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
