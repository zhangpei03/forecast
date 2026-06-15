from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --brand:#4F46E5; --brand-dark:#4338CA; --brand-soft:#EEF2FF;
          --text:#101828; --text-2:#475467; --muted:#667085; --subtle:#98A2B3;
          --line:#E4E7EC; --bg:#F7F8FA; --card:#FFFFFF;
          --success:#12B76A; --success-soft:#ECFDF3;
          --warning:#F79009; --warning-soft:#FFFAEB;
          --danger:#F04438; --danger-soft:#FEF3F2;
          --blue:#2E90FA; --blue-soft:#EFF8FF;
          --shadow:0 1px 2px rgba(16,24,40,.04);
        }
        .stApp { background: var(--bg); color: var(--text); }
        .main .block-container { max-width: 1360px; padding-top: 1.4rem; padding-bottom: 4rem; }
        h1, h2, h3 { letter-spacing: 0; color: var(--text); }
        div[data-testid="stMetric"] {
          background: var(--card); border:1px solid var(--line); border-radius:12px;
          padding:18px 18px 14px; box-shadow:var(--shadow); min-height:132px;
        }
        div[data-testid="stMetric"] label { color: var(--muted); }
        div[data-testid="stMetricValue"] { color: var(--text); font-weight:700; }
        div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
        .fl-card {
          background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:20px 22px; box-shadow:var(--shadow); margin-bottom:16px;
        }
        .fl-conclusion {
          background:linear-gradient(90deg,#FAFAFF 0%,#FFF 68%);
          border:1px solid #D9D6FE; border-radius:12px; padding:20px 22px;
          display:flex; gap:16px; align-items:flex-start; margin-bottom:16px;
        }
        .fl-icon {
          width:42px; height:42px; border-radius:11px; background:var(--brand-soft);
          color:var(--brand); display:grid; place-items:center; font-weight:800; flex:0 0 auto;
        }
        .fl-eyebrow { color:var(--brand); font-size:12px; font-weight:700; margin-bottom:5px; }
        .fl-title { color:var(--text); font-size:18px; font-weight:700; line-height:1.35; }
        .fl-desc { color:var(--muted); margin-top:6px; line-height:1.55; }
        .fl-badge {
          display:inline-flex; align-items:center; gap:6px; border-radius:999px;
          padding:4px 9px; font-size:12px; font-weight:700;
        }
        .fl-badge-success { color:#027A48; background:var(--success-soft); }
        .fl-badge-warning { color:#B54708; background:var(--warning-soft); }
        .fl-badge-danger { color:#B42318; background:var(--danger-soft); }
        .fl-badge-info { color:#175CD3; background:var(--blue-soft); }
        .fl-section-title { font-size:18px; font-weight:700; margin:0 0 4px; color:var(--text); }
        .fl-section-desc { color:var(--muted); font-size:12px; margin-bottom:12px; }
        .fl-risk {
          border:1px solid #F0F1F4; border-radius:10px; padding:13px; background:#FFF; margin-bottom:10px;
        }
        .fl-risk strong { color:var(--text); }
        .fl-risk p { color:var(--muted); margin:.25rem 0 0; font-size:12px; line-height:1.5; }
        .fl-empty {
          border:1px dashed #D0D5DD; border-radius:12px; padding:56px 20px; text-align:center;
          background:#FFF; color:var(--muted);
        }
        .fl-kv { display:flex; justify-content:space-between; gap:20px; padding:8px 0; color:var(--muted); }
        .fl-kv strong { color:var(--text-2); }
        .stButton > button, .stDownloadButton > button {
          border-radius:8px; font-weight:700; border-color:var(--line); box-shadow:var(--shadow);
        }
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
          background:var(--brand); border-color:var(--brand);
        }
        footer { visibility:hidden; }
        @media (max-width: 900px) {
          .main .block-container { padding-left:1rem; padding-right:1rem; }
          .fl-conclusion { flex-direction:column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
