from __future__ import annotations

# 在读取 settings / 初始化引擎之前加载 .env(DATABASE_URL / MAX_CONCURRENT_TRAINING 等)
from dotenv import load_dotenv as _load_dotenv

_load_dotenv(override=False)

import streamlit as st

from src.core.auth import current_user, is_sso_authenticated
from src.core.config import get_settings
from src.repositories.experiment_repository import ExperimentRepository
from src.storage.file_store import ensure_runtime_dirs
from src.ui.theme import apply_theme

settings = get_settings()
ensure_runtime_dirs(settings)
ExperimentRepository(settings.database_url)

st.set_page_config(
    page_title="Forecast Lab",
    page_icon="FL",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()

with st.sidebar:
    _user = current_user()
    if is_sso_authenticated():
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
              <span style="display:inline-flex;align-items:center;justify-content:center;
                width:28px;height:28px;border-radius:50%;background:#4F46E5;
                color:#fff;font-weight:700;font-size:13px;">{_user[0].upper()}</span>
              <div>
                <div style="font-size:14px;font-weight:600;color:#101828;">{_user}</div>
                <div style="font-size:11px;color:#667085;">滴滴内网 SSO 已登录</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<a href="/sso/logout" style="font-size:12px;color:#4F46E5;text-decoration:none;">退出登录</a>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"当前用户：{_user}")
        st.markdown(
            '<a href="/sso/login" style="font-size:12px;color:#4F46E5;text-decoration:none;">SSO 登录</a>',
            unsafe_allow_html=True,
        )

experiments_page = st.Page(
    "pages/experiments.py",
    title="实验列表",
    icon=":material/table_chart:",
    url_path="experiments",
)
new_page = st.Page(
    "pages/create_experiment.py",
    title="新建实验",
    icon=":material/add_circle:",
    url_path="new",
)
run_page = st.Page(
    "pages/run_status.py",
    title="运行状态",
    icon=":material/progress_activity:",
    url_path="run",
)
result_page = st.Page(
    "pages/result_analysis.py",
    title="结果分析",
    icon=":material/monitoring:",
    url_path="result",
)

navigation = st.navigation(
    [experiments_page, new_page, run_page, result_page],
    position="top",
)
navigation.run()
