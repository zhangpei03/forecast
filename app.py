from __future__ import annotations

# 在读取 settings / 初始化引擎之前加载 .env(DATABASE_URL / MAX_CONCURRENT_TRAINING 等)
from dotenv import load_dotenv as _load_dotenv

_load_dotenv(override=False)

import streamlit as st

from src.core.auth import current_user
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
    st.caption(f"当前用户：{current_user()}")

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
