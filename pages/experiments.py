from __future__ import annotations

import streamlit as st

from src.core.config import get_settings
from src.repositories.experiment_repository import ExperimentRepository
from src.ui.components import page_header, render_experiment_cards, status_badge

settings = get_settings()
repository = ExperimentRepository(settings.database_path)

page_header(
    "财务预测实验",
    "通过历史回测比较不同模型，判断财务指标是否具备稳定预测能力。",
)

left, right = st.columns([1, 0.18])
with right:
    if st.button("新建实验", type="primary", use_container_width=True):
        st.switch_page("pages/create_experiment.py")

experiments = repository.list_experiments()

if not experiments:
    st.markdown(
        """
        <div class="fl-empty">
          <div style="font-size:40px;margin-bottom:10px;">↗</div>
          <h3 style="margin:0;color:#101828;">还没有预测实验</h3>
          <p>上传一份历史数据，先验证它是否具有可预测性。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("创建第一个实验", type="primary"):
        st.switch_page("pages/create_experiment.py")
    st.stop()

with st.container(border=True):
    c1, c2, c3 = st.columns([0.5, 0.25, 0.25])
    search = c1.text_input("搜索实验名称", placeholder="输入实验名称或目标指标")
    status = c2.selectbox("状态", ["全部", "SUCCEEDED", "RUNNING", "FAILED", "VALIDATED", "QUEUED"])
    freq = c3.selectbox("数据频率", ["全部", "M", "W", "D"])

filtered = experiments
if search:
    filtered = [
        experiment
        for experiment in filtered
        if search.lower() in experiment.name.lower()
        or search.lower() in str(experiment.target_column or "").lower()
    ]
if status != "全部":
    filtered = [experiment for experiment in filtered if experiment.status.value == status]
if freq != "全部":
    filtered = [experiment for experiment in filtered if experiment.config.get("freq") == freq]

render_experiment_cards(filtered)

st.markdown("#### 快速进入")
for experiment in filtered[:5]:
    c1, c2, c3, c4 = st.columns([0.42, 0.16, 0.16, 0.26])
    c1.markdown(
        f"**{experiment.name}**  \n{experiment.data_start or '—'} 至 {experiment.data_end or '—'}"
    )
    c2.markdown(status_badge(experiment.status.value), unsafe_allow_html=True)
    c3.markdown(
        f"WAPE：**{experiment.best_wape * 100:.1f}%**" if experiment.best_wape else "WAPE：—"
    )
    with c4:
        target_page = (
            "pages/result_analysis.py"
            if experiment.status.value == "SUCCEEDED"
            else "pages/run_status.py"
        )
        if st.button("查看", key=f"open_{experiment.id}", use_container_width=True):
            st.query_params["experiment_id"] = experiment.id
            st.switch_page(target_page)
