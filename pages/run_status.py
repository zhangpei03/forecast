from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

from src.core.config import get_settings
from src.domain.enums import ExperimentStatus
from src.jobs.job_runner import start_training_job
from src.repositories.experiment_repository import ExperimentRepository
from src.storage.file_store import get_experiment_dir, read_json
from src.ui.components import page_header, status_badge

settings = get_settings()
repository = ExperimentRepository(settings.database_path)

experiment_id = st.query_params.get("experiment_id")
if not experiment_id:
    latest = repository.list_experiments()
    experiment_id = latest[0].id if latest else None

if not experiment_id:
    page_header("运行状态", "暂无正在运行的实验。")
    st.info("请先创建实验。")
    st.stop()

summary = repository.get_experiment(experiment_id)
if summary is None:
    st.error("实验不存在。")
    st.stop()

progress_path = get_experiment_dir(settings, experiment_id) / "progress.json"
progress = read_json(progress_path)

page_header(
    summary.name,
    f"数据范围：{summary.data_start or '—'} 至 {summary.data_end or '—'} · {summary.item_count} 条序列",
    badge="运行跟踪",
)

st.markdown(status_badge(progress.get("status", summary.status.value)), unsafe_allow_html=True)
st.progress(int(progress.get("progress", 0)), text=progress.get("message", "等待 Worker 更新状态"))

steps = [
    "LOAD_DATA",
    "TRAIN_BASELINES",
    "TRAIN_AUTOGLUON",
    "CALCULATE_METRICS",
    "GENERATE_FUTURE_FORECAST",
    "BUILD_EXPORT",
    "COMPLETE",
]
current_stage = progress.get("stage", "LOAD_DATA")
st.write(" → ".join([f"**{step}**" if step == current_stage else step for step in steps]))

c1, c2, c3, c4 = st.columns(4)
c1.metric("训练模式", summary.config.get("preset", "—"))
c2.metric("时间预算", f"{summary.config.get('time_limit_seconds', '—')} 秒")
c3.metric("预测周期", summary.config.get("prediction_length", "—"))
c4.metric("当前 PID", progress.get("worker_pid") or "—")

if progress.get("status") == ExperimentStatus.SUCCEEDED.value:
    st.success("评测完成。")
    if st.button("查看结果", type="primary"):
        st.query_params["experiment_id"] = experiment_id
        st.switch_page("pages/result_analysis.py")
elif progress.get("status") == ExperimentStatus.FAILED.value:
    st.error(progress.get("message", "任务失败"))
    st.caption(f"技术日志路径：{progress.get('log_path', '—')}")
    if st.button("重新运行评测", type="primary"):
        start_training_job(
            settings=settings,
            repository=repository,
            experiment_id=experiment_id,
        )
        st.rerun()
else:
    st.caption(f"最后更新：{progress.get('updated_at') or datetime.now():}")
    time.sleep(2)
    st.rerun()

with st.expander("运行日志", expanded=False):
    log_path = progress.get("log_path")
    if log_path:
        try:
            st.code(open(log_path, encoding="utf-8").read()[-6000:])
        except OSError:
            st.caption("日志尚未生成。")
