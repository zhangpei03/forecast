from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime

import streamlit as st

from src.core.config import get_settings
from src.core.constants import DEFAULT_PREDICTION_LENGTH, TRAINING_PRESETS
from src.domain.models import ExperimentConfig
from src.jobs.job_runner import start_training_job
from src.repositories.experiment_repository import ExperimentRepository
from src.services.data_quality_service import (
    estimate_supported_backtest_windows,
    profile_normalized_data,
)
from src.services.excel_service import (
    list_excel_sheets,
    normalize_finance_dataframe,
    read_excel_sheet,
)
from src.services.mapping_service import guess_mapping_columns
from src.storage.file_store import get_experiment_dir, sha256_file, write_json
from src.storage.parquet import write_parquet
from src.ui.components import page_header, profile_to_json, render_quality_profile

settings = get_settings()
repository = ExperimentRepository(settings.database_path)

page_header(
    "新建预测实验",
    "上传 Excel，完成字段映射和数据质量检查，再启动本地评测 Worker。",
)

step = st.radio(
    "创建步骤",
    ["1 上传数据", "2 字段与质量", "3 预测配置", "4 确认运行"],
    horizontal=True,
    label_visibility="collapsed",
)

if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None
if "raw_preview" not in st.session_state:
    st.session_state.raw_preview = None
if "normalized_data" not in st.session_state:
    st.session_state.normalized_data = None
if "data_profile" not in st.session_state:
    st.session_state.data_profile = None

if step == "1 上传数据":
    with st.container(border=True):
        uploaded = st.file_uploader("上传 .xlsx 文件", type=["xlsx"], accept_multiple_files=False)
        st.caption("单文件不超过 50MB。示例数据可通过 `make demo-data` 生成。")
        if uploaded:
            experiment_id = (
                st.session_state.get("draft_experiment_id") or f"exp_{uuid.uuid4().hex[:12]}"
            )
            st.session_state.draft_experiment_id = experiment_id
            upload_dir = settings.upload_dir / experiment_id
            upload_dir.mkdir(parents=True, exist_ok=True)
            uploaded_path = upload_dir / "source.xlsx"
            uploaded_path.write_bytes(uploaded.getbuffer())
            st.session_state.uploaded_path = str(uploaded_path)
            st.success(f"已上传：{uploaded.name} · SHA256 {sha256_file(uploaded_path)[:12]}")
            sheets = list_excel_sheets(uploaded_path)
            sheet = st.selectbox("选择 Sheet", sheets)
            raw = read_excel_sheet(uploaded_path, sheet)
            st.session_state.sheet_name = sheet
            st.session_state.raw_preview = raw
            st.dataframe(raw.head(50), use_container_width=True, hide_index=True)

if step == "2 字段与质量":
    if st.session_state.raw_preview is None:
        st.warning("请先上传 Excel 并选择 Sheet。")
        st.stop()
    raw = st.session_state.raw_preview
    guesses = guess_mapping_columns(raw)
    columns = list(raw.columns)
    left, right = st.columns([0.36, 0.64])
    with left:
        st.markdown("#### 字段映射")
        timestamp_column = st.selectbox(
            "时间字段",
            columns,
            index=columns.index(guesses["timestamp_column"]),
        )
        target_column = st.selectbox(
            "目标值字段", columns, index=columns.index(guesses["target_column"])
        )
        item_columns = st.multiselect(
            "序列维度字段",
            columns,
            default=[column for column in guesses["item_columns"] if column in columns],
        )
        known_covariates = st.multiselect(
            "已知未来变量", [c for c in columns if c not in item_columns]
        )
        past_covariates = st.multiselect("历史变量", [c for c in columns if c not in item_columns])
        static_features = st.multiselect("静态属性", [c for c in columns if c not in item_columns])
        duplicate_strategy = st.selectbox("重复记录处理", ["block", "sum", "last", "mean"], index=1)
        missing_strategy = st.selectbox(
            "目标缺失处理", ["block", "zero", "ffill", "interpolate"], index=0
        )
        run_quality = st.button("标准化并检查质量", type="primary", use_container_width=True)
    with right:
        st.markdown("#### 数据预览与质量")
        if run_quality:
            normalized = normalize_finance_dataframe(
                raw,
                timestamp_column=timestamp_column,
                target_column=target_column,
                item_columns=item_columns,
                known_covariates=known_covariates,
                past_covariates=past_covariates,
                static_features=static_features,
                duplicate_strategy=duplicate_strategy,
                missing_strategy=missing_strategy,
            )
            profile = profile_normalized_data(normalized)
            st.session_state.mapping = {
                "timestamp_column": timestamp_column,
                "target_column": target_column,
                "item_columns": item_columns,
                "known_covariates": known_covariates,
                "past_covariates": past_covariates,
                "static_features": static_features,
                "duplicate_strategy": duplicate_strategy,
                "missing_strategy": missing_strategy,
            }
            st.session_state.normalized_data = normalized
            st.session_state.data_profile = profile
        if st.session_state.normalized_data is not None:
            st.dataframe(
                st.session_state.normalized_data.head(20), use_container_width=True, hide_index=True
            )
            render_quality_profile(st.session_state.data_profile)

if step == "3 预测配置":
    if st.session_state.data_profile is None:
        st.warning("请先完成字段映射和数据质量检查。")
        st.stop()
    profile = st.session_state.data_profile
    default_length = DEFAULT_PREDICTION_LENGTH[profile.frequency]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 预测范围")
        experiment_name = st.text_input(
            "实验名称",
            value=f"{st.session_state.mapping['target_column']}-{datetime.now():%Y%m%d}",
        )
        prediction_length = st.number_input(
            "预测周期", min_value=1, max_value=60, value=default_length
        )
    with c2:
        st.markdown("#### 回测与指标")
        requested_windows = st.number_input("回测窗口", min_value=1, max_value=6, value=3)
        supported_windows = estimate_supported_backtest_windows(
            st.session_state.normalized_data,
            prediction_length=int(prediction_length),
            requested_windows=int(requested_windows),
        )
        st.caption(f"当前数据支持 {supported_windows} 个窗口；不足时会自动降级。")
        metric = st.selectbox("主评测指标", ["WAPE"], disabled=True)
    with c3:
        st.markdown("#### 训练模式")
        mode = st.selectbox("模式", list(TRAINING_PRESETS), index=1)
        preset_config = TRAINING_PRESETS[mode]
        time_limit = st.number_input(
            "时间预算（秒）",
            min_value=120,
            max_value=14400,
            value=int(preset_config["time_limit_seconds"]),
        )
        st.caption(preset_config["description"])
    st.session_state.experiment_runtime_config = {
        "experiment_name": experiment_name,
        "prediction_length": int(prediction_length),
        "num_val_windows": int(supported_windows),
        "val_step_size": int(prediction_length),
        "preset": preset_config["preset"],
        "time_limit_seconds": int(time_limit),
        "metric": metric,
        "mode": mode,
    }

if step == "4 确认运行":
    if st.session_state.get("experiment_runtime_config") is None:
        st.warning("请先完成预测配置。")
        st.stop()
    profile = st.session_state.data_profile
    config_view = st.session_state.experiment_runtime_config
    st.markdown('<div class="fl-card">', unsafe_allow_html=True)
    st.markdown("#### 启动前确认")
    st.write(
        {
            "数据范围": f"{profile.data_start} 至 {profile.data_end}",
            "序列数": profile.item_count,
            "目标字段": st.session_state.mapping["target_column"],
            "预测频率": profile.frequency,
            "预测周期": config_view["prediction_length"],
            "回测窗口": config_view["num_val_windows"],
            "训练模式": config_view["mode"],
            "时间预算": config_view["time_limit_seconds"],
        }
    )
    if profile.warning_count:
        st.warning("当前数据存在警告项，建议将结果视为方向性验证。")
    can_run = profile.blocking_issue_count == 0
    if not can_run:
        st.error("仍存在阻断问题，请返回字段与质量步骤处理后再运行。")
    if st.button("开始评测", type="primary", disabled=not can_run):
        experiment_id = st.session_state.draft_experiment_id
        experiment_dir = get_experiment_dir(settings, experiment_id)
        mapping = st.session_state.mapping
        runtime = st.session_state.experiment_runtime_config
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=runtime["experiment_name"],
            source_file=st.session_state.uploaded_path,
            sheet_name=st.session_state.sheet_name,
            timestamp_column=mapping["timestamp_column"],
            target_column=mapping["target_column"],
            item_columns=mapping["item_columns"],
            known_covariates=mapping["known_covariates"],
            past_covariates=mapping["past_covariates"],
            static_features=mapping["static_features"],
            freq=profile.frequency,
            prediction_length=runtime["prediction_length"],
            num_val_windows=runtime["num_val_windows"],
            val_step_size=runtime["val_step_size"],
            preset=runtime["preset"],
            time_limit_seconds=runtime["time_limit_seconds"],
            duplicate_strategy=mapping["duplicate_strategy"],
            missing_strategy=mapping["missing_strategy"],
        )
        write_parquet(st.session_state.normalized_data, experiment_dir / "normalized_data.parquet")
        write_json(experiment_dir / "config.json", asdict(config))
        write_json(experiment_dir / "data_profile.json", profile_to_json(profile))
        repository.create_experiment(
            config,
            data_start=profile.data_start,
            data_end=profile.data_end,
            item_count=profile.item_count,
            row_count=profile.row_count,
        )
        start_training_job(settings=settings, repository=repository, experiment_id=experiment_id)
        st.query_params["experiment_id"] = experiment_id
        st.switch_page("pages/run_status.py")
