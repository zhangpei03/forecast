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
from src.services.forecast_driver_service import (
    AVAILABILITY_HISTORICAL,
    AVAILABILITY_KNOWN_FUTURE,
    CONFIG_TYPE_CALENDAR_FACTOR,
    CONFIG_TYPE_COVARIATE,
    CONFIG_TYPE_GROWTH_RATE,
    CONFIG_TYPE_SCENARIO_COVARIATE,
    FUTURE_VALUE_LAST,
    FUTURE_VALUE_MEAN,
    historical_covariate_columns,
    known_covariate_columns,
    normalize_driver_configs,
    validate_driver_configs,
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
if "driver_configs" not in st.session_state:
    st.session_state.driver_configs = []

if step == "1 上传数据":
    with st.container(border=True):
        uploaded = st.file_uploader("上传 .xlsx 文件", type=["xlsx"], accept_multiple_files=False)
        st.caption("单文件不超过 50MB。示例数据可通过 `make demo-data` 生成。")
        if uploaded:
            if st.session_state.get("uploaded_source_name") != uploaded.name:
                st.session_state.driver_configs = []
                st.session_state.uploaded_source_name = uploaded.name
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
        candidate_columns = [
            column
            for column in columns
            if column not in {timestamp_column, target_column, *item_columns}
        ]
        covariate_candidates = st.multiselect(
            "影响因子与协变量候选字段",
            candidate_columns,
            help="仅保留数值字段。具体类型与未来值规则将在预测配置步骤维护。",
        )
        static_features = st.multiselect(
            "静态属性",
            [column for column in candidate_columns if column not in covariate_candidates],
        )
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
                known_covariates=covariate_candidates,
                past_covariates=[],
                static_features=static_features,
                duplicate_strategy=duplicate_strategy,
                missing_strategy=missing_strategy,
            )
            profile = profile_normalized_data(normalized)
            st.session_state.mapping = {
                "timestamp_column": timestamp_column,
                "target_column": target_column,
                "item_columns": item_columns,
                "covariate_candidates": covariate_candidates,
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

    st.markdown("#### 预测驱动配置")
    st.caption("协变量会进入支持该类型的模型；增长率按每预测期复利调整所有候选模型的预测区间。")
    driver_type = st.selectbox(
        "配置类型",
        ["协变量", "增长率", "节假日/事件影响", "手工情景协变量"],
        key="driver_config_type",
    )
    if driver_type == "协变量":
        candidate_columns = st.session_state.mapping.get("covariate_candidates", [])
        if not candidate_columns:
            st.info("请先在字段与质量步骤选择至少一个影响因子与协变量候选字段。")
        else:
            with st.form("add_covariate_driver"):
                left, middle, right = st.columns(3)
                name = left.text_input("配置名称", placeholder="例如：工作日数")
                column = middle.selectbox("数据字段", candidate_columns)
                availability_label = right.selectbox(
                    "可用性类型", ["已知未来", "历史滞后"], help="已知未来变量可作为预测期输入。"
                )
                strategy_label = "历史末值延续"
                if availability_label == "已知未来":
                    strategy_label = st.selectbox(
                        "未来值生成规则",
                        ["历史末值延续", "历史均值延续"],
                        help="未来预测时按该规则生成协变量；回测阶段使用对应期间的实际协变量。",
                    )
                submitted = st.form_submit_button("添加协变量")
            if submitted:
                if not name.strip():
                    st.error("请填写配置名称。")
                elif any(
                    item.get("name") == name.strip() for item in st.session_state.driver_configs
                ):
                    st.error(f"配置名称“{name.strip()}”已存在。")
                elif any(item.get("column") == column for item in st.session_state.driver_configs):
                    st.error(f"字段“{column}”已配置。")
                else:
                    st.session_state.driver_configs.append(
                        {
                            "name": name.strip(),
                            "config_type": CONFIG_TYPE_COVARIATE,
                            "column": column,
                            "availability": (
                                AVAILABILITY_KNOWN_FUTURE
                                if availability_label == "已知未来"
                                else AVAILABILITY_HISTORICAL
                            ),
                            "future_value_strategy": (
                                FUTURE_VALUE_MEAN
                                if strategy_label == "历史均值延续"
                                else FUTURE_VALUE_LAST
                            ),
                        }
                    )
                    st.rerun()
    elif driver_type == "增长率":
        with st.form("add_growth_rate_driver"):
            left, right = st.columns(2)
            name = left.text_input("配置名称", value="经营增长假设")
            growth_rate_percent = right.number_input(
                "每预测期增长率 (%)", min_value=-99.99, max_value=1000.0, value=0.0, step=0.1
            )
            submitted = st.form_submit_button("添加增长率")
        if submitted:
            if not name.strip():
                st.error("请填写配置名称。")
            elif any(item.get("name") == name.strip() for item in st.session_state.driver_configs):
                st.error(f"配置名称“{name.strip()}”已存在。")
            else:
                st.session_state.driver_configs.append(
                    {
                        "name": name.strip(),
                        "config_type": CONFIG_TYPE_GROWTH_RATE,
                        "growth_rate": float(growth_rate_percent) / 100,
                    }
                )
                st.rerun()
    elif driver_type == "节假日/事件影响":
        with st.form("add_calendar_factor_driver"):
            left, middle, right = st.columns(3)
            name = left.text_input("事件名称", value="五一与国庆影响")
            impact_months = middle.multiselect(
                "影响月份",
                options=list(range(1, 13)),
                default=[5, 10],
                format_func=lambda month: f"{month}月",
            )
            impact_rate_percent = right.number_input(
                "影响率 (%)", min_value=-99.99, max_value=1000.0, value=0.0, step=0.1
            )
            submitted = st.form_submit_button("添加事件影响")
        if submitted:
            if not name.strip() or not impact_months:
                st.error("请填写事件名称并选择至少一个影响月份。")
            elif any(item.get("name") == name.strip() for item in st.session_state.driver_configs):
                st.error(f"配置名称“{name.strip()}”已存在。")
            else:
                st.session_state.driver_configs.append(
                    {
                        "name": name.strip(),
                        "config_type": CONFIG_TYPE_CALENDAR_FACTOR,
                        "impact_months": impact_months,
                        "impact_rate": float(impact_rate_percent) / 100,
                    }
                )
                st.rerun()
    else:
        with st.form("add_scenario_covariate_driver"):
            left, middle, right = st.columns(3)
            name = left.text_input("协变量名称", placeholder="例如：计划订单数或燃油指数")
            base_value = middle.number_input("基准值", min_value=0.0001, value=100.0, step=1.0)
            scenario_growth_percent = right.number_input(
                "每期变化率 (%)", min_value=-99.99, max_value=1000.0, value=0.0, step=0.1
            )
            effect_rate_percent = st.number_input(
                "目标影响系数 (%)",
                min_value=-1000.0,
                max_value=1000.0,
                value=0.0,
                step=1.0,
                help="协变量每变化 1%，预测目标相应变化的百分比。例如 80% 表示订单数增长 1%，目标增长 0.8%。",
            )
            submitted = st.form_submit_button("添加手工情景协变量")
        if submitted:
            if not name.strip():
                st.error("请填写协变量名称。")
            elif any(item.get("name") == name.strip() for item in st.session_state.driver_configs):
                st.error(f"配置名称“{name.strip()}”已存在。")
            else:
                st.session_state.driver_configs.append(
                    {
                        "name": name.strip(),
                        "config_type": CONFIG_TYPE_SCENARIO_COVARIATE,
                        "base_value": float(base_value),
                        "scenario_growth_rate": float(scenario_growth_percent) / 100,
                        "effect_rate": float(effect_rate_percent) / 100,
                    }
                )
                st.rerun()

    if st.session_state.driver_configs:
        rows = []
        for driver in st.session_state.driver_configs:
            if driver["config_type"] == CONFIG_TYPE_COVARIATE:
                rows.append(
                    {
                        "名称": driver["name"],
                        "类型": "协变量",
                        "字段": driver["column"],
                        "可用性": "已知未来"
                        if driver["availability"] == AVAILABILITY_KNOWN_FUTURE
                        else "历史滞后",
                        "规则": "历史均值延续"
                        if driver.get("future_value_strategy") == FUTURE_VALUE_MEAN
                        else "历史末值延续",
                    }
                )
            elif driver["config_type"] == CONFIG_TYPE_GROWTH_RATE:
                rows.append(
                    {
                        "名称": driver["name"],
                        "类型": "增长率",
                        "字段": "—",
                        "可用性": "全部候选模型",
                        "规则": f"每期 {float(driver['growth_rate']) * 100:.2f}%",
                    }
                )
            elif driver["config_type"] == CONFIG_TYPE_CALENDAR_FACTOR:
                rows.append(
                    {
                        "名称": driver["name"],
                        "类型": "节假日/事件影响",
                        "字段": "—",
                        "可用性": "影响月份："
                        + "、".join(f"{month}月" for month in driver.get("impact_months", [])),
                        "规则": f"影响率 {float(driver.get('impact_rate') or 0) * 100:.2f}%",
                    }
                )
            else:
                rows.append(
                    {
                        "名称": driver["name"],
                        "类型": "手工情景协变量",
                        "字段": "无历史字段",
                        "可用性": "输出未来假设",
                        "规则": (
                            f"基准 {float(driver.get('base_value') or 0):,.2f}，"
                            f"每期 {float(driver.get('scenario_growth_rate') or 0) * 100:.2f}%，"
                            f"影响系数 {float(driver.get('effect_rate') or 0) * 100:.2f}%"
                        ),
                    }
                )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        delete_name = st.selectbox(
            "删除配置", [driver["name"] for driver in st.session_state.driver_configs]
        )
        if st.button("删除所选配置"):
            st.session_state.driver_configs = [
                driver
                for driver in st.session_state.driver_configs
                if driver["name"] != delete_name
            ]
            st.rerun()

    st.session_state.experiment_runtime_config = {
        "experiment_name": experiment_name,
        "prediction_length": int(prediction_length),
        "num_val_windows": int(supported_windows),
        "val_step_size": int(prediction_length),
        "preset": preset_config["preset"],
        "time_limit_seconds": int(time_limit),
        "metric": metric,
        "mode": mode,
        "driver_configs": list(st.session_state.driver_configs),
    }

if step == "4 确认运行":
    if st.session_state.get("experiment_runtime_config") is None:
        st.warning("请先完成预测配置。")
        st.stop()
    profile = st.session_state.data_profile
    config_view = st.session_state.experiment_runtime_config
    driver_configs = normalize_driver_configs(config_view["driver_configs"])
    driver_errors = validate_driver_configs(driver_configs, st.session_state.normalized_data)
    known_covariates = known_covariate_columns(driver_configs)
    historical_covariates = historical_covariate_columns(driver_configs)
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
            "已知未来协变量": "、".join(known_covariates) or "—",
            "历史滞后协变量": "、".join(historical_covariates) or "—",
            "增长率配置": "、".join(
                f"{driver.name}（{float(driver.growth_rate or 0) * 100:.2f}%/期）"
                for driver in driver_configs
                if driver.config_type == CONFIG_TYPE_GROWTH_RATE
            )
            or "—",
            "事件影响配置": "、".join(
                f"{driver.name}（{'/'.join(f'{month}月' for month in driver.impact_months)}，{float(driver.impact_rate or 0) * 100:.2f}%）"
                for driver in driver_configs
                if driver.config_type == CONFIG_TYPE_CALENDAR_FACTOR
            )
            or "—",
            "手工情景协变量": "、".join(
                f"{driver.name}（基准 {float(driver.base_value or 0):,.2f}，每期 {float(driver.scenario_growth_rate or 0) * 100:.2f}%）"
                for driver in driver_configs
                if driver.config_type == CONFIG_TYPE_SCENARIO_COVARIATE
            )
            or "—",
        }
    )
    if profile.warning_count:
        st.warning("当前数据存在警告项，建议将结果视为方向性验证。")
    if driver_errors:
        for error in driver_errors:
            st.error(error)
    can_run = profile.blocking_issue_count == 0 and not driver_errors
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
            known_covariates=known_covariates,
            past_covariates=historical_covariates,
            static_features=mapping["static_features"],
            driver_configs=driver_configs,
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
