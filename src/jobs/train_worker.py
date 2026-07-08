from __future__ import annotations

# 必须在 import pandas / autogluon / pyarrow / xgboost 等任何重库之前完成,
# 否则 macOS arm64 上 libomp + Arrow 多线程会触发 condition_variable 死锁。
import os as _os

for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    # 防止 macOS fork 时 Objective-C runtime 与多线程 BLAS 冲突
    ("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES"),
    # AutoGluon / Ray 的并行限制
    ("TOKENIZERS_PARALLELISM", "false"),
):
    _os.environ.setdefault(_k, _v)

import argparse
import os
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.core.constants import (
    MODEL_FAMILY_ALL,
    MODEL_FAMILY_AUTOGLUON,
    MODEL_FAMILY_BASELINE,
    MODEL_FAMILY_CUSTOM,
)
from src.core.config import AppSettings
from src.core.exceptions import ForecastLabError
from src.core.logging import configure_worker_logger
from src.domain.enums import ExperimentStatus, WorkerStage
from src.domain.models import ExperimentConfig
from src.repositories.experiment_repository import ExperimentRepository, utc_now_iso
from src.services.autogluon_service import (
    AutoGluonUnavailableError,
    generate_autogluon_backtest_predictions,
    generate_autogluon_future_forecast,
)
from src.services.baseline_service import (
    generate_baseline_backtest_predictions,
    generate_baseline_future_forecast,
)
from src.services.custom_model_service import (
    CUSTOM_MODEL_NAMES,
    generate_custom_model_backtest_predictions,
    generate_custom_model_future_forecast,
)
from src.services.evaluation_service import (
    build_business_conclusion,
    calculate_improvement,
    calculate_prediction_metrics,
    choose_best_baseline,
    choose_best_model,
    evaluate_by_series,
    evaluate_by_window,
    evaluate_models,
)
from src.services.export_service import export_evaluation_workbook
from src.services.forecast_driver_service import (
    apply_forecast_driver_adjustments,
    build_future_driver_assumptions,
    normalize_driver_configs,
)
from src.storage.file_store import get_experiment_dir, read_json, write_json
from src.storage.parquet import read_parquet, write_parquet


def main() -> None:
    args = _parse_args()
    settings = AppSettings(
        runtime_dir=Path(args.runtime_dir),
        database_url=args.database_url,
    )
    repository = ExperimentRepository(args.database_url)
    experiment_dir = get_experiment_dir(settings, args.owner_ldap, args.experiment_id)
    logger = configure_worker_logger(experiment_dir / "run.log")

    def progress(stage: WorkerStage, percent: int, message: str, status: str = "RUNNING") -> None:
        payload = {
            "experiment_id": args.experiment_id,
            "status": status,
            "stage": stage.value,
            "progress": percent,
            "message": message,
            "started_at": read_json(experiment_dir / "progress.json").get("started_at")
            or utc_now_iso(),
            "updated_at": utc_now_iso(),
            "worker_pid": os.getpid(),
            "log_path": str(experiment_dir / "run.log"),
        }
        write_json(experiment_dir / "progress.json", payload)
        repository.update_run(
            args.run_id,
            status=status,
            stage=stage.value,
            progress=percent,
            worker_pid=os.getpid(),
        )
        logger.info("%s %s", stage.value, message)

    try:
        repository.update_status(
            args.experiment_id, ExperimentStatus.RUNNING, owner_ldap=args.owner_ldap
        )
        progress(WorkerStage.LOAD_DATA, 5, "正在读取标准化财务数据")
        summary = repository.get_experiment(args.experiment_id, args.owner_ldap)
        if summary is None:
            raise ForecastLabError("实验不存在或已被删除")
        config = ExperimentConfig(**summary.config)
        driver_configs = normalize_driver_configs(config.driver_configs)
        selected_family = config.selected_model_family or MODEL_FAMILY_ALL
        selected_model = config.selected_model_name
        normalized_data = read_parquet(experiment_dir / "normalized_data.parquet")
        if normalized_data.empty:
            raise ForecastLabError("标准化数据不存在，请重新上传并保存实验")

        progress(WorkerStage.BUILD_BACKTEST_WINDOWS, 15, "正在构建时间顺序滚动回测窗口")
        run_all_models = selected_family == MODEL_FAMILY_ALL
        all_predictions: list[pd.DataFrame] = []

        if run_all_models or selected_family == MODEL_FAMILY_BASELINE:
            progress(WorkerStage.TRAIN_BASELINES, 28, "正在训练并评估业务基线")
            baseline_predictions = generate_baseline_backtest_predictions(
                data=normalized_data,
                freq=config.freq,
                prediction_length=config.prediction_length,
                num_windows=config.num_val_windows,
                models=None if run_all_models else [selected_model],
            )
            if not baseline_predictions.empty:
                all_predictions.append(baseline_predictions)

        if run_all_models or selected_family == MODEL_FAMILY_CUSTOM:
            progress(WorkerStage.TRAIN_BASELINES, 34, "正在训练并评估外部模型")
            custom_predictions, custom_failures = generate_custom_model_backtest_predictions(
                data=normalized_data,
                freq=config.freq,
                prediction_length=config.prediction_length,
                num_windows=config.num_val_windows,
                driver_configs=driver_configs,
                models=None if run_all_models else [selected_model],
            )
            if not custom_predictions.empty:
                all_predictions.append(custom_predictions)
            for failure in custom_failures:
                logger.warning("Custom model %s failed: %s", failure.model, failure.message)

        if run_all_models or selected_family == MODEL_FAMILY_AUTOGLUON:
            progress(WorkerStage.TRAIN_AUTOGLUON, 42, "正在训练 AutoGluon 候选模型")
            try:
                autogluon_predictions = generate_autogluon_backtest_predictions(
                    data=normalized_data,
                    config=config,
                    model_dir=experiment_dir / "models",
                    selected_model=None if run_all_models else selected_model,
                )
                if not autogluon_predictions.empty:
                    all_predictions.append(autogluon_predictions)
            except AutoGluonUnavailableError as exc:
                logger.warning("AutoGluon unavailable: %s", exc)
            except Exception as exc:
                logger.exception("AutoGluon training failed; continuing with available models: %s", exc)

        progress(WorkerStage.CALCULATE_METRICS, 64, "正在统一计算 WAPE、MAE、Bias 与覆盖率")
        backtest_predictions = apply_forecast_driver_adjustments(
            pd.concat(all_predictions, ignore_index=True), driver_configs
        )
        if backtest_predictions.empty:
            raise ForecastLabError("没有可用预测结果，请检查序列长度和预测周期")
        leaderboard = evaluate_models(backtest_predictions)
        best_model = choose_best_model(leaderboard)
        best_baseline = choose_best_baseline(leaderboard)
        baseline_model_name = (
            str(best_baseline["model"]) if best_baseline is not None else str(best_model["model"])
        )
        fallback_baseline_model = str(best_baseline["model"]) if best_baseline is not None else "Last Value"
        baseline_wape = (
            float(best_baseline["wape"])
            if best_baseline is not None and pd.notna(best_baseline["wape"])
            else None
        )

        window_metrics = evaluate_by_window(
            backtest_predictions,
            best_model=str(best_model["model"]),
            best_baseline=baseline_model_name,
        )
        improved_windows = int(window_metrics["improved"].sum()) if not window_metrics.empty else 0
        total_windows = int(window_metrics.shape[0])
        series_metrics = evaluate_by_series(backtest_predictions, str(best_model["model"]))
        high_risk_series_count = (
            int((series_metrics["wape"] > 0.30).sum()) if not series_metrics.empty else 0
        )
        aggregate_metric = calculate_prediction_metrics(
            backtest_predictions[backtest_predictions["model"].eq(best_model["model"])]
        )
        aggregate_metrics = pd.DataFrame([asdict(aggregate_metric)])
        conclusion = build_business_conclusion(
            best_model=str(best_model["model"]),
            best_wape=float(best_model["wape"]) if pd.notna(best_model["wape"]) else None,
            best_baseline=baseline_model_name,
            baseline_wape=baseline_wape,
            improved_windows=improved_windows,
            total_windows=total_windows,
            high_risk_series_count=high_risk_series_count,
        )

        progress(WorkerStage.GENERATE_FUTURE_FORECAST, 78, "正在生成未来预测区间")
        if str(best_model["model"]) in CUSTOM_MODEL_NAMES:
            try:
                future_forecast = generate_custom_model_future_forecast(
                    data=normalized_data,
                    freq=config.freq,
                    prediction_length=config.prediction_length,
                    model=str(best_model["model"]),
                    driver_configs=driver_configs,
                )
            except Exception as exc:
                logger.exception(
                    "Custom model future forecast failed; falling back to baseline: %s",
                    exc,
                )
                future_forecast = generate_baseline_future_forecast(
                    data=normalized_data,
                    freq=config.freq,
                    prediction_length=config.prediction_length,
                    model=fallback_baseline_model,
                )
        elif str(best_model["model_type"]) == "基线":
            future_forecast = generate_baseline_future_forecast(
                data=normalized_data,
                freq=config.freq,
                prediction_length=config.prediction_length,
                model=str(best_model["model"]),
            )
        else:
            try:
                future_forecast = generate_autogluon_future_forecast(
                    data=normalized_data,
                    config=config,
                    model_dir=experiment_dir / "models",
                    model_name=str(best_model["model"]),
                    selected_model=(
                        selected_model if selected_family == MODEL_FAMILY_AUTOGLUON else None
                    ),
                )
            except AutoGluonUnavailableError:
                future_forecast = generate_baseline_future_forecast(
                    data=normalized_data,
                    freq=config.freq,
                    prediction_length=config.prediction_length,
                    model=fallback_baseline_model,
                )
            except Exception as exc:
                logger.exception(
                    "AutoGluon future forecast failed; falling back to baseline: %s",
                    exc,
                )
                future_forecast = generate_baseline_future_forecast(
                    data=normalized_data,
                    freq=config.freq,
                    prediction_length=config.prediction_length,
                    model=fallback_baseline_model,
                )

        future_forecast = apply_forecast_driver_adjustments(future_forecast, driver_configs)
        future_driver_assumptions = build_future_driver_assumptions(
            normalized_data,
            driver_configs,
            freq=config.freq,
            prediction_length=config.prediction_length,
        )

        progress(WorkerStage.BUILD_EXPORT, 90, "正在落盘结果并生成 Excel 评测报告")
        quality_report = _quality_report_frame(experiment_dir / "data_profile.json")
        metadata = _runtime_metadata(config)
        write_json(
            experiment_dir / "results" / "conclusion.json", {"conclusion": conclusion, **metadata}
        )
        write_parquet(leaderboard, experiment_dir / "results" / "leaderboard.parquet")
        write_parquet(aggregate_metrics, experiment_dir / "results" / "aggregate_metrics.parquet")
        write_parquet(series_metrics, experiment_dir / "results" / "series_metrics.parquet")
        write_parquet(
            backtest_predictions, experiment_dir / "results" / "backtest_predictions.parquet"
        )
        write_parquet(future_forecast, experiment_dir / "results" / "future_forecast.parquet")
        write_parquet(
            future_driver_assumptions,
            experiment_dir / "results" / "future_driver_assumptions.parquet",
        )
        write_parquet(window_metrics, experiment_dir / "results" / "window_metrics.parquet")
        export_path = export_evaluation_workbook(
            output_dir=experiment_dir / "exports",
            experiment_name=config.name,
            conclusion=conclusion,
            leaderboard=leaderboard,
            aggregate_metrics=aggregate_metrics,
            series_metrics=series_metrics,
            backtest_predictions=backtest_predictions,
            future_forecast=future_forecast,
            future_driver_assumptions=future_driver_assumptions,
            quality_report=quality_report,
            config={**asdict(config), **metadata},
        )
        write_json(experiment_dir / "results" / "export.json", {"path": str(export_path)})

        improvement = calculate_improvement(
            float(best_model["wape"]) if pd.notna(best_model["wape"]) else None,
            baseline_wape,
        )
        repository.update_results(
            args.experiment_id,
            status=ExperimentStatus.SUCCEEDED,
            best_model=str(best_model["model"]),
            best_wape=float(best_model["wape"]) if pd.notna(best_model["wape"]) else None,
            baseline_wape=baseline_wape,
            improvement_rate=improvement,
            owner_ldap=args.owner_ldap,
        )
        progress(WorkerStage.COMPLETE, 100, "评测完成，可查看结果并导出报告", status="SUCCEEDED")
        repository.update_run(
            args.run_id,
            status="SUCCEEDED",
            stage=WorkerStage.COMPLETE.value,
            progress=100,
            ended_at=utc_now_iso(),
        )
    except ForecastLabError as exc:
        _fail(repository, args, experiment_dir, logger, exc.error_code, exc.message)
    except MemoryError:
        _fail(
            repository,
            args,
            experiment_dir,
            logger,
            "TRAINING_OOM",
            "内存不足，请减少模型模式或序列数量",
        )
    except TimeoutError:
        _fail(repository, args, experiment_dir, logger, "TRAINING_TIMEOUT", "达到训练时间上限")
    except Exception as exc:
        _fail(repository, args, experiment_dir, logger, "TRAINING_FAILED", f"训练失败：{exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--owner-ldap", required=True)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--runtime-dir", required=True)
    return parser.parse_args()


def _fail(
    repository: ExperimentRepository,
    args: argparse.Namespace,
    experiment_dir: Path,
    logger,
    error_code: str,
    error_message: str,
) -> None:
    logger.exception("%s %s", error_code, error_message)
    repository.update_status(
        args.experiment_id, ExperimentStatus.FAILED, owner_ldap=args.owner_ldap
    )
    repository.update_run(
        args.run_id,
        status="FAILED",
        stage="FAILED",
        progress=100,
        error_code=error_code,
        error_message=error_message,
        ended_at=utc_now_iso(),
    )
    write_json(
        experiment_dir / "progress.json",
        {
            "experiment_id": args.experiment_id,
            "status": "FAILED",
            "stage": "FAILED",
            "progress": 100,
            "message": error_message,
            "started_at": None,
            "updated_at": utc_now_iso(),
            "worker_pid": os.getpid(),
            "log_path": str(experiment_dir / "run.log"),
            "error_code": error_code,
        },
    )


def _quality_report_frame(profile_path: Path) -> pd.DataFrame:
    profile = read_json(profile_path)
    return pd.DataFrame(profile.get("issues", []))


def _runtime_metadata(config: ExperimentConfig) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "software_version": "0.1.0",
        "random_seed": str(config.random_seed),
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }


if __name__ == "__main__":
    main()
