from __future__ import annotations

APP_NAME = "Forecast Lab"
RUNTIME_DIR = "runtime"
MAX_UPLOAD_MB = 50
DEFAULT_RANDOM_SEED = 123

FREQ_LABELS = {
    "M": "月度",
    "W": "周度",
    "D": "日度",
}

DEFAULT_PREDICTION_LENGTH = {
    "M": 3,
    "W": 8,
    "D": 30,
}

MIN_TRAIN_LENGTH = {
    "M": 24,
    "W": 52,
    "D": 90,
}

RECOMMENDED_TRAIN_LENGTH = {
    "M": 36,
    "W": 104,
    "D": 365,
}

ROLLING_MEAN_WINDOW = {
    "M": 3,
    "W": 4,
    "D": 7,
}

SEASONAL_LAG = {
    "M": 12,
    "W": 52,
    "D": 7,
}

TRAINING_PRESETS = {
    "快速验证": {
        "preset": "fast_training",
        "time_limit_seconds": 600,
        "description": "首次判断数据是否具备预测性。",
    },
    "标准评测": {
        "preset": "medium_quality",
        "time_limit_seconds": 1800,
        "description": "阶段一默认，兼顾模型丰富度和运行时间。",
    },
    "深度评测": {
        "preset": "high_quality",
        "time_limit_seconds": 3600,
        "description": "适合更大训练预算和更充分的模型比较。",
    },
}

MODEL_FAMILY_ALL = "all"
MODEL_FAMILY_BASELINE = "baseline"
MODEL_FAMILY_CUSTOM = "custom"
MODEL_FAMILY_AUTOGLUON = "autogluon"

BASELINE_MODEL_NAMES = (
    "Last Value",
    "Seasonal Naive",
    "Rolling Mean",
    "YoY",
    "WoW",
    "MTD Daily Avg",
    "YoY Weekday WoW",
    "YoY Weekday WoW Original",
    "YoY Weekday WoW Hybrid",
    "YoY Weekday DoD",
)

CUSTOM_MODEL_NAMES = ("AutoARIMA", "Prophet", "XGBoost")

AUTOGLUON_MODEL_NAMES = (
    "SeasonalNaive",
    "RecursiveTabular",
    "DirectTabular",
    "ETS",
    "Theta",
    "DeepAR",
    "TemporalFusionTransformer",
    "PatchTST",
    "Chronos2",
)

QUANTILE_LEVELS = [0.1, 0.5, 0.9]
