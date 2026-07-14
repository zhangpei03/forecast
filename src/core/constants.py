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
    "YoY Weekday DoD (Holiday)",
    "YoY Weekday DoD (Weekday Only)",
    "YoY Weekday DoD (Anchor v1)",
    "YoY Weekday DoD (Anchor v2)",
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

# 供前端展示、按类别批量选择的统一模型目录。说明面向非算法用户，保持一行即可理解。
MODEL_CATEGORY_ORDER = (
    "业务基线",
    "统计模型",
    "机器学习",
    "深度学习",
    "Transformer",
    "预训练大模型",
)

MODEL_CATALOG = {
    (MODEL_FAMILY_BASELINE, "Last Value"): ("业务基线", "直接延续最近一期实际值，适合短期且变化平稳的指标。"),
    (MODEL_FAMILY_BASELINE, "Seasonal Naive"): (
        "业务基线",
        "复用上一季节相同周期的实际值，适合周度或月度规律稳定的指标。",
    ),
    (MODEL_FAMILY_BASELINE, "Rolling Mean"): (
        "业务基线",
        "使用最近若干期平均值平滑波动，适合没有明显趋势或季节性的指标。",
    ),
    (MODEL_FAMILY_BASELINE, "YoY"): ("业务基线", "以去年同期作为参考，适合年周期明显的日频指标。"),
    (MODEL_FAMILY_BASELINE, "WoW"): ("业务基线", "以最近一周同一天作为参考，适合周内节律稳定的日频指标。"),
    (MODEL_FAMILY_BASELINE, "MTD Daily Avg"): (
        "业务基线",
        "按当月已发生日期的平均水平外推，适合月内累计口径的日频指标。",
    ),
    (MODEL_FAMILY_BASELINE, "YoY Weekday WoW"): (
        "业务基线",
        "按去年同星期的周环比推演，并在比例异常时使用相邻同星期参考值修正。",
    ),
    (MODEL_FAMILY_BASELINE, "YoY Weekday DoD (Holiday)"): (
        "业务基线",
        "按去年同星期日环比推演；特殊假期时按相同假期代码对齐。",
    ),
    (MODEL_FAMILY_BASELINE, "YoY Weekday DoD (Weekday Only)"): (
        "业务基线",
        "只按去年同星期日环比推演，不使用特殊假期对齐。",
    ),
    (MODEL_FAMILY_BASELINE, "YoY Weekday DoD (Anchor v1)"): (
        "业务基线",
        "保留特殊假期日环比对齐，并将递推结果与去年同期水平锚点融合，抑制长窗口误差累积。",
    ),
    (MODEL_FAMILY_BASELINE, "YoY Weekday DoD (Anchor v2)"): (
        "业务基线",
        "按近期历史表现自适应融合日环比递推与年度水平锚点，分别处理特殊假期、假期后和普通日期。",
    ),
    (MODEL_FAMILY_CUSTOM, "AutoARIMA"): (
        "统计模型",
        "自动拟合自回归和移动平均关系，适合单序列趋势与周期较清晰的场景。",
    ),
    (MODEL_FAMILY_CUSTOM, "Prophet"): (
        "统计模型",
        "将趋势、季节性和节假日影响拆开建模，便于理解长期变化。",
    ),
    (MODEL_FAMILY_CUSTOM, "XGBoost"): (
        "机器学习",
        "用树模型学习历史滞后和业务因子的非线性组合关系。",
    ),
    (MODEL_FAMILY_AUTOGLUON, "SeasonalNaive"): (
        "统计模型",
        "AutoGluon 的季节性朴素模型，可作为自动化建模的稳定参照。",
    ),
    (MODEL_FAMILY_AUTOGLUON, "ETS"): (
        "统计模型",
        "用指数平滑同时刻画水平、趋势和季节性，适合规律较稳定的序列。",
    ),
    (MODEL_FAMILY_AUTOGLUON, "Theta"): (
        "统计模型",
        "通过趋势分解和指数平滑预测，适合中短期趋势较清晰的序列。",
    ),
    (MODEL_FAMILY_AUTOGLUON, "RecursiveTabular"): (
        "机器学习",
        "递归地用表格特征预测下一期，适合拥有可用协变量的连续预测场景。",
    ),
    (MODEL_FAMILY_AUTOGLUON, "DirectTabular"): (
        "机器学习",
        "为不同预测步长直接建模，可减少长预测链条的误差累积。",
    ),
    (MODEL_FAMILY_AUTOGLUON, "DeepAR"): (
        "深度学习",
        "用循环神经网络同时学习多条序列的共同模式和不确定性。",
    ),
    (MODEL_FAMILY_AUTOGLUON, "TemporalFusionTransformer"): (
        "Transformer",
        "用注意力机制结合历史和已知未来因子，适合多协变量的复杂时序。",
    ),
    (MODEL_FAMILY_AUTOGLUON, "PatchTST"): (
        "Transformer",
        "将时间序列切成片段后用 Transformer 学习长距离变化模式。",
    ),
    (MODEL_FAMILY_AUTOGLUON, "Chronos2"): (
        "预训练大模型",
        "预训练时序大模型，可迁移通用时间模式，适合希望比较基础模型与大模型的场景。",
    ),
}

PREDICTION_PROVENANCE_COLUMNS = (
    "预测依据",
    "预测基准日期",
    "预测基准值",
    "去年环比日期",
    "去年环比值",
    "去年环比对比日期",
    "去年环比对比值",
    "实际采用环比",
    "环比来源",
)

QUANTILE_LEVELS = [0.1, 0.5, 0.9]
