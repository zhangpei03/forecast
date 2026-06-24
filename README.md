# Forecast Lab 企业财务预测评测台

Forecast Lab 是一个阶段一 MVP：财务分析人员上传近三年 Excel 历史数据后，系统完成字段映射、数据质量检查、滚动回测、三类业务基线、AutoGluon TimeSeries 训练、指标评测、结果分析和 Excel 导出。

项目定位是评测“数据是否具备稳定预测性”，不是完整预算系统。

面向财务用户的完整操作、配置、指标和模型说明见：[财务预测使用说明](docs/财务预测使用说明.md)。

## 环境要求

- Python `>=3.12,<3.13`
- `uv`
- 本地 CPU 即可运行快速验证；AutoGluon 深度模型可能需要更长时间和网络下载依赖

## 安装与启动

```bash
uv sync
make demo-data
make run
```

启动后打开 Streamlit 显示的本地地址。

## 常用命令

```bash
make init-db        # 初始化 runtime/app.db
make demo-data     # 生成 sample_data 示例 Excel
make didi-example-data # 生成合成滴滴出行财务预测示例
make test          # 运行 pytest
make lint          # Ruff 检查与格式校验
make clean-runtime # 删除 runtime 目录
```

## Excel 字段要求

上传 `.xlsx` 文件后可映射任意列名，但标准化后必须包含：

- `timestamp`：月、周或日时间字段
- `target`：预测目标金额或指标
- `item_id`：由组织、科目、产品等维度拼接形成

示例字段：

| 期间 | 组织 | 科目 | 产品 | 实际值 | 工作日数 | 预算人数 | 是否春节 |
|---|---|---|---|---:|---:|---:|---:|

时间字段支持 `YYYY-MM-DD`、`YYYY/MM/DD`、`YYYY-MM`、`YYYYMM`、`YYYY年第M月`。金额字段支持千分位和括号负数。

## 训练模式

- 快速验证：`fast_training`，默认 600 秒
- 标准评测：`medium_quality`，默认 1800 秒
- 深度评测：`high_quality`，默认 3600 秒

Worker 独立子进程运行，页面通过 SQLite、`progress.json` 和 Parquet 结果文件轮询。

## 评测指标

- WAPE：整体绝对误差 / 整体实际值绝对值
- MAE：平均绝对误差
- Bias Rate：偏差金额 / 整体实际值绝对值
- Coverage：实际值落在 P10-P90 区间的比例
- Improvement：最佳模型相对最佳业务基线的 WAPE 改善率

三类业务基线始终参与统一排行榜：Last Value、Seasonal Naive、Rolling Mean。

## 数据隐私

所有上传文件、模型文件、Parquet 结果和导出 Excel 都保存在本机 `runtime/` 目录。项目不会上传财务明细到外部服务。

## 常见问题

AutoGluon 安装慢：首次 `uv sync` 会解析并下载较多机器学习依赖，可先使用已有环境或快速模式验证。

模型下载失败：切换到快速验证，或检查网络后重新运行。

内存不足：减少训练模式、序列数量或预测周期。

运行时间过长：降低时间预算，或先用基线结果判断数据质量。

清理本地数据：

```bash
make clean-runtime
```

## 合成滴滴出行示例

运行以下命令生成 [didi_finance_forecast_example.xlsx](sample_data/didi_finance_forecast_example.xlsx)：

```bash
make didi-example-data
```

该数据是可复现的合成数据，不代表滴滴实际经营或财务披露。工作簿包含 `财务月度` 与 `实验说明` 两个 Sheet，可按以下方式创建实验：

| 配置项 | 示例值 |
|---|---|
| 时间字段 | `期间` |
| 目标字段 | `实际金额（元）` |
| 序列维度 | `业务线`、`财务科目` |
| 影响因子候选 | `计划订单量（万单）`、`工作日数`、`燃油价格指数` |
| 静态属性 | `城市覆盖数` |
| 已知未来协变量 | `计划订单量（万单）`、`工作日数` |
| 历史滞后协变量 | `燃油价格指数` |
| 增长率 | 每预测期 `0.30%` |
| 事件影响 | 五一、国庆：维护影响月份 `5`、`10` 与影响率 |
| 协变量缺失 | 使用“手工情景协变量”，维护基准值、每期变化率、目标影响系数；系统单独输出未来假设 |
| 建议参数 | 月度、预测 6 期、回测 3 窗口、快速验证 |
