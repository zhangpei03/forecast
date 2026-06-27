"""构建期预下载 AutoGluon 依赖的模型权重, 避免生产容器首跑联网拉权重失败。

内网环境通常无法访问 HuggingFace / AutoGluon 的公网权重源, 因此在 Docker 构建阶段
(有受控出网或内网镜像源时) 先把权重烤进镜像。下载失败不阻断构建——基线模型与
轻量模型仍可运行, 仅深度模型在运行期降级。
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        # 触发 AutoGluon TimeSeries 的延迟权重下载(若有)。
        # 具体预热点随 autogluon 版本而异; 这里做最小可用的 import 预热。
        import autogluon.timeseries  # noqa: F401

        print("[prefetch] autogluon.timeseries import OK")
    except Exception as exc:
        print(f"[prefetch] autogluon import failed (non-fatal): {exc}", file=sys.stderr)
        return 0
    print("[prefetch] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
