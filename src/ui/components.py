from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import streamlit as st

from src.domain.models import DataProfile, ExperimentSummary


def page_header(title: str, subtitle: str, badge: str | None = None) -> None:
    badge_html = f' <span class="fl-badge fl-badge-success">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:18px;">
          <div>
            <h1 style="font-size:28px;line-height:1.25;margin:0;">{title}{badge_html}</h1>
            <div style="color:#667085;margin-top:8px;line-height:1.55;">{subtitle}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    mapping = {
        "SUCCEEDED": ("fl-badge-success", "成功"),
        "RUNNING": ("fl-badge-info", "运行中"),
        "QUEUED": ("fl-badge-info", "排队中"),
        "FAILED": ("fl-badge-danger", "失败"),
        "DRAFT": ("fl-badge-warning", "草稿"),
        "VALIDATED": ("fl-badge-warning", "已校验"),
        "CANCELLED": ("fl-badge-danger", "已取消"),
    }
    css_class, label = mapping.get(status, ("fl-badge-warning", status))
    return f'<span class="fl-badge {css_class}">{label}</span>'


def render_experiment_cards(experiments: list[ExperimentSummary]) -> None:
    rows = []
    for experiment in experiments:
        rows.append(
            {
                "实验名称": experiment.name,
                "目标指标": experiment.target_column or "—",
                "数据范围": _range_text(experiment.data_start, experiment.data_end),
                "序列数": experiment.item_count,
                "最佳模型": experiment.best_model or "—",
                "WAPE": _format_percent(experiment.best_wape),
                "相对基线提升": _format_percent(experiment.improvement_rate),
                "状态": experiment.status.value,
                "创建时间": experiment.created_at.strftime("%Y-%m-%d %H:%M"),
                "实验ID": experiment.id,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_quality_profile(profile: DataProfile) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("总记录数", f"{profile.row_count:,}")
    c2.metric("时间序列数", f"{profile.item_count:,}")
    c3.metric("时间范围", _range_text(profile.data_start, profile.data_end))
    c4.metric("平均长度", f"{profile.average_series_length:.1f}")
    c5.metric("阻断问题", profile.blocking_issue_count)
    c6.metric("警告", profile.warning_count)

    issue_rows = [
        {
            "问题类型": issue.issue_type,
            "级别": issue.severity,
            "说明": issue.message,
            "影响记录": issue.affected_count,
            "样例": "；".join(issue.sample),
        }
        for issue in profile.issues
    ]
    if issue_rows:
        st.dataframe(pd.DataFrame(issue_rows), hide_index=True, use_container_width=True)
    else:
        st.success("数据质量检查通过，未发现阻断问题。")


def profile_to_json(profile: DataProfile) -> dict:
    payload = asdict(profile)
    payload["issues"] = [asdict(issue) for issue in profile.issues]
    return payload


def _range_text(start: str | None, end: str | None) -> str:
    return f"{start or '—'} 至 {end or '—'}"


def _format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"
