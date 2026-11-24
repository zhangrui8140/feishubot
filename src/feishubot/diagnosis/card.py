from __future__ import annotations

from typing import Any

from feishubot.diagnosis.models import DiagnosisContext, DiagnosisStatus, FeedbackAction

_TEXT_BUDGET = 12_000
_QUESTION_BUDGET = 2_000
_STAGE_BUDGET = 2_000
_TRUNCATED = "\n\n内容过长，已截断"

_STATUS_STYLE = {
    DiagnosisStatus.ACCEPTED: ("分析中", "blue"),
    DiagnosisStatus.ANALYZING: ("分析中", "blue"),
    DiagnosisStatus.WAITING_SUPPLEMENT: ("等待补充", "orange"),
    DiagnosisStatus.COMPLETED: ("诊断完成", "green"),
    DiagnosisStatus.CANCELLED: ("已取消", "grey"),
    DiagnosisStatus.TIMED_OUT: ("诊断超时", "red"),
    DiagnosisStatus.PERMISSION_DENIED: ("权限不足", "red"),
    DiagnosisStatus.FAILED: ("诊断失败", "red"),
}
_PROGRESS_STYLE = {
    DiagnosisStatus.ACCEPTED: "▰▱▱▱  已受理",
    DiagnosisStatus.ANALYZING: "▰▰▱▱  正在分析",
    DiagnosisStatus.WAITING_SUPPLEMENT: "▰▰▰▱  等待补充",
    DiagnosisStatus.COMPLETED: "▰▰▰▰  已完成",
    DiagnosisStatus.CANCELLED: "▰▰▱▱  流程已取消",
    DiagnosisStatus.TIMED_OUT: "▰▰▱▱  流程已超时",
    DiagnosisStatus.PERMISSION_DENIED: "▰▰▱▱  权限校验未通过",
    DiagnosisStatus.FAILED: "▰▰▱▱  流程已失败",
}
FEEDBACK_STATUSES = {
    DiagnosisStatus.WAITING_SUPPLEMENT,
    DiagnosisStatus.COMPLETED,
    DiagnosisStatus.CANCELLED,
    DiagnosisStatus.TIMED_OUT,
    DiagnosisStatus.PERMISSION_DENIED,
    DiagnosisStatus.FAILED,
}
FEEDBACK_OPTIONS = (
    (FeedbackAction.CORRECT, "结论正确"),
    (FeedbackAction.SOLVED, "问题已解决"),
    (FeedbackAction.MISMATCH, "结论不符"),
    (FeedbackAction.CONTINUE, "继续排查"),
)
FEEDBACK_GUIDANCE = "回复本卡：1 结论正确 / 2 问题已解决 / 3 结论不符 / 4 继续排查"
_EVIDENCE_NOTE = "工具已发起不代表已取得证据，最终结论以诊断结果中的实际证据为准。"


def _tool_category(name: str) -> str | None:
    lowered = name.lower()
    if any(
        term in lowered
        for term in ("aliyun-dms", "executescript", "listtables", "gettabledetailinfo")
    ):
        return "ERP数据检查"
    if any(
        term in lowered
        for term in ("observability", "victorialogs", "prometheus", "traces")
    ):
        return "日志与链路检查"
    if lowered in {"read", "glob", "grep", "searchworkspace"}:
        return "源码检查"
    return None


def _activity_text(tool_names: tuple[str, ...]) -> str:
    categories: list[str] = []
    for name in tool_names:
        category = _tool_category(name)
        if category and category not in categories:
            categories.append(category)
    if not categories:
        return "尚未发起检查"
    return "\n".join(f"- {category}：已发起检查" for category in categories)


def _duration_text(duration_ms: int | None) -> str:
    if duration_ms is None:
        return "计算中"
    if duration_ms < 1_000:
        return f"{duration_ms} 毫秒"
    return f"{duration_ms / 1_000:.1f} 秒"


def _truncate(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    allowance = max(0, budget - len(_TRUNCATED))
    return text[:allowance] + _TRUNCATED


def _trim_sections(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    fixed = sum(len(title) for title, _ in sections)
    remaining = max(0, _TEXT_BUDGET - fixed)
    result: list[tuple[str, str]] = []
    for title, text in sections:
        if len(text) <= remaining:
            result.append((title, text))
            remaining -= len(text)
            continue
        if remaining > len(_TRUNCATED):
            result.append((title, _truncate(text, remaining)))
            remaining = 0
        else:
            if title == "诊断结果" or not result:
                result.append((title, _TRUNCATED.strip()))
            remaining = 0
    return result


def _markdown(title: str, text: str) -> dict[str, Any]:
    return {"tag": "markdown", "content": f"**{title}**\n{text}"}


def render_card(context: DiagnosisContext) -> dict[str, Any]:
    status_text, color = _STATUS_STYLE[context.status]
    projects = "、".join(context.projects) if context.projects else "未明确识别"
    sections = [
        ("问题摘要", _truncate(context.question or "未提供", _QUESTION_BUDGET)),
        ("环境", context.environment),
        ("实际项目", projects),
        ("诊断阶段", _PROGRESS_STYLE[context.status]),
        ("状态与耗时", f"{status_text}；耗时：{_duration_text(context.duration_ms)}"),
        ("活动账本", _activity_text(context.tool_names)),
    ]
    if context.tool_names:
        sections.append(("证据说明", _EVIDENCE_NOTE))
    if context.status in {DiagnosisStatus.ACCEPTED, DiagnosisStatus.ANALYZING}:
        sections.append(
            ("当前进展", _truncate(context.stage_text or "正在分析", _STAGE_BUDGET))
        )
    elif context.status in {DiagnosisStatus.WAITING_SUPPLEMENT, DiagnosisStatus.COMPLETED}:
        sections.append(("诊断结果", context.result_text or "暂无结果"))
    elif context.result_text:
        sections.append(("诊断结果", context.result_text))

    elements: list[dict[str, Any]] = [
        _markdown(title, text) for title, text in _trim_sections(sections)
    ]
    if context.status in FEEDBACK_STATUSES:
        elements.append({"tag": "markdown", "content": FEEDBACK_GUIDANCE})

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": color,
            "title": {"tag": "plain_text", "content": "ERP 诊断"},
        },
        "elements": elements,
    }
