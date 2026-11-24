from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from feishubot.diagnosis.card import FEEDBACK_GUIDANCE, render_card
from feishubot.diagnosis.models import DiagnosisContext, DiagnosisStatus


@pytest.fixture
def context() -> DiagnosisContext:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return DiagnosisContext(
        diagnosis_id="diag-1",
        source_message_id="source-1",
        card_message_id="card-1",
        chat_id="chat-1",
        chat_type="p2p",
        owner_open_id="owner-1",
        session_key="session-1",
        question="订单为什么查不到",
        created_at=now,
        updated_at=now,
        stage_text="正在检查订单查询链路",
    )


@pytest.mark.parametrize(
    ("status", "color", "text"),
    [
        (DiagnosisStatus.ACCEPTED, "blue", "分析中"),
        (DiagnosisStatus.ANALYZING, "blue", "分析中"),
        (DiagnosisStatus.WAITING_SUPPLEMENT, "orange", "等待补充"),
        (DiagnosisStatus.COMPLETED, "green", "诊断完成"),
        (DiagnosisStatus.CANCELLED, "grey", "已取消"),
        (DiagnosisStatus.TIMED_OUT, "red", "诊断超时"),
        (DiagnosisStatus.PERMISSION_DENIED, "red", "权限不足"),
        (DiagnosisStatus.FAILED, "red", "诊断失败"),
    ],
)
def test_status_color_and_text(context, status, color, text):
    card = render_card(replace(context, status=status, result_text="结果"))

    assert card["header"]["template"] == color
    assert text in json.dumps(card, ensure_ascii=False)


def test_empty_projects_and_running_stage(context):
    rendered = json.dumps(render_card(context), ensure_ascii=False)

    assert "未明确识别" in rendered
    assert context.stage_text in rendered


@pytest.mark.parametrize(
    ("status", "progress"),
    [
        (DiagnosisStatus.ACCEPTED, "▰▱▱▱  已受理"),
        (DiagnosisStatus.ANALYZING, "▰▰▱▱  正在分析"),
        (DiagnosisStatus.WAITING_SUPPLEMENT, "▰▰▰▱  等待补充"),
        (DiagnosisStatus.COMPLETED, "▰▰▰▰  已完成"),
        (DiagnosisStatus.CANCELLED, "▰▰▱▱  流程已取消"),
        (DiagnosisStatus.TIMED_OUT, "▰▰▱▱  流程已超时"),
        (DiagnosisStatus.PERMISSION_DENIED, "▰▰▱▱  权限校验未通过"),
        (DiagnosisStatus.FAILED, "▰▰▱▱  流程已失败"),
    ],
)
def test_progress_bar_tracks_lifecycle_without_percentages(context, status, progress):
    rendered = json.dumps(render_card(replace(context, status=status)), ensure_ascii=False)

    assert "诊断阶段" in rendered
    assert progress in rendered
    assert "%" not in rendered


def test_tools_only_claim_checks_started(context):
    card = render_card(
        replace(
            context,
            tool_names=(
                "Read",
                "mcp__aliyun-dms-mcp-server__executeScript",
                "observability.traces",
            ),
        )
    )
    rendered = json.dumps(card, ensure_ascii=False)

    assert "源码检查：已发起检查" in rendered
    assert "ERP数据检查：已发起检查" in rendered
    assert "日志与链路检查：已发起检查" in rendered
    assert "工具已发起不代表已取得证据" in rendered
    assert "成功" not in rendered
    assert "已取证" not in rendered


@pytest.mark.parametrize(
    "status",
    [
        DiagnosisStatus.WAITING_SUPPLEMENT,
        DiagnosisStatus.COMPLETED,
        DiagnosisStatus.CANCELLED,
        DiagnosisStatus.TIMED_OUT,
        DiagnosisStatus.PERMISSION_DENIED,
        DiagnosisStatus.FAILED,
    ],
)
def test_feedback_guidance_present_on_terminal_statuses(context, status):
    rendered = json.dumps(
        render_card(replace(context, status=status, result_text="完成")),
        ensure_ascii=False,
    )

    assert FEEDBACK_GUIDANCE in rendered
    assert '"tag": "action"' not in rendered


@pytest.mark.parametrize(
    "status", [DiagnosisStatus.ACCEPTED, DiagnosisStatus.ANALYZING]
)
def test_feedback_guidance_absent_while_running(context, status):
    rendered = json.dumps(render_card(replace(context, status=status)), ensure_ascii=False)

    assert FEEDBACK_GUIDANCE not in rendered
    assert '"tag": "action"' not in rendered


def test_long_content_is_trimmed_and_serializable(context):
    card = render_card(replace(context, question="长" * 20_000))
    serialized = json.dumps(card, ensure_ascii=False)

    assert "内容过长，已截断" in serialized
    visible = "".join(
        element.get("content", "")
        for element in card["elements"]
        if element.get("tag") == "markdown"
    )
    assert len(visible) < 12_500


def test_long_question_does_not_hide_final_result(context):
    card = render_card(
        replace(
            context,
            question="问" * 20_000,
            status=DiagnosisStatus.COMPLETED,
            result_text="核心结论必须保留",
        )
    )

    assert "核心结论必须保留" in json.dumps(card, ensure_ascii=False)
    assert FEEDBACK_GUIDANCE in json.dumps(card, ensure_ascii=False)
