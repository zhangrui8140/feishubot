from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from feishubot.diagnosis.models import DiagnosisContext, DiagnosisStatus, FeedbackAction
from feishubot.diagnosis.runtime import (
    DiagnosisRuntime,
    parse_feedback_text,
    record_text_feedback,
)
from feishubot.diagnosis.store import DiagnosisStore


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
        question="问题",
        created_at=now,
        updated_at=now,
        status=DiagnosisStatus.COMPLETED,
    )


@pytest.fixture
def runtime(tmp_path, context) -> DiagnosisRuntime:
    store = DiagnosisStore(tmp_path)
    store.save(context)
    return DiagnosisRuntime(SimpleNamespace(), store)


def make_reply(
    text: str,
    *,
    parent_id: str | None = "card-1",
    chat_id: str = "chat-1",
) -> SimpleNamespace:
    return SimpleNamespace(message_id="reply-1", chat_id=chat_id, parent_id=parent_id)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1", FeedbackAction.CORRECT),
        (" 2 ", FeedbackAction.SOLVED),
        ("3", FeedbackAction.MISMATCH),
        ("4", FeedbackAction.CONTINUE),
        ("结论正确", FeedbackAction.CORRECT),
        ("问题已解决", FeedbackAction.SOLVED),
        ("结论不符", FeedbackAction.MISMATCH),
        ("继续排查", FeedbackAction.CONTINUE),
    ],
)
def test_parse_feedback_text_maps_numbers_and_labels(text, expected):
    assert parse_feedback_text(text) is expected


@pytest.mark.parametrize("text", ["5", "0", "结论正确吗", "随便", "", "1 结论正确"])
def test_parse_feedback_text_unknown_returns_none(text):
    assert parse_feedback_text(text) is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1", "反馈已记录"),
        ("3", "请回复本卡片，补充具体差异、单号或截图后继续。"),
    ],
)
def test_record_text_feedback_writes_and_maps_reply(tmp_path, runtime, text, message):
    outcome = record_text_feedback(
        runtime.store, make_reply(text), text, "owner-1", runtime
    )

    assert outcome is not None
    assert outcome.ok
    assert outcome.message == message
    feedback_files = list((tmp_path / "feedback").rglob("*.json"))
    assert len(feedback_files) == 1


def test_record_text_feedback_without_reply_returns_none(runtime):
    outcome = record_text_feedback(
        runtime.store,
        make_reply("1", parent_id=None),
        "1",
        "owner-1",
        runtime,
    )

    assert outcome is None


def test_record_text_feedback_wrong_owner_returns_none(runtime):
    outcome = record_text_feedback(
        runtime.store, make_reply("1"), "1", "other-owner", runtime
    )

    assert outcome is None


def test_record_text_feedback_while_running_is_declined(tmp_path, runtime, context):
    running_context = replace(context, status=DiagnosisStatus.ANALYZING)
    runtime.store.save(running_context)

    outcome = record_text_feedback(
        runtime.store, make_reply("1"), "1", "owner-1", runtime
    )

    assert outcome is not None
    assert not outcome.ok
    assert outcome.message == "当前诊断无需反馈"
    assert not list((tmp_path / "feedback").rglob("*.json"))


def test_record_text_feedback_is_idempotent(tmp_path, runtime):
    first = record_text_feedback(
        runtime.store, make_reply("1"), "1", "owner-1", runtime
    )
    second = record_text_feedback(
        runtime.store, make_reply("1"), "1", "owner-1", runtime
    )

    assert first is not None and first.ok and not first.duplicate
    assert second is not None and second.ok and second.duplicate
    assert len(list((tmp_path / "feedback" / "diag-1").glob("*.json"))) == 1


def test_context_from_reply_and_latest_context(runtime, context):
    # Lookup by card_message_id
    msg_card = SimpleNamespace(message_id="reply-1", chat_id="chat-1", parent_id="card-1")
    found = runtime.context_from_reply(msg_card)
    assert found is not None
    assert found.diagnosis_id == context.diagnosis_id

    # Lookup by source_message_id
    msg_source = SimpleNamespace(message_id="reply-2", chat_id="chat-1", parent_id="source-1")
    found_source = runtime.context_from_reply(msg_source)
    assert found_source is not None
    assert found_source.diagnosis_id == context.diagnosis_id

    # Allow other members in group when check_owner=False
    found_other = runtime.context_from_reply(msg_card, user_open_id="other-user", check_owner=False)
    assert found_other is not None

    # Block other members when check_owner=True
    found_blocked = runtime.context_from_reply(msg_card, user_open_id="other-user", check_owner=True)
    assert found_blocked is None

    # Latest context by session_key
    latest = runtime.latest_context("session-1")
    assert latest is not None
    assert latest.diagnosis_id == context.diagnosis_id


def test_create_with_reuse_inherits_card_and_merges_projects(runtime, context):
    context_with_proj = replace(context, projects=("erp-mall",))
    runtime.store.save(context_with_proj)

    msg = SimpleNamespace(message_id="new-msg-1", chat_id="chat-1", chat_type="p2p", parent_id="card-1")
    async def fallback(text: str) -> None:
        pass

    controller = runtime.create(
        message=msg,
        question="分析 som-erp-distribute-prod 的异常",
        owner_open_id="owner-1",
        session_key="session-1",
        available_projects=("erp-mall", "som-erp-distribute-prod"),
        fallback=fallback,
        reuse=context_with_proj,
    )

    # Round incremented, card reused, projects merged
    assert controller.context.round_index == 1
    assert controller.context.card_message_id == "card-1"
    assert set(controller.context.projects) == {"erp-mall", "som-erp-distribute-prod"}
