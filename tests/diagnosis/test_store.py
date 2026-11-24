from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from feishubot.diagnosis.models import DiagnosisContext, FeedbackAction
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
        projects=("erp-mall",),
        created_at=now,
        updated_at=now,
    )


def test_roundtrip_and_card_index_survive_rebuild(tmp_path, context):
    store = DiagnosisStore(tmp_path)

    assert store.save(context)
    assert store.get(context.diagnosis_id) == context
    assert DiagnosisStore(tmp_path).get_by_card_message_id("card-1") == context
    assert DiagnosisStore(tmp_path).get_by_message_id("card-1") == context
    assert DiagnosisStore(tmp_path).get_by_message_id("source-1") == context
    assert DiagnosisStore(tmp_path).get_latest_by_session_key("session-1") == context
    assert DiagnosisStore(tmp_path).get_latest_by_session_key("unknown") is None


@pytest.mark.parametrize(
    ("operator", "card", "chat", "message"),
    [
        ("other", "card-1", "chat-1", "仅诊断发起人可反馈"),
        ("owner-1", "wrong", "chat-1", "卡片消息不匹配"),
        ("owner-1", "card-1", "wrong", "会话不匹配"),
    ],
)
def test_feedback_rejects_mismatch_without_writing(
    tmp_path, context, operator, card, chat, message
):
    store = DiagnosisStore(tmp_path)
    store.save(context)

    outcome = store.record_feedback("diag-1", operator, card, chat, "correct")

    assert not outcome.ok
    assert message in outcome.message
    assert not list((tmp_path / "feedback").rglob("*.json"))


@pytest.mark.parametrize("action", list(FeedbackAction))
def test_all_feedback_actions_are_idempotent_after_rebuild(tmp_path, context, action):
    store = DiagnosisStore(tmp_path)
    store.save(context)

    first = store.record_feedback("diag-1", "owner-1", "card-1", "chat-1", action)
    second = DiagnosisStore(tmp_path).record_feedback(
        "diag-1", "owner-1", "card-1", "chat-1", action
    )

    assert first.ok and not first.duplicate
    assert second.ok and second.duplicate
    assert len(list((tmp_path / "feedback" / "diag-1").glob("*.json"))) == 1


def test_disabled_store_is_safe(context):
    store = DiagnosisStore(None)

    assert not store.enabled
    assert not store.save(context)
    assert store.get("diag-1") is None
    assert not store.record_feedback("diag-1", "owner-1", "card-1", "chat-1", "correct").ok


def test_corrupt_files_fail_safely(tmp_path, context):
    store = DiagnosisStore(tmp_path)
    store.save(context)
    context_path = tmp_path / "contexts" / "diag-1.json"
    context_path.write_text("{broken", encoding="utf-8")

    assert store.get("diag-1") is None
    context_path.write_text("{}", encoding="utf-8")
    assert store.get_by_card_message_id("card-1") is None


def test_path_unsafe_diagnosis_id_is_rejected(tmp_path, context):
    store = DiagnosisStore(tmp_path)

    assert not store.save(replace(context, diagnosis_id="../escape"))
    assert store.get("../escape") is None
