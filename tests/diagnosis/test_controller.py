from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from feishubot.diagnosis.controller import DiagnosisController
from feishubot.diagnosis.models import DiagnosisContext, DiagnosisStatus
from feishubot.diagnosis.store import DiagnosisStore
from feishubot.qoder_agent.models import AgentProgress


class FakeChannel:
    def __init__(self) -> None:
        self.sent = []
        self.replied = []
        self.patched = []
        self.fail_reply = False
        self.fail_send = False
        self.fail_patch = False

    async def send_interactive_message(self, receive_id, card, receive_id_type="open_id"):
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent.append((receive_id, receive_id_type, card))
        return {"data": {"message_id": "card-created"}}

    async def reply_interactive_message(self, message_id, card):
        if self.fail_reply:
            raise RuntimeError("reply failed")
        self.replied.append((message_id, card))
        return {"data": {"message_id": "card-replied"}}

    async def patch_interactive_message(self, message_id, card):
        if self.fail_patch:
            raise RuntimeError("patch failed")
        self.patched.append((message_id, card))
        return {}


@pytest.fixture
def context() -> DiagnosisContext:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return DiagnosisContext(
        diagnosis_id="diag-1",
        source_message_id="source-1",
        card_message_id=None,
        chat_id="chat-1",
        chat_type="p2p",
        owner_open_id="owner-1",
        session_key="session-1",
        question="订单为什么查不到",
        created_at=now,
        updated_at=now,
    )


def make_controller(tmp_path, context, channel, fallback, **kwargs):
    async def send_fallback(text: str) -> None:
        fallback.append(text)

    patch_interval_seconds = kwargs.pop("patch_interval_seconds", 0)
    return DiagnosisController(
        channel,
        DiagnosisStore(tmp_path),
        context,
        send_fallback,
        patch_interval_seconds=patch_interval_seconds,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_p2p_start_creates_and_persists_card(tmp_path, context):
    channel = FakeChannel()
    fallback = []
    controller = make_controller(tmp_path, context, channel, fallback)

    await controller.start("已受理")

    assert channel.sent[0][:2] == ("chat-1", "chat_id")
    assert controller.context.card_message_id == "card-created"
    assert DiagnosisStore(tmp_path).get_by_card_message_id("card-created") == controller.context
    assert fallback == []


@pytest.mark.asyncio
async def test_group_start_replies_to_source_message(tmp_path, context):
    channel = FakeChannel()
    fallback = []
    controller = make_controller(
        tmp_path,
        replace(context, chat_type="group"),
        channel,
        fallback,
    )

    await controller.start("已受理")

    assert channel.replied[0][0] == "source-1"
    assert channel.sent == []
    assert controller.context.card_message_id == "card-replied"


@pytest.mark.asyncio
async def test_group_reply_failure_sends_card_to_chat(tmp_path, context):
    channel = FakeChannel()
    channel.fail_reply = True
    fallback = []
    controller = make_controller(
        tmp_path,
        replace(context, chat_type="group"),
        channel,
        fallback,
    )

    await controller.start("已受理")

    assert channel.sent[0][:2] == ("chat-1", "chat_id")
    assert fallback == []


@pytest.mark.asyncio
async def test_progress_and_finish_patch_same_card(tmp_path, context):
    channel = FakeChannel()
    fallback = []
    controller = make_controller(tmp_path, context, channel, fallback)
    await controller.start("已受理")

    await controller.progress(AgentProgress(kind="assistant_text", text="正在定位接口"))
    await controller.progress(AgentProgress(kind="tool_started", tool_name="Read"))
    await controller.finish(
        DiagnosisStatus.COMPLETED,
        "最终结论",
        duration_ms=1200,
        tool_names=("Read",),
    )

    assert {message_id for message_id, _card in channel.patched} == {"card-created"}
    assert controller.context.status == DiagnosisStatus.COMPLETED
    assert controller.context.result_text == "最终结论"
    assert fallback == []


@pytest.mark.asyncio
async def test_existing_card_is_reused_for_next_round(tmp_path, context):
    channel = FakeChannel()
    fallback = []
    controller = make_controller(
        tmp_path,
        replace(context, card_message_id="existing-card", round_index=1),
        channel,
        fallback,
    )

    await controller.start("继续分析")

    assert channel.sent == []
    assert channel.replied == []
    assert channel.patched[0][0] == "existing-card"
    assert DiagnosisStore(tmp_path).get_by_card_message_id("existing-card") == controller.context


@pytest.mark.asyncio
async def test_create_failure_falls_back_to_text(tmp_path, context):
    channel = FakeChannel()
    channel.fail_send = True
    fallback = []
    controller = make_controller(tmp_path, context, channel, fallback)

    await controller.start("已受理")
    await controller.progress(AgentProgress(kind="assistant_text", text="阶段结果"))
    await controller.finish(DiagnosisStatus.COMPLETED, "最终结果")

    assert fallback == ["已受理", "阶段结果", "最终结果"]
    assert controller.cards_available is False


@pytest.mark.asyncio
async def test_first_patch_failure_trips_card_updates_for_round(tmp_path, context):
    channel = FakeChannel()
    fallback = []
    controller = make_controller(tmp_path, context, channel, fallback)
    await controller.start("已受理")
    channel.fail_patch = True

    await controller.progress(AgentProgress(kind="assistant_text", text="阶段结果"))
    await controller.finish(DiagnosisStatus.FAILED, "分析失败")

    assert len(channel.patched) == 0
    assert fallback == ["阶段结果", "分析失败"]
    assert controller.cards_available is False


@pytest.mark.asyncio
async def test_throttled_progress_is_flushed_after_interval(tmp_path, context):
    channel = FakeChannel()
    fallback = []
    controller = make_controller(
        tmp_path,
        context,
        channel,
        fallback,
        patch_interval_seconds=0.5,
    )
    await controller.start("已受理")

    await controller.progress(AgentProgress(kind="assistant_text", text="最新阶段"))
    assert channel.patched == []
    await asyncio.sleep(0.55)

    assert len(channel.patched) == 1
    assert "最新阶段" in str(channel.patched[0][1])


@pytest.mark.asyncio
async def test_existing_card_patch_failure_falls_back_once(tmp_path, context):
    channel = FakeChannel()
    channel.fail_patch = True
    fallback = []
    controller = make_controller(
        tmp_path,
        replace(context, card_message_id="existing-card"),
        channel,
        fallback,
    )

    await controller.start("继续分析")

    assert fallback == ["继续分析"]


@pytest.mark.asyncio
async def test_terminal_state_ignores_late_progress(tmp_path, context):
    channel = FakeChannel()
    fallback = []
    controller = make_controller(tmp_path, context, channel, fallback)
    await controller.start("已受理")
    await controller.finish(DiagnosisStatus.CANCELLED, "已取消")
    patch_count = len(channel.patched)

    await controller.progress(AgentProgress(kind="assistant_text", text="迟到阶段"))
    await controller.finish(DiagnosisStatus.COMPLETED, "迟到结果")

    assert len(channel.patched) == patch_count
    assert controller.context.status == DiagnosisStatus.CANCELLED
