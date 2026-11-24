from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
import pytest

from feishubot.diagnosis.models import DiagnosisContext, DiagnosisStatus
from feishubot.diagnosis.runner import (
    handle_qoder_diagnosis,
    _build_in_flight_supplement_prompt,
    _build_followup_prompt,
    _is_followup_intent,
)
from feishubot.diagnosis.runtime import DiagnosisRuntime
from feishubot.diagnosis.store import DiagnosisStore
from feishubot.qoder_agent.models import AgentProgress, AgentResult, SessionKey
from feishubot.qoder_agent.service import SessionStatus


class MockCardChannel:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.patched: list[tuple[str, dict[str, Any]]] = []

    async def send_interactive_message(self, receive_id: str, card: dict[str, Any], receive_id_type: str = "open_id") -> dict[str, Any]:
        self.sent.append(card)
        return {"data": {"message_id": f"card-msg-{len(self.sent)}"}}

    async def reply_interactive_message(self, message_id: str, card: dict[str, Any]) -> dict[str, Any]:
        self.sent.append(card)
        return {"data": {"message_id": f"card-msg-{len(self.sent)}"}}

    async def patch_interactive_message(self, message_id: str, card: dict[str, Any]) -> dict[str, Any]:
        self.patched.append((message_id, card))
        return {"data": {"message_id": message_id}}


class MockService:
    def __init__(self) -> None:
        self.projects = ("erp-mall", "som-erp-distribute-prod")
        self.observability_enabled = True
        self._running = False
        self.cancelled_sessions: list[str] = []
        self.submitted_requests: list[Any] = []

    def status(self, session_key: str) -> SessionStatus:
        return SessionStatus(running=self._running, queued=0)

    async def cancel(self, session_key: str) -> bool:
        self.cancelled_sessions.append(session_key)
        self._running = False
        return True

    async def run(self, request: Any, on_accepted: Any = None, on_progress: Any = None) -> AgentResult:
        self.submitted_requests.append(request)
        if on_accepted:
            await on_accepted()
        if on_progress:
            await on_progress(AgentProgress(kind="assistant_text", text="排查中"))
        return AgentResult(
            text="最终分析结论：未发现明显异常。",
            session_id="sdk-session-1",
            duration_ms=1200,
            duration_api_ms=1000,
            tool_names=("tool_1",),
        )


@pytest.fixture
def store(tmp_path) -> DiagnosisStore:
    return DiagnosisStore(tmp_path)


@pytest.fixture
def channel() -> MockCardChannel:
    return MockCardChannel()


@pytest.fixture
def runtime(channel, store) -> DiagnosisRuntime:
    return DiagnosisRuntime(channel, store)


@pytest.mark.asyncio
async def test_in_flight_supplement_interrupts_and_continues(runtime, channel):
    """Scenario 1: While answering, user supplements information -> interrupt old analysis and continue with merged info."""
    service = MockService()
    session_key = SessionKey.build(chat_id="chat-1", user_open_id="user-1", message_id="msg-1")
    responses: list[tuple[Any, str]] = []

    async def respond(msg: Any, text: str) -> None:
        responses.append((msg, text))

    async def dummy_download(msg_id: str, keys: list[str]) -> tuple:
        return ()

    # 1. Start initial run, controller becomes active and running is True
    msg_1 = SimpleNamespace(message_id="msg-1", chat_id="chat-1", chat_type="p2p", parent_id=None, root_id=None)
    controller_1 = runtime.create(
        message=msg_1,
        question="查询 som-erp-distribute-prod NS666661051360000044 报错信息",
        owner_open_id="user-1",
        session_key=session_key.value,
        available_projects=service.projects,
        fallback=respond,
    )
    await controller_1.start("开始分析")
    service._running = True

    # Controller has recorded some progress
    await controller_1.progress(AgentProgress(kind="assistant_text", text="已初步定位到 distribute 模块"))

    # 2. User sends supplement mid-flight
    msg_2 = SimpleNamespace(message_id="msg-2", chat_id="chat-1", chat_type="p2p", parent_id="card-msg-1", root_id=None)
    await handle_qoder_diagnosis(
        message=msg_2,
        text="补充：是 10月27日 16:50 左右发生的报错",
        user_open_id="user-1",
        image_keys=None,
        session_key=session_key,
        service=service,
        runtime=runtime,
        download_images=dummy_download,
        respond=respond,
    )

    # Verification:
    # 1. Old session was cancelled
    assert session_key.value in service.cancelled_sessions
    # 2. Resubmitted request contains original question, stage progress, and user's new info
    assert len(service.submitted_requests) == 1
    submitted = service.submitted_requests[0]
    assert "【原排查目标】" in submitted.prompt
    assert "som-erp-distribute-prod NS666661051360000044" in submitted.prompt
    assert "【中断前已掌握的线索/阶段进展】" in submitted.prompt
    assert "已初步定位到 distribute 模块" in submitted.prompt
    assert "【用户最新补充/更正信息】" in submitted.prompt
    assert "10月27日 16:50 左右发生的报错" in submitted.prompt
    assert "【排查要求】" in submitted.prompt


@pytest.mark.asyncio
async def test_in_flight_new_question_queues_separate_card(runtime, channel):
    """A brand-new question during an in-flight analysis must NOT merge into the running card."""
    service = MockService()
    session_key = SessionKey.build(chat_id="chat-1", user_open_id="user-1", message_id="msg-1")
    responses: list[tuple[Any, str]] = []

    async def respond(msg: Any, text: str) -> None:
        responses.append((msg, text))

    async def dummy_download(msg_id: str, keys: list[str]) -> tuple:
        return ()

    msg_1 = SimpleNamespace(message_id="msg-1", chat_id="chat-1", chat_type="p2p", parent_id=None, root_id=None)
    controller_1 = runtime.create(
        message=msg_1,
        question="查询 erp-mall DD6666690950000739 实发数量为空",
        owner_open_id="user-1",
        session_key=session_key.value,
        available_projects=service.projects,
        fallback=respond,
    )
    await controller_1.start("开始分析")
    service._running = True

    # Unrelated new question that does NOT open with the supplement marker
    msg_2 = SimpleNamespace(message_id="msg-2", chat_id="chat-1", chat_type="p2p", parent_id=None, root_id=None)
    await handle_qoder_diagnosis(
        message=msg_2,
        text="erp-purchase 分析一下 PR202611040005 订货后 DIFC 为什么都是 0",
        user_open_id="user-1",
        image_keys=None,
        session_key=session_key,
        service=service,
        runtime=runtime,
        download_images=dummy_download,
        respond=respond,
    )

    # The running analysis was NOT interrupted/cancelled
    assert service.cancelled_sessions == []
    # Submitted as its own request carrying only the new question (no merged scaffolding)
    assert len(service.submitted_requests) == 1
    submitted = service.submitted_requests[0]
    assert "PR202611040005" in submitted.prompt
    assert "【原排查目标】" not in submitted.prompt
    assert "DD6666690950000739" not in submitted.prompt
    # A second, independent card was created instead of reusing the running one
    assert len(channel.sent) == 2


@pytest.mark.asyncio
async def test_completed_analysis_quote_reply_continues_investigation(runtime, store, channel):
    """Scenario 2A: Analysis completed, user clicks 'reply' on card to continue investigation."""
    service = MockService()
    session_key = SessionKey.build(chat_id="chat-1", user_open_id="user-1", message_id="msg-1")
    responses: list[tuple[Any, str]] = []

    async def respond(msg: Any, text: str) -> None:
        responses.append((msg, text))

    async def dummy_download(msg_id: str, keys: list[str]) -> tuple:
        return ()

    # Pre-populate completed diagnosis in store
    prev_context = DiagnosisContext(
        diagnosis_id="diag-prev-1",
        source_message_id="msg-1",
        card_message_id="card-msg-prev",
        chat_id="chat-1",
        chat_type="p2p",
        owner_open_id="user-1",
        session_key=session_key.value,
        question="查询 som-erp-distribute-prod NS666661051360000044 报错信息",
        projects=("som-erp-distribute-prod",),
        status=DiagnosisStatus.COMPLETED,
        result_text="TMS 接口返回 code 2005，提示退单不存在。",
    )
    store.save(prev_context)

    # User quotes the card with follow-up instruction
    msg_reply = SimpleNamespace(
        message_id="msg-reply-1",
        chat_id="chat-1",
        chat_type="p2p",
        parent_id="card-msg-prev",
        root_id=None,
    )
    await handle_qoder_diagnosis(
        message=msg_reply,
        text="再查查是不是 wms 审批接口传参不对导致退单查不到？",
        user_open_id="user-1",
        image_keys=None,
        session_key=session_key,
        service=service,
        runtime=runtime,
        download_images=dummy_download,
        respond=respond,
    )

    # Verification:
    assert len(service.submitted_requests) == 1
    submitted = service.submitted_requests[0]
    assert "【前序排查背景与问题】" in submitted.prompt
    assert "som-erp-distribute-prod NS666661051360000044" in submitted.prompt
    assert "【前序诊断已得出的结论与证据】" in submitted.prompt
    assert "TMS 接口返回 code 2005，提示退单不存在" in submitted.prompt
    assert "【用户最新追问/补充排查指令】" in submitted.prompt
    assert "再查查是不是 wms 审批接口传参不对导致退单查不到？" in submitted.prompt


@pytest.mark.asyncio
async def test_completed_analysis_followup_keyword_without_quote(runtime, store, channel):
    """Scenario 2B: Analysis completed, user types follow-up keyword (e.g. '继续排查') without quote."""
    service = MockService()
    session_key = SessionKey.build(chat_id="chat-1", user_open_id="user-1", message_id="msg-1")
    responses: list[tuple[Any, str]] = []

    async def respond(msg: Any, text: str) -> None:
        responses.append((msg, text))

    async def dummy_download(msg_id: str, keys: list[str]) -> tuple:
        return ()

    # Pre-populate completed diagnosis in store
    prev_context = DiagnosisContext(
        diagnosis_id="diag-prev-2",
        source_message_id="msg-1",
        card_message_id="card-msg-prev-2",
        chat_id="chat-1",
        chat_type="p2p",
        owner_open_id="user-1",
        session_key=session_key.value,
        question="查询 som-erp-distribute-prod 订单失败原因",
        projects=("som-erp-distribute-prod",),
        status=DiagnosisStatus.COMPLETED,
        result_text="第一轮排查发现超时异常。",
    )
    store.save(prev_context)

    # User types follow-up command in session without parent_id
    msg_followup = SimpleNamespace(
        message_id="msg-followup-1",
        chat_id="chat-1",
        chat_type="p2p",
        parent_id=None,
        root_id=None,
    )
    await handle_qoder_diagnosis(
        message=msg_followup,
        text="继续排查超时是哪台机器引起的",
        user_open_id="user-1",
        image_keys=None,
        session_key=session_key,
        service=service,
        runtime=runtime,
        download_images=dummy_download,
        respond=respond,
    )

    # Verification:
    assert len(service.submitted_requests) == 1
    submitted = service.submitted_requests[0]
    assert "【前序排查背景与问题】" in submitted.prompt
    assert "第一轮排查发现超时异常" in submitted.prompt
    assert "继续排查超时是哪台机器引起的" in submitted.prompt


class UnauthorizedObservabilityService(MockService):
    async def run(self, request: Any, on_accepted: Any = None, on_progress: Any = None) -> AgentResult:
        self.submitted_requests.append(request)
        if on_accepted:
            await on_accepted()
        return AgentResult(
            text="最终分析结论：未发现明显异常。",
            session_id="sdk-session-1",
            duration_ms=1200,
            duration_api_ms=1000,
            tool_names=("tool_1",),
            observability_unauthorized=True,
        )


@pytest.mark.asyncio
async def test_unauthorized_observability_appends_notice_to_result(runtime, channel, tmp_path):
    """Log source unavailable must be stated explicitly, not silently dropped."""
    service = UnauthorizedObservabilityService()
    session_key = SessionKey.build(chat_id="chat-1", user_open_id="user-1", message_id="msg-1")
    responses: list[tuple[Any, str]] = []

    async def respond(msg: Any, text: str) -> None:
        responses.append((msg, text))

    async def dummy_download(msg_id: str, keys: list[str]) -> tuple:
        return ()

    msg_1 = SimpleNamespace(message_id="msg-1", chat_id="chat-1", chat_type="p2p", parent_id=None, root_id=None)
    await handle_qoder_diagnosis(
        message=msg_1,
        text="查询 som-erp-distribute-prod NS666661051360000044 报错信息",
        user_open_id="user-1",
        image_keys=None,
        session_key=session_key,
        service=service,
        runtime=runtime,
        download_images=dummy_download,
        respond=respond,
    )

    assert channel.patched, "final card should be patched with the result"
    notice = "本次未能查询日志与链路"
    assert any(notice in json.dumps(card, ensure_ascii=False) for _, card in channel.patched)

    # A normal result must not carry the notice.
    normal = MockService()
    runtime_2 = DiagnosisRuntime(channel, DiagnosisStore(tmp_path))
    session_key_2 = SessionKey.build(chat_id="chat-2", user_open_id="user-2", message_id="msg-2")
    msg_2 = SimpleNamespace(message_id="msg-2", chat_id="chat-2", chat_type="p2p", parent_id=None, root_id=None)
    await handle_qoder_diagnosis(
        message=msg_2,
        text="查询 erp-mall DD6666690950000739 实发数量为空",
        user_open_id="user-2",
        image_keys=None,
        session_key=session_key_2,
        service=normal,
        runtime=runtime_2,
        download_images=dummy_download,
        respond=respond,
    )
    last_card = channel.patched[-1][1]
    assert notice not in json.dumps(last_card, ensure_ascii=False)
