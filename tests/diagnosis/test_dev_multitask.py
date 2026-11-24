from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from feishubot.diagnosis.runner import _is_explicit_supplement, handle_qoder_diagnosis
from feishubot.diagnosis.runtime import DiagnosisRuntime
from feishubot.diagnosis.store import DiagnosisStore
from feishubot.ai.orchestrator.chat_commands import dev_status_text
from feishubot.qoder_agent.models import AgentCancelledError, AgentResult, SessionKey
from feishubot.qoder_agent.service import SessionStatus


class DevMockService:
    def __init__(self, mode: str = "dev") -> None:
        self.mode = mode
        self.projects = ("erp-purchase-dev",)
        self.observability_enabled = False
        self.submitted: list[Any] = []
        self.cancelled: list[str] = []
        self._running: dict[str, int] = {}
        self._clock = 0

    def status(self, session_key: str) -> SessionStatus:
        return SessionStatus(running=session_key in self._running, queued=0)

    def task_statuses(self, scope_prefix: str) -> list[tuple[str, SessionStatus]]:
        return [
            (key, SessionStatus(running=True, queued=0))
            for key in self._running
            if key.startswith(scope_prefix)
        ]

    def latest_running_key(self, scope_prefix: str) -> str | None:
        candidates = [
            (stamp, key)
            for key, stamp in self._running.items()
            if key.startswith(scope_prefix)
        ]
        return max(candidates)[1] if candidates else None

    async def cancel(self, session_key: str) -> bool:
        self.cancelled.append(session_key)
        self._running.pop(session_key, None)
        return True

    async def supplement(self, request: Any, text: str) -> bool:
        return False

    async def run(self, request: Any, on_accepted: Any = None, on_progress: Any = None) -> AgentResult:
        self.submitted.append(request)
        self._clock += 1
        self._running[request.session_key.value] = self._clock
        if on_accepted:
            await on_accepted()
        return AgentResult(
            text="完成",
            session_id="sdk-session",
            duration_ms=1,
            duration_api_ms=1,
            tool_names=(),
        )


class DevBlockingService(DevMockService):
    """Keeps each run in flight until cancel() or an explicit gate release."""

    def __init__(self) -> None:
        super().__init__()
        self.gates: dict[str, asyncio.Event] = {}

    async def cancel(self, session_key: str) -> bool:
        cancelled = await super().cancel(session_key)
        gate = self.gates.get(session_key)
        if gate is not None:
            gate.set()
        return cancelled

    async def run(self, request: Any, on_accepted: Any = None, on_progress: Any = None) -> AgentResult:
        self.submitted.append(request)
        self._clock += 1
        key = request.session_key.value
        self._running[key] = self._clock
        if on_accepted:
            await on_accepted()
        gate = asyncio.Event()
        self.gates[key] = gate
        await gate.wait()
        if key in self.cancelled:
            raise AgentCancelledError(request.message_id)
        self._running.pop(key, None)
        return AgentResult(
            text="完成",
            session_id="sdk-session",
            duration_ms=1,
            duration_api_ms=1,
            tool_names=(),
        )


def _msg(message_id: str, parent_id: str | None = None) -> Any:
    return SimpleNamespace(
        message_id=message_id,
        chat_id="chat-1",
        chat_type="p2p",
        parent_id=parent_id,
        root_id=None,
    )


async def _noop_download(message_id: str, keys: list[str]) -> tuple:
    return ()


def _base_key() -> SessionKey:
    return SessionKey.build(chat_id="chat-1", user_open_id="user-1", message_id="m0")


def _no_card_runtime(tmp_path) -> DiagnosisRuntime:
    # Mirrors the production dev bot: cards disabled, contexts still recorded.
    return DiagnosisRuntime(
        _MockCardChannel(), DiagnosisStore(tmp_path), cards_enabled=False
    )


async def _send(service, runtime, session_key, message, text, responses):
    async def respond(msg: Any, reply: str) -> None:
        responses.append(reply)

    await handle_qoder_diagnosis(
        message=message,
        text=text,
        user_open_id="user-1",
        image_keys=None,
        session_key=session_key,
        service=service,
        runtime=runtime,
        download_images=_noop_download,
        respond=respond,
    )


async def _send_reply_ids(service, runtime, session_key, message, text, responses, reply_ids):
    async def respond(msg: Any, reply: str) -> str:
        responses.append(reply)
        reply_id = f"bot-reply-{len(responses)}"
        reply_ids.append(reply_id)
        return reply_id

    await handle_qoder_diagnosis(
        message=message,
        text=text,
        user_open_id="user-1",
        image_keys=None,
        session_key=session_key,
        service=service,
        runtime=runtime,
        download_images=_noop_download,
        respond=respond,
    )


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def test_for_task_and_parse_roundtrip():
    base = _base_key()
    task = base.for_task("m1")
    assert task.value.startswith(base.value)
    assert task.value != base.value
    parsed = SessionKey.parse(task.value)
    assert parsed == task
    assert SessionKey.parse(base.value) == base


@pytest.mark.asyncio
async def test_new_tasks_get_distinct_parallel_sessions(tmp_path):
    service = DevMockService()
    runtime = _no_card_runtime(tmp_path)
    base = _base_key()
    responses: list[str] = []
    await _send(service, runtime, base, _msg("m1"), "任务一", responses)
    await _send(service, runtime, base, _msg("m2"), "任务二", responses)
    keys = [request.session_key.value for request in service.submitted]
    assert len(keys) == 2
    assert len(set(keys)) == 2
    assert all(key.startswith(base.value) for key in keys)
    assert all(key != base.value for key in keys)


@pytest.mark.asyncio
async def test_supplement_targets_latest_running_task(tmp_path):
    service = DevMockService()
    runtime = _no_card_runtime(tmp_path)
    base = _base_key()
    responses: list[str] = []
    await _send(service, runtime, base, _msg("m1"), "任务一", responses)
    await _send(service, runtime, base, _msg("m2"), "任务二", responses)
    first_key = service.submitted[0].session_key.value
    await _send(service, runtime, base, _msg("m3"), "补充 追加约束", responses)
    assert service.submitted[2].session_key.value == service.submitted[1].session_key.value
    assert service.submitted[2].session_key.value != first_key


@pytest.mark.asyncio
async def test_cancel_targets_latest_running_task(tmp_path):
    service = DevMockService()
    runtime = _no_card_runtime(tmp_path)
    base = _base_key()
    responses: list[str] = []
    await _send(service, runtime, base, _msg("m1"), "任务一", responses)
    await _send(service, runtime, base, _msg("m2"), "任务二", responses)
    latest_key = service.submitted[1].session_key.value
    await _send(service, runtime, base, _msg("m3"), "终止", responses)
    assert service.cancelled == [latest_key]
    assert len(service.submitted) == 2
    assert responses[-1] == "正在终止当前任务。"


@pytest.mark.asyncio
async def test_status_lists_parallel_tasks(tmp_path):
    service = DevMockService()
    runtime = _no_card_runtime(tmp_path)
    base = _base_key()
    responses: list[str] = []
    await _send(service, runtime, base, _msg("m1"), "任务一", responses)
    await _send(service, runtime, base, _msg("m2"), "任务二", responses)
    await _send(service, runtime, base, _msg("m3"), "状态", responses)
    assert "2 个任务并行进行中" in responses[-1]
    assert len(service.submitted) == 2


@pytest.mark.asyncio
async def test_cancel_without_running_task(tmp_path):
    service = DevMockService()
    runtime = _no_card_runtime(tmp_path)
    base = _base_key()
    responses: list[str] = []
    await _send(service, runtime, base, _msg("m1"), "终止", responses)
    assert responses[-1] == "当前没有正在进行的任务。"
    assert service.cancelled == []


@pytest.mark.asyncio
async def test_quote_reply_continues_specific_task(tmp_path):
    channel = _MockCardChannel()
    runtime = DiagnosisRuntime(channel, DiagnosisStore(tmp_path))
    service = DevMockService()
    base = _base_key()
    responses: list[str] = []
    await _send(service, runtime, base, _msg("m1"), "任务一", responses)
    first_key = service.submitted[0].session_key.value
    card_message_id = channel.sent_ids[-1]
    await _send(service, runtime, base, _msg("m2", parent_id=card_message_id), "继续优化", responses)
    assert service.submitted[1].session_key.value == first_key


@pytest.mark.asyncio
async def test_quote_reply_to_text_reply_continues_task_without_cards(tmp_path):
    channel = _MockCardChannel()
    runtime = DiagnosisRuntime(channel, DiagnosisStore(tmp_path), cards_enabled=False)
    service = DevMockService()
    base = _base_key()
    responses: list[str] = []
    reply_ids: list[str] = []
    await _send_reply_ids(service, runtime, base, _msg("m1"), "任务一", responses, reply_ids)
    first_key = service.submitted[0].session_key.value
    assert reply_ids
    await _send_reply_ids(
        service, runtime, base, _msg("m2", parent_id=reply_ids[-1]), "继续优化", responses, reply_ids
    )
    assert service.submitted[1].session_key.value == first_key
    assert "【前序排查背景与问题】" in service.submitted[1].prompt
    assert channel.sent_ids == []


@pytest.mark.asyncio
async def test_no_card_supplement_interrupts_and_merges(tmp_path):
    runtime = _no_card_runtime(tmp_path)
    service = DevBlockingService()
    base = _base_key()
    responses: list[str] = []
    first = asyncio.create_task(_send(service, runtime, base, _msg("m1"), "任务一", responses))
    await _wait_for(lambda: len(service.submitted) >= 1)
    first_key = service.submitted[0].session_key.value

    second = asyncio.create_task(
        _send(service, runtime, base, _msg("m2"), "补充：追加约束", responses)
    )
    await _wait_for(lambda: len(service.submitted) >= 2)

    assert service.cancelled == [first_key]
    merged = service.submitted[1]
    assert merged.session_key.value == first_key
    assert "【原排查目标】" in merged.prompt
    assert "追加约束" in merged.prompt
    assert "已收到补充信息，正在中断当前任务……" in responses

    service.gates[merged.session_key.value].set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_supplement_without_separator_starts_new_task(tmp_path):
    service = DevMockService()
    runtime = _no_card_runtime(tmp_path)
    base = _base_key()
    responses: list[str] = []
    await _send(service, runtime, base, _msg("m1"), "任务一", responses)
    await _send(service, runtime, base, _msg("m2"), "补充单元测试", responses)
    keys = [request.session_key.value for request in service.submitted]
    assert keys[0] != keys[1]
    assert service.submitted[1].prompt == "补充单元测试"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("补充", True),
        ("补充：追加约束", True),
        ("补充: 追加约束", True),
        ("补充 追加约束", True),
        ("补充，追加约束", True),
        ("补充单元测试", False),
        ("补充说明一下", False),
        ("帮我补充文档", False),
    ],
)
def test_is_explicit_supplement_marker(text, expected):
    assert _is_explicit_supplement(text) is expected


@pytest.mark.asyncio
async def test_readonly_keeps_single_session():
    service = DevMockService(mode="readonly")
    base = _base_key()
    responses: list[str] = []
    await _send(service, None, base, _msg("m1"), "问题一", responses)
    await _send(service, None, base, _msg("m2"), "问题二", responses)
    keys = [request.session_key.value for request in service.submitted]
    assert keys == [base.value, base.value]


def test_dev_status_text_empty():
    service = DevMockService()
    assert "没有正在进行的任务" in dev_status_text(service, _base_key().value)


class _MockCardChannel:
    def __init__(self) -> None:
        self.sent_ids: list[str] = []

    async def send_interactive_message(self, receive_id, card, receive_id_type="open_id"):
        message_id = f"card-msg-{len(self.sent_ids) + 1}"
        self.sent_ids.append(message_id)
        return {"data": {"message_id": message_id}}

    async def reply_interactive_message(self, message_id, card):
        return {"data": {"message_id": message_id}}

    async def patch_interactive_message(self, message_id, card):
        return {"data": {"message_id": message_id}}