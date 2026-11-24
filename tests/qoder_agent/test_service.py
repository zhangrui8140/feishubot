from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from feishubot.qoder_agent import service as service_module
from feishubot.qoder_agent.models import (
    AgentCancelledError,
    AgentProgress,
    AgentRequest,
    AgentResult,
    DuplicateMessageError,
    ImageInput,
    SessionKey,
)
from feishubot.qoder_agent.service import QoderAgentService, QoderServiceConfig
from feishubot.qoder_agent.session_store import QoderSessionStore


class FakeHandle:
    def __init__(self, session_id: str, adapter) -> None:
        self.session_id = session_id
        self.adapter = adapter
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.disconnected = False
        self.interrupted = False
        self.injected: list[str] = []
        self.accept_injection = True
        self.progress_events: list[AgentProgress] = []

    async def inject(self, text: str) -> bool:
        self.injected.append(text)
        return self.accept_injection

    async def run(self, prompt: str, *, images=(), on_progress=None) -> AgentResult:
        self.adapter.images.append(images)
        if on_progress is not None:
            event = AgentProgress(kind="assistant_text", text="working")
            self.progress_events.append(event)
            await on_progress(event)
        self.started.set()
        self.adapter.running += 1
        self.adapter.max_running = max(self.adapter.max_running, self.adapter.running)
        try:
            if self.adapter.block_runs:
                await self.release.wait()
            await asyncio.sleep(0)
            return AgentResult(
                text=f"answer:{prompt}",
                session_id=self.session_id,
                duration_ms=10,
                duration_api_ms=5,
                tool_names=("Read",),
            )
        finally:
            self.adapter.running -= 1

    async def interrupt(self) -> None:
        self.interrupted = True

    async def disconnect(self) -> None:
        self.disconnected = True


class FakeAdapter:
    def __init__(self) -> None:
        self.handles: list[FakeHandle] = []
        self.resume_ids: list[str | None] = []
        self.running = 0
        self.max_running = 0
        self.block_runs = False
        self.images: list[tuple] = []

    def validate(self) -> None:
        return None

    async def open_session(self, resume_session_id: str | None = None) -> FakeHandle:
        self.resume_ids.append(resume_session_id)
        session_id = resume_session_id or f"session-{len(self.handles) + 1}"
        handle = FakeHandle(session_id, self)
        self.handles.append(handle)
        return handle


def make_config(tmp_path, **overrides) -> QoderServiceConfig:
    values = {
        "workspace_root": tmp_path,
        "model": "auto",
        "session_store_dir": tmp_path / "sessions",
        "run_timeout_seconds": 1.0,
        "cancel_grace_seconds": 0.1,
        "shutdown_grace_seconds": 0.2,
        "max_concurrent_runs": 2,
        "max_live_sessions": 4,
        "max_queued_per_session": 3,
        "session_idle_ttl_seconds": 60.0,
        "dedup_ttl_seconds": 60.0,
    }
    values.update(overrides)
    return QoderServiceConfig(**values)


def request(message_id: str, chat_id: str = "chat") -> AgentRequest:
    return AgentRequest(
        message_id=message_id,
        session_key=SessionKey(chat_id, "root", "user"),
        prompt=message_id,
    )


def test_passes_mcp_config_to_default_adapter(tmp_path, monkeypatch):
    captured = {}
    adapter = FakeAdapter()

    def create_adapter(**kwargs):
        captured.update(kwargs)
        return adapter

    monkeypatch.setattr(service_module, "QoderSdkAdapter", create_adapter)
    mcp_config = tmp_path / "mcp.json"

    QoderAgentService(
        make_config(tmp_path, mcp_servers_config_path=mcp_config),
    )

    assert captured["mcp_servers_config_path"] == mcp_config


@pytest.mark.asyncio
async def test_reuses_session_and_deduplicates_messages(tmp_path):
    adapter = FakeAdapter()
    service = QoderAgentService(make_config(tmp_path), adapter=adapter)
    await service.start()

    first = await service.run(request("message-1"))
    second = await service.run(request("message-2"))

    assert first.session_id == second.session_id
    assert len(adapter.handles) == 1
    with pytest.raises(DuplicateMessageError):
        await service.run(request("message-1"))
    await service.close()


@pytest.mark.asyncio
async def test_passes_request_images_to_session(tmp_path):
    adapter = FakeAdapter()
    service = QoderAgentService(make_config(tmp_path), adapter=adapter)
    await service.start()
    images = (ImageInput(media_type="image/png", data_base64="QUJD"),)

    await service.run(replace(request("message-1"), images=images))

    assert adapter.images == [images]
    await service.close()


@pytest.mark.asyncio
async def test_emits_started_before_handle_progress(tmp_path):
    adapter = FakeAdapter()
    service = QoderAgentService(make_config(tmp_path), adapter=adapter)
    await service.start()
    events: list[AgentProgress] = []

    async def collect(event: AgentProgress) -> None:
        events.append(event)

    await service.run(request("message-1"), on_progress=collect)

    assert events == [
        AgentProgress(kind="started"),
        AgentProgress(kind="assistant_text", text="working"),
    ]
    await service.close()


@pytest.mark.asyncio
async def test_started_callback_failure_does_not_abort_run(tmp_path):
    adapter = FakeAdapter()
    service = QoderAgentService(make_config(tmp_path), adapter=adapter)
    await service.start()
    calls = 0

    async def fail_started(event: AgentProgress) -> None:
        nonlocal calls
        calls += 1
        if event.kind == "started":
            raise RuntimeError("callback unavailable")

    result = await service.run(request("message-1"), on_progress=fail_started)

    assert calls == 2
    assert result.text.startswith("answer:")
    await service.close()


@pytest.mark.asyncio
async def test_serializes_same_session(tmp_path):
    adapter = FakeAdapter()
    adapter.block_runs = True
    service = QoderAgentService(make_config(tmp_path), adapter=adapter)
    await service.start()

    first_task = asyncio.create_task(service.run(request("message-1")))
    await asyncio.sleep(0)
    handle = adapter.handles[0]
    second_task = asyncio.create_task(service.run(request("message-2")))
    await asyncio.sleep(0.02)

    assert adapter.max_running == 1
    handle.release.set()
    await asyncio.gather(first_task, second_task)
    assert adapter.max_running == 1
    await service.close()


@pytest.mark.asyncio
async def test_cancels_active_session(tmp_path):
    adapter = FakeAdapter()
    adapter.block_runs = True
    service = QoderAgentService(make_config(tmp_path), adapter=adapter)
    await service.start()
    session_key = request("message-1").session_key

    task = asyncio.create_task(service.run(request("message-1")))
    await asyncio.sleep(0)
    handle = adapter.handles[0]
    handle.release.clear()
    await handle.started.wait()

    assert await service.cancel(session_key.value) is True
    with pytest.raises(AgentCancelledError):
        await task
    assert handle.interrupted is True
    assert handle.disconnected is True
    await service.close()


@pytest.mark.asyncio
async def test_resumes_persisted_session(tmp_path):
    store = QoderSessionStore(tmp_path / "sessions")
    store.set(request("message-1").session_key.value, "persisted-session")
    adapter = FakeAdapter()
    service = QoderAgentService(
        make_config(tmp_path),
        adapter=adapter,
        session_store=store,
    )
    await service.start()

    result = await service.run(request("message-1"))

    assert result.session_id == "persisted-session"
    assert adapter.resume_ids == ["persisted-session"]
    await service.close()


async def _start_blocked_run(tmp_path, store=None):
    adapter = FakeAdapter()
    adapter.block_runs = True
    service = QoderAgentService(make_config(tmp_path), adapter=adapter, session_store=store)
    await service.start()
    task = asyncio.create_task(service.run(request("message-1")))
    await asyncio.sleep(0)
    handle = adapter.handles[0]
    await handle.started.wait()
    return service, handle, task


@pytest.mark.asyncio
async def test_supplement_injects_into_running_analysis(tmp_path):
    service, handle, task = await _start_blocked_run(tmp_path)
    key = request("message-1").session_key.value

    assert service.status(key).running is True
    assert await service.supplement(request("message-2"), "【用户补充】只看 erp-mall-prod") is True
    assert handle.injected == ["【用户补充】只看 erp-mall-prod"]
    with pytest.raises(DuplicateMessageError):
        await service.supplement(request("message-2"), "重复")

    handle.release.set()
    await task
    assert service.status(key).running is False
    assert await service.supplement(request("message-3"), "补充") is False
    await service.close()


@pytest.mark.asyncio
async def test_rejected_supplement_can_still_run_as_new_request(tmp_path):
    service, handle, task = await _start_blocked_run(tmp_path)
    handle.accept_injection = False

    assert await service.supplement(request("message-2"), "补充") is False
    handle.release.set()
    await task
    result = await service.run(request("message-2"))
    assert result.text.startswith("answer:")
    await service.close()


@pytest.mark.asyncio
async def test_reset_refused_while_running_then_clears_session(tmp_path):
    store = QoderSessionStore(tmp_path / "sessions")
    service, handle, task = await _start_blocked_run(tmp_path, store)
    key = request("message-1").session_key.value

    assert await service.reset(key) is False
    handle.release.set()
    await task
    assert store.get(key) is not None

    assert await service.reset(key) is True
    assert store.get(key) is None
    assert handle.disconnected is True
    handle.adapter.block_runs = False
    await service.run(request("message-2"))
    assert service._sessions[key].handle is not handle
    await service.close()


def test_extracts_needs_input_marker_from_result():
    result = AgentResult(
        text="请提供订单号。\n【诊断状态】NEEDS_INPUT",
        session_id="session-1",
        duration_ms=10,
        duration_api_ms=5,
        tool_names=(),
    )

    parsed = QoderAgentService._extract_diagnosis_status(result)

    assert parsed.text == "请提供订单号。"
    assert parsed.needs_input is True


def test_plain_result_does_not_guess_needs_input_from_words():
    result = AgentResult(
        text="如果方便可以补充订单号，但现有证据已足够完成诊断。",
        session_id="session-1",
        duration_ms=10,
        duration_api_ms=5,
        tool_names=(),
    )

    assert QoderAgentService._extract_diagnosis_status(result) == result
