from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from feishubot.qoder_agent.domain_knowledge import DomainKnowledgeStore, split_record
from feishubot.qoder_agent.models import (
    AgentBusyError,
    AgentCancelledError,
    AgentPermissionError,
    AgentProgress,
    AgentRequest,
    AgentResult,
    AgentTimeoutError,
    AgentUnavailableError,
    DuplicateMessageError,
    ImageInput,
)
from feishubot.qoder_agent.projects import (
    list_dev_projects,
    list_projects,
    mentioned_projects,
)
from feishubot.qoder_agent.sdk_adapter import QoderSdkAdapter
from feishubot.qoder_agent.session_handle import (
    ProgressCallback,
    QoderSdkPermissionError,
    QoderSdkRunError,
)
from feishubot.qoder_agent.session_store import QoderSessionStore

logger = logging.getLogger(__name__)

_DIAGNOSIS_NEEDS_INPUT_MARKER = "【诊断状态】NEEDS_INPUT"


class SessionHandle(Protocol):
    session_id: str | None

    async def run(
        self,
        prompt: str,
        *,
        images: tuple[ImageInput, ...] = (),
        on_progress: ProgressCallback | None = None,
    ) -> AgentResult: ...

    async def inject(self, text: str) -> bool: ...

    async def interrupt(self) -> None: ...

    async def disconnect(self) -> None: ...


class AgentAdapter(Protocol):
    @property
    def observability_enabled(self) -> bool: ...

    def validate(self) -> None: ...

    async def open_session(self, resume_session_id: str | None = None) -> SessionHandle: ...


@dataclass(frozen=True, slots=True)
class QoderServiceConfig:
    workspace_root: Path
    model: str
    session_store_dir: Path
    personal_access_token: str = field(default="", repr=False)
    mcp_servers_config_path: Path | None = None
    run_timeout_seconds: float = 600.0
    cancel_grace_seconds: float = 5.0
    shutdown_grace_seconds: float = 30.0
    max_concurrent_runs: int = 2
    max_live_sessions: int = 4
    max_queued_per_session: int = 3
    session_idle_ttl_seconds: float = 1800.0
    dedup_ttl_seconds: float = 86400.0
    domain_knowledge_dir: Path | None = None
    # Shared rui knowledge base; None disables knowledge sharing.
    knowledge_dir: Path | None = None
    # "readonly" (prod analysis) or "dev" (dev worktree development tasks).
    mode: str = "readonly"
    max_buffer_size_bytes: int = 32 * 1024 * 1024


@dataclass(slots=True)
class _SessionState:
    session_key: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    handle: SessionHandle | None = None
    session_id: str | None = None
    active_task: asyncio.Task[object] | None = None
    pending_count: int = 0
    last_used_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class SessionStatus:
    running: bool
    queued: int


class QoderAgentService:
    def __init__(
        self,
        config: QoderServiceConfig,
        *,
        adapter: AgentAdapter | None = None,
        session_store: QoderSessionStore | None = None,
    ) -> None:
        self._config = config
        self._domain = (
            DomainKnowledgeStore(config.domain_knowledge_dir, config.workspace_root)
            if config.domain_knowledge_dir
            else None
        )
        self._adapter = adapter or QoderSdkAdapter(
            workspace_root=config.workspace_root,
            model=config.model,
            personal_access_token=config.personal_access_token,
            mcp_servers_config_path=config.mcp_servers_config_path,
            domain_knowledge_enabled=self._domain is not None,
            max_buffer_size=config.max_buffer_size_bytes,
            mode=config.mode,
            knowledge_dir=config.knowledge_dir,
        )
        self._session_store = session_store or QoderSessionStore(config.session_store_dir)
        self._semaphore = asyncio.Semaphore(config.max_concurrent_runs)
        self._sessions: dict[str, _SessionState] = {}
        self._seen_messages: dict[str, float] = {}
        self._run_tasks: set[asyncio.Task[object]] = set()
        self._maintenance_task: asyncio.Task[None] | None = None
        self._started = False
        self._closing = False
        self._model_override = self._load_model_override()
        if self._model_override:
            setter = getattr(self._adapter, "set_model", None)
            if callable(setter):
                setter(self._model_override)

    @property
    def observability_enabled(self) -> bool:
        return self._adapter.observability_enabled

    @property
    def mode(self) -> str:
        return self._config.mode

    @property
    def model(self) -> str:
        current = getattr(self._adapter, "model", None)
        return current if isinstance(current, str) and current else self._config.model

    async def set_model(self, model: str) -> str:
        """Switch the model for future sessions; idle sessions are rebuilt lazily."""
        cleaned = model.strip()
        if not cleaned:
            raise ValueError("模型名称不能为空")
        setter = getattr(self._adapter, "set_model", None)
        if callable(setter):
            setter(cleaned)
        self._model_override = cleaned
        self._save_model_override(cleaned)
        for state in list(self._sessions.values()):
            if (
                state.active_task is None
                and not state.lock.locked()
                and state.pending_count == 0
                and state.handle is not None
            ):
                await self._discard_handle(state)
        logger.info("Qoder model switched to %s", cleaned)
        return cleaned

    def _model_override_path(self) -> Path:
        return self._config.session_store_dir / "model.json"

    def _load_model_override(self) -> str:
        try:
            payload = json.loads(self._model_override_path().read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ""
        model = payload.get("model") if isinstance(payload, dict) else None
        return model.strip() if isinstance(model, str) else ""

    def _save_model_override(self, model: str) -> None:
        path = self._model_override_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": model}), encoding="utf-8")
        except OSError:
            logger.warning("Failed to persist model override: %s", path)

    @property
    def projects(self) -> tuple[str, ...]:
        if self._config.mode == "dev":
            return list_dev_projects(self._config.workspace_root)
        return list_projects(self._config.workspace_root)

    async def start(self) -> None:
        if self._started:
            return
        self._validate_config()
        self._adapter.validate()
        self._closing = False
        self._started = True
        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(),
            name="qoder-session-maintenance",
        )
        logger.info(
            "Qoder Agent service ready, workspace=%s, model=%s, max_concurrency=%s",
            self._config.workspace_root,
            self._config.model,
            self._config.max_concurrent_runs,
        )

    async def run(
        self,
        request: AgentRequest,
        *,
        on_accepted: Callable[[], Awaitable[None]] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> AgentResult:
        if not self._started or self._closing:
            raise AgentUnavailableError("Qoder Agent service is not accepting requests")

        self._prune_seen_messages()
        if request.message_id in self._seen_messages:
            raise DuplicateMessageError(request.message_id)
        self._seen_messages[request.message_id] = float("inf")

        task = asyncio.current_task()
        if task is not None:
            self._run_tasks.add(task)

        session_key = request.session_key.value
        state = self._sessions.setdefault(session_key, _SessionState(session_key=session_key))
        queued = False
        completed = False
        try:
            if state.pending_count >= self._config.max_queued_per_session:
                raise AgentBusyError("too many queued requests for this session")
            state.pending_count += 1
            queued = True
            if on_accepted is not None:
                await on_accepted()

            async with state.lock:
                state.pending_count -= 1
                queued = False
                async with self._semaphore:
                    await self._ensure_handle(state)
                    if state.handle is None:
                        raise AgentUnavailableError("Qoder session failed to initialize")

                    prompt = self._prompt_with_domain(request.prompt)
                    state.active_task = asyncio.current_task()
                    try:
                        await self._send_started(on_progress)
                        async with asyncio.timeout(self._config.run_timeout_seconds):
                            result = await state.handle.run(
                                prompt,
                                images=request.images,
                                on_progress=on_progress,
                            )
                    except TimeoutError as exc:
                        await self._interrupt_and_discard(state)
                        raise AgentTimeoutError("Qoder analysis timed out") from exc
                    except asyncio.CancelledError as exc:
                        await self._interrupt_and_discard(state)
                        raise AgentCancelledError("Qoder analysis was cancelled") from exc
                    except QoderSdkPermissionError as exc:
                        raise AgentPermissionError(str(exc)) from exc
                    except QoderSdkRunError as exc:
                        await self._discard_handle(state)
                        self._session_store.delete(session_key)
                        state.session_id = None
                        raise AgentUnavailableError(str(exc)) from exc
                    except Exception as exc:  # noqa: BLE001
                        await self._discard_handle(state)
                        self._session_store.delete(session_key)
                        state.session_id = None
                        raise AgentUnavailableError("Qoder SDK request failed") from exc
                    finally:
                        state.active_task = None
                        state.last_used_at = time.monotonic()

                    state.session_id = result.session_id
                    self._session_store.set(session_key, result.session_id)
                    result = self._record_domain(result)
                    result = self._extract_diagnosis_status(result)
                    completed = True
                    self._seen_messages[request.message_id] = (
                        time.monotonic() + self._config.dedup_ttl_seconds
                    )
                    logger.info(
                        "Qoder analysis completed, request=%s, duration_ms=%s, "
                        "api_duration_ms=%s, tools=%s",
                        request.message_id[-8:],
                        result.duration_ms,
                        result.duration_api_ms,
                        len(result.tool_names),
                    )
                    return result
        finally:
            if queued:
                state.pending_count -= 1
            if not completed:
                self._seen_messages.pop(request.message_id, None)
            if task is not None:
                self._run_tasks.discard(task)

    @staticmethod
    async def _send_started(on_progress: ProgressCallback | None) -> None:
        if on_progress is None:
            return
        try:
            await on_progress(AgentProgress(kind="started"))
        except Exception:  # noqa: BLE001
            logger.warning("Failed to deliver Qoder started progress event")

    def _prompt_with_domain(self, prompt: str) -> str:
        if self._domain is None:
            return prompt
        projects = self.projects
        scope = mentioned_projects(prompt, projects) or projects
        try:
            context = self._domain.build_context(prompt, scope)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load domain knowledge", exc_info=True)
            return prompt
        if not context:
            return prompt
        return f"{context}\n\n【用户问题】\n{prompt}"

    def _record_domain(self, result: AgentResult) -> AgentResult:
        if self._domain is None:
            return result
        text, payload = split_record(result.text)
        if payload is not None:
            try:
                self._domain.record(payload)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to record domain knowledge", exc_info=True)
        if text == result.text:
            return result
        # The record may arrive as its own final message after the pushed conclusion.
        return dataclasses.replace(result, text=text or "以上为本次分析结论。")

    @staticmethod
    def _extract_diagnosis_status(result: AgentResult) -> AgentResult:
        if _DIAGNOSIS_NEEDS_INPUT_MARKER not in result.text:
            return result
        text = result.text.replace(_DIAGNOSIS_NEEDS_INPUT_MARKER, "").strip()
        return dataclasses.replace(
            result,
            text=text or "请补充必要信息后继续诊断。",
            needs_input=True,
        )

    def status(self, session_key: str) -> SessionStatus:
        state = self._sessions.get(session_key)
        if state is None:
            return SessionStatus(running=False, queued=0)
        return SessionStatus(running=state.active_task is not None, queued=state.pending_count)

    def task_statuses(self, scope_prefix: str) -> list[tuple[str, SessionStatus]]:
        """Statuses of every session under a chat scope (dev multi-task)."""
        return [
            (
                key,
                SessionStatus(
                    running=state.active_task is not None,
                    queued=state.pending_count,
                ),
            )
            for key, state in self._sessions.items()
            if key.startswith(scope_prefix)
        ]

    def latest_running_key(self, scope_prefix: str) -> str | None:
        """Most recently active running session under a chat scope."""
        running = [
            (state.last_used_at, key)
            for key, state in self._sessions.items()
            if key.startswith(scope_prefix) and state.active_task is not None
        ]
        if not running:
            return None
        return max(running)[1]

    async def supplement(self, request: AgentRequest, text: str) -> bool:
        """Feed text into the session's running analysis; False means no run picked it up."""
        state = self._sessions.get(request.session_key.value)
        if state is None or state.active_task is None or state.handle is None:
            return False
        self._prune_seen_messages()
        if request.message_id in self._seen_messages:
            raise DuplicateMessageError(request.message_id)
        self._seen_messages[request.message_id] = float("inf")
        try:
            accepted = await state.handle.inject(text)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to supplement Qoder analysis, session=%s", state.session_key)
            accepted = False
        if accepted:
            self._seen_messages[request.message_id] = (
                time.monotonic() + self._config.dedup_ttl_seconds
            )
        else:
            self._seen_messages.pop(request.message_id, None)
        return accepted

    async def reset(self, session_key: str) -> bool:
        """Start the next question in a fresh Qoder session; refused while one is in flight."""
        state = self._sessions.get(session_key)
        if state is not None:
            if state.active_task is not None or state.lock.locked() or state.pending_count:
                return False
            await self._discard_handle(state)
            self._sessions.pop(session_key, None)
        self._session_store.delete(session_key)
        return True

    async def cancel(self, session_key: str) -> bool:
        state = self._sessions.get(session_key)
        if state is None or state.active_task is None or state.handle is None:
            return False

        try:
            await asyncio.wait_for(
                state.handle.interrupt(),
                timeout=self._config.cancel_grace_seconds,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to interrupt Qoder session, session=%s", session_key)
        state.active_task.cancel()
        return True

    async def close(self) -> None:
        if not self._started:
            return
        self._closing = True

        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
            self._maintenance_task = None

        tasks = [task for task in self._run_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            try:
                async with asyncio.timeout(self._config.shutdown_grace_seconds):
                    await asyncio.gather(*tasks, return_exceptions=True)
            except TimeoutError:
                logger.warning("Timed out waiting for Qoder runs to stop")

        handles = [state.handle for state in self._sessions.values() if state.handle]
        await asyncio.gather(
            *(self._safe_disconnect(handle) for handle in handles),
            return_exceptions=True,
        )
        for state in self._sessions.values():
            state.handle = None
            state.active_task = None

        self._started = False
        logger.info("Qoder Agent service stopped")

    async def _ensure_handle(self, state: _SessionState) -> None:
        if state.handle is not None:
            return

        await self._evict_for_capacity(exclude_session_key=state.session_key)
        resume_session_id = state.session_id or self._session_store.get(state.session_key)
        try:
            state.handle = await self._adapter.open_session(resume_session_id)
        except Exception:  # noqa: BLE001
            if not resume_session_id:
                raise
            logger.warning("Failed to resume Qoder session; creating a new session")
            self._session_store.delete(state.session_key)
            state.session_id = None
            state.handle = await self._adapter.open_session(None)

    async def _evict_for_capacity(self, *, exclude_session_key: str) -> None:
        live_states = [state for state in self._sessions.values() if state.handle is not None]
        if len(live_states) < self._config.max_live_sessions:
            return

        candidates = [
            state
            for state in live_states
            if state.session_key != exclude_session_key
            and state.active_task is None
            and not state.lock.locked()
        ]
        if not candidates:
            raise AgentBusyError("all Qoder sessions are active")

        oldest = min(candidates, key=lambda state: state.last_used_at)
        await self._discard_handle(oldest)

    async def _maintenance_loop(self) -> None:
        interval = min(60.0, max(1.0, self._config.session_idle_ttl_seconds / 2))
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            expired_keys: list[str] = []
            for session_key, state in list(self._sessions.items()):
                if (
                    state.active_task is None
                    and not state.lock.locked()
                    and state.pending_count == 0
                    and now - state.last_used_at >= self._config.session_idle_ttl_seconds
                ):
                    if state.handle is not None:
                        await self._discard_handle(state)
                    expired_keys.append(session_key)
            for session_key in expired_keys:
                self._sessions.pop(session_key, None)
            self._prune_seen_messages()

    async def _interrupt_and_discard(self, state: _SessionState) -> None:
        handle = state.handle
        if handle is None:
            return
        try:
            await asyncio.wait_for(
                handle.interrupt(),
                timeout=self._config.cancel_grace_seconds,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Qoder interrupt failed while cleaning up a run")
        await self._discard_handle(state)

    async def _discard_handle(self, state: _SessionState) -> None:
        handle = state.handle
        state.handle = None
        if handle is not None:
            await self._safe_disconnect(handle)

    @staticmethod
    async def _safe_disconnect(handle: SessionHandle) -> None:
        try:
            await handle.disconnect()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to disconnect Qoder session cleanly")

    def _prune_seen_messages(self) -> None:
        now = time.monotonic()
        expired = [
            message_id
            for message_id, expiry in self._seen_messages.items()
            if expiry <= now
        ]
        for message_id in expired:
            self._seen_messages.pop(message_id, None)

    def _validate_config(self) -> None:
        if not self._config.workspace_root.expanduser().resolve().is_dir():
            raise ValueError(f"Qoder workspace not found: {self._config.workspace_root}")
        numeric_values = {
            "run_timeout_seconds": self._config.run_timeout_seconds,
            "cancel_grace_seconds": self._config.cancel_grace_seconds,
            "shutdown_grace_seconds": self._config.shutdown_grace_seconds,
            "max_concurrent_runs": self._config.max_concurrent_runs,
            "max_live_sessions": self._config.max_live_sessions,
            "max_queued_per_session": self._config.max_queued_per_session,
            "session_idle_ttl_seconds": self._config.session_idle_ttl_seconds,
            "dedup_ttl_seconds": self._config.dedup_ttl_seconds,
        }
        invalid = [name for name, value in numeric_values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Qoder configuration values must be positive: {', '.join(invalid)}")
