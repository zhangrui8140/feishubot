from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from qoder_agent_sdk import (
    AssistantMessage,
    QoderSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
)

from feishubot.qoder_agent.models import AgentProgress, AgentResult, ImageInput
from feishubot.qoder_agent.tool_policy import OBSERVABILITY_SERVER

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[AgentProgress], Awaitable[None]]
_MAX_PROGRESS_MESSAGES = 10
_MAX_PROGRESS_CHARS = 1500
_STRAY_RESULT_DRAIN_SECONDS = 30.0

class QoderSdkRunError(Exception):
    pass


class QoderSdkPermissionError(QoderSdkRunError):
    pass


@dataclass(frozen=True, slots=True)
class _RunCommand:
    prompt: str
    images: tuple[ImageInput, ...]
    on_progress: ProgressCallback | None


async def _user_message_with_images(
    prompt: str, images: tuple[ImageInput, ...]
) -> AsyncIterator[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.media_type,
                "data": image.data_base64,
            },
        }
        for image in images
    )
    yield {
        "type": "user",
        "message": {"role": "user", "content": content},
        "parent_tool_use_id": None,
    }


class _ProgressRelay:
    def __init__(self, callback: ProgressCallback | None) -> None:
        self._callback = callback
        self._sent = 0

    async def send(self, event: AgentProgress) -> None:
        if self._callback is None or self._sent >= _MAX_PROGRESS_MESSAGES:
            return
        self._sent += 1
        if event.kind == "assistant_text" and event.text is not None:
            text = event.text
            if len(text) > _MAX_PROGRESS_CHARS:
                text = text[:_MAX_PROGRESS_CHARS].rstrip() + "……"
                event = AgentProgress(kind=event.kind, text=text)
        try:
            await self._callback(event)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to deliver Qoder progress message")


class QoderSessionHandle:
    def __init__(
        self,
        client: QoderSDKClient,
        session_id: str | None = None,
        *,
        include_current_time: bool = False,
    ) -> None:
        self._client = client
        self.session_id = session_id
        self._include_current_time = include_current_time
        self._commands: asyncio.Queue[
            tuple[str, _RunCommand | None, asyncio.Future[Any]]
        ] = asyncio.Queue()
        self._ready: asyncio.Future[None] | None = None
        self._owner_task: asyncio.Task[None] | None = None
        self._pending: set[asyncio.Future[Any]] = set()
        self._run_active = False
        self._inject_lock = asyncio.Lock()
        self._injections: list[tuple[str, asyncio.Future[bool]]] = []

    @property
    def running(self) -> bool:
        return self._run_active

    async def inject(self, text: str) -> bool:
        """Return True once the active turn has picked up the supplement."""
        async with self._inject_lock:
            if not self._run_active:
                return False
            future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            self._injections.append((text, future))
            try:
                # should_query=False keeps a late supplement from starting an unattended turn.
                await self._client.query(text, priority="next", should_query=False)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to inject supplement into Qoder session")
                self._injections.remove((text, future))
                return False
        return await asyncio.shield(future)

    async def start(self) -> None:
        if self._owner_task is not None:
            return
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._owner_task = asyncio.create_task(
            self._owner_loop(),
            name="qoder-sdk-session-owner",
        )
        await asyncio.shield(self._ready)

    async def run(
        self,
        prompt: str,
        *,
        images: tuple[ImageInput, ...] = (),
        on_progress: ProgressCallback | None = None,
    ) -> AgentResult:
        if self._owner_task is None or self._owner_task.done():
            raise QoderSdkRunError("Qoder SDK session is not connected")
        future: asyncio.Future[AgentResult] = asyncio.get_running_loop().create_future()
        self._pending.add(future)
        await self._commands.put(("run", _RunCommand(prompt, images, on_progress), future))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(self._consume_abandoned_result)
            raise
        finally:
            if future.done():
                self._pending.discard(future)

    async def interrupt(self) -> None:
        await self._client.interrupt()

    async def disconnect(self) -> None:
        owner = self._owner_task
        if owner is None:
            return
        if owner.done():
            await asyncio.gather(owner, return_exceptions=True)
            return

        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pending.add(future)
        await self._commands.put(("stop", None, future))
        try:
            await asyncio.shield(future)
        finally:
            self._pending.discard(future)
            await asyncio.gather(owner, return_exceptions=True)

    async def _owner_loop(self) -> None:
        stop_future: asyncio.Future[Any] | None = None
        try:
            await self._client.connect()
        except BaseException as exc:
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(exc)
            return

        if self._ready is not None and not self._ready.done():
            self._ready.set_result(None)

        try:
            while True:
                command, request, future = await self._commands.get()
                if command == "stop" or request is None:
                    stop_future = future
                    break
                try:
                    result = await self._execute_run(
                        request.prompt,
                        request.on_progress,
                        request.images,
                    )
                except BaseException as exc:
                    if not future.done():
                        future.set_exception(exc)
                else:
                    if not future.done():
                        future.set_result(result)
        finally:
            try:
                await self._client.disconnect()
            except BaseException:  # noqa: BLE001
                logger.warning("Qoder SDK client did not disconnect cleanly")
            if stop_future is not None and not stop_future.done():
                stop_future.set_result(None)
            error = QoderSdkRunError("Qoder SDK session stopped")
            for future in list(self._pending):
                if not future.done():
                    future.set_exception(error)

    async def _execute_run(
        self,
        prompt: str,
        on_progress: ProgressCallback | None = None,
        images: tuple[ImageInput, ...] = (),
    ) -> AgentResult:
        if self._include_current_time:
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            prompt = f"[当前时间 {now}]\n{prompt}"
        if images:
            await self._client.query(_user_message_with_images(prompt, images))
        else:
            await self._client.query(prompt)
        self._run_active = True
        try:
            result_message, latest_text, tool_names = await self._receive_turn(on_progress)
        except BaseException:
            await self._finish_injections(drain=False)
            raise
        await self._finish_injections(drain=True)

        if result_message is None:
            raise QoderSdkRunError("Qoder SDK response ended without a result message")
        if result_message.permission_denials:
            raise QoderSdkPermissionError("Qoder SDK denied a requested tool")
        if result_message.is_error:
            details = "; ".join(result_message.errors or [])
            raise QoderSdkRunError(details or result_message.subtype)

        text = (result_message.result or latest_text).strip()
        if not text:
            raise QoderSdkRunError("Qoder SDK returned an empty result")

        observability_unauthorized = False
        if self._include_current_time and any(
            name.startswith("mcp__observability") for name in tool_names
        ):
            observability_unauthorized = await self._observability_unauthorized()

        return AgentResult(
            text=text,
            session_id=result_message.session_id,
            duration_ms=result_message.duration_ms,
            duration_api_ms=result_message.duration_api_ms,
            tool_names=tuple(tool_names),
            observability_unauthorized=observability_unauthorized,
        )

    async def mcp_status(self) -> dict[str, Any]:
        return await self._client.get_mcp_status()

    async def _observability_unauthorized(self) -> bool:
        try:
            status = await self._client.get_mcp_status()
        except Exception:  # noqa: BLE001
            logger.warning("Could not read MCP status to check observability authorization")
            return False
        for server in status.get("mcpServers", []):
            if isinstance(server, dict) and server.get("name") == OBSERVABILITY_SERVER:
                return server.get("status") != "connected"
        return False

    async def _receive_turn(
        self, on_progress: ProgressCallback | None
    ) -> tuple[ResultMessage | None, str, list[str]]:
        latest_text = ""
        # Text is only known to be intermediate once more assistant output follows it.
        unsent_text = ""
        progress = _ProgressRelay(on_progress)
        result_message: ResultMessage | None = None
        tool_names: list[str] = []

        async for message in self._client.receive_response():
            if isinstance(message, SystemMessage) and message.subtype == "init":
                session_id = message.data.get("session_id")
                if isinstance(session_id, str) and session_id:
                    self.session_id = session_id
                continue

            if isinstance(message, UserMessage):
                self._confirm_injection(message)
                continue

            if isinstance(message, AssistantMessage):
                text_parts: list[str] = []
                tool_blocks: list[ToolUseBlock] = []
                if message.session_id:
                    self.session_id = message.session_id
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        tool_names.append(block.name)
                        tool_blocks.append(block)
                if unsent_text:
                    await progress.send(AgentProgress(kind="assistant_text", text=unsent_text))
                    unsent_text = ""
                text = "".join(text_parts).strip()
                if text:
                    latest_text = text
                    if tool_blocks:
                        await progress.send(AgentProgress(kind="assistant_text", text=text))
                    else:
                        unsent_text = text
                for block in tool_blocks:
                    await progress.send(
                        AgentProgress(kind="tool_started", tool_name=block.name)
                    )
                continue

            if isinstance(message, ResultMessage):
                result_message = message
                self.session_id = message.session_id
        return result_message, latest_text, tool_names

    def _confirm_injection(self, message: UserMessage) -> None:
        if isinstance(message.content, str):
            echoed = message.content
        else:
            echoed = "".join(b.text for b in message.content if isinstance(b, TextBlock))
        echoed = echoed.strip()
        if not echoed:
            return
        for entry in self._injections:
            text, future = entry
            if text.strip() in echoed:
                self._injections.remove(entry)
                if not future.done():
                    future.set_result(True)
                return

    async def _finish_injections(self, *, drain: bool) -> None:
        async with self._inject_lock:
            self._run_active = False
            missed, self._injections = self._injections, []
        for _, future in missed:
            if not future.done():
                future.set_result(False)
        if not drain:
            return
        # A supplement that arrived after the turn ended still yields its own empty result.
        for _ in missed:
            try:
                async with asyncio.timeout(_STRAY_RESULT_DRAIN_SECONDS):
                    async for _message in self._client.receive_response():
                        pass
            except TimeoutError:
                logger.warning("Timed out draining a late Qoder supplement result")
                return

    def _consume_abandoned_result(self, future: asyncio.Future[Any]) -> None:
        self._pending.discard(future)
        if not future.cancelled():
            future.exception()
