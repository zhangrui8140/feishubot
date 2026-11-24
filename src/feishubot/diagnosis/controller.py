from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any, Protocol

from feishubot.diagnosis.card import render_card
from feishubot.diagnosis.models import DiagnosisContext, DiagnosisStatus, utc_now
from feishubot.diagnosis.store import DiagnosisStore
from feishubot.qoder_agent.models import AgentProgress

logger = logging.getLogger(__name__)

_PATCH_INTERVAL_SECONDS = 0.75
_TERMINAL_STATUSES = {
    DiagnosisStatus.COMPLETED,
    DiagnosisStatus.CANCELLED,
    DiagnosisStatus.TIMED_OUT,
    DiagnosisStatus.PERMISSION_DENIED,
    DiagnosisStatus.FAILED,
}


class CardChannel(Protocol):
    async def send_interactive_message(
        self,
        receive_id: str,
        card: dict[str, Any],
        receive_id_type: str = "open_id",
    ) -> dict[str, Any]: ...

    async def reply_interactive_message(
        self, message_id: str, card: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def patch_interactive_message(
        self, message_id: str, card: dict[str, Any]
    ) -> dict[str, Any]: ...


class DiagnosisController:
    def __init__(
        self,
        channel: CardChannel,
        store: DiagnosisStore,
        context: DiagnosisContext,
        fallback: Callable[[str], Awaitable[None]],
        *,
        patch_interval_seconds: float = _PATCH_INTERVAL_SECONDS,
    ) -> None:
        self._channel = channel
        self._store = store
        self._context = context
        self._fallback = fallback
        self._patch_interval_seconds = patch_interval_seconds
        self._cards_available = True
        self._last_payload = ""
        self._last_patch_at = 0.0
        self._pending_patch: asyncio.Task[None] | None = None
        self._pending_fallback_text: str | None = None
        self._interrupted_for_supplement = False

    @property
    def is_interrupted_for_supplement(self) -> bool:
        return self._interrupted_for_supplement

    def mark_interrupted_for_supplement(self) -> None:
        self._interrupted_for_supplement = True

    def reset_for_continuation(self, new_question: str) -> None:
        self._interrupted_for_supplement = False
        self._replace(
            question=new_question,
            status=DiagnosisStatus.ANALYZING,
            stage_text="已结合新信息重新分析中……",
        )

    @property
    def context(self) -> DiagnosisContext:
        return self._context

    @property
    def cards_available(self) -> bool:
        return self._cards_available

    async def start(self, fallback_text: str) -> None:
        self._store.save(self._context)
        if self._context.card_message_id:
            await self._patch(force=True, fallback_text=fallback_text)
            return
        else:
            try:
                response = await self._create_card()
                card_message_id = _message_id(response)
                if not card_message_id:
                    raise RuntimeError("Feishu card response has no message_id")
                self._context = replace(
                    self._context,
                    card_message_id=card_message_id,
                    updated_at=utc_now(),
                )
                self._last_payload = self._payload()
                self._last_patch_at = time.monotonic()
                self._store.save(self._context)
                return
            except Exception:  # noqa: BLE001
                logger.warning("Failed to create diagnosis card", exc_info=True)
                self._cards_available = False
        await self._fallback(fallback_text)

    async def progress(self, event: AgentProgress) -> None:
        if self._context.status in _TERMINAL_STATUSES:
            return
        if event.kind == "tool_started" and event.tool_name:
            tools = self._context.tool_names
            if event.tool_name not in tools:
                tools += (event.tool_name,)
            self._replace(status=DiagnosisStatus.ANALYZING, tool_names=tools)
            await self._patch()
            return
        if event.kind == "assistant_text" and event.text:
            self._replace(status=DiagnosisStatus.ANALYZING, stage_text=event.text)
            await self._patch(fallback_text=event.text)
            return
        if event.kind == "started":
            self._replace(status=DiagnosisStatus.ANALYZING, stage_text="正在分析")
            await self._patch()

    async def supplement(self, text: str) -> None:
        if self._context.status in _TERMINAL_STATUSES:
            return
        self._replace(status=DiagnosisStatus.ANALYZING, stage_text=text)
        await self._patch(force=True, fallback_text=text)

    async def finish(
        self,
        status: DiagnosisStatus,
        result_text: str,
        *,
        duration_ms: int | None = None,
        tool_names: tuple[str, ...] | None = None,
    ) -> None:
        if self._context.status in _TERMINAL_STATUSES:
            return
        values: dict[str, Any] = {
            "status": status,
            "result_text": result_text,
            "duration_ms": duration_ms,
        }
        if tool_names is not None:
            values["tool_names"] = tool_names
        self._replace(**values)
        await self._patch(force=True, fallback_text=result_text)

    def _replace(self, **changes: Any) -> None:
        self._context = replace(self._context, updated_at=utc_now(), **changes)
        self._store.save(self._context)

    async def _create_card(self) -> dict[str, Any]:
        card = render_card(self._context)
        if self._context.chat_type == "group":
            try:
                return await self._channel.reply_interactive_message(
                    self._context.source_message_id,
                    card,
                )
            except Exception:  # noqa: BLE001
                logger.warning("Failed to reply with diagnosis card", exc_info=True)
        return await self._channel.send_interactive_message(
            self._context.chat_id,
            card,
            receive_id_type="chat_id",
        )

    async def _patch(self, *, force: bool = False, fallback_text: str | None = None) -> bool:
        if not self._cards_available or not self._context.card_message_id:
            if fallback_text:
                await self._fallback(fallback_text)
            return False
        payload = self._payload()
        if payload == self._last_payload:
            return True
        now = time.monotonic()
        if not force and now - self._last_patch_at < self._patch_interval_seconds:
            self._schedule_patch(fallback_text)
            return True
        if force:
            self._cancel_pending_patch()
        try:
            await self._channel.patch_interactive_message(
                self._context.card_message_id,
                render_card(self._context),
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to patch diagnosis card", exc_info=True)
            self._cards_available = False
            if fallback_text:
                await self._fallback(fallback_text)
            return False
        self._last_payload = payload
        self._last_patch_at = now
        return True

    def _schedule_patch(self, fallback_text: str | None) -> None:
        if fallback_text:
            self._pending_fallback_text = fallback_text
        if self._pending_patch is not None and not self._pending_patch.done():
            return
        delay = max(
            0.0,
            self._patch_interval_seconds - (time.monotonic() - self._last_patch_at),
        )
        self._pending_patch = asyncio.create_task(self._flush_patch(delay))

    async def _flush_patch(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            fallback_text = self._pending_fallback_text
            self._pending_patch = None
            self._pending_fallback_text = None
            await self._patch(force=True, fallback_text=fallback_text)
        except asyncio.CancelledError:
            return

    def _cancel_pending_patch(self) -> None:
        task = self._pending_patch
        self._pending_patch = None
        self._pending_fallback_text = None
        if task is not None and not task.done():
            task.cancel()

    def _payload(self) -> str:
        return json.dumps(render_card(self._context), ensure_ascii=False, sort_keys=True)


def _message_id(response: dict[str, Any]) -> str | None:
    data = response.get("data")
    if not isinstance(data, dict):
        return None
    message_id = data.get("message_id")
    return message_id if isinstance(message_id, str) and message_id else None
