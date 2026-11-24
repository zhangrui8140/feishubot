from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from feishubot.diagnosis.card import FEEDBACK_OPTIONS, FEEDBACK_STATUSES
from feishubot.diagnosis.controller import CardChannel, DiagnosisController
from feishubot.diagnosis.models import DiagnosisContext, FeedbackAction, utc_now
from feishubot.diagnosis.store import DiagnosisStore, FeedbackOutcome
from feishubot.qoder_agent.projects import mentioned_projects


class DiagnosisRuntime:
    def __init__(
        self,
        channel: CardChannel,
        store: DiagnosisStore,
        *,
        cards_enabled: bool = True,
    ) -> None:
        self._channel = channel
        self._store = store
        self._cards_enabled = cards_enabled
        self._active: dict[str, DiagnosisController] = {}

    @property
    def store(self) -> DiagnosisStore:
        return self._store

    def active(self, session_key: str) -> DiagnosisController | None:
        return self._active.get(session_key)

    def create(
        self,
        *,
        message: Any,
        question: str,
        owner_open_id: str,
        session_key: str,
        available_projects: tuple[str, ...],
        fallback: Callable[[str], Awaitable[str | None]],
        reuse: DiagnosisContext | None = None,
    ) -> DiagnosisController:
        previous = reuse or self.context_from_reply(message, owner_open_id, check_owner=False)
        now = utc_now()
        projects = mentioned_projects(question, available_projects)
        if previous and previous.projects:
            projects = tuple(sorted(set(projects) | set(previous.projects)))
        can_reuse_card = previous is not None and previous.owner_open_id == owner_open_id
        context = DiagnosisContext(
            diagnosis_id=uuid4().hex,
            source_message_id=str(message.message_id),
            card_message_id=previous.card_message_id if can_reuse_card else None,
            chat_id=str(message.chat_id),
            chat_type=str(message.chat_type),
            owner_open_id=owner_open_id,
            session_key=session_key,
            question=question,
            projects=projects,
            created_at=now,
            updated_at=now,
            round_index=previous.round_index + 1 if previous else 0,
        )
        controller = DiagnosisController(
            self._channel,
            self._store,
            context,
            fallback,
            cards_enabled=self._cards_enabled,
        )
        self._active[session_key] = controller
        return controller

    def release(self, session_key: str, controller: DiagnosisController) -> None:
        if self._active.get(session_key) is controller:
            self._active.pop(session_key, None)

    def context_from_reply(
        self, message: Any, user_open_id: str | None = None, *, check_owner: bool = False
    ) -> DiagnosisContext | None:
        for attribute in ("parent_id", "root_id"):
            message_id = getattr(message, attribute, None)
            if not message_id:
                continue
            context = self._store.get_by_message_id(str(message_id))
            if context is None:
                context = self._store.get_by_card_message_id(str(message_id))
            if context is None:
                continue
            if check_owner and user_open_id and context.owner_open_id != user_open_id:
                continue
            if context.chat_id != str(message.chat_id):
                continue
            return context
        return None

    def latest_context(self, session_key: str) -> DiagnosisContext | None:
        return self._store.get_latest_by_session_key(session_key)

    def _context_from_reply(
        self, message: Any, owner_open_id: str
    ) -> DiagnosisContext | None:
        return self.context_from_reply(message, owner_open_id, check_owner=True)


def create_store(directory: str | Path | None) -> DiagnosisStore:
    if directory is None or not str(directory).strip():
        return DiagnosisStore(None)
    return DiagnosisStore(Path(directory).expanduser().resolve())


def parse_feedback_text(text: str) -> FeedbackAction | None:
    normalized = text.strip()
    if normalized in {"1", "2", "3", "4"}:
        return FEEDBACK_OPTIONS[int(normalized) - 1][0]
    for action, label in FEEDBACK_OPTIONS:
        if normalized == label:
            return action
    return None


def record_text_feedback(
    store: DiagnosisStore,
    message: Any,
    text: str,
    user_open_id: str | None,
    runtime: DiagnosisRuntime,
) -> FeedbackOutcome | None:
    action = parse_feedback_text(text)
    if action is None:
        return None
    context = runtime._context_from_reply(message, user_open_id or "")
    if context is None:
        return None
    if context.status not in FEEDBACK_STATUSES:
        return FeedbackOutcome(False, False, "当前诊断无需反馈")
    outcome = store.record_feedback(
        context.diagnosis_id,
        context.owner_open_id,
        context.card_message_id or "",
        context.chat_id,
        action,
    )
    if not outcome.ok:
        return outcome
    if action in {FeedbackAction.MISMATCH, FeedbackAction.CONTINUE}:
        return replace(
            outcome,
            message="请回复本卡片，补充具体差异、单号或截图后继续。",
        )
    return outcome
