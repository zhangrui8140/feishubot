from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from feishubot.diagnosis.models import DiagnosisContext, FeedbackAction


@dataclass(frozen=True, slots=True)
class FeedbackOutcome:
    ok: bool
    duplicate: bool
    message: str


class DiagnosisStore:
    def __init__(self, directory: str | Path | None) -> None:
        self._directory = Path(directory).expanduser().resolve() if directory is not None else None
        self._lock = RLock()

    @property
    def enabled(self) -> bool:
        return self._directory is not None

    def save(self, context: DiagnosisContext) -> bool:
        if self._directory is None or not _safe_id(context.diagnosis_id):
            return False
        with self._lock:
            try:
                self._atomic_write(self._context_path(context.diagnosis_id), context.to_dict())
                if context.card_message_id:
                    self._atomic_write(
                        self._card_path(context.card_message_id),
                        {"diagnosis_id": context.diagnosis_id},
                    )
                    self._atomic_write(
                        self._message_path(context.card_message_id),
                        {"diagnosis_id": context.diagnosis_id},
                    )
                if context.source_message_id:
                    self._atomic_write(
                        self._message_path(context.source_message_id),
                        {"diagnosis_id": context.diagnosis_id},
                    )
                if context.session_key:
                    self._atomic_write(
                        self._session_path(context.session_key),
                        {
                            "diagnosis_id": context.diagnosis_id,
                            "updated_at": context.updated_at.isoformat(),
                        },
                    )
            except OSError:
                return False
        return True

    def get(self, diagnosis_id: str) -> DiagnosisContext | None:
        if self._directory is None or not _safe_id(diagnosis_id):
            return None
        with self._lock:
            payload = self._read_json(self._context_path(diagnosis_id))
            if not isinstance(payload, dict):
                return None
            try:
                return DiagnosisContext.from_dict(payload)
            except (KeyError, TypeError, ValueError):
                return None

    def get_by_card_message_id(self, card_message_id: str) -> DiagnosisContext | None:
        if self._directory is None or not card_message_id:
            return None
        with self._lock:
            index = self._read_json(self._card_path(card_message_id))
            if not isinstance(index, dict) or not isinstance(index.get("diagnosis_id"), str):
                return None
            return self.get(index["diagnosis_id"])

    def get_by_message_id(self, message_id: str) -> DiagnosisContext | None:
        if self._directory is None or not message_id:
            return None
        with self._lock:
            index = self._read_json(self._message_path(message_id))
            if isinstance(index, dict) and isinstance(index.get("diagnosis_id"), str):
                return self.get(index["diagnosis_id"])
            return self.get_by_card_message_id(message_id)

    def get_latest_by_session_key(self, session_key: str) -> DiagnosisContext | None:
        if self._directory is None or not session_key:
            return None
        with self._lock:
            index = self._read_json(self._session_path(session_key))
            if isinstance(index, dict) and isinstance(index.get("diagnosis_id"), str):
                return self.get(index["diagnosis_id"])
            return None

    def record_feedback(
        self,
        diagnosis_id: str,
        operator_open_id: str,
        card_message_id: str,
        chat_id: str,
        action: FeedbackAction | str,
    ) -> FeedbackOutcome:
        if self._directory is None:
            return FeedbackOutcome(False, False, "诊断存储未启用")
        try:
            parsed_action = FeedbackAction(action)
        except ValueError:
            return FeedbackOutcome(False, False, "反馈动作无效")
        with self._lock:
            context = self.get(diagnosis_id)
            mismatch = _validate_feedback(context, operator_open_id, card_message_id, chat_id)
            if mismatch:
                return FeedbackOutcome(False, False, mismatch)
            key = f"{diagnosis_id}:{operator_open_id}:{parsed_action.value}"
            record = {
                "key": key,
                "diagnosis_id": diagnosis_id,
                "operator_open_id": operator_open_id,
                "card_message_id": card_message_id,
                "chat_id": chat_id,
                "action": parsed_action.value,
            }
            try:
                created = self._atomic_create(self._feedback_path(diagnosis_id, key), record)
            except OSError:
                return FeedbackOutcome(False, False, "反馈保存失败")
            if not created:
                return FeedbackOutcome(True, True, "反馈已记录")
            return FeedbackOutcome(True, False, "反馈已记录")

    def _context_path(self, diagnosis_id: str) -> Path:
        return self._required_directory() / "contexts" / f"{diagnosis_id}.json"

    def _card_path(self, card_message_id: str) -> Path:
        digest = hashlib.sha256(card_message_id.encode("utf-8")).hexdigest()
        return self._required_directory() / "card_index" / f"{digest}.json"

    def _message_path(self, message_id: str) -> Path:
        digest = hashlib.sha256(message_id.encode("utf-8")).hexdigest()
        return self._required_directory() / "message_index" / f"{digest}.json"

    def _session_path(self, session_key: str) -> Path:
        digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
        return self._required_directory() / "session_index" / f"{digest}.json"

    def _feedback_path(self, diagnosis_id: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._required_directory() / "feedback" / diagnosis_id / f"{digest}.json"

    def _required_directory(self) -> Path:
        if self._directory is None:
            raise RuntimeError("diagnosis store is disabled")
        return self._directory

    @staticmethod
    def _read_json(path: Path) -> Any | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _atomic_write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_create(path: Path, payload: Any) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
            return True
        finally:
            temporary.unlink(missing_ok=True)


def _safe_id(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and all(
        character.isalnum() or character in "-_" for character in value
    )


def _validate_feedback(
    context: DiagnosisContext | None,
    operator_open_id: str,
    card_message_id: str,
    chat_id: str,
) -> str | None:
    if context is None:
        return "诊断记录不存在"
    if context.owner_open_id != operator_open_id:
        return "仅诊断发起人可反馈"
    if context.card_message_id != card_message_id:
        return "卡片消息不匹配"
    if context.chat_id != chat_id:
        return "会话不匹配"
    return None
