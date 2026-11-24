from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from threading import RLock
from typing import Any


class QoderSessionStore:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory).expanduser().resolve()
        self._path = self._directory / "sessions.json"
        self._lock = RLock()

    def get(self, session_key: str) -> str | None:
        with self._lock:
            record = self._load().get(session_key)
            if not isinstance(record, dict):
                return None
            session_id = record.get("session_id")
            return session_id if isinstance(session_id, str) and session_id else None

    def set(self, session_key: str, session_id: str) -> None:
        with self._lock:
            records = self._load()
            records[session_key] = {"session_id": session_id}
            self._save(records)

    def delete(self, session_key: str) -> None:
        with self._lock:
            records = self._load()
            if records.pop(session_key, None) is not None:
                self._save(records)

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save(self, records: dict[str, Any]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        temp_path = self._directory / f"sessions.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temp_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self._path)
