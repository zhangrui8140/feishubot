from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class DiagnosisStatus(StrEnum):
    ACCEPTED = "accepted"
    ANALYZING = "analyzing"
    WAITING_SUPPLEMENT = "waiting_supplement"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    PERMISSION_DENIED = "permission_denied"
    FAILED = "failed"


class FeedbackAction(StrEnum):
    CORRECT = "correct"
    SOLVED = "solved"
    MISMATCH = "mismatch"
    CONTINUE = "continue"


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class DiagnosisContext:
    diagnosis_id: str
    source_message_id: str
    card_message_id: str | None
    chat_id: str
    chat_type: str
    owner_open_id: str
    session_key: str
    question: str
    environment: str = "prod"
    projects: tuple[str, ...] = ()
    status: DiagnosisStatus = DiagnosisStatus.ACCEPTED
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    stage_text: str = ""
    result_text: str = ""
    duration_ms: int | None = None
    tool_names: tuple[str, ...] = ()
    round_index: int = 0

    def __post_init__(self) -> None:
        if self.environment != "prod":
            raise ValueError("environment must be prod")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["projects"] = list(self.projects)
        data["tool_names"] = list(self.tool_names)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosisContext:
        values = dict(data)
        values["status"] = DiagnosisStatus(values["status"])
        values["created_at"] = datetime.fromisoformat(values["created_at"])
        values["updated_at"] = datetime.fromisoformat(values["updated_at"])
        values["projects"] = tuple(values.get("projects", ()))
        values["tool_names"] = tuple(values.get("tool_names", ()))
        return cls(**values)
