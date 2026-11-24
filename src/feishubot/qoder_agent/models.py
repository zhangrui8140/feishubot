from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AgentProgressKind = Literal["started", "assistant_text", "tool_started"]


@dataclass(frozen=True, slots=True)
class AgentProgress:
    kind: AgentProgressKind
    text: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class SessionKey:
    chat_id: str
    thread_scope: str
    user_open_id: str

    @classmethod
    def build(
        cls,
        *,
        chat_id: str,
        user_open_id: str,
        message_id: str,
        thread_id: str | None = None,
    ) -> SessionKey:
        # Quote replies stay in-session; only Feishu topics create a separate scope.
        thread_scope = thread_id or "root"
        isolated_user = user_open_id or f"anonymous-{message_id}"
        return cls(
            chat_id=chat_id,
            thread_scope=thread_scope,
            user_open_id=isolated_user,
        )

    @property
    def value(self) -> str:
        return f"feishu:{self.chat_id}:{self.thread_scope}:{self.user_open_id}"

    def for_task(self, message_id: str) -> "SessionKey":
        """Independent session scope for one parallel task (dev multi-task).

        The suffix keeps ``value`` prefixed by the chat-scope value so
        running tasks can be listed with a startswith scan.
        """
        return SessionKey(
            chat_id=self.chat_id,
            thread_scope=self.thread_scope,
            user_open_id=f"{self.user_open_id}:t:{message_id}",
        )

    @classmethod
    def parse(cls, value: str) -> "SessionKey":
        _, chat_id, thread_scope, user_open_id = value.split(":", 3)
        return cls(
            chat_id=chat_id,
            thread_scope=thread_scope,
            user_open_id=user_open_id,
        )


@dataclass(frozen=True, slots=True)
class ImageInput:
    media_type: str
    data_base64: str


@dataclass(frozen=True, slots=True)
class AgentRequest:
    message_id: str
    session_key: SessionKey
    prompt: str
    images: tuple[ImageInput, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentResult:
    text: str
    session_id: str
    duration_ms: int
    duration_api_ms: int
    tool_names: tuple[str, ...]
    needs_input: bool = False
    observability_unauthorized: bool = False


class AgentError(Exception):
    pass


class DuplicateMessageError(AgentError):
    pass


class AgentBusyError(AgentError):
    pass


class AgentTimeoutError(AgentError):
    pass


class AgentCancelledError(AgentError):
    pass


class AgentUnavailableError(AgentError):
    pass


class AgentPermissionError(AgentError):
    pass
