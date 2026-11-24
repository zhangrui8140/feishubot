from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Channel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def send_text_message(
        self, receive_id: str, text: str, receive_id_type: str = "open_id"
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def reply_text_message(self, message_id: str, text: str) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} channel does not support replies")

    async def send_interactive_message(
        self,
        receive_id: str,
        card: dict[str, Any],
        receive_id_type: str = "open_id",
    ) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} channel does not support interactive messages")

    async def reply_interactive_message(
        self, message_id: str, card: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} channel does not support interactive replies")

    async def patch_interactive_message(
        self, message_id: str, card: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError(f"{self.name} channel does not support interactive updates")

    async def get_message_image_data_url(self, *, message_id: str, file_key: str) -> str:
        raise NotImplementedError(f"{self.name} channel does not support image download")
