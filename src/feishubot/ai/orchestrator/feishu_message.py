from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

_MENTION_PATTERN = re.compile(r"@_(?:user_\d+|all)")
_FORWARDED_FIELDS = ("message_id", "chat_id", "chat_type", "thread_id", "root_id", "parent_id")


def forward_payload(
    message: Any, text: str, user_open_id: str | None, image_keys: list[str] | None
) -> dict[str, Any]:
    payload = {name: getattr(message, name, None) for name in _FORWARDED_FIELDS}
    payload.update(text=text, user_open_id=user_open_id, image_keys=image_keys or [])
    return payload


def message_from_payload(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**{name: payload.get(name) for name in _FORWARDED_FIELDS})


def strip_mentions(text: str) -> str:
    """Drop Feishu @ placeholders so commands like "@机器人 终止" still match."""
    if not _MENTION_PATTERN.search(text):
        return text
    lines = (" ".join(_MENTION_PATTERN.sub(" ", line).split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def extract_text(raw_content: Any) -> str | None:
    if not isinstance(raw_content, str):
        return None
    text = raw_content.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(payload, dict):
        parsed_text = payload.get("text")
        if isinstance(parsed_text, str) and parsed_text.strip():
            return parsed_text.strip()
    return text


def _load_json_object(raw_content: Any) -> dict[str, Any]:
    if not isinstance(raw_content, str):
        return {}
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def extract_image_message(message_type: str, raw_content: Any) -> tuple[str, list[str]]:
    payload = _load_json_object(raw_content)
    if message_type == "image":
        image_key = payload.get("image_key")
        return "", [image_key] if isinstance(image_key, str) and image_key else []

    body = payload
    if "content" not in body:
        body = next(
            (value for value in payload.values() if isinstance(value, dict) and "content" in value),
            {},
        )
    lines: list[str] = []
    title = body.get("title")
    if isinstance(title, str) and title.strip():
        lines.append(title.strip())
    image_keys: list[str] = []
    paragraphs = body.get("content")
    for paragraph in paragraphs if isinstance(paragraphs, list) else []:
        parts: list[str] = []
        for element in paragraph if isinstance(paragraph, list) else []:
            if not isinstance(element, dict):
                continue
            if element.get("tag") == "img" and isinstance(element.get("image_key"), str):
                image_keys.append(element["image_key"])
            elif isinstance(element.get("text"), str):
                parts.append(element["text"])
        line = "".join(parts).strip()
        if line:
            lines.append(line)
    return "\n".join(lines), image_keys
