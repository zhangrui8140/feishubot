from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feishubot.env_config import (
    ConfigConflictError,
    ConfigValidationError,
    atomic_write,
)

DMS_SERVER = "aliyun-dms-mcp-server"
OBSERVABILITY_SERVER = "observability"
MANAGED_SERVERS = (DMS_SERVER, OBSERVABILITY_SERVER)

AUTH_MODE_STATIC = "static_token"
AUTH_MODE_OAUTH = "oauth"
SERVER_AUTH_MODES = {
    DMS_SERVER: AUTH_MODE_STATIC,
    OBSERVABILITY_SERVER: AUTH_MODE_OAUTH,
}

BACKUP_SUFFIX = ".bak-admin"
_BEARER_PREFIX = "Bearer "


@dataclass(frozen=True)
class ManagedMcpConfig:
    servers: dict[str, Any]
    revision: str
    parse_error: str | None = None


def mcp_revision(path: Path) -> str:
    content = path.read_bytes() if path.exists() else b""
    return hashlib.sha256(content).hexdigest()


def read_mcp_config(path: Path) -> ManagedMcpConfig:
    revision = mcp_revision(path)
    payload = _load_payload(path)
    if payload is None:
        return ManagedMcpConfig(
            servers=_empty_servers(),
            revision=revision,
            parse_error="mcp.json is not a valid JSON object; saving rewrites the whole file",
        )
    raw_servers = payload.get("mcpServers")
    if not isinstance(raw_servers, dict):
        return ManagedMcpConfig(
            servers=_empty_servers(),
            revision=revision,
            parse_error="mcp.json must contain an mcpServers object; saving rewrites it",
        )
    return ManagedMcpConfig(
        servers={
            DMS_SERVER: _sanitize_http(DMS_SERVER, raw_servers.get(DMS_SERVER)),
            OBSERVABILITY_SERVER: _sanitize_http(
                OBSERVABILITY_SERVER, raw_servers.get(OBSERVABILITY_SERVER)
            ),
        },
        revision=revision,
    )


def save_mcp_config(
    path: Path,
    *,
    expected_revision: str,
    dms: dict[str, Any],
    observability: dict[str, Any],
) -> ManagedMcpConfig:
    if mcp_revision(path) != expected_revision:
        raise ConfigConflictError("MCP configuration changed since it was loaded")

    payload = _load_payload(path) or {}
    raw_servers = payload.get("mcpServers")
    current_servers = raw_servers if isinstance(raw_servers, dict) else {}

    entries: dict[str, dict[str, Any] | None] = {
        DMS_SERVER: _build_http_entry(
            DMS_SERVER,
            dms,
            current_servers.get(DMS_SERVER),
            server_type="sse",
            auth_mode=SERVER_AUTH_MODES[DMS_SERVER],
            required=True,
        ),
        OBSERVABILITY_SERVER: _build_http_entry(
            OBSERVABILITY_SERVER,
            observability,
            current_servers.get(OBSERVABILITY_SERVER),
            server_type="http",
            auth_mode=SERVER_AUTH_MODES[OBSERVABILITY_SERVER],
        ),
    }

    new_servers = {
        name: entry for name, entry in current_servers.items() if name not in MANAGED_SERVERS
    }
    for name, entry in entries.items():
        if entry is not None:
            new_servers[name] = entry

    if path.is_file():
        shutil.copy2(path, path.with_name(path.name + BACKUP_SUFFIX))

    payload["mcpServers"] = new_servers
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    atomic_write(path, content.encode("utf-8"))
    return read_mcp_config(path)


def _load_payload(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _empty_servers() -> dict[str, Any]:
    return {
        DMS_SERVER: _sanitize_http(DMS_SERVER, None),
        OBSERVABILITY_SERVER: _sanitize_http(OBSERVABILITY_SERVER, None),
    }


def _build_http_entry(
    name: str,
    update: dict[str, Any],
    current: Any,
    *,
    server_type: str,
    auth_mode: str,
    required: bool = False,
) -> dict[str, Any] | None:
    if not update.get("enabled"):
        if required:
            raise ConfigValidationError(f"{name} cannot be disabled")
        return None
    url = _clean_text(update.get("url"), f"{name} url")
    if not url.startswith(("http://", "https://")):
        raise ConfigValidationError(f"{name} url must start with http:// or https://")
    if auth_mode == AUTH_MODE_OAUTH:
        # qodercli owns this token; a stored static header would shadow it.
        return {"url": url, "type": server_type}
    token = _resolved_secret(
        update.get("token"),
        name,
        "token",
        _current_authorization(current),
    )
    if not token.startswith(_BEARER_PREFIX):
        token = f"{_BEARER_PREFIX}{token}"
    return {"url": url, "type": server_type, "headers": {"Authorization": token}}


def _sanitize_http(name: str, current: Any) -> dict[str, Any]:
    url = current.get("url") if isinstance(current, dict) else None
    return {
        "kind": "http",
        "enabled": isinstance(current, dict),
        "url": url if isinstance(url, str) else "",
        "auth_mode": SERVER_AUTH_MODES.get(name, AUTH_MODE_STATIC),
        "token_configured": bool(_current_authorization(current)),
    }


def _clean_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ConfigValidationError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ConfigValidationError(f"{field} cannot be empty")
    _reject_control_characters(text, field)
    return text


def _resolved_secret(update: Any, name: str, label: str, current_value: str) -> str:
    if not isinstance(update, dict):
        raise ConfigValidationError(f"{name} {label} update is invalid")
    action = update.get("action")
    value = update.get("value", "")
    if action == "keep":
        if not current_value:
            raise ConfigValidationError(
                f"{name} {label} has no stored value; choose replace to set it"
            )
        return current_value
    if action == "replace":
        if not isinstance(value, str) or not value.strip():
            raise ConfigValidationError(f"{name} {label} cannot be empty")
        return value
    if action == "clear":
        raise ConfigValidationError(
            f"{name} {label} cannot be cleared while the server is enabled; "
            "disable the server instead"
        )
    raise ConfigValidationError(f"Invalid {name} {label} action")


def _current_authorization(current: Any) -> str:
    if not isinstance(current, dict):
        return ""
    headers = current.get("headers")
    if not isinstance(headers, dict):
        return ""
    value = headers.get("Authorization")
    return value if isinstance(value, str) else ""


def _reject_control_characters(value: str, field: str) -> None:
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfigValidationError(f"{field} contains control characters")
