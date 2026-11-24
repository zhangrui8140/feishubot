from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feishubot.config import Settings

EDITABLE_FIELDS = (
    "LOG_LEVEL",
    "FEISHU_APP_ID",
    "QODER_MODEL",
    "QODER_WORKSPACE_ROOT",
    "QODER_SESSION_STORE_DIR",
    "QODER_RUN_TIMEOUT_SECONDS",
    "QODER_CANCEL_GRACE_SECONDS",
    "QODER_SHUTDOWN_GRACE_SECONDS",
    "QODER_MAX_CONCURRENT_RUNS",
    "QODER_MAX_LIVE_SESSIONS",
    "QODER_MAX_QUEUED_PER_SESSION",
    "QODER_SESSION_IDLE_TTL_SECONDS",
    "QODER_DEDUP_TTL_SECONDS",
)

SECRET_FIELDS = (
    "FEISHU_APP_SECRET",
    "FEISHU_VERIFICATION_TOKEN",
    "FEISHU_ENCRYPT_KEY",
    "QODER_PERSONAL_ACCESS_TOKEN",
)

_FIELD_ATTRIBUTES = {
    "LOG_LEVEL": "log_level",
    "FEISHU_APP_ID": "feishu_app_id",
    "QODER_MODEL": "qoder_model",
    "QODER_WORKSPACE_ROOT": "qoder_workspace_root",
    "QODER_SESSION_STORE_DIR": "qoder_session_store_dir",
    "QODER_RUN_TIMEOUT_SECONDS": "qoder_run_timeout_seconds",
    "QODER_CANCEL_GRACE_SECONDS": "qoder_cancel_grace_seconds",
    "QODER_SHUTDOWN_GRACE_SECONDS": "qoder_shutdown_grace_seconds",
    "QODER_MAX_CONCURRENT_RUNS": "qoder_max_concurrent_runs",
    "QODER_MAX_LIVE_SESSIONS": "qoder_max_live_sessions",
    "QODER_MAX_QUEUED_PER_SESSION": "qoder_max_queued_per_session",
    "QODER_SESSION_IDLE_TTL_SECONDS": "qoder_session_idle_ttl_seconds",
    "QODER_DEDUP_TTL_SECONDS": "qoder_dedup_ttl_seconds",
}

_NUMERIC_LIMITS: dict[str, tuple[float, float]] = {
    "QODER_RUN_TIMEOUT_SECONDS": (1, 3600),
    "QODER_CANCEL_GRACE_SECONDS": (0, 300),
    "QODER_SHUTDOWN_GRACE_SECONDS": (0, 600),
    "QODER_MAX_CONCURRENT_RUNS": (1, 64),
    "QODER_MAX_LIVE_SESSIONS": (1, 256),
    "QODER_MAX_QUEUED_PER_SESSION": (0, 100),
    "QODER_SESSION_IDLE_TTL_SECONDS": (60, 604800),
    "QODER_DEDUP_TTL_SECONDS": (60, 604800),
}

_INTEGER_FIELDS = {
    "QODER_MAX_CONCURRENT_RUNS",
    "QODER_MAX_LIVE_SESSIONS",
    "QODER_MAX_QUEUED_PER_SESSION",
}


class ConfigConflictError(ValueError):
    pass


class ConfigValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ManagedConfig:
    values: dict[str, Any]
    secrets: dict[str, bool]
    revision: str


@dataclass(frozen=True)
class _AssignmentSpan:
    key: str
    start: int
    end: int
    line_ending: str


def load_env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return _parse_env_values(path.read_text(encoding="utf-8"))


def env_revision(path: Path) -> str:
    content = path.read_bytes() if path.exists() else b""
    return hashlib.sha256(content).hexdigest()


def read_managed_config(path: Path) -> ManagedConfig:
    raw_values = load_env_values(path)
    configured = Settings(_env_file=path if path.exists() else None)
    values = {
        key: getattr(configured, attribute)
        for key, attribute in _FIELD_ATTRIBUTES.items()
    }
    secrets = {key: bool(raw_values.get(key, "")) for key in SECRET_FIELDS}
    return ManagedConfig(values=values, secrets=secrets, revision=env_revision(path))


def save_managed_config(
    path: Path,
    *,
    expected_revision: str,
    values: dict[str, Any],
    secret_updates: dict[str, dict[str, str]],
) -> ManagedConfig:
    unknown_fields = set(values) - set(EDITABLE_FIELDS)
    unknown_secrets = set(secret_updates) - set(SECRET_FIELDS)
    if unknown_fields or unknown_secrets:
        unknown = sorted(unknown_fields | unknown_secrets)
        raise ConfigValidationError(f"Unknown configuration fields: {', '.join(unknown)}")

    if env_revision(path) != expected_revision:
        raise ConfigConflictError("Configuration changed since it was loaded")

    current = load_env_values(path)
    normalized = _validate_editable_values(values, current)
    updates: dict[str, str] = dict(normalized)
    for key, update in secret_updates.items():
        action = update.get("action", "")
        value = update.get("value", "")
        if action == "keep":
            continue
        if action == "clear":
            updates[key] = ""
            continue
        if action == "replace":
            _validate_text(key, value, allow_empty=False)
            updates[key] = value
            continue
        raise ConfigValidationError(f"Invalid secret action for {key}")

    merged = dict(current)
    merged.update(updates)
    _validate_with_settings(merged)
    write_env_values(path, updates, expected_revision=expected_revision)
    return read_managed_config(path)


def write_env_values(
    path: Path,
    values: dict[str, Any],
    *,
    expected_revision: str | None = None,
) -> str:
    current_revision = env_revision(path)
    if expected_revision is not None and current_revision != expected_revision:
        raise ConfigConflictError("Configuration changed since it was loaded")

    content = path.read_text(encoding="utf-8") if path.exists() else ""
    updates = {key: _stringify(value) for key, value in values.items()}
    rendered = _apply_updates(content, updates)
    atomic_write(path, rendered.encode("utf-8"))
    return env_revision(path)


def _validate_editable_values(
    values: dict[str, Any], current: dict[str, str]
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, raw_value in values.items():
        value = _stringify(raw_value).strip()
        _validate_text(key, value, allow_empty=False)
        if key == "LOG_LEVEL" and value.upper() not in {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        }:
            raise ConfigValidationError("LOG_LEVEL is invalid")
        if key in _NUMERIC_LIMITS:
            value = _validate_numeric(key, value)
        normalized[key] = value

    merged = dict(current)
    merged.update(normalized)
    workspace_raw = merged.get("QODER_WORKSPACE_ROOT", "").strip()
    _validate_text("QODER_WORKSPACE_ROOT", workspace_raw, allow_empty=False)
    workspace = Path(workspace_raw).expanduser()
    if not workspace.exists() or not workspace.is_dir():
        raise ConfigValidationError("QODER_WORKSPACE_ROOT must be an existing directory")

    session_dir_raw = merged.get("QODER_SESSION_STORE_DIR", "").strip()
    _validate_text("QODER_SESSION_STORE_DIR", session_dir_raw, allow_empty=False)
    session_dir = Path(session_dir_raw).expanduser()
    if session_dir.exists() and not session_dir.is_dir():
        raise ConfigValidationError("QODER_SESSION_STORE_DIR must be a directory")

    model = merged.get("QODER_MODEL", "").strip()
    _validate_text("QODER_MODEL", model, allow_empty=False)
    if len(model) > 128:
        raise ConfigValidationError("QODER_MODEL is too long")

    app_id = merged.get("FEISHU_APP_ID", "").strip()
    if len(app_id) > 255:
        raise ConfigValidationError("FEISHU_APP_ID is too long")
    return normalized


def _validate_numeric(key: str, value: str) -> str:
    try:
        number = int(value) if key in _INTEGER_FIELDS else float(value)
    except ValueError as exc:
        raise ConfigValidationError(f"{key} must be a number") from exc
    minimum, maximum = _NUMERIC_LIMITS[key]
    if number < minimum or number > maximum:
        raise ConfigValidationError(f"{key} must be between {minimum:g} and {maximum:g}")
    return str(number) if key in _INTEGER_FIELDS else format(number, "g")


def _validate_text(key: str, value: str, *, allow_empty: bool) -> None:
    if not allow_empty and not value:
        raise ConfigValidationError(f"{key} cannot be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfigValidationError(f"{key} contains control characters")


def _validate_with_settings(values: dict[str, str]) -> None:
    kwargs = {
        key.lower(): value
        for key, value in values.items()
        if key.lower() in Settings.model_fields
    }
    try:
        Settings(_env_file=None, **kwargs)
    except ValueError as exc:
        raise ConfigValidationError("Configuration values are invalid") from exc


def _parse_env_values(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    lines = content.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            index += 1
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("\"", "'")):
            quote = value[0]
            if not (len(value) >= 2 and value.endswith(quote)):
                collected = [value]
                index += 1
                while index < len(lines):
                    collected.append(lines[index])
                    if lines[index].rstrip().endswith(quote):
                        break
                    index += 1
                value = "\n".join(collected)
        if value.startswith("\"") and value.endswith("\"") and len(value) >= 2:
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        elif value.startswith("'") and value.endswith("'") and len(value) >= 2:
            value = value[1:-1]
        values[key] = value
        index += 1
    return values


def _apply_updates(content: str, updates: dict[str, str]) -> str:
    spans = _assignment_spans(content)
    by_key: dict[str, _AssignmentSpan] = {}
    for span in spans:
        by_key[span.key] = span

    replacements: list[tuple[int, int, str]] = []
    missing: list[tuple[str, str]] = []
    for key, value in updates.items():
        span = by_key.get(key)
        assignment = f"{key}={_format_env_value(value)}"
        if span is None:
            missing.append((key, assignment))
        else:
            replacements.append((span.start, span.end, assignment + span.line_ending))

    rendered = content
    for start, end, replacement in sorted(replacements, reverse=True):
        rendered = rendered[:start] + replacement + rendered[end:]

    if missing:
        if rendered and not rendered.endswith(("\n", "\r")):
            rendered += "\n"
        rendered += "".join(f"{assignment}\n" for _, assignment in missing)
    return rendered


def _assignment_spans(content: str) -> list[_AssignmentSpan]:
    lines = content.splitlines(keepends=True)
    spans: list[_AssignmentSpan] = []
    offset = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        start = offset
        end = offset + len(line)
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            offset = end
            index += 1
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("\"", "'")):
            quote = value[0]
            while not (len(value) >= 2 and value.endswith(quote)) and index + 1 < len(lines):
                index += 1
                offset = end
                line = lines[index]
                end += len(line)
                value += line.rstrip("\r\n")
        line_ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        spans.append(_AssignmentSpan(key=key, start=start, end=end, line_ending=line_ending))
        offset = end
        index += 1
    return spans


def _format_env_value(value: str) -> str:
    needs_quote = any(character.isspace() for character in value) or any(
        character in value for character in ["#", '"', "'"]
    )
    if not needs_quote:
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        raise ConfigValidationError("Boolean values are not supported")
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ConfigValidationError("Configuration values must be strings or numbers")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
