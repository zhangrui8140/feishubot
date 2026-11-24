"""Observability MCP authorization via qodercli's own OAuth flow.

qodercli owns the token: it dynamically registers a client, binds a random
loopback port for the redirect, exchanges the code and persists the token in
its own store, then silently refreshes it on later runs. This module only
drives that flow and reports state -- it never handles PKCE, refresh or token
storage itself.

Two constraints shape the design:

* The CLI's loopback callback server only exists while its process lives, so an
  authorization flow must hold one client open until the human finishes SSO.
* The client is configured with the observability server alone. Loading the
  whole mcp.json would also reconnect the DMS SSE endpoint, which rate-limits
  rapid reconnects with 429.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qoder_agent_sdk import (
    QoderAgentOptions,
    QoderSDKClient,
    access_token,
    access_token_from_env,
)

logger = logging.getLogger(__name__)

OBSERVABILITY_SERVER = "observability"
CONNECTED = "connected"
DISABLED = "disabled"

FLOW_TIMEOUT_SECONDS = 600.0
POLL_INTERVAL_SECONDS = 3.0
PROBE_CACHE_SECONDS = 20.0


class ObservabilityAuthError(RuntimeError):
    """The authorization flow could not be started, queried or cleared."""


@dataclass
class AuthorizerSettings:
    workspace_root: Path
    model: str
    mcp_config_path: Path
    personal_access_token: str = ""


@dataclass
class _Flow:
    client: QoderSDKClient
    auth_url: str
    started_at: float
    status: dict[str, Any]
    task: asyncio.Task[None] | None = None

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.started_at) > FLOW_TIMEOUT_SECONDS

    def snapshot(self) -> dict[str, Any]:
        return {
            "flowActive": True,
            "authUrl": self.auth_url,
            **self.status,
        }


def _summarize(status: Any) -> dict[str, Any]:
    servers = status.get("mcpServers") if isinstance(status, dict) else None
    for server in servers or []:
        if not isinstance(server, dict) or server.get("name") != OBSERVABILITY_SERVER:
            continue
        state = str(server.get("status") or "unknown")
        return {
            "status": state,
            "authorized": state == CONNECTED,
            "error": server.get("error"),
            "tools": len(server.get("tools") or []),
        }
    return {
        "status": "absent",
        "authorized": False,
        "error": f"{OBSERVABILITY_SERVER} is not configured",
        "tools": 0,
    }


def _disabled_status() -> dict[str, Any]:
    return {
        "status": DISABLED,
        "authorized": False,
        "error": None,
        "tools": 0,
    }


class ObservabilityAuthorizer:
    """Drives and reports the observability OAuth flow from the gateway."""

    def __init__(self, settings: AuthorizerSettings) -> None:
        self._settings = settings
        self._flow_lock = asyncio.Lock()
        self._probe_lock = asyncio.Lock()
        self._flow: _Flow | None = None
        self._probe_cache: tuple[float, dict[str, Any]] | None = None
        self._probe_task: asyncio.Task[None] | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> ObservabilityAuthorizer:
        token = settings.qoder_personal_access_token
        return cls(
            AuthorizerSettings(
                workspace_root=Path(settings.qoder_workspace_root).expanduser(),
                model=settings.qoder_model,
                mcp_config_path=Path(settings.mcp_servers_config_path).expanduser(),
                personal_access_token=token.get_secret_value() if hasattr(token, "get_secret_value") else str(token or ""),
            )
        )

    async def start(self) -> dict[str, Any]:
        """Begin an authorization flow, or return the one already in flight."""
        async with self._flow_lock:
            flow = self._flow
            if flow is not None and not flow.expired:
                return flow.snapshot()
            await self._close_flow_locked()

            entry = self._observability_entry()
            if entry is None:
                raise ObservabilityAuthError(
                    f"{OBSERVABILITY_SERVER} is not enabled in the MCP configuration"
                )
            client = self._build_client(entry)
            try:
                await client.connect()
                result = await client.mcp_authenticate(OBSERVABILITY_SERVER)
            except Exception as exc:  # noqa: BLE001
                await self._quiet_disconnect(client)
                raise ObservabilityAuthError(f"Could not start authorization: {exc}") from exc

            flow = _Flow(
                client=client,
                auth_url=str(result.get("authUrl") or ""),
                started_at=time.monotonic(),
                status={
                    "status": "pending",
                    "authorized": not result.get("requiresUserAction", True),
                    "error": None,
                    "tools": 0,
                },
            )
            self._flow = flow
            flow.task = asyncio.create_task(self._watch(flow))
            self._probe_cache = None
            return flow.snapshot()

    async def status(self) -> dict[str, Any]:
        flow = self._flow
        if flow is not None and not flow.expired:
            return flow.snapshot()
        cached = self._probe_cache
        if cached and (time.monotonic() - cached[0]) < PROBE_CACHE_SECONDS:
            return {"flowActive": False, **cached[1]}
        if cached:
            # A probe costs a full CLI connect (~30s), so answer with the stale
            # value and refresh in the background instead of blocking callers.
            self._schedule_probe_refresh()
            return {"flowActive": False, **cached[1]}
        async with self._probe_lock:
            cached = self._probe_cache
            if cached and (time.monotonic() - cached[0]) < PROBE_CACHE_SECONDS:
                return {"flowActive": False, **cached[1]}
            result = await self._probe()
            self._probe_cache = (time.monotonic(), result)
            return {"flowActive": False, **result}

    def _schedule_probe_refresh(self) -> None:
        if self._probe_task is not None and not self._probe_task.done():
            return
        self._probe_task = asyncio.create_task(self._refresh_probe())

    async def _refresh_probe(self) -> None:
        async with self._probe_lock:
            try:
                result = await self._probe()
            except Exception:  # noqa: BLE001
                logger.warning("observability status refresh failed", exc_info=True)
                return
            self._probe_cache = (time.monotonic(), result)

    async def clear(self) -> None:
        """Drop the stored token so the next analysis reports the log source as unavailable."""
        async with self._flow_lock:
            await self._close_flow_locked()
        entry = self._observability_entry()
        if entry is None:
            self._probe_cache = (time.monotonic(), _disabled_status())
            return
        client = self._build_client(entry)
        try:
            await client.connect()
            await client.mcp_clear_auth(OBSERVABILITY_SERVER)
        except Exception as exc:  # noqa: BLE001
            raise ObservabilityAuthError(f"Could not clear authorization: {exc}") from exc
        finally:
            await self._quiet_disconnect(client)
        self._probe_cache = None

    async def close(self) -> None:
        async with self._flow_lock:
            await self._close_flow_locked()
        if self._probe_task is not None:
            self._probe_task.cancel()
            self._probe_task = None

    async def _watch(self, flow: _Flow) -> None:
        summary = flow.status
        while not flow.expired:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            try:
                summary = _summarize(await flow.client.get_mcp_status())
            except Exception as exc:  # noqa: BLE001
                logger.warning("observability auth poll failed: %s", exc)
                continue
            flow.status = summary
            if summary.get("status") == CONNECTED:
                logger.info("observability authorization completed")
                break
        # No lock here: _close_flow_locked cancels this task while holding it.
        # The check and the clear are atomic because nothing awaits between them.
        if self._flow is flow:
            self._flow = None
            self._probe_cache = None
        await self._quiet_disconnect(flow.client)

    async def _probe(self) -> dict[str, Any]:
        entry = self._observability_entry()
        if entry is None:
            return _disabled_status()
        client = self._build_client(entry)
        try:
            await client.connect()
            return _summarize(await client.get_mcp_status())
        except Exception as exc:  # noqa: BLE001
            raise ObservabilityAuthError(f"Could not read authorization status: {exc}") from exc
        finally:
            await self._quiet_disconnect(client)

    async def _close_flow_locked(self) -> None:
        flow = self._flow
        self._flow = None
        self._probe_cache = None
        if flow is None:
            return
        if flow.task is not None and not flow.task.done():
            flow.task.cancel()
        await self._quiet_disconnect(flow.client)

    @staticmethod
    async def _quiet_disconnect(client: QoderSDKClient) -> None:
        with contextlib.suppress(Exception):
            await client.disconnect()

    def _observability_entry(self) -> dict[str, Any] | None:
        """The observability server config, without any static Authorization header.

        A leftover static header would shadow the OAuth token, so it is dropped
        here regardless of what mcp.json still contains.
        """
        path = self._settings.mcp_config_path
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        servers = payload.get("mcpServers") if isinstance(payload, dict) else None
        entry = servers.get(OBSERVABILITY_SERVER) if isinstance(servers, dict) else None
        if not isinstance(entry, dict):
            return None
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            return None
        return {
            "url": url,
            "type": entry.get("type") or "http",
        }

    def _build_client(self, entry: dict[str, Any]) -> QoderSDKClient:
        options = QoderAgentOptions(
            auth=self._auth(),
            cwd=self._settings.workspace_root,
            model=self._settings.model,
            permission_mode="dontAsk",
            setting_sources=[],
            mcp_servers={OBSERVABILITY_SERVER: entry},
            allowed_mcp_server_names=[OBSERVABILITY_SERVER],
            env={
                "QODER_AGENT_SDK_ENTRYPOINT": None,
                "QODER_AGENT_SDK_VERSION": None,
            },
            stderr=_ignore_stderr,
        )
        return QoderSDKClient(options=options)

    def _auth(self) -> Any:
        if self._settings.personal_access_token:
            return access_token(self._settings.personal_access_token)
        return access_token_from_env()


def _ignore_stderr(line: str) -> None:
    clean = line.strip()
    if clean:
        logger.debug("[Qoder CLI STDERR] %s", clean)
