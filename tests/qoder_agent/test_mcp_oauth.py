from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from feishubot.qoder_agent import mcp_oauth
from feishubot.qoder_agent.mcp_oauth import (
    AuthorizerSettings,
    ObservabilityAuthError,
    ObservabilityAuthorizer,
)


class FakeClient:
    instances: list["FakeClient"] = []
    default_status: dict = {
        "mcpServers": [
            {"name": "observability", "status": "needs-auth", "tools": []}
        ]
    }

    def __init__(self, options) -> None:
        self.options = options
        self.connected = False
        self.cleared = False
        self.authenticate_calls = 0
        self.status_response = dict(self.default_status)
        self.authenticate_result: dict = {
            "requiresUserAction": True,
            "authUrl": "https://sso.example.com/authorize?client_id=x",
        }
        FakeClient.instances.append(self)

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def mcp_authenticate(self, server_name: str, redirect_uri=None) -> dict:
        self.authenticate_calls += 1
        return dict(self.authenticate_result)

    async def get_mcp_status(self) -> dict:
        return self.status_response

    async def mcp_clear_auth(self, server_name: str) -> None:
        self.cleared = True


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    FakeClient.instances = []
    FakeClient.default_status = {
        "mcpServers": [
            {"name": "observability", "status": "needs-auth", "tools": []}
        ]
    }
    monkeypatch.setattr(mcp_oauth, "QoderSDKClient", FakeClient)
    monkeypatch.setattr(mcp_oauth, "POLL_INTERVAL_SECONDS", 0.02)
    return FakeClient


def make_authorizer(tmp_path: Path, config: dict | None) -> ObservabilityAuthorizer:
    config_path = tmp_path / "mcp.json"
    if config is not None:
        config_path.write_text(json.dumps(config), encoding="utf-8")
    return ObservabilityAuthorizer(
        AuthorizerSettings(
            workspace_root=tmp_path,
            model="test-model",
            mcp_config_path=config_path,
            personal_access_token="token",
        )
    )


def mcp_config(**obs_extra) -> dict:
    return {
        "mcpServers": {
            "aliyun-dms-mcp-server": {"url": "https://dms/sse", "type": "sse"},
            "observability": {"url": "https://obs/mcp", "type": "http", **obs_extra},
        }
    }


def mcp_config_without_obs() -> dict:
    return {
        "mcpServers": {
            "aliyun-dms-mcp-server": {"url": "https://dms/sse", "type": "sse"},
        }
    }


@pytest.mark.asyncio
async def test_start_returns_auth_url_and_holds_client(tmp_path):
    authorizer = make_authorizer(tmp_path, mcp_config())

    result = await authorizer.start()

    assert result["authUrl"].startswith("https://sso.example.com/authorize")
    assert result["flowActive"] is True
    client = FakeClient.instances[0]
    assert client.connected is True
    assert client.options.mcp_servers["observability"] == {
        "url": "https://obs/mcp",
        "type": "http",
    }
    await authorizer.close()


@pytest.mark.asyncio
async def test_start_reuses_in_flight_flow_instead_of_second_client(tmp_path):
    authorizer = make_authorizer(tmp_path, mcp_config())

    first = await authorizer.start()
    second = await authorizer.start()

    assert second["authUrl"] == first["authUrl"]
    assert len(FakeClient.instances) == 1
    assert FakeClient.instances[0].authenticate_calls == 1
    await authorizer.close()


@pytest.mark.asyncio
async def test_flow_completes_when_status_turns_connected(tmp_path):
    authorizer = make_authorizer(tmp_path, mcp_config())

    await authorizer.start()
    client = FakeClient.instances[0]
    client.status_response = {
        "mcpServers": [{"name": "observability", "status": "connected", "tools": ["a"]}]
    }
    await asyncio.sleep(0.1)

    assert authorizer._flow is None
    assert client.connected is False


@pytest.mark.asyncio
async def test_status_probes_transient_client_and_caches(tmp_path):
    authorizer = make_authorizer(tmp_path, mcp_config())

    first = await authorizer.status()
    second = await authorizer.status()

    assert first["status"] == "needs-auth"
    assert first["authorized"] is False
    assert second == first
    assert len(FakeClient.instances) == 1


@pytest.mark.asyncio
async def test_status_reports_connected_when_authorized(tmp_path):
    authorizer = make_authorizer(tmp_path, mcp_config())
    FakeClient.default_status = {
        "mcpServers": [{"name": "observability", "status": "connected", "tools": ["a"]}]
    }

    result = await authorizer.status()

    assert result["authorized"] is True
    assert result["tools"] == 1


@pytest.mark.asyncio
async def test_stale_status_returns_immediately_and_refreshes_in_background(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_oauth, "PROBE_CACHE_SECONDS", 0.05)
    authorizer = make_authorizer(tmp_path, mcp_config())

    first = await authorizer.status()
    assert first["status"] == "needs-auth"
    await asyncio.sleep(0.1)

    FakeClient.default_status = {
        "mcpServers": [{"name": "observability", "status": "connected", "tools": ["a"]}]
    }
    started = time.monotonic()
    stale = await authorizer.status()
    assert time.monotonic() - started < 0.5
    assert stale["status"] == "needs-auth"

    for _ in range(50):
        await asyncio.sleep(0.02)
        if authorizer._probe_cache and authorizer._probe_cache[1]["status"] == "connected":
            break
    assert authorizer._probe_cache[1]["status"] == "connected"
    assert len(FakeClient.instances) == 2
    await authorizer.close()


@pytest.mark.asyncio
async def test_clear_uses_cli_and_invalidates_cache(tmp_path):
    authorizer = make_authorizer(tmp_path, mcp_config())

    await authorizer.clear()

    client = FakeClient.instances[0]
    assert client.cleared is True
    assert client.connected is False


@pytest.mark.asyncio
async def test_missing_entry_raises_on_start_and_reports_disabled_on_status(tmp_path):
    authorizer = make_authorizer(tmp_path, mcp_config_without_obs())

    with pytest.raises(ObservabilityAuthError):
        await authorizer.start()
    status = await authorizer.status()
    assert status["status"] == "disabled"

    authorizer = make_authorizer(tmp_path, None)
    with pytest.raises(ObservabilityAuthError):
        await authorizer.start()
    assert (await authorizer.status())["status"] == "disabled"


def test_observability_entry_strips_static_headers(tmp_path):
    authorizer = make_authorizer(tmp_path, mcp_config(headers={"Authorization": "Bearer dead"}))

    entry = authorizer._observability_entry()

    assert entry == {"url": "https://obs/mcp", "type": "http"}
