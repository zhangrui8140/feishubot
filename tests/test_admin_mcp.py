from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from feishubot import admin
from feishubot.app import app
from feishubot.qoder_agent.mcp_oauth import ObservabilityAuthError

ORIGIN = {"Origin": "http://127.0.0.1:8000"}


def _write_mcp_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "aliyun-dms-mcp-server": {
                        "url": "https://dms.example.com/sse",
                        "type": "sse",
                        "headers": {"Authorization": "Bearer super-secret-dms-token"},
                    },
                    "observability": {
                        "url": "https://observability.example.com/mcp",
                        "type": "http",
                        "headers": {"Authorization": "Bearer super-secret-obs-token"},
                    },
                    "extra-server": {"command": "keep-me"},
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return config_path


def _client(host: str = "127.0.0.1") -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1:8000", client=(host, 50000))


def _dms_update(**overrides) -> dict:
    update = {
        "enabled": True,
        "url": "https://dms.example.com/sse",
        "token": {"action": "keep", "value": ""},
    }
    update.update(overrides)
    return update


def _obs_update(**overrides) -> dict:
    update = {
        "enabled": True,
        "url": "https://observability.example.com/mcp",
    }
    update.update(overrides)
    return update


def _payload(config: dict, **overrides) -> dict:
    servers = config["servers"]
    payload = {
        "revision": config["revision"],
        "dms": _dms_update(),
        "observability": _obs_update(enabled=servers["observability"]["enabled"]),
    }
    payload.update(overrides)
    return payload


class _StubAuthorizer:
    def __init__(self, *, start=None, status=None, clear=None, error=None) -> None:
        self.start_result = start or {
            "flowActive": True,
            "status": "pending",
            "authorized": False,
            "error": None,
            "tools": 0,
            "authUrl": "https://sso.example.com/authorize?client_id=x",
        }
        self.status_result = status or {
            "flowActive": False,
            "status": "connected",
            "authorized": True,
            "error": None,
            "tools": 11,
        }
        self.clear_result = clear
        self.error = error

    async def start(self) -> dict:
        if self.error:
            raise ObservabilityAuthError(self.error)
        return self.start_result

    async def status(self) -> dict:
        if self.error:
            raise ObservabilityAuthError(self.error)
        return self.status_result

    async def clear(self) -> None:
        if self.error:
            raise ObservabilityAuthError(self.error)


def test_get_mcp_config_reports_auth_mode_without_secret_values(tmp_path, monkeypatch):
    config_path = _write_mcp_config(tmp_path)
    monkeypatch.setattr(admin, "resolve_mcp_file", lambda: config_path)

    response = _client().get("/admin/api/mcp")

    assert response.status_code == 200
    body = response.text
    assert "super-secret-dms-token" not in body
    assert "super-secret-obs-token" not in body
    servers = response.json()["servers"]
    assert servers["aliyun-dms-mcp-server"]["enabled"] is True
    assert servers["aliyun-dms-mcp-server"]["url"] == "https://dms.example.com/sse"
    assert servers["aliyun-dms-mcp-server"]["token_configured"] is True
    assert servers["aliyun-dms-mcp-server"]["auth_mode"] == "static_token"
    assert servers["observability"]["auth_mode"] == "oauth"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"


def test_get_mcp_config_reports_unconfigured_path(monkeypatch):
    monkeypatch.setattr(admin, "resolve_mcp_file", lambda: None)

    response = _client().get("/admin/api/mcp")

    assert response.status_code == 200
    body = response.json()
    assert body["servers"] is None
    assert body["config_path"] == ""


def test_put_mcp_config_writes_oauth_entry_without_headers(tmp_path, monkeypatch):
    config_path = _write_mcp_config(tmp_path)
    monkeypatch.setattr(admin, "resolve_mcp_file", lambda: config_path)
    client = _client()
    config = client.get("/admin/api/mcp").json()
    original_content = config_path.read_text(encoding="utf-8")

    payload = _payload(config, dms=_dms_update(url="https://new-dms.example.com/sse"))
    response = client.put("/admin/api/mcp", headers=ORIGIN, json=payload)

    assert response.status_code == 200
    assert response.json()["effective_immediately"] is True
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    dms = saved["mcpServers"]["aliyun-dms-mcp-server"]
    assert dms["url"] == "https://new-dms.example.com/sse"
    assert dms["type"] == "sse"
    assert dms["headers"]["Authorization"] == "Bearer super-secret-dms-token"  # noqa: S105
    obs = saved["mcpServers"]["observability"]
    assert obs == {"url": "https://observability.example.com/mcp", "type": "http"}
    assert "headers" not in obs
    assert saved["mcpServers"]["extra-server"] == {"command": "keep-me"}
    assert config_path.read_text(encoding="utf-8").endswith("\n")
    backup = config_path.with_name("mcp.json.bak-admin")
    assert backup.read_text(encoding="utf-8") == original_content


def test_put_mcp_config_requires_same_origin_and_current_revision(tmp_path, monkeypatch):
    config_path = _write_mcp_config(tmp_path)
    monkeypatch.setattr(admin, "resolve_mcp_file", lambda: config_path)
    client = _client()
    config = client.get("/admin/api/mcp").json()
    payload = _payload(config)

    assert client.put("/admin/api/mcp", json=payload).status_code == 403
    assert (
        client.put(
            "/admin/api/mcp",
            headers={"Origin": "http://localhost:8000"},
            json=payload,
        ).status_code
        == 403
    )
    assert (
        _client("192.168.1.20").put("/admin/api/mcp", headers=ORIGIN, json=payload).status_code
        == 403
    )

    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert client.put("/admin/api/mcp", headers=ORIGIN, json=payload).status_code == 409


def test_put_mcp_config_rejects_invalid_updates(tmp_path, monkeypatch):
    config_path = _write_mcp_config(tmp_path)
    monkeypatch.setattr(admin, "resolve_mcp_file", lambda: config_path)
    client = _client()
    config = client.get("/admin/api/mcp").json()

    cases = [
        _payload(config, dms=_dms_update(enabled=False)),
        _payload(config, dms=_dms_update(url=" ")),
        _payload(config, dms=_dms_update(url="ftp://bad.example.com")),
        _payload(
            config,
            dms=_dms_update(token={"action": "clear", "value": ""}),
        ),
        _payload(
            config,
            dms=_dms_update(token={"action": "replace", "value": " "}),
        ),
        _payload(config, observability=_obs_update(url=" ")),
        _payload(config, observability=_obs_update(url="ftp://bad.example.com")),
        # observability no longer accepts a token: extra field is rejected
        _payload(config, observability=_obs_update(token={"action": "keep", "value": ""})),
    ]
    for payload in cases:
        response = client.put("/admin/api/mcp", headers=ORIGIN, json=payload)
        assert response.status_code == 422, response.text

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["mcpServers"]["aliyun-dms-mcp-server"]["url"] == "https://dms.example.com/sse"
    assert not (tmp_path / "mcp.json.bak-admin").exists()


def test_put_mcp_config_disables_and_reenables_observability(tmp_path, monkeypatch):
    config_path = _write_mcp_config(tmp_path)
    monkeypatch.setattr(admin, "resolve_mcp_file", lambda: config_path)
    client = _client()

    config = client.get("/admin/api/mcp").json()
    payload = _payload(config, observability=_obs_update(enabled=False))
    assert client.put("/admin/api/mcp", headers=ORIGIN, json=payload).status_code == 200
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert "observability" not in saved["mcpServers"]

    config = client.get("/admin/api/mcp").json()
    assert config["servers"]["observability"]["enabled"] is False

    response = client.put(
        "/admin/api/mcp",
        headers=ORIGIN,
        json=_payload(config, observability=_obs_update()),
    )
    assert response.status_code == 200
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    obs = saved["mcpServers"]["observability"]
    assert obs == {"url": "https://observability.example.com/mcp", "type": "http"}
    assert "headers" not in obs


def test_put_mcp_config_repairs_broken_file_and_dms_keeps_token(tmp_path, monkeypatch):
    config_path = tmp_path / "mcp.json"
    config_path.write_text("{broken json", encoding="utf-8")
    monkeypatch.setattr(admin, "resolve_mcp_file", lambda: config_path)
    client = _client()

    config = client.get("/admin/api/mcp").json()
    assert config["parse_error"]
    assert config["servers"]["aliyun-dms-mcp-server"]["enabled"] is False

    payload = _payload(
        config,
        dms=_dms_update(token={"action": "replace", "value": "raw-dms-token"}),
        observability=_obs_update(),
    )
    response = client.put("/admin/api/mcp", headers=ORIGIN, json=payload)

    assert response.status_code == 200
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    dms = saved["mcpServers"]["aliyun-dms-mcp-server"]
    assert dms["headers"]["Authorization"] == "Bearer raw-dms-token"
    assert dms["type"] == "sse"
    obs = saved["mcpServers"]["observability"]
    assert obs == {"url": "https://observability.example.com/mcp", "type": "http"}


def test_put_mcp_config_without_configured_path_returns_503(monkeypatch):
    monkeypatch.setattr(admin, "resolve_mcp_file", lambda: None)
    payload = {
        "revision": "a" * 64,
        "dms": _dms_update(token={"action": "replace", "value": "dms-token"}),
        "observability": _obs_update(enabled=False),
    }

    response = _client().put("/admin/api/mcp", headers=ORIGIN, json=payload)

    assert response.status_code == 503


def test_authorize_observability_requires_same_origin(monkeypatch):
    monkeypatch.setattr(admin, "_get_authorizer", lambda: _StubAuthorizer())

    client = _client()
    assert client.post("/admin/api/mcp/observability/authorize").status_code == 403
    assert client.post("/admin/api/mcp/observability/clear").status_code == 403


def test_authorize_observability_returns_auth_url(monkeypatch):
    monkeypatch.setattr(admin, "_get_authorizer", lambda: _StubAuthorizer())

    response = _client().post("/admin/api/mcp/observability/authorize", headers=ORIGIN)

    assert response.status_code == 200
    body = response.json()
    assert body["flowActive"] is True
    assert body["authUrl"].startswith("https://sso.example.com/authorize")


def test_authorize_observability_maps_cli_failure_to_502(monkeypatch):
    monkeypatch.setattr(
        admin,
        "_get_authorizer",
        lambda: _StubAuthorizer(error="Could not start authorization: boom"),
    )

    response = _client().post("/admin/api/mcp/observability/authorize", headers=ORIGIN)

    assert response.status_code == 502
    assert "boom" in response.json()["detail"]


def test_observability_status_reports_connection_state(monkeypatch):
    monkeypatch.setattr(admin, "_get_authorizer", lambda: _StubAuthorizer())

    response = _client().get("/admin/api/mcp/observability/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "connected"
    assert body["authorized"] is True


def test_clear_observability_authorization(monkeypatch):
    monkeypatch.setattr(admin, "_get_authorizer", lambda: _StubAuthorizer())

    response = _client().post("/admin/api/mcp/observability/clear", headers=ORIGIN)

    assert response.status_code == 200
    assert response.json() == {"cleared": True}
