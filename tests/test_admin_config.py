from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from feishubot import admin
from feishubot.app import app
from feishubot.config import settings


def _write_env(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# keep this comment\n"
        "UNKNOWN_SETTING=keep-me\n"
        "LOG_LEVEL=INFO\n"
        "FEISHU_APP_ID=cli_test\n"
        "FEISHU_APP_SECRET=super-secret-app-value\n"
        "FEISHU_VERIFICATION_TOKEN=super-secret-verification-value\n"
        "FEISHU_ENCRYPT_KEY=super-secret-encrypt-value\n"
        "FEISHU_AGENT_BACKEND=qoder_sdk\n"
        "QODER_PERSONAL_ACCESS_TOKEN=super-secret-pat-value\n"
        f"QODER_WORKSPACE_ROOT={workspace}\n"
        "QODER_MODEL=auto\n"
        f"QODER_SESSION_STORE_DIR={tmp_path / 'sessions'}\n"
        "QODER_RUN_TIMEOUT_SECONDS=600\n"
        "QODER_CANCEL_GRACE_SECONDS=5\n"
        "QODER_SHUTDOWN_GRACE_SECONDS=30\n"
        "QODER_MAX_CONCURRENT_RUNS=2\n"
        "QODER_MAX_LIVE_SESSIONS=4\n"
        "QODER_MAX_QUEUED_PER_SESSION=3\n"
        "QODER_SESSION_IDLE_TTL_SECONDS=1800\n"
        "QODER_DEDUP_TTL_SECONDS=86400\n",
        encoding="utf-8",
    )
    return env_path


def _client(host: str = "127.0.0.1") -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1:8000", client=(host, 50000))


def _payload(config: dict) -> dict:
    return {
        "revision": config["revision"],
        "values": dict(config["values"]),
        "secrets": {
            key: {"action": "keep", "value": ""}
            for key in config["secrets"]
        },
    }


def test_admin_routes_allow_loopback_and_reject_non_loopback(tmp_path, monkeypatch):
    env_path = _write_env(tmp_path)
    monkeypatch.setattr(admin, "resolve_env_file", lambda: env_path)

    assert _client().get("/admin/api/status").status_code == 200
    assert _client("192.168.1.20").get("/admin/api/status").status_code == 403
    assert _client("192.168.1.20").get("/admin/config").status_code == 403


def test_config_response_never_contains_secret_values(tmp_path, monkeypatch):
    env_path = _write_env(tmp_path)
    monkeypatch.setattr(admin, "resolve_env_file", lambda: env_path)

    response = _client().get("/admin/api/config")

    assert response.status_code == 200
    body = response.text
    assert "super-secret-app-value" not in body
    assert "super-secret-verification-value" not in body
    assert "super-secret-encrypt-value" not in body
    assert "super-secret-pat-value" not in body
    assert all(item["configured"] for item in response.json()["secrets"].values())


def test_put_saves_file_without_mutating_runtime_settings(tmp_path, monkeypatch):
    env_path = _write_env(tmp_path)
    monkeypatch.setattr(admin, "resolve_env_file", lambda: env_path)
    client = _client()
    config = client.get("/admin/api/config").json()
    payload = _payload(config)
    payload["values"]["QODER_MODEL"] = "saved-model"
    payload["values"]["QODER_MAX_CONCURRENT_RUNS"] = 7
    payload["secrets"]["QODER_PERSONAL_ACCESS_TOKEN"] = {
        "action": "replace",
        "value": "replacement-pat",
    }
    runtime_model = settings.qoder_model

    response = client.put(
        "/admin/api/config",
        headers={"Origin": "http://127.0.0.1:8000"},
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["restart_required"] is True
    content = env_path.read_text(encoding="utf-8")
    assert "# keep this comment" in content
    assert "UNKNOWN_SETTING=keep-me" in content
    assert "QODER_MODEL=saved-model" in content
    assert "QODER_MAX_CONCURRENT_RUNS=7" in content
    assert "QODER_PERSONAL_ACCESS_TOKEN=replacement-pat" in content
    assert settings.qoder_model == runtime_model


def test_put_requires_same_origin_and_current_revision(tmp_path, monkeypatch):
    env_path = _write_env(tmp_path)
    monkeypatch.setattr(admin, "resolve_env_file", lambda: env_path)
    client = _client()
    config = client.get("/admin/api/config").json()
    payload = _payload(config)

    assert client.put("/admin/api/config", json=payload).status_code == 403
    assert client.put(
        "/admin/api/config",
        headers={"Origin": "http://localhost:8000"},
        json=payload,
    ).status_code == 403

    changed_content = env_path.read_text(encoding="utf-8") + "EXTERNAL=change\n"
    env_path.write_text(changed_content, encoding="utf-8")
    response = client.put(
        "/admin/api/config",
        headers={"Origin": "http://127.0.0.1:8000"},
        json=payload,
    )
    assert response.status_code == 409


def test_put_rejects_unknown_fields(tmp_path, monkeypatch):
    env_path = _write_env(tmp_path)
    monkeypatch.setattr(admin, "resolve_env_file", lambda: env_path)
    client = _client()
    config = client.get("/admin/api/config").json()
    payload = _payload(config)
    payload["values"]["UNMANAGED_VALUE"] = "not-allowed"

    response = client.put(
        "/admin/api/config",
        headers={"Origin": "http://127.0.0.1:8000"},
        json=payload,
    )

    assert response.status_code == 422
    assert "UNMANAGED_VALUE" not in env_path.read_text(encoding="utf-8")


def test_admin_responses_include_security_headers(tmp_path, monkeypatch):
    env_path = _write_env(tmp_path)
    monkeypatch.setattr(admin, "resolve_env_file", lambda: env_path)

    for path in ("/admin/config", "/admin/api/config", "/admin/api/status"):
        response = _client().get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_business_api_key_authentication_is_unchanged():
    original_key = settings.gateway_internal_api_key
    settings.gateway_internal_api_key = "required-key"
    try:
        response = _client().post(
            "/api/feishu/push",
            headers={"x-api-key": "wrong-key"},
            json={"receive_id": "user", "text": "hello"},
        )
    finally:
        settings.gateway_internal_api_key = original_key

    assert response.status_code == 401
