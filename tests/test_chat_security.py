from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from feishubot.app import app
from feishubot.config import settings


@pytest.fixture
def client():
    return TestClient(app)


def test_chat_endpoint_disallows_get(client):
    response = client.get("/api/chat?message=hello")
    # GET 方法应该被拒绝 (405 Method Not Allowed)
    assert response.status_code == 405


def test_chat_endpoint_requires_auth_outside_dev(client, monkeypatch):
    monkeypatch.setattr(settings, "app_env", "prod")
    monkeypatch.setattr(settings, "gateway_internal_api_key", "secret-token-123")

    # 未带 key 访问
    res_no_key = client.post("/api/chat", json={"message": "ping"})
    assert res_no_key.status_code == 401

    # 带错误 key 访问
    res_wrong_key = client.post(
        "/api/chat",
        json={"message": "ping"},
        headers={"x-api-key": "wrong-token"},
    )
    assert res_wrong_key.status_code == 401
