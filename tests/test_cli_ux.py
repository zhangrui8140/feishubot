from __future__ import annotations

import getpass

import pytest

from feishubot.cli import _prompt_secret


def test_prompt_secret_uses_getpass(monkeypatch):
    called = []

    def mock_getpass(prompt=""):
        called.append(prompt)
        return "my-secret-key"

    monkeypatch.setattr(getpass, "getpass", mock_getpass)

    res = _prompt_secret("API_KEY")
    assert res == "my-secret-key"
    assert len(called) == 1
    assert "API_KEY" in called[0]


def test_prompt_secret_fallback_to_default(monkeypatch):
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "")
    res = _prompt_secret("API_KEY", default="default-key")
    assert res == "default-key"
