from __future__ import annotations

from feishubot.qoder_agent.session_store import QoderSessionStore


def test_session_store_persists_only_session_mapping(tmp_path):
    store = QoderSessionStore(tmp_path)

    store.set("feishu:chat:root:user", "session-1")

    assert store.get("feishu:chat:root:user") == "session-1"
    content = (tmp_path / "sessions.json").read_text(encoding="utf-8")
    assert "session-1" in content
    assert "prompt" not in content
    assert "token" not in content


def test_session_store_deletes_mapping(tmp_path):
    store = QoderSessionStore(tmp_path)
    store.set("key", "session-1")

    store.delete("key")

    assert store.get("key") is None
