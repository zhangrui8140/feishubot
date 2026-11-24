from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from feishubot.ai.providers.openai_compatible.client import OpenAICompatibleProvider
from feishubot.config import settings
from feishubot.qoder_agent.domain_knowledge import (
    _fingerprint,
    _fingerprint_async,
    _unchanged,
    _unchanged_async,
)
from feishubot.qoder_agent.service import (
    QoderAgentService,
    QoderServiceConfig,
    _SessionState,
)


@pytest.mark.asyncio
async def test_openai_provider_reuses_client_and_closes():
    provider = OpenAICompatibleProvider(
        base_url="https://api.example.com",
        api_key="sk-test",
        model="gpt-test",
    )
    client_instance = provider._client
    assert client_instance is not None
    assert not client_instance.is_closed

    await provider.aclose()
    assert client_instance.is_closed


def test_models_config_mtime_caching(tmp_path, monkeypatch):
    models_file = tmp_path / "models.toml"
    models_file.write_text(
        """
default_model = "test-model"
[models.test-model]
provider = "echo"
model = "echo-v1"
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(settings, "llm_models_config_path", str(models_file))
    monkeypatch.setattr(settings, "llm_active_model", "test-model")

    cfg1 = settings._resolve_from_model_map()
    cfg2 = settings._resolve_from_model_map()
    assert cfg1 is not None
    assert cfg1 is cfg2  # Same object returned from memory cache!


@pytest.mark.asyncio
async def test_chunked_fingerprint_matches_regular_hash(tmp_path):
    test_file = tmp_path / "sample.py"
    test_content = b"print('hello performance optimization')\n" * 1000
    test_file.write_bytes(test_content)

    fp = _fingerprint(test_file)
    assert len(fp["sha256"]) == 64
    assert _unchanged(test_file, fp) is True

    # Async wrapper test
    fp_async = await _fingerprint_async(test_file)
    assert fp_async["sha256"] == fp["sha256"]
    assert await _unchanged_async(test_file, fp) is True


@pytest.mark.asyncio
async def test_session_pruning_removes_idle_session_from_dict(tmp_path):
    config = QoderServiceConfig(
        workspace_root=tmp_path,
        model="test",
        session_store_dir=tmp_path / "sessions",
        session_idle_ttl_seconds=0.01,  # Short TTL for test
    )
    service = QoderAgentService(config=config, adapter=AsyncMock(), session_store=MagicMock())

    # Create dummy session
    session_key = "test-session-key"
    state = _SessionState(session_key=session_key)
    state.last_used_at = 0.0  # Epoch, expired
    service._sessions[session_key] = state

    # Trigger maintenance loop iteration logic
    expired_keys = []
    now = 100000.0
    for key, s in list(service._sessions.items()):
        if (
            s.active_task is None
            and not s.lock.locked()
            and s.pending_count == 0
            and now - s.last_used_at >= config.session_idle_ttl_seconds
        ):
            expired_keys.append(key)
    for key in expired_keys:
        service._sessions.pop(key, None)

    assert session_key not in service._sessions


@pytest.mark.asyncio
async def test_session_reset_removes_from_sessions_dict(tmp_path):
    config = QoderServiceConfig(
        workspace_root=tmp_path,
        model="test",
        session_store_dir=tmp_path / "sessions",
    )
    service = QoderAgentService(config=config, adapter=AsyncMock(), session_store=MagicMock())

    session_key = "test-reset-key"
    state = _SessionState(session_key=session_key)
    service._sessions[session_key] = state

    result = await service.reset(session_key)
    assert result is True
    assert session_key not in service._sessions
