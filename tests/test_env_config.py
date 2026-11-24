from __future__ import annotations

import os
from pathlib import Path

import pytest

from feishubot.config import Settings
from feishubot.env_config import (
    ConfigConflictError,
    ConfigValidationError,
    env_revision,
    load_env_values,
    read_managed_config,
    save_managed_config,
    write_env_values,
)


def _write_env(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# retained comment\n"
        "UNKNOWN_SETTING=keep-me\n"
        "LOG_LEVEL=INFO\n"
        "FEISHU_APP_ID=cli_test\n"
        "FEISHU_APP_SECRET=original-secret\n"
        "FEISHU_VERIFICATION_TOKEN=verify-secret\n"
        "FEISHU_ENCRYPT_KEY=encrypt-secret\n"
        "FEISHU_AGENT_BACKEND=qoder_sdk\n"
        "QODER_PERSONAL_ACCESS_TOKEN=pat-secret\n"
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
        "QODER_DEDUP_TTL_SECONDS=86400\n"
        "WINDOWS_PATH=D:\\workspace\\erp\n",
        encoding="utf-8",
    )
    return env_path, workspace


def test_write_preserves_comments_unknown_keys_order_and_windows_paths(tmp_path):
    env_path, _ = _write_env(tmp_path)

    write_env_values(env_path, {"LOG_LEVEL": "DEBUG", "QODER_MODEL": "auto"})

    content = env_path.read_text(encoding="utf-8")
    assert content.startswith("# retained comment\nUNKNOWN_SETTING=keep-me\nLOG_LEVEL=DEBUG\n")
    assert "WINDOWS_PATH=D:\\workspace\\erp\n" in content
    assert content.index("UNKNOWN_SETTING") < content.index("LOG_LEVEL")


def test_atomic_write_failure_keeps_original_and_removes_temp_file(tmp_path, monkeypatch):
    env_path, _ = _write_env(tmp_path)
    original = env_path.read_bytes()

    def fail_replace(source, destination):  # noqa: ARG001
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_env_values(env_path, {"LOG_LEVEL": "DEBUG"})

    assert env_path.read_bytes() == original
    assert list(tmp_path.glob("..env.*.tmp")) == []


def test_revision_conflict_does_not_overwrite_newer_content(tmp_path):
    env_path, _ = _write_env(tmp_path)
    stale_revision = env_revision(env_path)
    changed_content = env_path.read_text(encoding="utf-8") + "EXTERNAL=change\n"
    env_path.write_text(changed_content, encoding="utf-8")

    with pytest.raises(ConfigConflictError):
        save_managed_config(
            env_path,
            expected_revision=stale_revision,
            values={"QODER_MODEL": "new-model"},
            secret_updates={},
        )

    assert "EXTERNAL=change" in env_path.read_text(encoding="utf-8")
    assert "QODER_MODEL=auto" in env_path.read_text(encoding="utf-8")


def test_secret_keep_replace_and_clear(tmp_path):
    env_path, _ = _write_env(tmp_path)

    first = save_managed_config(
        env_path,
        expected_revision=env_revision(env_path),
        values={},
        secret_updates={"FEISHU_APP_SECRET": {"action": "keep", "value": ""}},
    )
    assert load_env_values(env_path)["FEISHU_APP_SECRET"].startswith("original-")

    replacement = "new-" + "secret"
    second = save_managed_config(
        env_path,
        expected_revision=first.revision,
        values={},
        secret_updates={"FEISHU_APP_SECRET": {"action": "replace", "value": replacement}},
    )
    assert load_env_values(env_path)["FEISHU_APP_SECRET"] == replacement

    final = save_managed_config(
        env_path,
        expected_revision=second.revision,
        values={},
        secret_updates={"FEISHU_APP_SECRET": {"action": "clear", "value": ""}},
    )
    assert load_env_values(env_path)["FEISHU_APP_SECRET"] == ""
    assert final.secrets["FEISHU_APP_SECRET"] is False


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"UNKNOWN": "value"}, "Unknown configuration fields"),
        ({"LOG_LEVEL": "VERBOSE"}, "LOG_LEVEL is invalid"),
        ({"QODER_MAX_CONCURRENT_RUNS": "0"}, "must be between"),
        ({"QODER_MODEL": "bad\nmodel"}, "control characters"),
        ({"QODER_WORKSPACE_ROOT": "Z:/missing/workspace"}, "existing directory"),
    ],
)
def test_invalid_managed_values_are_rejected(tmp_path, values, message):
    env_path, _ = _write_env(tmp_path)

    with pytest.raises(ConfigValidationError, match=message):
        save_managed_config(
            env_path,
            expected_revision=env_revision(env_path),
            values=values,
            secret_updates={},
        )


def test_saved_values_load_through_settings(tmp_path):
    env_path, workspace = _write_env(tmp_path)

    saved = save_managed_config(
        env_path,
        expected_revision=env_revision(env_path),
        values={
            "QODER_WORKSPACE_ROOT": str(workspace),
            "QODER_MODEL": "analysis-model",
            "QODER_MAX_CONCURRENT_RUNS": 6,
        },
        secret_updates={},
    )

    loaded = Settings(_env_file=env_path)
    assert loaded.qoder_model == "analysis-model"
    assert loaded.qoder_max_concurrent_runs == 6
    assert read_managed_config(env_path).revision == saved.revision
