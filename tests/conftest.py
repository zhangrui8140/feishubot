from __future__ import annotations

import pytest

from feishubot.config import settings


@pytest.fixture(autouse=True)
def _isolate_runtime_dirs(tmp_path, monkeypatch):
    # Keep tests away from the live bot's state under ~/.feishubot.
    for name in (
        "qoder_session_store_dir",
        "qoder_domain_knowledge_dir",
        "qoder_handoff_dir",
        "qoder_diagnosis_store_dir",
    ):
        monkeypatch.setattr(settings, name, str(tmp_path / "runtime" / name))
