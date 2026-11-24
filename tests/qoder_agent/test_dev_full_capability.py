from __future__ import annotations

import json
from pathlib import Path

import pytest

from feishubot.ai.orchestrator.chat_commands import _parse_model_command, handle_command
from feishubot.qoder_agent import sdk_adapter
from feishubot.qoder_agent.sdk_adapter import QoderSdkAdapter
from feishubot.qoder_agent.service import QoderAgentService, QoderServiceConfig
from feishubot.qoder_agent.tool_policy import DevToolPolicy


@pytest.fixture()
def policy(tmp_path: Path) -> DevToolPolicy:
    return DevToolPolicy(
        tmp_path,
        ("aliyun-dms-mcp-server", "mysql-erp-dev", "mysql-scm-dev", "bi", "claude-mem"),
    )


def test_agent_tools_allowed(policy: DevToolPolicy) -> None:
    for tool in ("Skill", "Task", "TodoWrite", "WebSearch", "WebFetch", "Agent"):
        assert policy.denial_reason(tool, {}) is None


def test_unknown_mcp_tool_denied(policy: DevToolPolicy) -> None:
    assert policy.denial_reason("mcp__evil-server__drop_table", {}) is not None


def test_mysql_execute_sql_readonly(policy: DevToolPolicy) -> None:
    assert policy.denial_reason("mcp__mysql-erp-dev__execute_sql", {"query": "SELECT 1"}) is None
    assert (
        policy.denial_reason("mcp__mysql-scm-dev__execute_sql", {"query": "UPDATE t SET a=1"})
        is not None
    )
    assert policy.denial_reason("mcp__bi__execute_sql", {"query": "DELETE FROM t"}) is not None


def test_mysql_meta_tools_allowed(policy: DevToolPolicy) -> None:
    assert policy.denial_reason("mcp__mysql-erp-dev__get_schema_info", {"table_name": "t"}) is None
    assert policy.denial_reason("mcp__bi__get_table_sample", {"table_name": "t"}) is None


def test_claude_mem_allowed(policy: DevToolPolicy) -> None:
    assert policy.denial_reason("mcp__claude-mem__save_memory", {"content": "x"}) is None
    assert policy.denial_reason("mcp__claude-mem__search", {"query": "x"}) is None


def test_read_user_rules_and_skills_allowed(policy: DevToolPolicy) -> None:
    rules = Path.home() / ".qoder" / "rules" / "coding-standard.md"
    assert policy.denial_reason("Read", {"file_path": str(rules)}) is None
    skills = Path.home() / ".qoder" / "skills" / "tdd" / "SKILL.md"
    assert policy.denial_reason("Read", {"file_path": str(skills)}) is None


def test_write_user_rules_denied(policy: DevToolPolicy) -> None:
    rules = Path.home() / ".qoder" / "rules" / "coding-standard.md"
    assert policy.denial_reason("Write", {"file_path": str(rules)}) is not None


def test_read_outside_workspace_still_denied(policy: DevToolPolicy) -> None:
    assert (
        policy.denial_reason("Read", {"file_path": "C:/Windows/system32/config/sam"}) is not None
    )


def test_dms_still_readonly(policy: DevToolPolicy) -> None:
    assert (
        policy.denial_reason(
            "mcp__aliyun-dms-mcp-server__executeScript",
            {"database_id": "64839638", "script": "SELECT 1"},
        )
        is None
    )
    assert (
        policy.denial_reason(
            "mcp__aliyun-dms-mcp-server__executeScript",
            {"database_id": "64839638", "script": "DROP TABLE t"},
        )
        is not None
    )


def test_parse_model_command() -> None:
    assert _parse_model_command("模型") == (True, "")
    assert _parse_model_command("模型 GLM-5.3") == (True, "GLM-5.3")
    assert _parse_model_command("模型：GLM-5.3") == (True, "GLM-5.3")
    assert _parse_model_command("/model GLM-5.3") == (True, "GLM-5.3")
    assert _parse_model_command("model the data flow") == (False, "")
    assert _parse_model_command("修复模型校验问题") == (False, "")


class FakeModelService:
    mode = "dev"

    def __init__(self) -> None:
        self.model = "GLM-5.3-Flash"
        self.switched: list[str] = []

    async def set_model(self, model: str) -> str:
        self.switched.append(model)
        self.model = model
        return model


@pytest.mark.asyncio
async def test_handle_model_command() -> None:
    service = FakeModelService()
    reply = await handle_command(service, "k", "模型")
    assert reply is not None and "GLM-5.3-Flash" in reply
    reply = await handle_command(service, "k", "模型 GLM-5.3")
    assert service.switched == ["GLM-5.3"]
    assert reply is not None and "GLM-5.3" in reply


class FakeAdapter:
    observability_enabled = False

    def __init__(self) -> None:
        self.model = "m1"

    def set_model(self, model: str) -> None:
        self.model = model

    def validate(self) -> None:
        pass

    async def open_session(self, resume_session_id=None):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_service_set_model_persists(tmp_path: Path) -> None:
    config = QoderServiceConfig(
        workspace_root=tmp_path,
        model="m1",
        session_store_dir=tmp_path / "store",
        mode="dev",
    )
    service = QoderAgentService(config, adapter=FakeAdapter())
    await service.set_model("m2")
    assert (tmp_path / "store" / "model.json").exists()
    service2 = QoderAgentService(config, adapter=FakeAdapter())
    assert service2.model == "m2"


@pytest.mark.asyncio
async def test_dev_adapter_full_capability_options(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "scm-dev").mkdir()
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "aliyun-dms-mcp-server": {"type": "sse", "url": "https://dms.example/sse"}
                }
            }
        ),
        encoding="utf-8",
    )
    user_home = tmp_path / "home"
    (user_home / ".qoder").mkdir(parents=True)
    (user_home / ".qoder" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "mysql-erp-dev": {"command": "uvx", "args": ["mysql_mcp_server"]},
                    "claude-mem": {"command": "node", "args": ["mem.cjs"]},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: user_home)
    adapter = QoderSdkAdapter(
        workspace_root=workspace,
        model="m",
        mcp_servers_config_path=mcp_config,
        mode="dev",
    )
    captured = {}

    class CapturingClient:
        def __init__(self, *, options):
            captured["options"] = options

    class FakeHandle:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            pass

        async def disconnect(self):
            pass

    monkeypatch.setattr(sdk_adapter, "QoderSDKClient", CapturingClient)
    monkeypatch.setattr(sdk_adapter, "QoderSessionHandle", FakeHandle)
    monkeypatch.setattr(QoderSdkAdapter, "_auth", lambda self: object())

    await adapter.open_session()
    options = captured["options"]
    assert options.setting_sources == ["user"]
    assert options.strict_mcp_config is True
    assert options.tools is None
    assert "mysql-erp-dev" in options.allowed_mcp_server_names
    assert "claude-mem" in options.allowed_mcp_server_names
    assert isinstance(options.mcp_servers, dict)
    assert "claude-mem" in options.mcp_servers
    assert "mysql-erp-dev" in options.mcp_servers
    assert "用户级规则库" in options.system_prompt
    assert "mysql-erp-dev" in adapter._policy.mcp_server_names
