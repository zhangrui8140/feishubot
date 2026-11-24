from __future__ import annotations

from pathlib import Path

import pytest

from feishubot.qoder_agent.tool_policy import (
    DMS_SERVER,
    OBSERVABILITY_SERVER,
    ToolPolicy,
)

START = "2026-10-13T09:00:00+08:00"
END = "2026-10-13T10:00:00+08:00"
LOG_QUERY = '"hxl.erp.storeorder.checkandadjust" AND ERROR'


def obs(name: str) -> str:
    return f"mcp__observability__{name}"


def dms(name: str) -> str:
    return f"mcp__aliyun-dms-mcp-server__{name}"


@pytest.fixture
def policy(tmp_path) -> ToolPolicy:
    return ToolPolicy(tmp_path, (DMS_SERVER, OBSERVABILITY_SERVER))


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("victorialogs_query", {"query": LOG_QUERY, "start": START, "end": END, "limit": 20}),
        ("victorialogs_stats", {"query": LOG_QUERY, "start": START, "end": END}),
        ("victoriatraces_query", {"query": "*", "start": START, "end": END, "limit": 200}),
        ("promql_query", {"query": "up"}),
        ("promql_query", {"query": "up", "time": "2026-10-13T02:00:00Z"}),
        ("promql_range", {"query": "up", "start": START, "end": END, "step": "5m"}),
        ("list_skills", {}),
        ("read_skill", {"name": "victorialogs"}),
        ("read_skill_reference", {"name": "victorialogs", "ref": "logsql.md"}),
        ("read_skill_document", {"name": "victorialogs", "path": "docs/query.md"}),
        ("search_skills", {"query": "logsql"}),
        ("search_skills", {"query": "logsql", "max_results": 5}),
    ],
)
def test_allows_valid_observability_calls(policy, tool, tool_input):
    assert policy.denial_reason(obs(tool), tool_input) is None


def log_query(**overrides):
    tool_input = {"query": LOG_QUERY, "start": START, "end": END, "limit": 20}
    tool_input.update(overrides)
    return {key: value for key, value in tool_input.items() if value is not None}


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("health", {}),
        ("flags", {}),
        ("victorialogs_delete", {}),
        ("VictoriaLogs_Query", log_query()),
        ("victorialogs_query", "not-an-object"),
        ("victorialogs_query", log_query(tenant="1:0")),
        ("victorialogs_query", log_query(start=None)),
        ("victorialogs_query", log_query(start="2026-10-13 09:00:00")),
        ("victorialogs_query", log_query(start="2026-10-13T09:00:00")),
        ("victorialogs_query", log_query(end="2026-10-14T09:00:01+08:00")),
        ("victorialogs_query", log_query(end="2026-10-13T08:59:59+08:00")),
        ("victorialogs_query", log_query(limit=None)),
        ("victorialogs_query", log_query(limit=0)),
        ("victorialogs_query", log_query(limit=201)),
        ("victorialogs_query", log_query(limit=True)),
        ("victorialogs_query", log_query(limit="20")),
        ("victorialogs_query", log_query(query="")),
        ("victorialogs_query", log_query(query="   ")),
        ("victorialogs_query", log_query(query="x" * 2001)),
        ("victorialogs_query", log_query(query="ERROR\x00")),
        ("promql_query", {"query": "up", "time": "now"}),
        ("promql_range", {"query": "up", "start": START, "end": END, "step": "0s"}),
        ("promql_range", {"query": "up", "start": START, "end": END, "step": "5 minutes"}),
        ("search_skills", {"query": "logsql", "max_results": 21}),
        ("read_skill", {"name": "../secrets"}),
        ("read_skill_document", {"name": "victorialogs", "path": "docs/../../etc"}),
        ("read_skill_document", {"name": "victorialogs", "path": "/etc/passwd"}),
    ],
)
def test_rejects_unsafe_observability_calls(policy, tool, tool_input):
    assert policy.denial_reason(obs(tool), tool_input) is not None


def test_missing_limit_hints_the_agent(policy):
    reason = policy.denial_reason(obs("victorialogs_query"), log_query(limit=None))

    assert reason is not None
    assert "limit <= 200" in reason


def test_rejects_disguised_server_prefix(policy):
    assert policy.denial_reason("mcp__observability_victorialogs_query", log_query()) is not None
    assert policy.denial_reason("mcp__Observability__victorialogs_query", log_query()) is not None


def test_mcp_call_applies_the_same_validation(policy):
    for tool_input in (log_query(), log_query(limit=None), log_query(tenant="1:0")):
        direct = policy.denial_reason(obs("victorialogs_query"), tool_input)
        nested = policy.denial_reason(
            "mcp_call",
            {"toolName": obs("victorialogs_query"), "arguments": tool_input},
        )
        assert (direct is None) == (nested is None)

    assert policy.denial_reason("mcp_get", {"toolName": obs("victorialogs_query")}) is None
    assert policy.denial_reason("mcp_get", {"toolName": obs("health")}) is not None


def test_observability_tools_are_denied_when_server_disabled(tmp_path):
    policy = ToolPolicy(tmp_path)

    assert policy.observability_enabled is False
    assert policy.mcp_server_names == (DMS_SERVER,)
    assert policy.denial_reason(obs("victorialogs_query"), log_query()) is not None
    assert (
        policy.denial_reason(
            "mcp_call",
            {"toolName": obs("victorialogs_query"), "arguments": log_query()},
        )
        is not None
    )


def test_enabled_policy_lists_observability_tools(policy):
    assert policy.observability_enabled is True
    assert policy.mcp_server_names == (DMS_SERVER, OBSERVABILITY_SERVER)
    observability_tools = {
        name for name in policy.allowed_tools if name.startswith("mcp__observability__")
    }
    assert len(observability_tools) == 10
    assert obs("health") not in observability_tools


def test_enabled_policy_lists_dms_read_only_tools(policy):
    dms_tools = {
        name
        for name in policy.allowed_tools
        if name.startswith("mcp__aliyun-dms-mcp-server__")
    }
    assert dms_tools == {
        dms("listTables"),
        dms("getTableDetailInfo"),
        dms("executeScript"),
        dms("answerSqlSyntax"),
        dms("fixSql"),
        dms("optimizeSql"),
    }
    # write / order / server-generated-SQL tools are never whitelisted
    for blocked in (
        "createDataChangeOrder",
        "submitOrderApproval",
        "approveOrder",
        "getOrderInfo",
        "askDatabase",
    ):
        assert dms(blocked) not in policy.allowed_tools


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        ("executeScript", {"database_id": "64839638", "script": "SELECT * FROM store_order WHERE fid = 'PR1'"}),
        ("executeScript", {"database_id": "64839638", "script": "SHOW TABLES"}),
        ("executeScript", {"database_id": "64839638", "script": "DESCRIBE store_order"}),
        ("executeScript", {"database_id": "64839638", "script": "WITH t AS (SELECT 1) SELECT * FROM t"}),
        ("listTables", {"database_id": "64839638"}),
        ("listTables", {"database_id": "64839638", "search_name": "store", "page_number": 1, "page_size": 50}),
        ("listTables", {"database_id": "64839638", "search_name": None}),
        ("getTableDetailInfo", {"table_guid": "123.erp_read.store_order"}),
        ("answerSqlSyntax", {"database_id": "64839638", "question": "how to group by day"}),
        ("answerSqlSyntax", {"database_id": "64839638", "question": "x", "model": "qwen"}),
        ("fixSql", {"database_id": "64839638", "sql": "SELECT 1", "error": "syntax error"}),
        ("optimizeSql", {"database_id": "64839638", "sql": "SELECT * FROM t"}),
    ],
)
def test_allows_valid_dms_calls(policy, tool, tool_input):
    assert policy.denial_reason(dms(tool), tool_input) is None


@pytest.mark.parametrize(
    ("tool", "tool_input"),
    [
        # write / order / askDatabase tools are not in the read-only whitelist
        ("createDataChangeOrder", {"script": "UPDATE store_order SET state = 1"}),
        ("submitOrderApproval", {"order_id": "1"}),
        ("approveOrder", {"workflow_instance_id": 1, "approval_type": "AGREE"}),
        ("getOrderInfo", {"order_id": "1"}),
        ("askDatabase", {"question": "how many orders"}),
        # database_id is mandatory and must be a positive numeric string
        ("executeScript", {"script": "SELECT 1"}),
        ("executeScript", {"database_id": "", "script": "SELECT 1"}),
        ("executeScript", {"database_id": "abc", "script": "SELECT 1"}),
        ("executeScript", {"database_id": 64839638, "script": "SELECT 1"}),
        ("listTables", {}),
        ("listTables", {"database_id": None}),
        ("answerSqlSyntax", {"question": "x"}),
        ("optimizeSql", {"sql": "SELECT 1"}),
        # executeScript must be a single read-only statement
        ("executeScript", {"database_id": "64839638", "script": "DELETE FROM store_order"}),
        ("executeScript", {"database_id": "64839638", "script": "UPDATE store_order SET state = 1"}),
        ("executeScript", {"database_id": "64839638", "script": "DROP TABLE store_order"}),
        ("executeScript", {"database_id": "64839638", "script": "SELECT 1; SELECT 2"}),
        ("executeScript", {"database_id": "64839638", "script": "SELECT * FROM t FOR UPDATE"}),
        ("executeScript", {"database_id": "64839638", "script": "SELECT 1 -- comment"}),
        ("executeScript", {"database_id": "64839638", "script": ""}),
        ("executeScript", {"database_id": "64839638", "script": "   "}),
        ("executeScript", {"database_id": "64839638", "script": 123}),
        ("executeScript", {}),
        ("executeScript", {"database_id": "64839638", "script": "SELECT 1", "extra": "x"}),
        # metadata / advisory field validation
        ("getTableDetailInfo", {}),
        ("getTableDetailInfo", {"table_guid": ""}),
        ("getTableDetailInfo", {"table_guid": "x", "extra": 1}),
        ("listTables", {"database_id": "64839638", "page_size": 0}),
        ("listTables", {"database_id": "64839638", "page_size": 201}),
        ("listTables", {"database_id": "64839638", "page_number": 0}),
        ("listTables", {"database_id": "64839638", "search_name": 123}),
        ("answerSqlSyntax", {}),
        ("answerSqlSyntax", {"database_id": "64839638", "question": ""}),
        ("fixSql", {"database_id": "64839638", "sql": "SELECT 1"}),
        ("optimizeSql", {}),
        ("optimizeSql", {"database_id": "64839638", "sql": "x" * 20001}),
    ],
)
def test_rejects_unsafe_dms_calls(policy, tool, tool_input):
    assert policy.denial_reason(dms(tool), tool_input) is not None


def test_dms_mcp_call_applies_the_same_validation(policy):
    for tool_input in (
        {"database_id": "64839638", "script": "SELECT 1"},
        {"database_id": "64839638", "script": "DELETE FROM t"},
    ):
        direct = policy.denial_reason(dms("executeScript"), tool_input)
        nested = policy.denial_reason(
            "mcp_call",
            {"toolName": dms("executeScript"), "arguments": tool_input},
        )
        assert (direct is None) == (nested is None)

    assert policy.denial_reason("mcp_get", {"toolName": dms("executeScript")}) is None
    assert policy.denial_reason("mcp_get", {"toolName": dms("askDatabase")}) is not None


def test_dms_tools_are_denied_when_server_disabled(tmp_path):
    policy = ToolPolicy(tmp_path, (OBSERVABILITY_SERVER,))

    assert DMS_SERVER not in policy.mcp_server_names
    assert dms("executeScript") not in policy.allowed_tools
    assert (
        policy.denial_reason(dms("executeScript"), {"script": "SELECT 1"}) is not None
    )
    assert (
        policy.denial_reason(
            "mcp_call",
            {"toolName": dms("executeScript"), "arguments": {"script": "SELECT 1"}},
        )
        is not None
    )
    assert policy.denial_reason("mcp_get", {"toolName": dms("executeScript")}) is not None


@pytest.mark.parametrize(
    ("tool", "key", "relative"),
    [
        ("Read", "file_path", "feishubot/.env"),
        ("Read", "file_path", "erp-mall-prod/.ENV.local"),
        ("Grep", "path", "feishubot/.env"),
    ],
)
def test_environment_secret_files_are_denied(policy, tmp_path, tool, key, relative):
    reason = policy.denial_reason(tool, {key: str(tmp_path / relative)})

    assert reason is not None
    assert "secret" in reason


def test_env_example_and_sibling_environments_follow_workspace_rules(policy, tmp_path):
    assert policy.denial_reason("Read", {"file_path": str(tmp_path / ".env.example")}) is None
    sibling = tmp_path.parent / "dev" / "erp-mall-dev" / "pom.xml"
    assert "outside the approved workspace" in policy.denial_reason(
        "Read", {"file_path": str(sibling)}
    )


def test_observability_prod_only_configuration(tmp_path):
    policy = ToolPolicy(tmp_path, ("observability-prod",))
    assert policy.observability_enabled is True
    assert policy.mcp_server_names == ("observability-prod",)
    obs_tools = {name for name in policy.allowed_tools if name.startswith("mcp__observability-prod__")}
    assert len(obs_tools) == 10
    assert "mcp__observability-prod__victorialogs_query" in obs_tools
    assert "mcp__aliyun-dms-mcp-server__executeScript" not in policy.allowed_tools
    # Test valid query on observability-prod
    denial = policy.denial_reason(
        "mcp__observability-prod__victorialogs_query",
        {"query": LOG_QUERY, "start": START, "end": END, "limit": 20},
    )
    assert denial is None


@pytest.fixture
def knowledge_setup(tmp_path):
    workspace = tmp_path / "prod"
    workspace.mkdir()
    knowledge = tmp_path / "rui"
    knowledge.mkdir()
    policy = ToolPolicy(workspace, (DMS_SERVER,), knowledge_dir=knowledge)
    return policy, workspace, knowledge


def test_knowledge_read_and_rules_root_allowed(knowledge_setup):
    policy, _workspace, knowledge = knowledge_setup
    assert policy.denial_reason(
        "Read", {"file_path": str(knowledge / "erp-purchase-dev" / "index.md")}
    ) is None
    rules_file = Path.home() / ".qoder" / "rules" / "00-rule-index.md"
    assert policy.denial_reason("Read", {"file_path": str(rules_file)}) is None
    assert policy.denial_reason("Read", {"file_path": str(knowledge / ".." / "evil.txt")}) is not None


def test_knowledge_write_limited_to_md(knowledge_setup):
    policy, _workspace, knowledge = knowledge_setup
    assert policy.denial_reason("Write", {"file_path": str(knowledge / "scm-dev" / "modules" / "x.md")}) is None
    assert policy.denial_reason("Edit", {"file_path": "scm-dev/modules/x.md"}) is None
    assert policy.denial_reason("Write", {"file_path": str(knowledge / "x.java")}) is not None
    assert policy.denial_reason("Write", {"file_path": str(knowledge / ".env")}) is not None
    assert policy.denial_reason("Write", {"file_path": str(knowledge / ".." / "outside.md")}) is not None


def test_knowledge_disabled_keeps_write_ban(policy):
    assert policy.denial_reason("Write", {"file_path": "a.md"}) is not None
    assert policy.denial_reason("Bash", {"command": "git log"}) is not None
    assert "Write" not in policy.allowed_tools


def test_knowledge_enabled_tools_and_glob(knowledge_setup):
    policy, _workspace, knowledge = knowledge_setup
    assert {"Write", "Edit", "Bash"} <= set(policy.allowed_tools)
    assert policy.denial_reason("Glob", {"pattern": str(knowledge / "**" / "*.md")}) is None
    assert policy.denial_reason("Glob", {"pattern": r"C:\other\**\*"}) is not None


@pytest.mark.parametrize(
    "command",
    [
        "git log --since=2026-11-16 --oneline -- src/A.java",
        "git -C erp-mall-prod log --oneline -5",
        "git diff prod...HEAD --name-only",
        "git show HEAD --stat",
        "git status",
        "git rev-parse --abbrev-ref HEAD",
        "git ls-files",
    ],
)
def test_readonly_bash_allows_single_readonly_git(knowledge_setup, command):
    policy, _workspace, _knowledge = knowledge_setup
    assert policy.denial_reason("Bash", {"command": command}) is None


@pytest.mark.parametrize(
    "command",
    [
        "git push origin HEAD:feature-x",
        "git commit -m x",
        "git checkout prod",
        "git log; rm -rf /",
        "git log && git push",
        "git log | tee out.txt",
        "git log > out.txt",
        "git -c core.pager=evil log",
        "git diff --output=evil.patch",
        "git diff --no-index a b",
        "git -C ../../ log",
        "ls -la",
        "echo $(git log)",
        "git log\nrm -rf /",
    ],
)
def test_readonly_bash_denies_everything_else(knowledge_setup, command):
    policy, _workspace, _knowledge = knowledge_setup
    assert policy.denial_reason("Bash", {"command": command}) is not None

