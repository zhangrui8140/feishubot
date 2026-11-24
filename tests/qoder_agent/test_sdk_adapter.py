from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from qoder_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
)

from feishubot.qoder_agent import sdk_adapter, session_handle
from feishubot.qoder_agent.models import AgentProgress, ImageInput
from feishubot.qoder_agent.sdk_adapter import QoderSdkAdapter, QoderSessionHandle


class FakeClient:
    def __init__(self, messages):
        self.messages = messages
        self.queries: list[str] = []
        self.interrupted = False
        self.disconnected = False

    async def connect(self) -> None:
        return None

    async def query(self, prompt) -> None:
        if isinstance(prompt, str):
            self.queries.append(prompt)
            return
        async for message in prompt:
            self.queries.append(message)

    async def receive_response(self) -> AsyncIterator[object]:
        for message in self.messages:
            yield message

    async def interrupt(self) -> None:
        self.interrupted = True

    async def disconnect(self) -> None:
        self.disconnected = True


def make_adapter(tmp_path) -> tuple[QoderSdkAdapter, object]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text(
        '{"mcpServers":{"aliyun-dms-mcp-server":{"url":"https://dms.example/sse"}}}',
        encoding="utf-8",
    )
    return (
        QoderSdkAdapter(
            workspace_root=workspace,
            model="auto",
            mcp_servers_config_path=mcp_config,
        ),
        mcp_config,
    )


@pytest.mark.asyncio
async def test_session_handle_collects_result_and_tools():
    client = FakeClient(
        [
            SystemMessage(subtype="init", data={"session_id": "session-1"}),
            AssistantMessage(
                content=[ToolUseBlock(id="tool-1", name="Grep", input={})],
                model="auto",
            ),
            AssistantMessage(content=[TextBlock(text="最终答案")], model="auto"),
            ResultMessage(
                subtype="success",
                duration_ms=120,
                duration_api_ms=80,
                is_error=False,
                num_turns=2,
                session_id="session-1",
                result="最终答案",
            ),
        ]
    )
    handle = QoderSessionHandle(client)
    await handle.start()

    result = await handle.run("分析流程")
    await handle.disconnect()

    assert client.queries == ["分析流程"]
    assert client.disconnected is True
    assert result.text == "最终答案"
    assert result.session_id == "session-1"
    assert result.tool_names == ("Grep",)


@pytest.mark.asyncio
async def test_permission_denials_with_successful_result_are_tolerated():
    client = FakeClient(
        [
            ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="session-1",
                result="已改用只读方式完成",
                permission_denials=[{"tool_name": "Bash", "message": "denied"}],
            )
        ]
    )
    handle = QoderSessionHandle(client)
    await handle.start()
    try:
        outcome = await handle.run("任务")
    finally:
        await handle.disconnect()

    assert outcome.text == "已改用只读方式完成"


@pytest.mark.asyncio
async def test_permission_denials_with_error_result_raise_permission_error():
    client = FakeClient(
        [
            ResultMessage(
                subtype="error_during_execution",
                duration_ms=1,
                duration_api_ms=1,
                is_error=True,
                num_turns=1,
                session_id="session-1",
                permission_denials=[{"tool_name": "Bash", "message": "denied"}],
            )
        ]
    )
    handle = QoderSessionHandle(client)
    await handle.start()
    try:
        with pytest.raises(session_handle.QoderSdkPermissionError):
            await handle.run("任务")
    finally:
        await handle.disconnect()


def _result(text: str) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=120,
        duration_api_ms=80,
        is_error=False,
        num_turns=3,
        session_id="session-1",
        result=text,
    )


def _text(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], model="auto")


def _tool(tool_id: str, text: str | None = None) -> AssistantMessage:
    content: list[object] = [TextBlock(text=text)] if text else []
    content.append(ToolUseBlock(id=tool_id, name="Grep", input={}))
    return AssistantMessage(content=content, model="auto")


async def _run_with_progress(messages, callback=None) -> tuple[str, list[AgentProgress]]:
    sent: list[AgentProgress] = []

    async def collect(event: AgentProgress) -> None:
        sent.append(event)

    handle = QoderSessionHandle(FakeClient(messages))
    await handle.start()
    try:
        result = await handle.run("分析", on_progress=callback or collect)
    finally:
        await handle.disconnect()
    return result.text, sent


@pytest.mark.asyncio
async def test_session_handle_sends_images_as_content_blocks():
    client = FakeClient([_result("ok")])
    handle = QoderSessionHandle(client)
    await handle.start()
    try:
        await handle.run(
            "看截图",
            images=(ImageInput(media_type="image/png", data_base64="QUJD"),),
        )
    finally:
        await handle.disconnect()

    (message,) = client.queries
    assert message["type"] == "user"
    content = message["message"]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"].endswith("看截图")
    assert content[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }


@pytest.mark.asyncio
async def test_session_handle_pushes_intermediate_text_but_not_final_answer():
    text, sent = await _run_with_progress(
        [
            _tool("t1", "【分析思路】先定位接口"),
            _text("【阶段结果】找到校验逻辑"),
            _tool("t2"),
            _text("【最终结论】原因是校验失败"),
            _result("【最终结论】原因是校验失败"),
        ]
    )

    assert sent == [
        AgentProgress(kind="assistant_text", text="【分析思路】先定位接口"),
        AgentProgress(kind="tool_started", tool_name="Grep"),
        AgentProgress(kind="assistant_text", text="【阶段结果】找到校验逻辑"),
        AgentProgress(kind="tool_started", tool_name="Grep"),
    ]
    assert text == "【最终结论】原因是校验失败"


class InjectingClient(FakeClient):
    def __init__(self, *, echo: bool) -> None:
        super().__init__([])
        self.echo = echo
        self.injected: list[tuple[str, dict]] = []
        self.tool_seen = asyncio.Event()
        self.inject_written = asyncio.Event()
        self.receive_calls = 0

    async def query(self, prompt, **kwargs) -> None:
        if kwargs:
            self.injected.append((prompt, kwargs))
            self.inject_written.set()
            return
        await super().query(prompt)

    async def receive_response(self) -> AsyncIterator[object]:
        self.receive_calls += 1
        if self.receive_calls > 1:
            yield _result("")
            return
        yield _tool("t1")
        self.tool_seen.set()
        await self.inject_written.wait()
        if self.echo:
            yield UserMessage(content=[TextBlock(text=self.injected[0][0])])
            yield _result("merged")
        else:
            yield _result("late")


async def _inject_during_run(client: InjectingClient) -> tuple[bool, str]:
    handle = QoderSessionHandle(client)
    await handle.start()
    try:
        run_task = asyncio.create_task(handle.run("分析"))
        await client.tool_seen.wait()
        assert handle.running is True
        accepted = await handle.inject("【用户补充】只看 erp-mall-prod")
        result = await run_task
    finally:
        await handle.disconnect()
    assert handle.running is False
    return accepted, result.text


@pytest.mark.asyncio
async def test_session_handle_injects_supplement_into_active_turn():
    client = InjectingClient(echo=True)

    accepted, text = await _inject_during_run(client)

    assert accepted is True
    assert text == "merged"
    assert client.injected == [
        ("【用户补充】只看 erp-mall-prod", {"priority": "next", "should_query": False})
    ]
    assert client.receive_calls == 1


@pytest.mark.asyncio
async def test_session_handle_drains_result_of_supplement_missed_by_turn():
    client = InjectingClient(echo=False)

    accepted, text = await _inject_during_run(client)

    assert accepted is False
    assert text == "late"
    assert client.receive_calls == 2


@pytest.mark.asyncio
async def test_session_handle_rejects_injection_when_idle():
    client = FakeClient([_result("ok")])
    handle = QoderSessionHandle(client)
    await handle.start()
    try:
        assert await handle.inject("补充") is False
    finally:
        await handle.disconnect()
    assert client.queries == []


@pytest.mark.asyncio
async def test_session_handle_caps_progress_and_only_truncates_text():
    long_tool_name = "G" * 2000
    messages = [
        AssistantMessage(
            content=[
                TextBlock(text="阶段0" + "x" * 2000),
                ToolUseBlock(id="t0", name=long_tool_name, input={}),
            ],
            model="auto",
        )
    ]
    messages += [_tool(f"t{i}", f"阶段{i}") for i in range(1, 12)]
    messages += [_text("最终"), _result("最终")]

    _text_result, sent = await _run_with_progress(messages)

    assert len(sent) == session_handle._MAX_PROGRESS_MESSAGES
    assert sent[0].kind == "assistant_text"
    assert sent[0].text is not None
    assert len(sent[0].text) <= session_handle._MAX_PROGRESS_CHARS + 2
    assert sent[0].text.startswith("阶段0") and sent[0].text.endswith("……")
    assert sent[1] == AgentProgress(kind="tool_started", tool_name=long_tool_name)


@pytest.mark.asyncio
async def test_session_handle_survives_progress_delivery_failure():
    async def failing(_event: AgentProgress) -> None:
        raise RuntimeError("feishu down")

    text, _sent = await _run_with_progress(
        [_tool("t1", "【分析思路】查代码"), _text("最终"), _result("最终")],
        callback=failing,
    )

    assert text == "最终"


def test_adapter_rejects_write_tools_and_external_paths(tmp_path):
    adapter, _mcp_config = make_adapter(tmp_path)

    assert adapter._tool_denial_reason("Write", {"file_path": "a.py"}) is not None
    assert adapter._tool_denial_reason("Read", {"file_path": "../secret.txt"}) is not None
    assert adapter._tool_denial_reason("Glob", {"pattern": "../**/*"}) is not None
    assert adapter._tool_denial_reason("Read", {"file_path": "src/a.py"}) is None


@pytest.mark.asyncio
async def test_adapter_passes_restricted_mcp_options(tmp_path, monkeypatch):
    adapter, mcp_config = make_adapter(tmp_path)
    captured = {}

    class CapturingClient(FakeClient):
        def __init__(self, *, options):
            super().__init__([])
            captured["options"] = options

    monkeypatch.setattr(sdk_adapter, "QoderSDKClient", CapturingClient)
    monkeypatch.setattr(QoderSdkAdapter, "_auth", lambda self: object())

    handle = await adapter.open_session()
    await handle.disconnect()

    options = captured["options"]
    assert options.mcp_servers == mcp_config.resolve()
    assert options.allowed_mcp_server_names == ["aliyun-dms-mcp-server"]
    assert set(options.allowed_tools) == {
        "Read",
        "Glob",
        "Grep",
        "mcp__aliyun-dms-mcp-server__listTables",
        "mcp__aliyun-dms-mcp-server__getTableDetailInfo",
        "mcp__aliyun-dms-mcp-server__executeScript",
        "mcp__aliyun-dms-mcp-server__answerSqlSyntax",
        "mcp__aliyun-dms-mcp-server__fixSql",
        "mcp__aliyun-dms-mcp-server__optimizeSql",
        "mcp_list",
        "mcp_get",
        "mcp_call",
    }
    assert set(options.tools) == set(options.allowed_tools)
    # The user rules index mentions observability as a routing entry; what must
    # stay out without the server is the concrete query guidance.
    assert "victorialogs_query" not in options.system_prompt
    assert "aliyun-dms-mcp-server" in options.system_prompt
    assert "executeScript" in options.system_prompt
    assert "createDataChangeOrder" in options.system_prompt
    assert "分析范围仅限 prod 环境项目" not in options.system_prompt
    assert adapter.observability_enabled is False


@pytest.mark.asyncio
async def test_adapter_prompt_limits_analysis_to_prod_projects(tmp_path, monkeypatch):
    adapter, _ = make_adapter(tmp_path)
    for name in ("erp-mall-prod", "scm-prod", "rui"):
        (tmp_path / "workspace" / name).mkdir()
    captured = {}

    class CapturingClient(FakeClient):
        def __init__(self, *, options):
            super().__init__([])
            captured["options"] = options

    monkeypatch.setattr(sdk_adapter, "QoderSDKClient", CapturingClient)
    monkeypatch.setattr(QoderSdkAdapter, "_auth", lambda self: object())

    handle = await adapter.open_session()
    await handle.disconnect()

    assert adapter.projects == ("erp-mall-prod", "scm-prod")
    prompt = captured["options"].system_prompt
    assert "分析范围仅限 prod 环境项目：erp-mall-prod、scm-prod" in prompt
    assert "dev、pre、test 等其他环境或其他目录一律不读取" in prompt


@pytest.mark.asyncio
async def test_adapter_enables_observability_when_configured(tmp_path, monkeypatch):
    adapter, mcp_config = make_adapter(tmp_path)
    mcp_config.write_text(
        '{"mcpServers":{"aliyun-dms-mcp-server":{"url":"https://dms.example/sse"},"observability":'
        '{"url":"https://obs.example/mcp","headers":{"Authorization":"Bearer secret-token"}}}}',
        encoding="utf-8",
    )
    captured = {}

    class CapturingClient(FakeClient):
        def __init__(self, *, options):
            super().__init__(
                [
                    ResultMessage(
                        subtype="success",
                        duration_ms=1,
                        duration_api_ms=1,
                        is_error=False,
                        num_turns=1,
                        session_id="session-obs",
                        result="ok",
                    )
                ]
            )
            captured["options"] = options
            captured["client"] = self

    monkeypatch.setattr(sdk_adapter, "QoderSDKClient", CapturingClient)
    monkeypatch.setattr(QoderSdkAdapter, "_auth", lambda self: object())

    handle = await adapter.open_session()
    await handle.run("查日志")
    await handle.disconnect()

    options = captured["options"]
    assert adapter.observability_enabled is True
    assert options.allowed_mcp_server_names == ["aliyun-dms-mcp-server", "observability"]
    observability_tools = {
        name for name in options.allowed_tools if name.startswith("mcp__observability__")
    }
    assert len(observability_tools) == 10
    assert "mcp__observability__victorialogs_query" in observability_tools
    assert "mcp__observability__health" not in observability_tools
    assert "victorialogs_query" in options.system_prompt
    assert "secret-token" not in options.system_prompt
    assert captured["client"].queries[0].startswith("[当前时间 ")
    assert captured["client"].queries[0].endswith("\n查日志")
    assert (
        adapter._tool_denial_reason(
            "mcp__observability__victorialogs_query",
            {"query": "ERROR", "start": "2026-10-13T09:00:00Z", "end": "2026-10-13T10:00:00Z"},
        )
        is not None
    )


@pytest.mark.asyncio
async def test_adapter_enables_observability_prod_only(tmp_path, monkeypatch):
    adapter, mcp_config = make_adapter(tmp_path)
    mcp_config.write_text(
        '{"mcpServers":{"observability-prod":'
        '{"url":"https://observability-mcp.observability.ali-test.xlbsoft.com/mcp",'
        '"headers":{"Authorization":"Bearer 293800ad31aefb7956de0cff552e28c323a0f1e86579e7a066e3c465a9840306"}}}}',
        encoding="utf-8",
    )
    captured = {}

    class CapturingClient(FakeClient):
        def __init__(self, *, options):
            super().__init__(
                [
                    ResultMessage(
                        subtype="success",
                        duration_ms=1,
                        duration_api_ms=1,
                        is_error=False,
                        num_turns=1,
                        session_id="session-obs-prod",
                        result="ok",
                    )
                ]
            )
            captured["options"] = options
            captured["client"] = self

    monkeypatch.setattr(sdk_adapter, "QoderSDKClient", CapturingClient)
    monkeypatch.setattr(QoderSdkAdapter, "_auth", lambda self: object())

    handle = await adapter.open_session()
    await handle.run("查生产日志")
    await handle.disconnect()

    options = captured["options"]
    assert adapter.observability_enabled is True
    assert options.allowed_mcp_server_names == ["observability-prod"]
    obs_tools = {
        name for name in options.allowed_tools if name.startswith("mcp__observability-prod__")
    }
    assert len(obs_tools) == 10
    assert "mcp__observability-prod__victorialogs_query" in obs_tools
    assert "mcp__aliyun-dms-mcp-server__executeScript" not in options.allowed_tools
    assert "victorialogs_query" in options.system_prompt
    assert "293800ad" not in options.system_prompt
    assert "aliyun-dms-mcp-server" not in options.system_prompt



@pytest.mark.parametrize(
    "observability",
    ['"bearer secret-token"', "{}", '{"url":""}', '{"headers":{"Authorization":"secret-token"}}'],
)
def test_adapter_rejects_invalid_observability_config_without_leaking(
    tmp_path,
    monkeypatch,
    observability,
):
    adapter, mcp_config = make_adapter(tmp_path)
    mcp_config.write_text(
        f'{{"mcpServers":{{"aliyun-dms-mcp-server":{{"url":"https://dms.example/sse"}},"observability":{observability}}}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(QoderSdkAdapter, "_auth", lambda self: object())

    with pytest.raises(ValueError, match="observability must be an object") as exc_info:
        adapter.validate()

    assert "secret-token" not in str(exc_info.value)
    assert adapter.observability_enabled is False


@pytest.mark.parametrize(
    ("content", "expected_message"),
    [
        ("not-json-database-password", "not valid JSON"),
        ("[]", "JSON object"),
        ("{}", "mcpServers object"),
        ('{"mcpServers":{}}', "define at least one valid server"),
    ],
)
def test_adapter_rejects_invalid_mcp_config_without_leaking_content(
    tmp_path,
    monkeypatch,
    content,
    expected_message,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text(content, encoding="utf-8")
    adapter = QoderSdkAdapter(
        workspace_root=workspace,
        model="auto",
        mcp_servers_config_path=mcp_config,
    )
    monkeypatch.setattr(QoderSdkAdapter, "_auth", lambda self: object())

    with pytest.raises(ValueError) as exc_info:
        adapter.validate()

    assert expected_message in str(exc_info.value)
    assert "database-password" not in str(exc_info.value)


def test_adapter_requires_existing_mcp_config(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = QoderSdkAdapter(workspace_root=workspace, model="auto")

    with pytest.raises(ValueError, match="MCP config path is required"):
        adapter.validate()


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM store_order WHERE note = 'delete; -- text'",
        "WITH recent AS (SELECT id FROM store_order) SELECT * FROM recent",
        "WITH RECURSIVE ids AS (SELECT 1) SELECT * FROM ids",
        "SHOW TABLES",
        "DESCRIBE store_order",
        "DESC store_order",
    ],
)
def test_adapter_allows_read_only_sql(tmp_path, query):
    adapter, _mcp_config = make_adapter(tmp_path)

    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__executeScript",
            {"database_id": "64839638", "script": query},
        )
        is None
    )


@pytest.mark.parametrize(
    "query",
    [
        "",
        "INSERT INTO store_order(id) VALUES (1)",
        "UPDATE store_order SET state = 1",
        "DELETE FROM store_order",
        "DROP TABLE store_order",
        "START TRANSACTION",
        "SET @value = 1",
        "CALL rebuild_order()",
        "SELECT 1;",
        "SELECT 1; SELECT 2",
        "SELECT * FROM store_order -- comment",
        "SELECT * FROM store_order /* comment */",
        "SELECT * FROM store_order INTO OUTFILE '/tmp/orders'",
        "SELECT * FROM store_order FOR UPDATE",
        "SELECT SLEEP(1)",
        "SELECT BENCHMARK(10, 1)",
        "SELECT @value",
        "EXPLAIN SELECT * FROM store_order",
        "WITH changed AS (UPDATE store_order SET state = 1) SELECT * FROM changed",
        "SELECT 'unterminated",
        "SELECT (1",
    ],
)
def test_adapter_rejects_unsafe_or_ambiguous_sql(tmp_path, query):
    adapter, _mcp_config = make_adapter(tmp_path)

    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__executeScript",
            {"database_id": "64839638", "script": query},
        )
        is not None
    )


def test_adapter_restricts_mcp_tool_names_and_inputs(tmp_path):
    adapter, _mcp_config = make_adapter(tmp_path)

    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__getTableDetailInfo",
            {"table_guid": "123.erp_read.store_order"},
        )
        is None
    )
    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__listTables",
            {"database_id": "64839638", "search_name": "store", "page_size": 50},
        )
        is None
    )
    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__optimizeSql",
            {"database_id": "64839638", "sql": "SELECT * FROM store_order"},
        )
        is None
    )
    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__executeScript",
            {"script": "SELECT 1"},
        )
        is not None
    )
    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__listTables",
            {"database_id": "not-a-number"},
        )
        is not None
    )
    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__askDatabase", {"question": "count orders"}
        )
        is not None
    )
    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__createDataChangeOrder",
            {"script": "UPDATE store_order SET state = 1"},
        )
        is not None
    )
    assert (
        adapter._tool_denial_reason("mcp__aliyun-dms-mcp-server__dropTable", {})
        is not None
    )
    assert (
        adapter._tool_denial_reason("mcp__aliyun-dms-mcp-server__getTableDetailInfo", {})
        is not None
    )
    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__listTables", {"page_size": 201}
        )
        is not None
    )


def test_adapter_validates_mcp_meta_tool_targets_and_nested_arguments(tmp_path):
    adapter, _mcp_config = make_adapter(tmp_path)
    adapter._validate_mcp_config()

    assert adapter._tool_denial_reason("mcp_list", {}) is None
    assert (
        adapter._tool_denial_reason(
            "mcp_get",
            {"toolName": "mcp__aliyun-dms-mcp-server__executeScript"},
        )
        is None
    )
    assert (
        adapter._tool_denial_reason(
            "mcp_call",
            {
                "toolName": "mcp__aliyun-dms-mcp-server__executeScript",
                "arguments": {"database_id": "64839638", "script": "SELECT 1"},
            },
        )
        is None
    )
    assert (
        adapter._tool_denial_reason(
            "mcp_call",
            {
                "toolName": "mcp__aliyun-dms-mcp-server__executeScript",
                "arguments": {"database_id": "64839638", "script": "DELETE FROM store_order"},
            },
        )
        is not None
    )
    assert (
        adapter._tool_denial_reason(
            "mcp_call",
            {
                "toolName": "mcp__aliyun-dms-mcp-server__askDatabase",
                "arguments": {"question": "count orders"},
            },
        )
        is not None
    )
    assert (
        adapter._tool_denial_reason(
            "mcp_get",
            {"toolName": "mcp__aliyun-dms-mcp-server__createDataChangeOrder"},
        )
        is not None
    )


def test_adapter_excludes_dms_when_not_configured(tmp_path):
    adapter, mcp_config = make_adapter(tmp_path)
    mcp_config.write_text(
        '{"mcpServers":{"observability":{"url":"https://obs.example/mcp"}}}',
        encoding="utf-8",
    )

    adapter._validate_mcp_config()

    assert adapter._policy.mcp_server_names == ("observability",)
    assert "mcp__aliyun-dms-mcp-server__executeScript" not in adapter._policy.allowed_tools
    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__executeScript", {"script": "SELECT 1"}
        )
        is not None
    )


def test_adapter_includes_dms_when_configured(tmp_path):
    adapter, _mcp_config = make_adapter(tmp_path)

    adapter._validate_mcp_config()

    assert adapter._policy.mcp_server_names == ("aliyun-dms-mcp-server",)
    assert "mcp__aliyun-dms-mcp-server__executeScript" in adapter._policy.allowed_tools
    assert (
        adapter._tool_denial_reason(
            "mcp__aliyun-dms-mcp-server__executeScript",
            {"database_id": "64839638", "script": "SELECT 1"},
        )
        is None
    )


@pytest.mark.asyncio
async def test_permission_hook_and_callback_share_policy(tmp_path):
    adapter, _mcp_config = make_adapter(tmp_path)
    tool_input = {"script": "DELETE FROM store_order"}

    callback_result = await adapter._can_use_tool(
        "mcp__aliyun-dms-mcp-server__executeScript",
        tool_input,
        None,
    )
    hook_result = await adapter._validate_pre_tool_use(
        {
            "tool_name": "mcp__aliyun-dms-mcp-server__executeScript",
            "tool_input": tool_input,
        },
        None,
        None,
    )

    assert callback_result.message
    assert hook_result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert hook_result["hookSpecificOutput"]["permissionDecisionReason"] == callback_result.message


def _knowledge_adapter(tmp_path, *, mode="readonly", with_knowledge=True):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    knowledge = tmp_path / "rui"
    knowledge.mkdir(exist_ok=True)
    mcp_config = tmp_path / "mcp.json"
    mcp_config.write_text(
        '{"mcpServers":{"aliyun-dms-mcp-server":{"url":"https://dms.example/sse"}}}',
        encoding="utf-8",
    )
    return QoderSdkAdapter(
        workspace_root=workspace,
        model="auto",
        mcp_servers_config_path=mcp_config,
        mode=mode,
        knowledge_dir=knowledge if with_knowledge else None,
    )


@pytest.mark.asyncio
async def test_readonly_knowledge_sharing_opens_guarded_tools(tmp_path, monkeypatch):
    adapter = _knowledge_adapter(tmp_path)
    captured = {}

    class CapturingClient(FakeClient):
        def __init__(self, *, options):
            super().__init__([])
            captured["options"] = options

    monkeypatch.setattr(sdk_adapter, "QoderSDKClient", CapturingClient)
    monkeypatch.setattr(QoderSdkAdapter, "_auth", lambda self: object())

    handle = await adapter.open_session()
    await handle.disconnect()

    options = captured["options"]
    assert {"Write", "Edit", "Bash"} <= set(options.allowed_tools)
    assert options.disallowed_tools == []
    prompt = options.system_prompt
    assert "共享知识库" in prompt
    assert "RUI-02" in prompt
    assert "用户级规则库" in prompt
    assert "ERP 代码仍只读" in prompt


@pytest.mark.asyncio
async def test_readonly_without_knowledge_keeps_write_ban(tmp_path, monkeypatch):
    adapter = _knowledge_adapter(tmp_path, with_knowledge=False)
    captured = {}

    class CapturingClient(FakeClient):
        def __init__(self, *, options):
            super().__init__([])
            captured["options"] = options

    monkeypatch.setattr(sdk_adapter, "QoderSDKClient", CapturingClient)
    monkeypatch.setattr(QoderSdkAdapter, "_auth", lambda self: object())

    handle = await adapter.open_session()
    await handle.disconnect()

    options = captured["options"]
    assert set(options.disallowed_tools) == {"Write", "Edit", "Bash"}
    assert "共享知识库" not in options.system_prompt
    # User rules stay aligned with the IDE even without knowledge sharing.
    assert "用户级规则库" in options.system_prompt


@pytest.mark.asyncio
async def test_dev_knowledge_prompt_marks_base_as_non_project(tmp_path, monkeypatch):
    adapter = _knowledge_adapter(tmp_path, mode="dev")
    captured = {}

    class CapturingClient(FakeClient):
        def __init__(self, *, options):
            super().__init__([])
            captured["options"] = options

    monkeypatch.setattr(sdk_adapter, "QoderSDKClient", CapturingClient)
    monkeypatch.setattr(QoderSdkAdapter, "_auth", lambda self: object())

    handle = await adapter.open_session()
    await handle.disconnect()

    prompt = captured["options"].system_prompt
    assert "共享知识库" in prompt
    assert "知识库不是开发项目" in prompt
    assert "RUI-03" in prompt
