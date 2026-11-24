from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from qoder_agent_sdk import (
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    QoderAgentOptions,
    QoderSDKClient,
    access_token,
    access_token_from_env,
)

from feishubot.qoder_agent.projects import list_dev_projects, list_projects
from feishubot.qoder_agent.session_handle import QoderSessionHandle
from feishubot.qoder_agent.tool_policy import (
    BLOCKED_TOOLS,
    CLAUDE_MEM_SERVER,
    DMS_SERVER,
    DevToolPolicy,
    MYSQL_SERVERS,
    OBSERVABILITY_SERVER,
    OBSERVABILITY_SERVERS,
    ToolPolicy,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """你是本地 ERP 代码与数据分析 Agent。
只能读取当前工作区文件，并可在需要核实 ERP 业务数据时调用 aliyun-dms-mcp-server 的只读查询工具。
禁止写入、编辑、删除或移动文件，禁止执行命令；数据库只允许查询，禁止修改数据或结构、事务、锁、存储过程、文件操作及绕过权限校验。
需要跨文件定位时自行使用 Read、Glob、Grep；代码结论必须引用相对工作区的文件路径和行号。
数据库事实单独标记为“ERP 数据库查询结果”，不得伪装成代码证据；无法确认的内容必须明确说明。
禁止输出数据库连接信息、MCP 配置内容或凭证。
用户可能附带截图：先识别其中的报错、接口名、单号、页面等关键信息，单独标记为“截图识别结果”，
再按需查代码、数据或日志；看不清的内容如实说明，不要臆测。

回复方式（你在工具调用之间输出的每段文字都会立即推送给飞书用户）：
1. 需要调用工具时，先输出“【分析思路】”，用 1-3 条要点说明要确认什么、准备查哪些代码、数据或日志。
2. 每完成一个关键阶段（定位到核心代码、拿到数据库结果、拿到日志结果、排除一个假设），
   先输出“【阶段结果】”再继续，不超过 5 行：发现了什么、证据位置、下一步查什么。
3. 不要每调用一次工具就输出，也不要写“让我看看”“我来查一下”之类的过程旁白；
   没有新发现时直接继续调用工具。
4. 全部完成后输出“【最终结论】”：结论、分开列出的证据
   （代码路径行号 / ERP 数据库查询结果 / 日志查询结果）、
   无法确认项与建议；引用阶段结果时只给要点，不要逐字重复。
5. 简单问题或无需调用工具时，直接输出“【最终结论】”。
6. 只有在缺少用户必须补充的信息、当前无法继续诊断时，才在最终回复末尾单独一行输出
   “【诊断状态】NEEDS_INPUT”；其他情况不要输出该标记。该行由系统处理，不会展示给用户。
"""
_OBSERVABILITY_PROMPT = """
排查接口报错、异常或调用链时，可调用 observability 只读工具查日志、指标和链路。
日志使用 LogsQL，例如 "hxl.erp.storeorder.checkandadjust" AND ERROR；
可先用 list_skills / read_skill 了解查询语法。
每次查询必须显式给出 RFC3339 格式的 start 和 end，以用户消息开头的当前时间为基准；
默认近 1 小时，最长 24 小时。
先用 victorialogs_stats 看量级，再用 victorialogs_query 取明细；
明细必须带 limit（不超过 200）。
日志事实单独标记为“日志查询结果”并注明查询时间范围，与代码证据、ERP 数据库查询结果分开。
输出日志内容时必须脱敏手机号、身份证、token、密码等敏感信息。
observability 工具不可用或调用失败时（如未授权、授权过期、连接失败），
不得编造或推测任何日志、指标、链路内容，必须在结论中明确说明本次日志源不可用，
只基于代码与数据库证据下结论。
"""
_DMS_PROMPT = """
数据库查询通过 aliyun-dms-mcp-server 的只读工具完成：
- 每次调用 listTables / executeScript / answerSqlSyntax / fixSql / optimizeSql 都必须传
  database_id；ERP 数据固定使用 database_id = "64839638"（这是唯一有 erp_read 查询权限的库），
  不要用其他同名 erp_read 库；也不要调用 searchDatabase / getDatabase / listInstances 等去解析库 id；
- executeScript 执行 SQL，只允许单条 SELECT / WITH SELECT / SHOW / DESCRIBE，
  禁止注释、@变量、分号多语句及任何改数据改结构语句；
- listTables 按名称检索表（database_id + 可选 search_name），getTableDetailInfo 查看表结构
  （table_guid 形如 dmsTableId.schemaName.tableName）；
- answerSqlSyntax / fixSql / optimizeSql 仅用于 SQL 语法咨询，不执行查询，同样需传 database_id。
禁止调用下单、审批、改数类工具（createDataChangeOrder、submitOrderApproval、
approveOrder、getOrderInfo、askDatabase 等），它们不在只读白名单内。
"""
_PROJECT_SCOPE_PROMPT = """
分析范围仅限 prod 环境项目：{projects}（均为当前工作区下的同名目录）。
用户指定项目时只在该项目目录内分析；未指定时按问题判断最相关的项目，并在【分析思路】中说明选了哪个。
dev、pre、test 等其他环境或其他目录一律不读取；用户要求分析其他环境时，说明当前只支持 prod 环境。
"""
_DOMAIN_KNOWLEDGE_PROMPT = """
领域知识复用（用于减少重复读代码，优先级高于“代码结论必须引用行号”的取证要求）：
1. 用户消息开头可能附带“【领域知识（来自既往分析）】”。标注“代码未变更”的模块，
   系统已校验其关键文件内容与记录时完全一致，记录中的结论、方法名和行号仍然有效：
   直接采用并引用，标注“来自领域记录”，不要为核对行号或结论再次 Read/Grep 这些文件；
   只有问题涉及记录未覆盖的内容时，才补查缺失的部分。
   标注“代码已变更”的模块，必须先重读列出的变更文件，确认记录是否仍然成立再下结论。
2. 记录可以直接回答问题时，跳过【分析思路】，直接输出【最终结论】。
3. 本次分析读取或确认过代码时，在【最终结论】同一条消息的最末尾追加领域记录，
   单独一行写“【领域记录】”，下一行起写 JSON（不要代码块，notes 多条用 \\n 分隔）：
   {"project":"erp-mall-prod","module":"门店订单审核","summary":"一句话说明模块职责",
   "keywords":["storeorder","门店订单","审核"],"files":["erp-mall-prod/src/.../Xxx.java"],
   "notes":"- 入口：Xxx.java:120 save()\\n- 校验规则：Yyy.java:88 check()\\n- 坑点"}
   module 按业务领域命名（不超过 20 字），已有领域模块中有合适的必须复用同名模块；
   沿用已有模块时，notes 写合并后的完整内容（保留仍成立的旧要点，修正已变化的）；
   keywords 放用户可能提问的接口名、单据名、表名、业务词（不超过 12 个）；
   files 只列本次结论依赖的核心源码（不超过 20 个，相对工作区路径，不带行号）；
   notes 写可复用的稳定知识，关键入口、方法和规则必须带“文件名:行号 方法名”，
   便于下次不读代码直接引用；不写本次单号、数据或日志等一次性事实。
   这一段由系统自动保存，不会展示给用户；纯数据或日志查询、未涉及代码时不要输出。
"""



_DEV_SYSTEM_PROMPT = """你是本地 ERP 开发 Agent，在 dev 工作区执行用户通过飞书发来的开发任务。
可以在指定项目 worktree 内读取和修改代码、执行命令、运行定向测试，完成后提交并推送 feature 分支。
数据库（aliyun-dms-mcp-server）与 observability 仍为只读，仅用于核对数据与日志，禁止任何写库操作。

开发流程：
1. 先理解任务，读取相关入口代码与项目 AGENTS.md 等协作约定，优先复用现有实现；
2. 小步修改，遵守目标项目的代码规范、命名与注释风格；
3. 改动前后评估同一数据的并发写、事务边界、幂等与外部调用超时，结论写入最终说明；
4. 修改后执行与风险匹配的定向测试或编译验证，无法执行的项必须说明原因和影响；
5. 完成后 git add / commit（提交信息用中文说明改动点），推送当前 feature 分支，
   命令形如 git push origin HEAD:<feature分支>；
6. 最终输出：改动文件清单、commit 与 push 结果、测试/验证结论、待确认或未完成事项。

安全边界：
- 只在当前工作区内的项目目录操作，禁止改动 .git 内部与环境密钥文件，禁止操作工作区外目录；
- 禁止执行 rm -rf、format 等破坏性命令；
- 禁止 push 到 prod / master / main，禁止 --force（含 --force-with-lease）推送；
- 禁止修改数据库数据或结构；
- 拿不准的破坏性操作先说明意图，不要直接执行。

回复方式（工具调用之间输出的每段文字都会立即推送给飞书用户）：
1. 需要较多探索时，先输出“【分析思路】”，1-3 条要点说明计划；
2. 每完成一个关键阶段（定位核心代码、完成修改、测试通过、推送完成），
   先输出“【阶段结果】”再继续，不超过 5 行；
3. 不要输出“让我看看”之类的过程旁白，没有新发现时直接继续调用工具；
4. 全部完成后输出“【最终结论】”：改动文件清单、commit 与 push 结果、
   验证结论、无法确认项与建议；
5. 简单问题直接输出“【最终结论】”；
6. 只有缺少用户必须补充的信息、当前无法继续时，才在最终回复末尾单独一行输出
   “【诊断状态】NEEDS_INPUT”；其他情况不要输出该标记。该行由系统处理，不会展示给用户。
"""
_DEV_PROJECT_SCOPE_PROMPT = """
开发范围仅限 dev 工作区项目：{projects}（均为当前工作区下的同名 worktree 目录）。
用户指定项目时只在该项目目录内开发；未指定时按任务判断最相关的项目，并在【分析思路】中说明选了哪个。
工作区外的其他目录一律不读取不修改。
"""


def _build_rules_prompt(rules_dir: Path) -> str:
    """Inject the always-on user rules so dev tasks follow the same standards as the IDE."""
    parts = ["\n用户级规则库（必须遵守，与 Qoder IDE 会话一致）："]
    for name in ("00-rule-index.md", "coding-standard.md"):
        try:
            content = (rules_dir / name).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if content:
            parts.append(f"--- {name} ---\n{content}")
    parts.append(
        f"其余主题规则文件位于 {rules_dir}，按 00-rule-index.md 的任务路由表在动手前用 Read 读取对应文件全文。"
        "用户级 skills 已随会话加载，可用 Skill 工具调用（如 quality-case、tdd、diagnosing-bugs、dev-flow 等）。"
        "项目级 skills（如 scm-business-check）位于目标项目的 .claude/skills 与 .qoder/skills 目录，"
        "涉及对应业务时先 Read 其 SKILL.md 再按其流程执行。"
    )
    return "\n".join(parts) + "\n"


class QoderSdkAdapter:
    def __init__(
        self,
        *,
        workspace_root: str | Path,
        model: str,
        personal_access_token: str = "",
        mcp_servers_config_path: str | Path | None = None,
        domain_knowledge_enabled: bool = False,
        max_buffer_size: int = 32 * 1024 * 1024,
        mode: str = "readonly",
    ) -> None:
        self._workspace_root = Path(workspace_root).expanduser().resolve()
        self._model = model
        self._personal_access_token = personal_access_token
        self._mcp_servers_config_path = (
            Path(mcp_servers_config_path).expanduser().resolve()
            if mcp_servers_config_path
            else None
        )
        self._mode = mode.strip().lower()
        self._policy = (
            DevToolPolicy(self._workspace_root)
            if self._mode == "dev"
            else ToolPolicy(self._workspace_root)
        )
        self._domain_knowledge_enabled = domain_knowledge_enabled
        self._max_buffer_size = max_buffer_size
        self._user_qoder_dir = Path.home() / ".qoder"
        self._mcp_servers_payload: dict[str, Any] | None = None

    @property
    def observability_enabled(self) -> bool:
        return self._policy.observability_enabled

    @property
    def projects(self) -> tuple[str, ...]:
        if self._mode == "dev":
            return list_dev_projects(self._workspace_root)
        return list_projects(self._workspace_root)

    @property
    def model(self) -> str:
        return self._model

    def set_model(self, model: str) -> None:
        cleaned = model.strip()
        if cleaned:
            self._model = cleaned

    def validate(self) -> None:
        if not self._workspace_root.is_dir():
            raise ValueError(f"Qoder workspace does not exist: {self._workspace_root}")
        self._validate_mcp_config()
        self._auth()

    async def open_session(self, resume_session_id: str | None = None) -> QoderSessionHandle:
        self._validate_mcp_config()
        policy = self._policy
        if self._mode == "dev":
            system_prompt = _DEV_SYSTEM_PROMPT + _build_rules_prompt(
                self._user_qoder_dir / "rules"
            )
            if DMS_SERVER in policy.mcp_server_names:
                system_prompt += _DMS_PROMPT
            if policy.observability_enabled:
                system_prompt += _OBSERVABILITY_PROMPT
            if projects := self.projects:
                system_prompt += _DEV_PROJECT_SCOPE_PROMPT.format(
                    projects="、".join(projects)
                )
        else:
            system_prompt = _SYSTEM_PROMPT
            if DMS_SERVER not in policy.mcp_server_names:
                # No database server: drop the DB verification guidance.
                system_prompt = system_prompt.replace(
                    "并可在需要核实 ERP 业务数据时调用 aliyun-dms-mcp-server 的只读查询工具。",
                    "重点排查代码实现与逻辑。"
                ).replace("数据库只允许查询，禁止修改数据或结构、事务、锁、存储过程、文件操作及绕过权限校验。\n", "")
            else:
                system_prompt += _DMS_PROMPT
            if policy.observability_enabled:
                system_prompt += _OBSERVABILITY_PROMPT
            if projects := self.projects:
                system_prompt += _PROJECT_SCOPE_PROMPT.format(projects="、".join(projects))
            if self._domain_knowledge_enabled:
                system_prompt += _DOMAIN_KNOWLEDGE_PROMPT
        options = QoderAgentOptions(
            auth=self._auth(),
            cwd=self._workspace_root,
            model=self._model,
            tools=None if self._mode == "dev" else list(policy.allowed_tools),
            allowed_tools=["Skill"] if self._mode == "dev" else list(policy.allowed_tools),
            disallowed_tools=[] if self._mode == "dev" else list(BLOCKED_TOOLS),
            permission_mode="dontAsk",
            system_prompt=system_prompt,
            resume=resume_session_id,
            setting_sources=["user"] if self._mode == "dev" else [],
            strict_mcp_config=self._mode == "dev",
            mcp_servers=(
                self._mcp_servers_payload
                if self._mode == "dev" and self._mcp_servers_payload is not None
                else self._mcp_servers_config_path or {}
            ),
            allowed_mcp_server_names=list(policy.mcp_server_names),
            env={
                "QODER_AGENT_SDK_ENTRYPOINT": None,
                "QODER_AGENT_SDK_VERSION": None,
            },
            hooks={"PreToolUse": [HookMatcher(hooks=[self._validate_pre_tool_use])]},
            can_use_tool=self._can_use_tool,
            stderr=self._ignore_stderr,
            max_buffer_size=self._max_buffer_size,
        )
        client = QoderSDKClient(options=options)
        handle = QoderSessionHandle(
            client=client,
            session_id=resume_session_id,
            include_current_time=policy.observability_enabled,
        )
        await handle.start()
        return handle

    def _validate_mcp_config(self) -> None:
        config_path = self._mcp_servers_config_path
        if config_path is None:
            raise ValueError("Qoder MCP config path is required")
        if not config_path.is_file():
            raise ValueError(f"Qoder MCP config does not exist: {config_path}")
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Qoder MCP config is not valid JSON: {config_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Qoder MCP config must be a JSON object")
        servers = payload.get("mcpServers")
        if not isinstance(servers, dict):
            raise ValueError("Qoder MCP config must contain an mcpServers object")
        server_names = []
        if DMS_SERVER in servers:
            dms = servers[DMS_SERVER]
            if not isinstance(dms, dict) or not any(
                isinstance(dms.get(key), str) and dms[key].strip()
                for key in ("url", "command")
            ):
                raise ValueError(
                    f"Qoder MCP config {DMS_SERVER} must be an object with url or command"
                )
            server_names.append(DMS_SERVER)
        for obs_name in OBSERVABILITY_SERVERS:
            if obs_name in servers:
                observability = servers[obs_name]
                if not isinstance(observability, dict) or not any(
                    isinstance(observability.get(key), str) and observability[key].strip()
                    for key in ("url", "command")
                ):
                    raise ValueError(
                        f"Qoder MCP config {obs_name} must be an object with url or command"
                    )
                server_names.append(obs_name)
        if self._mode == "dev":
            servers = self._merge_user_mcp_servers(servers)
            server_names.extend(
                name
                for name in servers
                if name in (*MYSQL_SERVERS, CLAUDE_MEM_SERVER)
                and name not in server_names
            )
            self._mcp_servers_payload = servers
        if not server_names:
            raise ValueError("Qoder MCP config must define at least one valid server")
        policy_cls = DevToolPolicy if self._mode == "dev" else ToolPolicy
        self._policy = policy_cls(self._workspace_root, server_names)

    def _merge_user_mcp_servers(self, servers: dict[str, Any]) -> dict[str, Any]:
        """Merge user-level MCP servers (~/.qoder/mcp.json) for IDE parity.

        Agent-configured servers win on name conflicts; read failures degrade to
        the agent-only config instead of blocking the session.
        """
        merged = dict(servers)
        user_config = self._user_qoder_dir / "mcp.json"
        if not user_config.is_file():
            return merged
        try:
            payload = json.loads(user_config.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            logger.warning("Failed to read user MCP config: %s", user_config)
            return merged
        user_servers = payload.get("mcpServers") if isinstance(payload, dict) else None
        if not isinstance(user_servers, dict):
            return merged
        for name, config in user_servers.items():
            if name not in merged and isinstance(config, dict):
                merged[name] = config
        return merged

    async def _can_use_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        _context: Any,
    ) -> PermissionResultAllow | PermissionResultDeny:
        denial = self._tool_denial_reason(tool_name, tool_input)
        if denial:
            return PermissionResultDeny(message=denial, interrupt=False)
        return PermissionResultAllow()

    async def _validate_pre_tool_use(
        self,
        hook_input: dict[str, Any],
        _tool_use_id: str | None,
        _context: Any,
    ) -> dict[str, Any]:
        denial = self._tool_denial_reason(
            str(hook_input.get("tool_name", "")),
            hook_input.get("tool_input", {}),
        )
        decision = "deny" if denial else "allow"
        output: dict[str, Any] = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
            }
        }
        if denial:
            output["hookSpecificOutput"]["permissionDecisionReason"] = denial
        return output

    def _tool_denial_reason(self, tool_name: str, tool_input: Any) -> str | None:
        return self._policy.denial_reason(tool_name, tool_input)

    def _auth(self) -> Any:
        if self._personal_access_token:
            return access_token(self._personal_access_token)
        return access_token_from_env()

    @staticmethod
    def _ignore_stderr(line: str) -> None:
        clean = line.strip()
        if clean:
            logger.warning("[Qoder CLI STDERR] %s", clean)
