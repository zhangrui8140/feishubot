# Handoff：feishubot dev 机器人（飞书直发开发任务）

日期：2026-11-24
交接目标：让新会话无缝接手「飞书私聊机器人 = Qoder IDE 全能力开发代理」这条线的后续工作。

## 1. 背景与总目标

用户要把"在 Qoder IDE 开会话写任务"改为**通过飞书私聊机器人直接发开发任务**。已确认决策：
- 私聊触发；全自动执行含 git push（仅 feature 分支，禁 prod/master/main、禁 force）。
- 用独立飞书应用 `cli_aab640f195365bb4` 跑独立 dev 进程，与现有只读分析机器人（`cli_aa4368d29db85bdd`）完全隔离，readonly 路径零改动。
- 凭证已验证有效并写入 `.env.dev`（App Secret 属敏感信息，此处不记录；直接读该文件即可）。

## 2. 项目与关键路径

- 仓库：`D:\workspace\projects\qoderwork\agent\feishubot`（Python 3.12 venv，uv 管理，qoder_agent_sdk 1.0.15 后端，lark-oapi websocket 长连接）。**agent 目录已于 2026-11-24 由用户加入 IDE 工作区，仓库内文件可直接用 Write/SearchReplace 编辑，不再需要提权补丁中转。**
- 环境文件：`D:\workspace\projects\qoderwork\agent\feishubot\.env.dev`（QODER_AGENT_MODE=dev、QODER_WORKSPACE_ROOT=D:\workspace\projects\qoderwork\dev、QODER_MODEL=GLM-5.3-Flash、并发 4/8）
- 启动脚本：`start-feishubot-dev.cmd`；实际运行方式：`$env:FEISHUBOT_ENV_FILE="...\.env.dev"; .venv\Scripts\python.exe -c "from feishubot.cli import main; main(['events','--no-fallback-webhook','--log-level','INFO'])"`（后台）
- 核心源码（均已改完并验证）：
  - `src/feishubot/qoder_agent/sdk_adapter.py` — dev 全能力 options、规则注入 `_build_rules_prompt`、用户 MCP 合并 `_merge_user_mcp_servers`、model property/set_model
  - `src/feishubot/qoder_agent/tool_policy.py` — DevToolPolicy 守卫（注意此文件是 CRLF）
  - `src/feishubot/qoder_agent/service.py` — 模型持久化（session_store_dir/model.json）、`task_statuses()`/`latest_running_key()` 跨会话查询
  - `src/feishubot/qoder_agent/models.py` — `SessionKey.for_task(message_id)` / `parse()`
  - `src/feishubot/diagnosis/runner.py` — `_handle_dev_task` 多任务路由（dev 专用）
  - `src/feishubot/ai/orchestrator/chat_commands.py` — 「模型」「状态」「帮助」等命令
  - `src/feishubot/ai/orchestrator/feishu_events.py` — dev 私聊结果引用回复原消息
- 测试：`tests/qoder_agent/test_dev_full_capability.py`（13 个）、`tests/diagnosis/test_dev_multitask.py`（9 个）；全套 **440 通过**。
- 历史补丁/冒烟脚本（提权时代的产物，仅作参考，不必重跑）：`d:\workspace\projects\qoderwork\dev\rui\.tmp-build\feishubot-dev\`（patch_dev_full.py、patch_multitask_a/b.py、smoke_skills.py 等）
- 长期记忆：`87a9237d-95c4-425d-84bc-f19ac9f69b2e`「feishubot dev 全能力与多任务并行配置」（已含全部机制摘要，先 SearchMemory fetch 它可省大量探索）

## 3. 已完成的三阶段改造（勿重做）

1. **全能力对齐**：dev options `tools=None / allowed_tools=["Skill"] / setting_sources=["user"] / strict_mcp_config=True`；合并 `~/.qoder/mcp.json`（mysql-erp-dev / mysql-scm-dev / bi / claude-mem + agent 侧 DMS / observability，共 6 个）；dev prompt 注入 `~/.qoder/rules` 的 00-rule-index.md + coding-standard.md 全文及项目级 skills 指引。冒烟 get_server_info：**98 skills / 6 MCP / 137 commands**。
   - 曾补 24 个 junction：`~/.qoder/skills\<名>` → `~/.agents/skills\<名>`（CLI 用户级只发现前者）。守卫 `_is_in_dev_read_roots` 已做 resolved+absolute 双路径匹配兼容 junction 穿透。
2. **模型可选**：飞书「模型」查看 /「模型 <名称>」或「/model <名称>」切换；持久化 model.json，重启恢复；丢弃空闲会话 handle 使下一任务生效。模型名无校验列表（CLI 不在 PATH，无法枚举），无效名会在会话报错回传。
3. **多任务并行**：每条新任务独立 SessionKey 并行跑；引用回复定向补充/终止某任务；「补充」并入最近任务；「状态」列并行任务清单；并发 max_concurrent_runs=4 / max_live_sessions=8。结果引用回复原消息以对应任务。

守卫保留（不可放松，除非用户明确要求）：写限 dev 工作区、禁 .git/密钥文件、Bash 禁破坏命令、push 禁保护分支/force、全部数据库 MCP 只读（mysql execute_sql 走 sql_guard）、rules/skills 目录只读；非 MCP agent 工具（Skill/Task/WebSearch 等）默认放行。

## 4. 运行状态

- dev 机器人已用最新代码后台重启并 connected（2026-11-24 09:39:18，会话 terminal 1；若终端已失效需按 §2 启动方式重启）。
- 进程管理注意：venv python.exe 是 launcher，实际解释器是 uv cpython 子进程（同 CreationDate 父子树），Stop-Process 要杀整棵树；用 `Get-CimInstance Win32_Process | Where CommandLine -like '*feishubot*'` 定位。
- 用户尚未回报飞书端到端实测结果（多任务/模型切换）。若用户发来测试反馈，先 GetTerminalOutput 核对事件与会话日志。

## 5. 未决事项（下一步的真正焦点）

用户最新问题：**"如何教当前机器人调用的 agent 一些工作流，比如提交代码后如何发布、点击任务完成等"**。上一会话已答复教学入口（Skills 推荐 / Rules / 项目 AGENTS.md / dev 系统提示），并指出关键能力缺口：

- 机器人 MCP 配置里**没有 browser-use**（IDE 侧有，bot 侧没有）→ agent 不能点网页按钮，只能 Bash 调 CLI/API。
- 「任务完成」若是飞书待办 → lark-task skill + lark-cli 可直接做；若是坚果平台 → quality-case / nut-auto-test 走 API 可做；若是纯网页按钮 → 需补 browser-use MCP 或找 API。
- 已建议新建工作流 skill `dev-delivery-flow`：commit+push feature → 提测（nut/quality-case）→ 发布 → 任务完成（lark-task），每步写触发条件、命令/API、成功判据、失败回退。

**正等用户回答两个问题**（新会话应先确认是否已有答案，没有则重新问）：
1. 「发布」怎么触发？哪个平台（Jenkins / 妙搭 / 内部发布系统 / 其它）？网页点按钮还是有 API/CLI/流水线？
2. 「任务完成」指飞书待办、坚果测试平台任务，还是别的系统按钮？

拿到答案后：判断 API 可达性 → 用 create-skill 写 `dev-delivery-flow`（放 `~/.qoder/skills\`，该目录在工作区外，写入仍需提权，见 §6）→ 若需浏览器操作则给 bot 的 MCP 合并配置补 browser-use（改 `_merge_user_mcp_servers` 数据源或 agent mcp.json，并同步 DevToolPolicy 白名单——browser-use 属写操作类工具，守卫策略需与用户确认）→ 冒烟验证 skill 被发现 → 飞书实测。

## 6. 操作手法备忘（本环境已验证）

- **仓库内写入**：agent 目录已加入工作区（2026-11-24），feishubot 源码/测试/文档可直接 Write/SearchReplace 编辑。
- **工作区外写入**（`~/.qoder/skills`、`~/.qoder/rules`、`~/.agents` 等用户目录仍需）：在工作区 `rui\.tmp-build\` 写 python 补丁脚本 → Bash `required_permissions='all'` 提权执行 → 精确锚点替换（count 必须=1）→ EOL 自适应（tool_policy.py 是 CRLF，其余 LF）。新建文件用提权 python shutil.copy 或 Copy-Item。
- **PowerShell 5.1**：不支持 `&&`，用 `;`；测试命令 `.venv\Scripts\python.exe -m pytest -q`（在 feishubot 目录下；Bash 沙箱若拦截仓库外执行则提权）。
- **验证顺序**：compileall → pytest 全套 → 冒烟（smoke_skills.py 模式：open_session 后 get_server_info）→ 杀旧进程树 → 后台重启 → GetTerminalOutput 确认 connected。
- 用户规则库要求：新规则必须固化到 `C:\Users\zhang\.qoder\rules\`（不能只写记忆）；skills 属用户资产，修改前按 rule-management 备份约定处理。

## 7. Suggested skills（新会话按需用 Skill 工具调用）

| 场景 | skill |
|---|---|
| 编写 dev-delivery-flow 工作流 skill（下一步主任务） | `create-skill`（骨架）+ `writing-for-agents`（写法规范） |
| 任务完成=飞书待办的实现细节 | `lark-task` |
| 提测/提交执行结果=坚果平台 | `quality-case`、`nut-auto-test` |
| 发布=妙搭平台时 | `lark-apps` |
| 需要给 bot 补浏览器能力时评估 MCP 封装 | `mcp-builder`（仅当决定自建；优先复用 IDE 现成 browser-use 配置） |
| 接手时恢复跨会话上下文 | `claude-mem-handoff`（查/存进度）；另先 SearchMemory fetch 记忆 87a9237d |
| 用户改口要走完整开发工序时 | `dev-flow` |

## 8. 敏感信息说明

本文档不含 App Secret、Personal Access Token、数据库凭证。它们分别在 `.env.dev`、`~/.qoder/mcp.json`、`D:\workspace\projects\qoderwork\agent\mcp.json` 中，需要时直接读文件，勿把值写进任何交付文档或日志。