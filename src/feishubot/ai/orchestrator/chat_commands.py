from __future__ import annotations

from feishubot.qoder_agent.projects import mentioned_dev_projects, mentioned_projects
from feishubot.qoder_agent.service import QoderAgentService

CANCEL_COMMANDS = {"/cancel", "/stop", "取消", "停止", "终止", "停止分析", "终止分析", "取消分析"}
HELP_COMMANDS = {"/help", "help", "帮助", "指令", "命令"}
STATUS_COMMANDS = {"/status", "状态", "进度"}
RESET_COMMANDS = {"/new", "/reset", "新会话", "新对话", "重置", "重置会话"}
HELP_TEXT = "\n".join(
    [
        "可用指令：",
        "· 分析进行中发送「补充：…」：并入当前分析（如「补充：erp-mall-prod」）；发送新问题则排队、单独成卡回答",
        "· 终止 / 停止 / 取消：停止当前分析",
        "· 状态：查看当前是否有分析在进行",
        "· 新会话 / 重置：清空上下文，下一个问题重新开始",
        "· 帮助：查看本说明",
    ]
)
COMMAND_HINT = "分析中发送「补充：…」可并入当前分析，新问题将排队单独回答；「终止」停止，「帮助」查看全部指令。"
DEV_HELP_TEXT = "\n".join(
    [
        "可用指令（开发任务模式 · 支持多任务并行）：",
        "· 直接发送任务：如「erp-purchase-dev-大仓端堆活动 修复xxx校验问题」；每条新任务独立并行执行（最多 4 个），完成后提交并推送 feature 分支（禁推 prod/master/main）",
        "· 引用回复某条任务消息：对该任务补充或追问；任务进行中发送「补充：…」并入最近一个任务",
        "· 终止 / 停止 / 取消：停止最近一个任务；引用回复某条任务再发「终止」可单独停止它",
        "· 状态：查看当前并行任务列表",
        "· 模型：查看当前模型；「模型 <名称>」切换（如「模型 GLM-5.3」），下一个任务生效",
        "· 帮助：查看本说明",
    ]
)
DEV_COMMAND_HINT = "多任务并行：新任务独立执行；「补充：…」并入最近任务，引用回复可定向某任务；「终止」停止最近任务，「状态」查看列表。"


def dev_status_text(service: QoderAgentService, scope_prefix: str) -> str:
    statuses = service.task_statuses(scope_prefix)
    running = [(key, status) for key, status in statuses if status.running]
    if not running:
        return "当前没有正在进行的任务。直接发送新任务即可并行执行。"
    lines = [f"当前有 {len(running)} 个任务并行进行中："]
    for index, (_key, status) in enumerate(running, 1):
        queued = f"，另有 {status.queued} 个排队" if status.queued else ""
        lines.append(f"· 任务{index}{queued}")
    lines.append("「终止」停止最近一个任务；引用回复某条任务可单独补充或终止它。")
    return "\n".join(lines)


def build_progress_text(
    text: str,
    projects: tuple[str, ...],
    *,
    observability_enabled: bool,
    mode: str = "readonly",
) -> str:
    if mode == "dev":
        if selected := mentioned_dev_projects(text, projects):
            lines = [
                f"已收到，正在 dev 工作区 {'、'.join(selected)} 执行开发任务，"
                "完成后回传改动清单与推送结果……"
            ]
        else:
            lines = ["已收到，正在 dev 工作区执行开发任务，请稍候……"]
            if projects:
                lines.append("可开发项目（dev 工作区）：")
                lines.extend(f"· {project}" for project in projects)
                lines.append(f"任务里带上项目名可缩小范围，例如「{projects[0]} 修复xxx」。")
        lines.append(DEV_COMMAND_HINT)
        return "\n".join(lines)
    scope = "代码、ERP 数据和日志" if observability_enabled else "代码和 ERP 数据"
    if selected := mentioned_projects(text, projects):
        lines = [f"已收到，正在分析 prod 环境 {'、'.join(selected)} 的{scope}，请稍候……"]
    else:
        lines = [f"已收到，正在分析 prod 环境的{scope}，请稍候……"]
        if projects:
            lines.append("可分析项目（仅 prod 环境）：")
            lines.extend(f"· {project}" for project in projects)
            lines.append(f"提问时带上项目名可缩小范围，例如「{projects[0]} 保存报错」。")
    lines.append(COMMAND_HINT)
    return "\n".join(lines)


def _parse_model_command(command: str) -> tuple[bool, str]:
    stripped = command.strip()
    if stripped == "模型":
        return True, ""
    if stripped.startswith(("模型 ", "模型:", "模型：")):
        return True, stripped[2:].lstrip(" :：").strip()
    lowered = stripped.lower()
    if lowered == "/model":
        return True, ""
    if lowered.startswith(("/model ", "/model:", "/model：")):
        return True, stripped[7:].lstrip(" :：").strip()
    return False, ""


async def handle_command(service: QoderAgentService, session_key: str, command: str) -> str | None:
    is_model, model_arg = _parse_model_command(command)
    if is_model:
        current = getattr(service, "model", None)
        current_text = current if isinstance(current, str) and current else "未知"
        if not model_arg:
            return (
                f"当前模型：{current_text}\n"
                "发送「模型 <名称>」切换，例如「模型 GLM-5.3」；切换后下一个任务生效。"
            )
        setter = getattr(service, "set_model", None)
        if not callable(setter):
            return "当前服务不支持切换模型。"
        try:
            new_model = await setter(model_arg)
        except ValueError as exc:
            return f"切换失败：{exc}"
        return f"模型已切换为 {new_model}，下一个任务生效。"
    if command in CANCEL_COMMANDS:
        cancelled = await service.cancel(session_key)
        return "正在终止当前分析。" if cancelled else "当前会话没有正在运行的分析任务。"
    if command in HELP_COMMANDS:
        mode = getattr(service, "mode", "readonly")
        return DEV_HELP_TEXT if mode == "dev" else HELP_TEXT
    if command in STATUS_COMMANDS:
        status = service.status(session_key)
        if not status.running:
            return "当前没有正在进行的分析。"
        queued = f"，另有 {status.queued} 个问题排队" if status.queued else ""
        return f"当前有分析正在进行{queued}。发送「补充：…」可并入当前分析，新问题将排队单独回答；「终止」停止。"
    if command in RESET_COMMANDS:
        if await service.reset(session_key):
            return "已开启新会话，之前的对话上下文已清空。"
        return "当前分析进行中，请先发送「终止」再重置会话。"
    return None
