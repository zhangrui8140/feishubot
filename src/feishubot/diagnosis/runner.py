from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from feishubot.ai.orchestrator.chat_commands import (
    CANCEL_COMMANDS,
    HELP_COMMANDS,
    RESET_COMMANDS,
    STATUS_COMMANDS,
    _parse_model_command,
    build_progress_text,
    dev_status_text,
    handle_command,
)
from feishubot.diagnosis.controller import DiagnosisController
from feishubot.diagnosis.models import DiagnosisStatus
from feishubot.diagnosis.runtime import DiagnosisRuntime, record_text_feedback
from feishubot.qoder_agent import (
    AgentBusyError,
    AgentCancelledError,
    AgentPermissionError,
    AgentRequest,
    AgentTimeoutError,
    AgentUnavailableError,
    DuplicateMessageError,
    ImageInput,
    QoderAgentService,
    SessionKey,
)

logger = logging.getLogger(__name__)

_SUPPLEMENT_PREFIX = "【用户补充】"
_IMAGE_ONLY_PROMPT = (
    "用户发送了截图，请识别其中的报错、接口名、单号或页面信息，"
    "并结合本地代码、ERP 数据或日志分析问题。"
)

# User-facing wording per mode; the task flow itself is shared.
_MODE_TEXTS = {
    "readonly": {
        "busy": "当前会话已有较多分析任务，请稍后再试。",
        "timeout": "本次代码分析已超时并停止，请缩小问题范围后重试。",
        "cancelled": "当前代码分析已取消。",
        "permission": "请求包含只读范围之外的操作，已拒绝执行。",
        "unavailable": "Qoder 代码分析暂时不可用，请稍后重试。请求ID：{request}",
        "failed": "代码分析失败，请稍后重试。请求ID：{request}",
        "interrupt_notice": "已收到补充信息，正在中断当前分析……",
        "interrupt_progress": "已结合补充信息重新排查，请稍候……",
        "queued_progress": "已收到新问题，将在当前分析完成后单独开始排查……",
        "followup_progress": "已关联上一轮排查结论，正在结合新信息深入排查……",
    },
    "dev": {
        "busy": "并行任务已达上限，请稍后再试。",
        "timeout": "本次任务已超时并停止，请缩小范围后重试。",
        "cancelled": "当前任务已取消。",
        "permission": "请求包含允许范围之外的操作，已拒绝执行。",
        "unavailable": "Qoder 开发任务暂时不可用，请稍后重试。请求ID：{request}",
        "failed": "开发任务失败，请稍后重试。请求ID：{request}",
        "interrupt_notice": "已收到补充信息，正在中断当前任务……",
        "interrupt_progress": "已结合补充信息重新执行，请稍候……",
        "queued_progress": "已收到新问题，将在当前任务完成后单独开始……",
        "followup_progress": "已关联上一轮任务结论，正在结合新信息继续……",
    },
}


async def handle_qoder_diagnosis(
    *,
    message: Any,
    text: str,
    user_open_id: str | None,
    image_keys: list[str] | None,
    session_key: SessionKey,
    service: QoderAgentService,
    runtime: DiagnosisRuntime | None,
    download_images: Callable[[str, list[str]], Awaitable[tuple[ImageInput, ...]]],
    respond: Callable[[Any, str], Awaitable[str | None]],
) -> None:
    message_id = str(message.message_id)
    mode = getattr(service, "mode", "readonly")
    is_dev = mode == "dev"
    texts = _MODE_TEXTS["dev" if is_dev else "readonly"]
    base_key = session_key
    reply_ctx = (
        runtime.context_from_reply(message, user_open_id) if runtime is not None else None
    )

    if not image_keys:
        command = text.strip().lower()
        if is_dev:
            reply = await _handle_dev_command(service, runtime, base_key, reply_ctx, command)
            if reply is not None:
                await respond(message, reply)
                return
        else:
            active = runtime.active(session_key.value) if runtime else None
            if active is not None and command in CANCEL_COMMANDS:
                await active.supplement("正在终止当前分析")
            reply = await handle_command(service, session_key.value, command)
            if reply is not None:
                if active is None or reply != "正在终止当前分析。":
                    await respond(message, reply)
                return

    if not image_keys and runtime is not None:
        outcome = record_text_feedback(
            runtime.store, message, text, user_open_id, runtime
        )
        if outcome is not None:
            await respond(message, outcome.message)
            return

    try:
        images = await _images(message_id, image_keys, download_images)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to download Feishu image, request=%s", message_id[-8:])
        await respond(message, "图片下载失败或超过大小限制，请压缩后重发，或用文字描述问题。")
        return

    # Dev multi-task: every new task gets its own session so tasks run in parallel.
    # Quote-reply or an explicit supplement targets an existing task instead.
    task_key = (
        _resolve_dev_task_key(service, base_key, message_id, text, image_keys, reply_ctx)
        if is_dev
        else base_key
    )
    active = runtime.active(task_key.value) if runtime is not None else None

    reuse = None
    if active is not None and _is_explicit_supplement(text):
        # In-flight task + explicit supplement: interrupt the old run and resubmit merged.
        logger.info("Task %s active, interrupting for supplement", task_key.value)
        old_question = active.context.question
        stage_text = (
            active.context.stage_text
            if (active.context.stage_text and active.context.stage_text not in ("正在分析", "正在排查"))
            else ""
        )
        active.mark_interrupted_for_supplement()
        await active.supplement(texts["interrupt_notice"])

        await service.cancel(task_key.value)
        for _ in range(50):
            if not service.status(task_key.value).running:
                break
            await asyncio.sleep(0.1)

        effective_prompt = _build_in_flight_supplement_prompt(
            old_question=old_question,
            stage_text=stage_text,
            new_input=text,
        )
        reuse = active.context
        active.reset_for_continuation(effective_prompt)
        progress_text = texts["interrupt_progress"]
    elif active is not None:
        # A brand-new, unrelated question arrived while a task is in flight.
        # Keep the running task untouched and queue this as its own diagnosis
        # (the service serialises same-session runs, so it starts afterwards).
        logger.info(
            "New question during in-flight task for session %s; queueing separately",
            task_key.value,
        )
        effective_prompt = text or _IMAGE_ONLY_PROMPT
        reuse = None
        progress_text = texts["queued_progress"]
    elif not is_dev and not images and service.status(task_key.value).running:
        # Readonly fallback for plain chat / sessions without an active controller.
        await respond(message, "已收到补充，将在当前分析中参考。")
        try:
            if await service.supplement(
                AgentRequest(
                    message_id=message_id,
                    session_key=task_key,
                    prompt=text or _IMAGE_ONLY_PROMPT,
                    images=images,
                ),
                f"{_SUPPLEMENT_PREFIX}{text}",
            ):
                return
        except DuplicateMessageError:
            logger.info("Ignored duplicate Feishu message, request=%s", message_id[-8:])
            return
        effective_prompt = text or _IMAGE_ONLY_PROMPT
        progress_text = "上一轮分析已结束，将结合补充内容继续分析，请稍候……"
    else:
        # Completed task followup / quote reply, or a brand-new task.
        reply_context = reply_ctx
        if not is_dev and reply_context is None and runtime is not None:
            latest = runtime.latest_context(task_key.value)
            if latest is not None and _is_followup_intent(text):
                reply_context = latest

        if reply_context is not None:
            logger.info(
                "Session %s continuing from reply_context %s",
                task_key.value,
                reply_context.diagnosis_id,
            )
            effective_prompt = _build_followup_prompt(
                previous_question=reply_context.question,
                previous_result=reply_context.result_text or reply_context.stage_text or "",
                new_input=text,
            )
            user_id = user_open_id or task_key.user_open_id
            reuse = reply_context if reply_context.owner_open_id == user_id else None
            progress_text = texts["followup_progress"]
        else:
            effective_prompt = text or _IMAGE_ONLY_PROMPT
            reuse = None
            progress_text = build_progress_text(
                text,
                service.projects,
                observability_enabled=service.observability_enabled,
                mode=mode,
            )

    request = AgentRequest(
        message_id=message_id,
        session_key=task_key,
        prompt=effective_prompt,
        images=images,
    )

    controller = _create_controller(
        runtime,
        message,
        effective_prompt,
        user_open_id,
        task_key,
        service.projects,
        respond,
        reuse,
    )

    async def accepted() -> None:
        if controller is not None:
            await controller.start(progress_text)
        else:
            await respond(message, progress_text)

    async def progress(event: Any) -> None:
        if controller is not None:
            await controller.progress(event)
        elif getattr(event, "kind", None) == "assistant_text" and event.text:
            await respond(message, event.text)

    async def finish(status: DiagnosisStatus, result_text: str) -> None:
        if controller is not None:
            await controller.finish(status, result_text)
        else:
            await respond(message, result_text)

    try:
        try:
            result = await service.run(request, on_accepted=accepted, on_progress=progress)
        except DuplicateMessageError:
            logger.info("Ignored duplicate Feishu message, request=%s", message_id[-8:])
            return
        except AgentBusyError:
            await finish(DiagnosisStatus.FAILED, texts["busy"])
            return
        except AgentTimeoutError:
            await finish(DiagnosisStatus.TIMED_OUT, texts["timeout"])
            return
        except AgentCancelledError:
            if controller is not None and controller.is_interrupted_for_supplement:
                logger.info("Run interrupted for supplement; keeping context for resubmission")
                return
            await finish(DiagnosisStatus.CANCELLED, texts["cancelled"])
            return
        except AgentPermissionError:
            await finish(DiagnosisStatus.PERMISSION_DENIED, texts["permission"])
            return
        except AgentUnavailableError:
            logger.exception("Qoder Agent request failed, request=%s", message_id[-8:])
            await finish(
                DiagnosisStatus.FAILED,
                texts["unavailable"].format(request=message_id[-8:]),
            )
            return
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected Qoder Agent failure, request=%s", message_id[-8:])
            await finish(
                DiagnosisStatus.FAILED,
                texts["failed"].format(request=message_id[-8:]),
            )
            return

        status = (
            DiagnosisStatus.WAITING_SUPPLEMENT
            if result.needs_input
            else DiagnosisStatus.COMPLETED
        )
        result_text = result.text
        if result.observability_unauthorized:
            result_text += (
                "\n\n本次未能查询日志与链路：observability 未授权或授权已过期，"
                "结论仅基于代码与 ERP 数据；可在本地管理页对 observability 重新授权。"
            )
        if controller is not None:
            await controller.finish(
                status,
                result_text,
                duration_ms=result.duration_ms,
                tool_names=result.tool_names,
            )
        else:
            await respond(message, result_text)
    finally:
        if runtime is not None and controller is not None:
            if not controller.is_interrupted_for_supplement:
                runtime.release(task_key.value, controller)


def _resolve_dev_task_key(
    service: QoderAgentService,
    base_key: SessionKey,
    message_id: str,
    text: str,
    image_keys: list[str] | None,
    reply_ctx: Any,
) -> SessionKey:
    if reply_ctx is not None:
        return SessionKey.parse(reply_ctx.session_key)
    if not image_keys and _is_explicit_supplement(text):
        target = service.latest_running_key(base_key.value)
        if target is not None:
            return SessionKey.parse(target)
    return base_key.for_task(message_id)


async def _handle_dev_command(
    service: QoderAgentService,
    runtime: DiagnosisRuntime | None,
    base_key: SessionKey,
    reply_ctx: Any,
    command: str,
) -> str | None:
    if command in HELP_COMMANDS or command in RESET_COMMANDS or _parse_model_command(command)[0]:
        return await handle_command(service, base_key.value, command)
    if command in STATUS_COMMANDS:
        return dev_status_text(service, base_key.value)
    if command in CANCEL_COMMANDS:
        target = (
            reply_ctx.session_key
            if reply_ctx is not None
            else service.latest_running_key(base_key.value)
        )
        if target is None:
            return "当前没有正在进行的任务。"
        if runtime is not None:
            active = runtime.active(target)
            if active is not None:
                await active.supplement("正在终止当前任务")
        cancelled = await service.cancel(target)
        return "正在终止当前任务。" if cancelled else "该任务已结束。"
    return None


async def _images(
    message_id: str,
    image_keys: list[str] | None,
    download_images: Callable[[str, list[str]], Awaitable[tuple[ImageInput, ...]]],
) -> tuple[ImageInput, ...]:
    if not image_keys:
        return ()
    return await download_images(message_id, image_keys)


def _create_controller(
    runtime: DiagnosisRuntime | None,
    message: Any,
    question: str,
    user_open_id: str | None,
    session_key: SessionKey,
    projects: tuple[str, ...],
    respond: Callable[[Any, str], Awaitable[str | None]],
    reuse: Any,
) -> DiagnosisController | None:
    if runtime is None:
        return None

    async def fallback(text: str) -> str | None:
        return await respond(message, text)

    return runtime.create(
        message=message,
        question=question,
        owner_open_id=user_open_id or session_key.user_open_id,
        session_key=session_key.value,
        available_projects=projects,
        fallback=fallback,
        reuse=reuse,
    )


def _build_in_flight_supplement_prompt(
    *,
    old_question: str,
    stage_text: str,
    new_input: str,
) -> str:
    parts: list[str] = []
    if old_question:
        parts.extend(["【原排查目标】", old_question.strip(), ""])
    if stage_text:
        parts.extend(["【中断前已掌握的线索/阶段进展】", stage_text.strip(), ""])
    parts.extend([
        "【用户最新补充/更正信息】",
        new_input.strip() or "（用户补充了截图，请结合原排查目标分析）",
        "",
        "【排查要求】",
        "用户在排查中途补充了上述新信息，已中断前序分析。请结合原排查目标以及用户补充的最新信息，调整思路继续排查并给出最终结论。",
    ])
    return "\n".join(parts)


def _build_followup_prompt(
    *,
    previous_question: str,
    previous_result: str,
    new_input: str,
) -> str:
    parts: list[str] = []
    if previous_question:
        parts.extend(["【前序排查背景与问题】", previous_question.strip(), ""])
    if previous_result:
        parts.extend(["【前序诊断已得出的结论与证据】", previous_result.strip(), ""])
    parts.extend([
        "【用户最新追问/补充排查指令】",
        new_input.strip() or "（用户补充了截图，请结合前序结果深入分析）",
        "",
        "【排查要求】",
        "请在充分已知前序排查结论和证据的基础上，结合用户的最新指令与补充信息，继续深入排查并给出最终结论。",
    ])
    return "\n".join(parts)


_FOLLOWUP_KEYWORDS = (
    "继续",
    "排查",
    "深入",
    "接着",
    "再查",
    "后续",
    "还有",
    "原因",
    "进一步",
    "看看",
    "查下",
    "为什么",
    "分析",
)


def _is_followup_intent(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in _FOLLOWUP_KEYWORDS)


_SUPPLEMENT_SEPARATORS = " :：,，、"


def _is_explicit_supplement(text: str) -> bool:
    """Only「补充」alone or followed by a separator merges into an in-flight task.

    「补充单元测试」opens a new task instead of supplementing the running one.
    """
    normalized = text.strip()
    if normalized == "补充":
        return True
    return (
        normalized.startswith("补充")
        and len(normalized) > 2
        and normalized[2] in _SUPPLEMENT_SEPARATORS
    )