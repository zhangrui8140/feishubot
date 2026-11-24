from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

import lark_oapi as lark

from feishubot.ai.orchestrator.agent_loop import AgentLoop
from feishubot.ai.orchestrator.chat_commands import HELP_TEXT
from feishubot.ai.orchestrator.feishu_message import (
    extract_image_message,
    extract_text,
    forward_payload,
    message_from_payload,
    strip_mentions,
)
from feishubot.ai.orchestrator.handoff import Handoff, HandoffSupervisor, ws_client_controls
from feishubot.ai.prompts import build_system_prompt
from feishubot.ai.providers import create_active_provider
from feishubot.ai.tools import ToolRuntime
from feishubot.channel import Channel, create_default_channel
from feishubot.config import settings
from feishubot.diagnosis import DiagnosisRuntime, create_store
from feishubot.diagnosis.runner import handle_qoder_diagnosis
from feishubot.qoder_agent import (
    AgentUnavailableError,
    ImageInput,
    QoderAgentService,
    QoderServiceConfig,
    SessionKey,
)

logger = logging.getLogger(__name__)

_MAX_IMAGES_PER_MESSAGE = 4

channel_client: Channel | None = None
_event_worker_loop: asyncio.AbstractEventLoop | None = None
_event_worker_thread: threading.Thread | None = None
_qoder_agent_service: QoderAgentService | None = None
_diagnosis_runtime: DiagnosisRuntime | None = None
_handoff: Handoff | None = None


def _get_channel_client() -> Channel:
    global channel_client
    if channel_client is None:
        channel_client = create_default_channel()
    return channel_client


def _get_qoder_agent_service() -> QoderAgentService:
    if _qoder_agent_service is None:
        raise AgentUnavailableError("Qoder Agent service is not initialized")
    return _qoder_agent_service


def start_event_worker_loop() -> None:
    global _event_worker_loop, _event_worker_thread
    if _event_worker_loop is not None:
        return

    loop = asyncio.new_event_loop()

    def _run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, name="feishu-event-worker", daemon=True)
    thread.start()
    _event_worker_loop = loop
    _event_worker_thread = thread

    future = asyncio.run_coroutine_threadsafe(_initialize_worker_runtime(), loop)
    try:
        future.result(timeout=60)
    except Exception:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        _event_worker_loop = None
        _event_worker_thread = None
        raise
    logger.info("Started event worker loop thread: feishu-event-worker")


def stop_event_worker_loop() -> None:
    global _event_worker_loop, _event_worker_thread, _qoder_agent_service
    global _diagnosis_runtime, _handoff
    loop = _event_worker_loop
    thread = _event_worker_thread
    if loop is None:
        return

    future = asyncio.run_coroutine_threadsafe(_shutdown_worker_runtime(), loop)
    try:
        future.result(timeout=settings.qoder_shutdown_grace_seconds + 10)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to stop event worker runtime cleanly")
    finally:
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=10)
        _event_worker_loop = None
        _event_worker_thread = None
        _qoder_agent_service = None
        _diagnosis_runtime = None
        _handoff = None
        logger.info("Stopped event worker loop thread")


def start_handoff(ws_client: Any) -> threading.Event:
    """Run the version-handoff supervisor; the returned event is set once this process yields."""
    handed_over = threading.Event()
    if _handoff is None or _event_worker_loop is None:
        return handed_over
    connected, disconnect, stop = ws_client_controls(ws_client)

    def _stop() -> None:
        handed_over.set()
        stop()

    supervisor = HandoffSupervisor(
        _handoff,
        connected=connected,
        process=_process_forwarded,
        disconnect=disconnect,
        stop=_stop,
        drain_timeout=settings.qoder_run_timeout_seconds
        + settings.qoder_shutdown_grace_seconds
        + 60,
    )
    asyncio.run_coroutine_threadsafe(supervisor.run(), _event_worker_loop)
    return handed_over


async def _initialize_worker_runtime() -> None:
    global _qoder_agent_service, _diagnosis_runtime, _handoff
    backend = settings.feishu_agent_backend.strip().lower()
    if backend == "qoder_sdk":
        mcp_config_value = settings.mcp_servers_config_path.strip()
        mcp_config_path = (
            Path(mcp_config_value).expanduser().resolve() if mcp_config_value else None
        )
        domain_value = settings.qoder_domain_knowledge_dir.strip()
        domain_dir = Path(domain_value).expanduser().resolve() if domain_value else None
        knowledge_value = settings.qoder_knowledge_dir.strip()
        knowledge_dir = (
            Path(knowledge_value).expanduser().resolve() if knowledge_value else None
        )
        config = QoderServiceConfig(
            workspace_root=Path(settings.qoder_workspace_root).expanduser().resolve(),
            model=settings.qoder_model,
            session_store_dir=Path(settings.qoder_session_store_dir).expanduser().resolve(),
            personal_access_token=settings.qoder_personal_access_token.get_secret_value(),
            mcp_servers_config_path=mcp_config_path,
            run_timeout_seconds=settings.qoder_run_timeout_seconds,
            cancel_grace_seconds=settings.qoder_cancel_grace_seconds,
            shutdown_grace_seconds=settings.qoder_shutdown_grace_seconds,
            max_concurrent_runs=settings.qoder_max_concurrent_runs,
            max_live_sessions=settings.qoder_max_live_sessions,
            max_queued_per_session=settings.qoder_max_queued_per_session,
            session_idle_ttl_seconds=settings.qoder_session_idle_ttl_seconds,
            dedup_ttl_seconds=settings.qoder_dedup_ttl_seconds,
            domain_knowledge_dir=domain_dir,
            knowledge_dir=knowledge_dir,
            max_buffer_size_bytes=settings.qoder_max_buffer_size_bytes,
            mode=settings.qoder_agent_mode.strip().lower(),
        )
        _qoder_agent_service = QoderAgentService(config)
        await _qoder_agent_service.start()
        # The runtime keeps task contexts even when card UI is disabled, so
        # quote-reply continuation and supplement merging never depend on cards.
        _diagnosis_runtime = DiagnosisRuntime(
            _get_channel_client(),
            create_store(settings.qoder_diagnosis_store_dir),
            cards_enabled=settings.qoder_diagnosis_cards_enabled,
        )
        handoff_value = settings.qoder_handoff_dir.strip()
        if handoff_value:
            _handoff = Handoff(Path(handoff_value).expanduser().resolve())
        return

    if backend == "legacy":
        from feishubot.ai.mcp import init_mcp_tools

        await init_mcp_tools()
        return

    raise ValueError(f"Unsupported FEISHU_AGENT_BACKEND: {settings.feishu_agent_backend}")


async def _shutdown_worker_runtime() -> None:
    if _qoder_agent_service is not None:
        await _qoder_agent_service.close()
        return

    if settings.feishu_agent_backend.strip().lower() == "legacy":
        from feishubot.ai.mcp import get_mcp_client

        await get_mcp_client().disconnect()


def _submit_event_task(coro: Any) -> None:
    if _event_worker_loop is None:
        start_event_worker_loop()
    if _event_worker_loop is None:
        raise RuntimeError("event worker loop failed to start")

    future = asyncio.run_coroutine_threadsafe(coro, _event_worker_loop)

    def _log_task_result(completed_future: Any) -> None:
        try:
            completed_future.result()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to handle Feishu message event")

    future.add_done_callback(_log_task_result)


async def _download_images(message_id: str, image_keys: list[str]) -> tuple[ImageInput, ...]:
    images: list[ImageInput] = []
    for image_key in image_keys[:_MAX_IMAGES_PER_MESSAGE]:
        data_url = await _get_channel_client().get_message_image_data_url(
            message_id=message_id,
            file_key=image_key,
        )
        header, _, data = data_url.partition(";base64,")
        images.append(ImageInput(media_type=header.removeprefix("data:"), data_base64=data))
    return tuple(images)


async def _run_legacy_agent(message: str, user_id: str | None) -> str:
    active = settings.active_llm_config()
    agent_loop = AgentLoop(
        model_provider=create_active_provider(),
        tool_runtime=ToolRuntime(),
        system_prompt=build_system_prompt(active.system_prompt, include_core=False),
    )
    return await agent_loop.run(user_input=message, user_id=user_id)


async def _send_chat_message(chat_id: str, text: str) -> dict[str, Any]:
    return await _get_channel_client().send_text_message(
        receive_id=chat_id,
        text=text,
        receive_id_type="chat_id",
    )


def _response_message_id(response: dict[str, Any]) -> str | None:
    data = response.get("data")
    if isinstance(data, dict):
        message_id = data.get("message_id")
        if isinstance(message_id, str) and message_id:
            return message_id
    return None


async def _respond(message: Any, text: str) -> str | None:
    """Reply to the triggering message so concurrent tasks/askers can tell answers apart."""
    quote_reply = (
        getattr(message, "chat_type", None) == "group"
        or settings.qoder_agent_mode.strip().lower() == "dev"
    )
    if quote_reply:
        try:
            response = await _get_channel_client().reply_text_message(
                message_id=str(message.message_id),
                text=text,
            )
            return _response_message_id(response)
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to reply to Feishu message, request=%s; sending to chat instead",
                str(message.message_id)[-8:],
                exc_info=True,
            )
    response = await _send_chat_message(str(message.chat_id), text)
    return _response_message_id(response)


def _build_session_key(message: Any, user_open_id: str | None) -> SessionKey:
    return SessionKey.build(
        chat_id=str(message.chat_id),
        user_open_id=user_open_id or "",
        message_id=str(message.message_id),
        thread_id=getattr(message, "thread_id", None),
    )


async def _process_forwarded(payload: dict[str, Any]) -> None:
    await _process_qoder_text(
        message_from_payload(payload),
        payload.get("text") or "",
        payload.get("user_open_id"),
        payload.get("image_keys") or None,
    )


async def _process_qoder_text(
    message: Any,
    text: str,
    user_open_id: str | None,
    image_keys: list[str] | None = None,
) -> None:
    handoff = _handoff
    if handoff is None:
        await _handle_qoder_text(message, text, user_open_id, image_keys)
        return
    session_key = _build_session_key(message, user_open_id).value
    target = handoff.forward_target(session_key)
    if target is not None:
        handoff.forward(target, forward_payload(message, text, user_open_id, image_keys))
        logger.warning(
            "Forwarded message to pid=%s during handoff, request=%s",
            target,
            str(message.message_id)[-8:],
        )
        return
    handoff.begin(session_key)
    try:
        await _handle_qoder_text(message, text, user_open_id, image_keys)
    finally:
        handoff.end(session_key)


def _dev_sender_allowed(user_open_id: str | None) -> bool:
    allowed = {
        item.strip()
        for item in settings.qoder_dev_allowed_open_ids.split(",")
        if item.strip()
    }
    return bool(user_open_id) and user_open_id in allowed


async def _handle_qoder_text(
    message: Any,
    text: str,
    user_open_id: str | None,
    image_keys: list[str] | None = None,
) -> None:
    if settings.qoder_agent_mode.strip().lower() == "dev" and not _dev_sender_allowed(
        user_open_id
    ):
        logger.warning(
            "Rejected dev task from unauthorized sender, request=%s",
            str(message.message_id)[-8:],
        )
        await _respond(message, "该机器人仅对授权用户开放。")
        return
    await handle_qoder_diagnosis(
        message=message,
        text=text,
        user_open_id=user_open_id,
        image_keys=image_keys,
        session_key=_build_session_key(message, user_open_id),
        service=_get_qoder_agent_service(),
        runtime=_diagnosis_runtime,
        download_images=_download_images,
        respond=_respond,
    )


async def process_p2_im_message_receive_v1(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
    event = data.event
    if event is None or event.message is None:
        logger.warning("Skipped Feishu event with no message payload")
        return

    message = event.message
    message_id = message.message_id
    chat_id = message.chat_id
    user_open_id = None
    if event.sender is not None and event.sender.sender_id is not None:
        user_open_id = event.sender.sender_id.open_id

    logger.info(
        "Received Feishu message, request=%s, type=%s",
        str(message_id)[-8:],
        message.message_type,
    )

    if not message_id or not chat_id:
        logger.warning("Skipped Feishu event missing message_id or chat_id")
        return

    is_qoder = settings.feishu_agent_backend.strip().lower() == "qoder_sdk"

    if message.message_type == "text":
        raw_text = extract_text(message.content)
        text = strip_mentions(raw_text) if raw_text else ""
        if not text:
            if is_qoder and raw_text:
                await _respond(message, HELP_TEXT)
                return
            logger.warning("Skipped empty Feishu text message, request=%s", str(message_id)[-8:])
            return

        if is_qoder:
            await _process_qoder_text(message, text, user_open_id)
            return

        reply = await _run_legacy_agent(text, user_open_id)
        await _send_chat_message(str(chat_id), reply)
        return

    if is_qoder and message.message_type in {"image", "post"}:
        text, image_keys = extract_image_message(message.message_type, message.content)
        text = strip_mentions(text)
        if not text and not image_keys:
            logger.warning("Skipped empty Feishu message, request=%s", str(message_id)[-8:])
            return
        await _process_qoder_text(message, text, user_open_id, image_keys)
        return

    if message.message_type == "image":
        await _send_chat_message(
            str(chat_id),
            "当前版本仅支持文本形式的本地代码分析，请用文字描述问题或粘贴报错。",
        )
        return

    logger.info(
        "Ignored unsupported Feishu message type, request=%s, type=%s",
        str(message_id)[-8:],
        message.message_type,
    )


def on_p2_im_chat_access_event_bot_p2p_chat_entered_v1(
    data: lark.im.v1.P2ImChatAccessEventBotP2pChatEnteredV1,
) -> None:
    logger.debug("Received bot_p2p_chat_entered_v1 event")


def on_p2_im_message_receive_v1(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
    _submit_event_task(process_p2_im_message_receive_v1(data))


def build_event_dispatcher(
    log_level: lark.LogLevel = lark.LogLevel.INFO,
) -> lark.EventDispatcherHandler:
    return (
        lark.EventDispatcherHandler.builder(
            settings.feishu_encrypt_key,
            settings.feishu_verification_token,
            log_level,
        )
        .register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(
            on_p2_im_chat_access_event_bot_p2p_chat_entered_v1
        )
        .register_p2_im_message_receive_v1(on_p2_im_message_receive_v1)
        .build()
    )
