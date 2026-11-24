from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from feishubot.ai.orchestrator import chat_commands, feishu_events
from feishubot.diagnosis import DiagnosisRuntime, DiagnosisStore
from feishubot.diagnosis.card import FEEDBACK_GUIDANCE
from feishubot.diagnosis.runner import _IMAGE_ONLY_PROMPT
from feishubot.qoder_agent.models import (
    AgentProgress,
    AgentResult,
    DuplicateMessageError,
    ImageInput,
)
from feishubot.qoder_agent.service import SessionStatus


class FakeChannel:
    name = "fake"

    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str]] = []
        self.replies: list[tuple[str, str]] = []
        self.interactive_messages: list[tuple[str, str, dict]] = []
        self.interactive_replies: list[tuple[str, dict]] = []
        self.interactive_patches: list[tuple[str, dict]] = []
        self.downloads: list[tuple[str, str]] = []
        self.fail_download = False
        self.fail_reply = False
        self.fail_card_reply = False
        self.fail_card_send = False
        self.fail_card_patch = False

    async def reply_text_message(self, *, message_id: str, text: str) -> dict:
        if self.fail_reply:
            raise RuntimeError("reply failed")
        self.replies.append((message_id, text))
        return {}

    async def send_interactive_message(
        self, receive_id: str, card: dict, receive_id_type: str = "open_id"
    ) -> dict:
        if self.fail_card_send:
            raise RuntimeError("card send failed")
        self.interactive_messages.append((receive_id, receive_id_type, card))
        return {"data": {"message_id": "card-created"}}

    async def reply_interactive_message(self, message_id: str, card: dict) -> dict:
        if self.fail_card_reply:
            raise RuntimeError("card reply failed")
        self.interactive_replies.append((message_id, card))
        return {"data": {"message_id": "card-replied"}}

    async def patch_interactive_message(self, message_id: str, card: dict) -> dict:
        if self.fail_card_patch:
            raise RuntimeError("card patch failed")
        self.interactive_patches.append((message_id, card))
        return {}

    async def get_message_image_data_url(self, *, message_id: str, file_key: str) -> str:
        self.downloads.append((message_id, file_key))
        if self.fail_download:
            raise RuntimeError("image too large")
        return "data:image/png;base64,QUJD"

    async def send_text_message(
        self,
        receive_id: str,
        text: str,
        receive_id_type: str = "open_id",
    ) -> dict:
        self.messages.append((receive_id, text, receive_id_type))
        return {}


_ACK_LINE = "已收到，正在分析 prod 环境的代码和 ERP 数据，请稍候……"
_ACK = "\n".join([_ACK_LINE, chat_commands.COMMAND_HINT])


class FakeService:
    def __init__(self) -> None:
        self.requests = []
        self.cancelled_keys: list[str] = []
        self.duplicate = False
        self.observability_enabled = False
        self.projects: tuple[str, ...] = ()
        self.stage_texts: list[str] = []
        self.running = False
        self.queued = 0
        self.supplement_accepted = True
        self.supplements: list[str] = []
        self.supplement_keys: list[str] = []
        self.status_keys: list[str] = []
        self.reset_keys: list[str] = []
        self.needs_input = False

    def status(self, session_key: str) -> SessionStatus:
        self.status_keys.append(session_key)
        return SessionStatus(running=self.running, queued=self.queued)

    async def supplement(self, request, text: str) -> bool:
        self.supplements.append(text)
        self.supplement_keys.append(request.session_key.value)
        return self.supplement_accepted

    async def reset(self, session_key: str) -> bool:
        if self.running:
            return False
        self.reset_keys.append(session_key)
        return True

    async def run(self, request, *, on_accepted=None, on_progress=None):
        if self.duplicate:
            raise DuplicateMessageError(request.message_id)
        self.requests.append(request)
        if on_accepted is not None:
            await on_accepted()
        if on_progress is not None:
            for stage_text in self.stage_texts:
                await on_progress(AgentProgress(kind="assistant_text", text=stage_text))
        return AgentResult(
            text="分析结果",
            session_id="session-1",
            duration_ms=100,
            duration_api_ms=60,
            tool_names=("Read",),
            needs_input=self.needs_input,
        )

    async def cancel(self, session_key: str) -> bool:
        self.cancelled_keys.append(session_key)
        return True


def make_event(
    *,
    text: str = "分析订单流程",
    message_type: str = "text",
    content: str | None = None,
    chat_type: str = "p2p",
    message_id: str = "message-12345678",
    parent_id: str | None = None,
    root_id: str | None = None,
    user_open_id: str = "user-1",
):
    message = SimpleNamespace(
        message_id=message_id,
        chat_id="chat-1",
        chat_type=chat_type,
        thread_id=None,
        root_id=root_id,
        parent_id=parent_id,
        message_type=message_type,
        content=content if content is not None else f'{{"text": "{text}"}}',
    )
    sender = SimpleNamespace(sender_id=SimpleNamespace(open_id=user_open_id))
    return SimpleNamespace(event=SimpleNamespace(message=message, sender=sender))


@pytest.fixture
def runtime(monkeypatch):
    channel = FakeChannel()
    service = FakeService()
    monkeypatch.setattr(feishu_events.settings, "feishu_agent_backend", "qoder_sdk")
    monkeypatch.setattr(feishu_events, "channel_client", channel)
    monkeypatch.setattr(feishu_events, "_qoder_agent_service", service)
    monkeypatch.setattr(feishu_events, "_diagnosis_runtime", None)
    monkeypatch.setattr(feishu_events, "_handoff", None)
    return channel, service


@pytest.fixture
def card_runtime(runtime, tmp_path, monkeypatch):
    channel, service = runtime
    diagnoses = DiagnosisRuntime(channel, DiagnosisStore(tmp_path / "diagnoses"))
    monkeypatch.setattr(feishu_events, "_diagnosis_runtime", diagnoses)
    return channel, service, diagnoses


@pytest.mark.asyncio
async def test_text_message_sends_progress_and_result(runtime):
    channel, service = runtime

    await feishu_events.process_p2_im_message_receive_v1(make_event())

    assert len(service.requests) == 1
    assert service.requests[0].session_key.value == "feishu:chat-1:root:user-1"
    assert [message[1] for message in channel.messages] == [
        _ACK,
        "分析结果",
    ]


@pytest.mark.asyncio
async def test_progress_mentions_logs_when_observability_enabled(runtime):
    channel, service = runtime
    service.observability_enabled = True

    await feishu_events.process_p2_im_message_receive_v1(make_event())

    assert channel.messages[0][1].startswith("已收到，正在分析 prod 环境的代码、ERP 数据和日志")


@pytest.mark.asyncio
async def test_progress_lists_selectable_prod_projects(runtime):
    channel, service = runtime
    service.projects = ("erp-mall-prod", "scm-prod")

    await feishu_events.process_p2_im_message_receive_v1(make_event())

    assert channel.messages[0][1] == "\n".join(
        [
            _ACK_LINE,
            "可分析项目（仅 prod 环境）：",
            "· erp-mall-prod",
            "· scm-prod",
            "提问时带上项目名可缩小范围，例如「erp-mall-prod 保存报错」。",
            chat_commands.COMMAND_HINT,
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["erp-mall-prod 门店订单保存报错", "查一下erp-mall的报错"])
async def test_progress_names_project_mentioned_by_user(runtime, text):
    channel, service = runtime
    service.projects = ("erp-mall-prod", "som-erp-mall-prod", "scm-prod")

    await feishu_events.process_p2_im_message_receive_v1(make_event(text=text))

    assert channel.messages[0][1] == "\n".join(
        [
            "已收到，正在分析 prod 环境 erp-mall-prod 的代码和 ERP 数据，请稍候……",
            chat_commands.COMMAND_HINT,
        ]
    )


@pytest.mark.asyncio
async def test_stage_results_are_sent_before_final_result(runtime):
    channel, service = runtime
    service.stage_texts = ["【分析思路】先定位保存接口", "【阶段结果】找到校验逻辑"]

    await feishu_events.process_p2_im_message_receive_v1(make_event())

    assert [message[1] for message in channel.messages] == [
        _ACK,
        "【分析思路】先定位保存接口",
        "【阶段结果】找到校验逻辑",
        "分析结果",
    ]


@pytest.mark.asyncio
async def test_diagnosis_uses_one_card_for_acceptance_and_result(card_runtime):
    channel, service, _diagnoses = card_runtime
    service.stage_texts = ["【分析思路】定位接口", "【阶段结果】找到校验"]

    await feishu_events.process_p2_im_message_receive_v1(make_event())

    assert len(channel.interactive_messages) == 1
    assert channel.messages == []
    assert channel.replies == []
    assert {message_id for message_id, _card in channel.interactive_patches} == {
        "card-created"
    }
    final_card = channel.interactive_patches[-1][1]
    assert final_card["header"]["template"] == "green"
    assert "分析结果" in json.dumps(final_card, ensure_ascii=False)


@pytest.mark.asyncio
async def test_group_diagnosis_replies_with_card_to_source(card_runtime):
    channel, _service, _diagnoses = card_runtime

    await feishu_events.process_p2_im_message_receive_v1(make_event(chat_type="group"))

    assert channel.interactive_replies[0][0] == "message-12345678"
    assert channel.interactive_messages == []
    assert {message_id for message_id, _card in channel.interactive_patches} == {
        "card-replied"
    }


@pytest.mark.asyncio
async def test_controller_remains_active_until_final_card_patch(card_runtime):
    channel, _service, diagnoses = card_runtime
    session_key = "feishu:chat-1:root:user-1"
    active_during_final_patch = []
    patch = channel.patch_interactive_message

    async def observe_final_patch(message_id, card):
        if card["header"]["template"] == "green":
            active_during_final_patch.append(diagnoses.active(session_key) is not None)
        return await patch(message_id, card)

    channel.patch_interactive_message = observe_final_patch

    await feishu_events.process_p2_im_message_receive_v1(make_event())

    assert active_during_final_patch == [True]
    assert diagnoses.active(session_key) is None


@pytest.mark.asyncio
async def test_waiting_for_input_uses_orange_card_with_feedback(card_runtime):
    channel, service, _diagnoses = card_runtime
    service.needs_input = True

    await feishu_events.process_p2_im_message_receive_v1(make_event())

    final_card = channel.interactive_patches[-1][1]
    assert final_card["header"]["template"] == "orange"
    rendered = json.dumps(final_card, ensure_ascii=False)
    assert FEEDBACK_GUIDANCE in rendered
    assert '"tag": "action"' not in rendered


@pytest.mark.asyncio
async def test_reply_to_completed_card_reuses_same_message(card_runtime):
    channel, _service, _diagnoses = card_runtime
    await feishu_events.process_p2_im_message_receive_v1(make_event())

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(
            text="补充订单号 SO-1",
            message_id="message-87654321",
            parent_id="card-created",
        )
    )

    assert len(channel.interactive_messages) == 1
    assert {message_id for message_id, _card in channel.interactive_patches} == {
        "card-created"
    }


@pytest.mark.asyncio
async def test_other_user_cannot_reuse_completed_card(card_runtime):
    channel, _service, _diagnoses = card_runtime
    await feishu_events.process_p2_im_message_receive_v1(make_event(chat_type="group"))

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(
            text="我也要排查",
            chat_type="group",
            message_id="message-other",
            parent_id="card-replied",
            user_open_id="user-2",
        )
    )

    assert len(channel.interactive_replies) == 2


@pytest.mark.asyncio
async def test_card_patch_failure_falls_back_to_final_text(card_runtime):
    channel, _service, _diagnoses = card_runtime
    channel.fail_card_patch = True

    await feishu_events.process_p2_im_message_receive_v1(make_event())

    assert [message[1] for message in channel.messages] == ["分析结果"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply_text", "expected"),
    [
        ("1", "反馈已记录"),
        ("结论正确", "反馈已记录"),
        ("3", "请回复本卡片，补充具体差异、单号或截图后继续。"),
        ("继续排查", "请回复本卡片，补充具体差异、单号或截图后继续。"),
    ],
)
async def test_text_feedback_reply_records_without_new_run(
    card_runtime, tmp_path, reply_text, expected
):
    channel, service, _diagnoses = card_runtime
    await feishu_events.process_p2_im_message_receive_v1(make_event())

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(
            text=reply_text,
            message_id="message-feedback",
            parent_id="card-created",
        )
    )

    assert len(service.requests) == 1
    assert channel.messages[-1][1] == expected
    feedback_files = list((tmp_path / "diagnoses" / "feedback").rglob("*.json"))
    assert len(feedback_files) == 1


@pytest.mark.asyncio
async def test_text_feedback_without_card_reply_starts_new_run(card_runtime, tmp_path):
    channel, service, _diagnoses = card_runtime
    await feishu_events.process_p2_im_message_receive_v1(make_event())

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(text="1", message_id="message-plain")
    )

    assert len(service.requests) == 2
    assert not list((tmp_path / "diagnoses" / "feedback").rglob("*.json"))


@pytest.mark.asyncio
async def test_text_feedback_by_other_user_falls_through(card_runtime, tmp_path):
    channel, service, _diagnoses = card_runtime
    await feishu_events.process_p2_im_message_receive_v1(make_event(chat_type="group"))

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(
            text="1",
            chat_type="group",
            message_id="message-other",
            parent_id="card-replied",
            user_open_id="user-2",
        )
    )

    assert len(service.requests) == 2
    assert not list((tmp_path / "diagnoses" / "feedback").rglob("*.json"))


@pytest.mark.asyncio
async def test_unknown_text_reply_to_card_falls_through(card_runtime, tmp_path):
    channel, service, _diagnoses = card_runtime
    await feishu_events.process_p2_im_message_receive_v1(make_event())

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(
            text="结论正确吗",
            message_id="message-unknown",
            parent_id="card-created",
        )
    )

    assert len(service.requests) == 2
    assert not list((tmp_path / "diagnoses" / "feedback").rglob("*.json"))


@pytest.mark.asyncio
async def test_worker_runtime_passes_mcp_config_path(tmp_path, monkeypatch):
    captured = {}

    class CapturingService:
        def __init__(self, config):
            captured["config"] = config

        async def start(self):
            captured["started"] = True

    mcp_config = tmp_path / "mcp.json"
    monkeypatch.setattr(feishu_events.settings, "feishu_agent_backend", "qoder_sdk")
    monkeypatch.setattr(
        feishu_events.settings,
        "mcp_servers_config_path",
        str(mcp_config),
    )
    monkeypatch.setattr(feishu_events, "QoderAgentService", CapturingService)
    monkeypatch.setattr(feishu_events, "_qoder_agent_service", None)
    monkeypatch.setattr(feishu_events, "_diagnosis_runtime", None)
    monkeypatch.setattr(feishu_events, "channel_client", FakeChannel())
    monkeypatch.setattr(feishu_events, "_handoff", None)
    monkeypatch.setattr(feishu_events.settings, "qoder_handoff_dir", str(tmp_path / "handoff"))
    monkeypatch.setattr(
        feishu_events.settings, "qoder_diagnosis_store_dir", str(tmp_path / "diagnoses")
    )

    await feishu_events._initialize_worker_runtime()

    assert feishu_events._handoff is not None
    assert feishu_events._diagnosis_runtime is not None
    assert captured["config"].mcp_servers_config_path == mcp_config.resolve()
    assert captured["started"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["取消", "终止", "停止", "/stop"])
async def test_cancel_command_interrupts_current_session(runtime, command):
    channel, service = runtime

    await feishu_events.process_p2_im_message_receive_v1(make_event(text=command))

    assert service.cancelled_keys == ["feishu:chat-1:root:user-1"]
    assert channel.messages[0][1] == "正在终止当前分析。"


@pytest.mark.asyncio
async def test_follow_up_while_running_supplements_current_analysis(runtime):
    channel, service = runtime
    service.running = True

    await feishu_events.process_p2_im_message_receive_v1(make_event(text="补充 erp-mall-prod"))

    assert service.supplements == ["【用户补充】补充 erp-mall-prod"]
    assert service.requests == []
    assert [message[1] for message in channel.messages] == ["已收到补充，将在当前分析中参考。"]


@pytest.mark.asyncio
async def test_supplement_missed_by_finished_run_starts_new_analysis(runtime):
    channel, service = runtime
    service.running = True
    service.supplement_accepted = False

    await feishu_events.process_p2_im_message_receive_v1(make_event(text="补充 erp-mall-prod"))

    assert service.requests[0].prompt == "补充 erp-mall-prod"
    assert [message[1] for message in channel.messages] == [
        "已收到补充，将在当前分析中参考。",
        "上一轮分析已结束，将结合补充内容继续分析，请稍候……",
        "分析结果",
    ]


@pytest.mark.asyncio
async def test_help_command_lists_commands(runtime):
    channel, service = runtime

    await feishu_events.process_p2_im_message_receive_v1(make_event(text="帮助"))

    assert channel.messages[0][1] == chat_commands.HELP_TEXT
    assert service.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("running", "queued", "expected"),
    [
        (False, 0, "当前没有正在进行的分析。"),
        (
            True,
            2,
            "当前有分析正在进行，另有 2 个问题排队。发送「补充 …」可并入当前分析，新问题将排队单独回答；「终止」停止。",
        ),
    ],
)
async def test_status_command_reports_session_state(runtime, running, queued, expected):
    channel, service = runtime
    service.running = running
    service.queued = queued

    await feishu_events.process_p2_im_message_receive_v1(make_event(text="状态"))

    assert channel.messages[0][1] == expected


@pytest.mark.asyncio
async def test_reset_command_clears_session(runtime):
    channel, service = runtime

    await feishu_events.process_p2_im_message_receive_v1(make_event(text="新会话"))

    assert service.reset_keys == ["feishu:chat-1:root:user-1"]
    assert channel.messages[0][1] == "已开启新会话，之前的对话上下文已清空。"


@pytest.mark.asyncio
async def test_reset_command_refused_while_running(runtime):
    channel, service = runtime
    service.running = True

    await feishu_events.process_p2_im_message_receive_v1(make_event(text="重置"))

    assert service.reset_keys == []
    assert "请先发送「终止」" in channel.messages[0][1]


@pytest.mark.asyncio
async def test_duplicate_message_sends_no_reply(runtime):
    channel, service = runtime
    service.duplicate = True

    await feishu_events.process_p2_im_message_receive_v1(make_event())

    assert channel.messages == []


@pytest.mark.asyncio
async def test_image_message_sends_image_to_qoder(runtime):
    channel, service = runtime

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(message_type="image", content='{"image_key": "img-1"}')
    )

    assert channel.downloads == [("message-12345678", "img-1")]
    request = service.requests[0]
    assert request.prompt == _IMAGE_ONLY_PROMPT
    assert request.images == (ImageInput(media_type="image/png", data_base64="QUJD"),)
    assert channel.messages[-1][1] == "分析结果"


@pytest.mark.asyncio
@pytest.mark.parametrize("wrap_locale", [False, True])
async def test_post_message_combines_text_and_images(runtime, wrap_locale):
    channel, service = runtime
    body = {
        "title": "保存报错",
        "content": [
            [
                {"tag": "text", "text": "门店订单保存失败 "},
                {"tag": "a", "text": "链接", "href": "x"},
            ],
            [{"tag": "img", "image_key": "img-1"}, {"tag": "img", "image_key": "img-2"}],
        ],
    }
    payload = {"zh_cn": body} if wrap_locale else body

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(message_type="post", content=json.dumps(payload, ensure_ascii=False))
    )

    request = service.requests[0]
    assert request.prompt == "保存报错\n门店订单保存失败 链接"
    assert len(request.images) == 2
    assert [key for _mid, key in channel.downloads] == ["img-1", "img-2"]


@pytest.mark.asyncio
async def test_image_download_failure_replies_without_calling_qoder(runtime):
    channel, service = runtime
    channel.fail_download = True

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(message_type="image", content='{"image_key": "img-1"}')
    )

    assert service.requests == []
    assert "图片下载失败" in channel.messages[0][1]


@pytest.mark.asyncio
async def test_image_message_rejected_on_legacy_backend(runtime, monkeypatch):
    channel, service = runtime
    monkeypatch.setattr(feishu_events.settings, "feishu_agent_backend", "legacy")

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(message_type="image", content='{"image_key": "img-1"}')
    )

    assert service.requests == []
    assert "仅支持文本" in channel.messages[0][1]


@pytest.mark.asyncio
async def test_group_message_replies_to_original_message(runtime):
    channel, service = runtime

    await feishu_events.process_p2_im_message_receive_v1(make_event(chat_type="group"))

    assert channel.messages == []
    assert channel.replies == [("message-12345678", _ACK), ("message-12345678", "分析结果")]


@pytest.mark.asyncio
async def test_group_reply_failure_falls_back_to_chat_message(runtime):
    channel, service = runtime
    channel.fail_reply = True

    await feishu_events.process_p2_im_message_receive_v1(make_event(chat_type="group"))

    assert [message[1] for message in channel.messages] == [_ACK, "分析结果"]


@pytest.mark.asyncio
async def test_group_mention_is_stripped_before_command_matching(runtime):
    channel, service = runtime

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(text="@_user_1 终止", chat_type="group")
    )

    assert service.cancelled_keys == ["feishu:chat-1:root:user-1"]
    assert channel.replies == [("message-12345678", "正在终止当前分析。")]


@pytest.mark.asyncio
async def test_group_mention_is_stripped_from_prompt(runtime):
    channel, service = runtime

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(text="@_user_1 门店订单保存报错", chat_type="group")
    )

    assert service.requests[0].prompt == "门店订单保存报错"


@pytest.mark.asyncio
async def test_bare_mention_replies_with_help(runtime):
    channel, service = runtime

    await feishu_events.process_p2_im_message_receive_v1(
        make_event(text="@_user_1", chat_type="group")
    )

    assert service.requests == []
    assert channel.replies == [("message-12345678", chat_commands.HELP_TEXT)]

