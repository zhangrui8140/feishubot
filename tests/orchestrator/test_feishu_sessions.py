from __future__ import annotations

import pytest
from test_feishu_events import make_event
from test_feishu_events import runtime as runtime

from feishubot.ai.orchestrator import feishu_events
from feishubot.ai.orchestrator import handoff as handoff_module

_SESSION = "feishu:chat-1:root:user-1"


@pytest.fixture
def handoffs(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff_module, "pid_alive", lambda pid: True)
    old = handoff_module.Handoff(tmp_path, pid=111)
    new = handoff_module.Handoff(tmp_path, pid=222)
    return old, new


@pytest.mark.asyncio
async def test_message_for_session_running_elsewhere_is_forwarded(runtime, handoffs, monkeypatch):
    channel, service = runtime
    old, new = handoffs
    old.begin(_SESSION)
    monkeypatch.setattr(feishu_events, "_handoff", new)

    await feishu_events.process_p2_im_message_receive_v1(make_event(text="补充 scm-prod"))

    assert service.requests == []
    assert channel.messages == []
    [payload] = old.take_inbox()
    assert payload["text"] == "补充 scm-prod"

    monkeypatch.setattr(feishu_events, "_handoff", old)
    service.running = True
    await feishu_events._process_forwarded(payload)

    assert service.supplements == ["【用户补充】补充 scm-prod"]
    assert service.supplement_keys == [_SESSION]


@pytest.mark.asyncio
async def test_draining_process_forwards_new_sessions_to_active(runtime, handoffs, monkeypatch):
    channel, service = runtime
    old, new = handoffs
    new.claim()
    old.draining = True
    monkeypatch.setattr(feishu_events, "_handoff", old)

    await feishu_events.process_p2_im_message_receive_v1(make_event(chat_type="group"))

    assert service.requests == []
    [payload] = new.take_inbox()
    monkeypatch.setattr(feishu_events, "_handoff", new)
    await feishu_events._process_forwarded(payload)

    assert len(service.requests) == 1
    assert channel.replies[-1] == ("message-12345678", "分析结果")
    assert not new.busy


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", ["p2p", "group"])
async def test_quote_reply_continues_current_session(runtime, chat_type):
    channel, service = runtime
    service.running = True
    event = make_event(text="补充 scm-prod", chat_type=chat_type)
    event.event.message.root_id = "bot-reply-1"
    event.event.message.parent_id = "bot-reply-1"

    await feishu_events.process_p2_im_message_receive_v1(event)

    assert service.supplements == ["【用户补充】补充 scm-prod"]
    assert service.status_keys == [_SESSION]
    assert service.supplement_keys == [_SESSION]
    assert service.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_id", [None, "omt-topic-1"])
async def test_quote_reply_cancel_targets_current_session(runtime, thread_id):
    channel, service = runtime
    event = make_event(text="@_user_1 终止", chat_type="group")
    event.event.message.thread_id = thread_id
    event.event.message.root_id = "original-question"
    event.event.message.parent_id = "bot-reply-1"

    await feishu_events.process_p2_im_message_receive_v1(event)

    assert service.cancelled_keys == [f"feishu:chat-1:{thread_id or 'root'}:user-1"]
    assert channel.replies == [("message-12345678", "正在终止当前分析。")]
    assert service.requests == []


@pytest.mark.asyncio
async def test_topic_thread_gets_its_own_session(runtime):
    channel, service = runtime
    event = make_event()
    event.event.message.thread_id = "omt-topic-1"
    event.event.message.root_id = "original-question"
    event.event.message.parent_id = "bot-reply-1"

    await feishu_events.process_p2_im_message_receive_v1(event)

    assert service.requests[0].session_key.value == "feishu:chat-1:omt-topic-1:user-1"


@pytest.mark.asyncio
async def test_users_in_same_group_and_topic_keep_separate_sessions(runtime):
    channel, service = runtime
    for user in ("user-1", "user-2"):
        event = make_event(chat_type="group")
        event.event.message.thread_id = "omt-topic-1"
        event.event.message.root_id = "original-question"
        event.event.message.parent_id = "bot-reply-1"
        event.event.sender.sender_id.open_id = user
        await feishu_events.process_p2_im_message_receive_v1(event)

    assert [request.session_key.value for request in service.requests] == [
        "feishu:chat-1:omt-topic-1:user-1",
        "feishu:chat-1:omt-topic-1:user-2",
    ]
