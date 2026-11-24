from __future__ import annotations

import asyncio
import os

import pytest

from feishubot.ai.orchestrator import handoff as handoff_module
from feishubot.ai.orchestrator.handoff import Handoff, HandoffSupervisor, pid_alive

_KEY = "feishu:chat-1:root:user-1"


@pytest.fixture
def alive(monkeypatch):
    pids = {111, 222, 333}
    monkeypatch.setattr(handoff_module, "pid_alive", lambda pid: pid in pids)
    return pids


def test_pid_alive_detects_current_and_missing_process():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(0) is False
    assert pid_alive(2**31 - 2) is False


def test_newer_claim_supersedes_older_process(tmp_path, alive):
    old = Handoff(tmp_path, pid=111)
    new = Handoff(tmp_path, pid=222)
    old.claim()
    assert old.superseded() is False

    new.claim()

    assert old.superseded() is True
    assert new.superseded() is False
    alive.discard(222)
    assert old.superseded() is False


def test_session_marker_tracks_nested_runs(tmp_path, alive):
    old = Handoff(tmp_path, pid=111)
    new = Handoff(tmp_path, pid=222)

    old.begin(_KEY)
    old.begin(_KEY)
    old.end(_KEY)
    assert new.owner(_KEY) == 111
    assert old.owner(_KEY) is None

    old.end(_KEY)
    assert not old.busy
    assert new.owner(_KEY) is None


def test_marker_of_exited_process_is_cleaned_up(tmp_path, alive):
    old = Handoff(tmp_path, pid=111)
    new = Handoff(tmp_path, pid=222)
    old.begin(_KEY)
    alive.discard(111)

    assert new.owner(_KEY) is None
    assert list((tmp_path / "sessions").iterdir()) == []


def test_forward_target_during_drain(tmp_path, alive):
    old = Handoff(tmp_path, pid=111)
    new = Handoff(tmp_path, pid=222)
    new.claim()
    old.begin("running")

    assert old.forward_target("fresh") is None
    old.draining = True
    assert old.forward_target("fresh") == 222
    assert old.forward_target("running") is None
    assert new.forward_target("running") == 111

    alive.discard(222)
    assert old.forward_target("fresh") is None


def test_inbox_is_delivered_once_and_orphans_are_adopted(tmp_path, alive):
    first = Handoff(tmp_path, pid=111)
    second = Handoff(tmp_path, pid=222)
    first.forward(111, {"text": "a"})
    first.forward(111, {"text": "b"})
    first.forward(333, {"text": "orphan"})

    assert second.take_inbox() == []
    assert first.take_inbox() == [{"text": "a"}, {"text": "b"}]
    assert first.take_inbox() == []

    alive.discard(333)
    assert second.take_inbox() == [{"text": "orphan"}]
    assert not (tmp_path / "inbox" / "333").exists()


def _supervisor(handoff, events, *, connected=True, process=None, drain_timeout=60.0):
    async def _noop(payload):
        events.append(("processed", payload))

    return HandoffSupervisor(
        handoff,
        connected=lambda: connected,
        process=process or _noop,
        disconnect=lambda: events.append("disconnect"),
        stop=lambda: events.append("stop"),
        drain_timeout=drain_timeout,
        interval=0,
        idle_ticks_to_stop=2,
    )


@pytest.mark.asyncio
async def test_supervisor_claims_only_after_connection(tmp_path, alive):
    handoff = Handoff(tmp_path, pid=111)
    events: list = []

    assert _supervisor(handoff, events, connected=False)._tick() is False
    assert handoff.active_pid() is None

    assert _supervisor(handoff, events)._tick() is False
    assert handoff.active_pid() == 111
    assert events == []


@pytest.mark.asyncio
async def test_supervisor_drains_running_work_then_stops(tmp_path, alive):
    old = Handoff(tmp_path, pid=111)
    events: list = []
    supervisor = _supervisor(old, events)
    supervisor._tick()
    old.begin(_KEY)

    Handoff(tmp_path, pid=222).claim()
    assert supervisor._tick() is False
    assert old.draining is True
    assert events == ["disconnect"]
    assert supervisor._tick() is False

    old.end(_KEY)
    assert supervisor._tick() is False
    assert supervisor._tick() is True
    assert events == ["disconnect", "stop"]


@pytest.mark.asyncio
async def test_supervisor_stops_after_drain_timeout(tmp_path, alive):
    old = Handoff(tmp_path, pid=111)
    events: list = []
    supervisor = _supervisor(old, events, drain_timeout=-1)
    supervisor._tick()
    old.begin(_KEY)
    Handoff(tmp_path, pid=222).claim()

    assert supervisor._tick() is True
    assert events == ["disconnect", "stop"]


@pytest.mark.asyncio
async def test_supervisor_processes_forwarded_messages(tmp_path, alive):
    handoff = Handoff(tmp_path, pid=111)
    handoff.forward(111, {"text": "补充"})
    events: list = []
    supervisor = _supervisor(handoff, events)

    supervisor._tick()
    await asyncio.sleep(0)

    assert events == [("processed", {"text": "补充"})]
