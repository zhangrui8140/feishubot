"""Hand a running bot over to a newly started version without dropping in-flight analyses.

The new process claims ``active.json`` once its long connection is up; the old process then
disconnects, finishes its running analyses and exits. Messages for sessions still running in
the old process are forwarded to its inbox so supplements and cancels keep working.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ACTIVE_FILE = "active.json"
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
                return False
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_pid(path: Path) -> int | None:
    payload = _read_json(path)
    pid = payload.get("pid") if payload else None
    return pid if isinstance(pid, int) else None


class Handoff:
    def __init__(self, directory: Path, pid: int | None = None) -> None:
        self._directory = directory
        self._sessions_dir = directory / "sessions"
        self._inbox_root = directory / "inbox"
        self.pid = pid or os.getpid()
        self.draining = False
        self._running: Counter[str] = Counter()

    @property
    def busy(self) -> bool:
        return bool(self._running)

    def handles(self, session_key: str) -> bool:
        return session_key in self._running

    def claim(self) -> None:
        _write_json(self._directory / _ACTIVE_FILE, {"pid": self.pid, "claimed_at": time.time()})

    def active_pid(self) -> int | None:
        return _read_pid(self._directory / _ACTIVE_FILE)

    def superseded(self) -> bool:
        active = self.active_pid()
        return active is not None and active != self.pid and pid_alive(active)

    def begin(self, session_key: str) -> None:
        self._running[session_key] += 1
        if self._running[session_key] == 1:
            _write_json(self._marker(session_key), {"pid": self.pid})

    def end(self, session_key: str) -> None:
        self._running[session_key] -= 1
        if self._running[session_key] > 0:
            return
        del self._running[session_key]
        marker = self._marker(session_key)
        if _read_pid(marker) == self.pid:
            marker.unlink(missing_ok=True)

    def owner(self, session_key: str) -> int | None:
        """Return another live process that is still analysing this session."""
        marker = self._marker(session_key)
        pid = _read_pid(marker)
        if pid is None or pid == self.pid:
            return None
        if pid_alive(pid):
            return pid
        marker.unlink(missing_ok=True)
        return None

    def forward_target(self, session_key: str) -> int | None:
        """Pick the process that must handle this session during a version handoff."""
        owner = self.owner(session_key)
        if owner is not None:
            return owner
        if not self.draining or self.handles(session_key):
            return None
        active = self.active_pid()
        return active if active is not None and active != self.pid and pid_alive(active) else None

    def forward(self, pid: int, payload: dict[str, Any]) -> None:
        name = f"{time.time_ns()}-{uuid.uuid4().hex}.json"
        _write_json(self._inbox_root / str(pid) / name, payload)

    def take_inbox(self) -> list[dict[str, Any]]:
        """Collect messages forwarded to this process, adopting inboxes of exited processes."""
        if not self._inbox_root.is_dir():
            return []
        payloads: list[dict[str, Any]] = []
        for inbox in sorted(self._inbox_root.iterdir()):
            own = inbox.name == str(self.pid)
            if not inbox.is_dir() or not inbox.name.isdigit():
                continue
            if not own and pid_alive(int(inbox.name)):
                continue
            for path in sorted(inbox.glob("*.json")):
                payload = _read_json(path)
                path.unlink(missing_ok=True)
                if payload is not None:
                    payloads.append(payload)
            if not own:
                try:
                    inbox.rmdir()
                except OSError:
                    pass
        return payloads

    def _marker(self, session_key: str) -> Path:
        digest = hashlib.sha256(session_key.encode("utf-8")).hexdigest()[:32]
        return self._sessions_dir / f"{digest}.json"


class HandoffSupervisor:
    def __init__(
        self,
        handoff: Handoff,
        *,
        connected: Callable[[], bool],
        process: Callable[[dict[str, Any]], Awaitable[None]],
        disconnect: Callable[[], None],
        stop: Callable[[], None],
        drain_timeout: float,
        interval: float = 1.0,
        idle_ticks_to_stop: int = 3,
    ) -> None:
        self._handoff = handoff
        self._connected = connected
        self._process = process
        self._disconnect = disconnect
        self._stop = stop
        self._drain_timeout = drain_timeout
        self._interval = interval
        self._idle_ticks_to_stop = idle_ticks_to_stop
        self._claimed = False
        self._drain_started: float | None = None
        self._idle_ticks = 0
        self._tasks: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        while True:
            try:
                if self._tick():
                    return
            except Exception:  # noqa: BLE001
                logger.exception("Handoff supervisor tick failed")
            await asyncio.sleep(self._interval)

    def _tick(self) -> bool:
        for payload in self._handoff.take_inbox():
            task = asyncio.create_task(self._process_safely(payload))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        if not self._claimed and self._connected():
            self._handoff.claim()
            self._claimed = True
            logger.warning("Long connection is up; this process now serves new messages")

        if self._claimed and self._drain_started is None and self._handoff.superseded():
            self._drain_started = time.monotonic()
            self._handoff.draining = True
            self._disconnect()
            logger.warning(
                "Superseded by pid=%s; finishing in-flight analyses before exit",
                self._handoff.active_pid(),
            )

        if self._drain_started is None:
            return False
        idle = not self._handoff.busy and not self._tasks
        self._idle_ticks = self._idle_ticks + 1 if idle else 0
        timed_out = time.monotonic() - self._drain_started > self._drain_timeout
        if self._idle_ticks < self._idle_ticks_to_stop and not timed_out:
            return False
        logger.warning("Handoff complete (timed_out=%s); stopping this process", timed_out)
        self._stop()
        return True

    async def _process_safely(self, payload: dict[str, Any]) -> None:
        try:
            await self._process(payload)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to process forwarded Feishu message")


def ws_client_controls(
    client: Any,
) -> tuple[Callable[[], bool], Callable[[], None], Callable[[], None]]:
    """Connection probes for lark_oapi.ws.Client, which exposes no public stop API."""
    from lark_oapi.ws import client as ws_module

    ws_loop = ws_module.loop

    def connected() -> bool:
        return client._conn is not None

    def disconnect() -> None:
        client._auto_reconnect = False
        asyncio.run_coroutine_threadsafe(client._disconnect(), ws_loop)

    def stop() -> None:
        ws_loop.call_soon_threadsafe(ws_loop.stop)

    return connected, disconnect, stop
