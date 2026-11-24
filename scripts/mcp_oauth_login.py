"""Interactively (re)authorize an MCP server via Qoder OAuth.

Normally the Qoder CLI refreshes the stored MCP OAuth token in
~/.qoder/mcp-oauth-tokens.json automatically, so feishubot's qoder_sdk backend
connects without any static Bearer token. Only run this script when the
refresh token itself has expired (MCP tools fail with 401/unauthorized and a
fresh client cannot connect silently): it opens a browser for SSO and waits
until the server reaches `connected`, then the persisted token works again
for both feishubot and Qoder IDE.

Usage:
  .venv/Scripts/python.exe scripts/mcp_oauth_login.py [--server observability] [--wait 300] [--force]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qoder_agent_sdk import QoderAgentOptions, QoderSDKClient, access_token

from feishubot.config import settings

DEFAULT_SERVER = "observability"


def _server_config_path() -> Path:
    configured = settings.mcp_servers_config_path.strip()
    if not configured:
        raise SystemExit("MCP_SERVERS_CONFIG_PATH is not configured in .env")
    return Path(configured).expanduser().resolve()


def _print_status(status: dict, server: str) -> dict:
    for entry in status.get("mcpServers", []):
        if entry.get("name") == server:
            summary = {
                "name": entry.get("name"),
                "status": entry.get("status"),
                "error": entry.get("error"),
                "tools": len(entry.get("tools") or []),
            }
            print("[status]", json.dumps(summary, ensure_ascii=False))
            return summary
    print(f"[status] server '{server}' not found in MCP status")
    return {"status": "absent"}


async def _run(server: str, wait_seconds: int, force: bool) -> int:
    options = QoderAgentOptions(
        auth=access_token(settings.qoder_personal_access_token.get_secret_value()),
        cwd=Path(settings.qoder_workspace_root),
        model=settings.qoder_model,
        permission_mode="dontAsk",
        setting_sources=[],
        mcp_servers=_server_config_path(),
        allowed_mcp_server_names=[server],
        env={
            "QODER_AGENT_SDK_ENTRYPOINT": None,
            "QODER_AGENT_SDK_VERSION": None,
        },
        stderr=lambda line: None,
    )
    async with QoderSDKClient(options=options) as client:
        current = _print_status(await client.get_mcp_status(), server)
        if current.get("status") == "connected" and not force:
            print("[done] already connected; stored OAuth token is valid, nothing to do")
            return 0

        result = await client.mcp_authenticate(server)
        auth_url = str(result.get("authUrl") or "")
        if result.get("requiresUserAction") and auth_url:
            print("[auth] browser login required, opening:", auth_url)
            try:
                webbrowser.open(auth_url)
            except Exception as exc:  # noqa: BLE001
                print(f"[auth] could not open browser automatically ({exc}); open the URL manually")
        elif result.get("requiresUserAction"):
            print("[auth] refresh failed silently; re-run once a browser login is possible")
        else:
            print("[auth] CLI refreshed the stored token without user action")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_seconds
        while loop.time() < deadline:
            await asyncio.sleep(3)
            current = _print_status(await client.get_mcp_status(), server)
            if current.get("status") == "connected":
                print("[done] connected; OAuth token persisted for future runs")
                return 0
        print(f"[fail] not connected after {wait_seconds}s; complete the browser login and re-run")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-authorize an MCP server via Qoder OAuth.")
    parser.add_argument(
        "--server", default=DEFAULT_SERVER, help=f"MCP server name (default: {DEFAULT_SERVER})"
    )
    parser.add_argument(
        "--wait", type=int, default=300, help="seconds to wait for connected (default: 300)"
    )
    parser.add_argument(
        "--force", action="store_true", help="run authenticate even if already connected"
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args.server, args.wait, args.force)))


if __name__ == "__main__":
    main()
