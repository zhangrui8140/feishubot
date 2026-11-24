from __future__ import annotations

import importlib.metadata
import importlib.resources
import ipaddress
import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from feishubot.config import resolve_env_file, settings
from feishubot.env_config import (
    ConfigConflictError,
    ConfigValidationError,
    read_managed_config,
    save_managed_config,
)
from feishubot.mcp_admin import read_mcp_config, save_mcp_config
from feishubot.qoder_agent.mcp_oauth import (
    ObservabilityAuthError,
    ObservabilityAuthorizer,
)

logger = logging.getLogger(__name__)

_admin_html_cache: str | None = None
_authorizer: ObservabilityAuthorizer | None = None

_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "connect-src 'self'; img-src 'self' data:; font-src 'self'; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


class SecretUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["keep", "replace", "clear"]
    value: str = ""


class ConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    values: dict[str, str | int | float]
    secrets: dict[str, SecretUpdate]


class McpSecretUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["keep", "replace", "clear"]
    value: str = ""


class McpHttpServerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    url: str = ""
    token: McpSecretUpdate


class McpObservabilityServerUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    url: str = ""


class McpConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    dms: McpHttpServerUpdate
    observability: McpObservabilityServerUpdate


def _get_authorizer() -> ObservabilityAuthorizer:
    global _authorizer
    if _authorizer is None:
        _authorizer = ObservabilityAuthorizer.from_settings(settings)
    return _authorizer


class AdminSecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        if request.url.path == "/admin" or request.url.path.startswith("/admin/"):
            for name, value in _SECURITY_HEADERS.items():
                response.headers[name] = value
        return response


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client is not None else ""
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise HTTPException(status_code=403, detail="Local access only")


def _require_same_origin(request: Request) -> None:
    origin = request.headers.get("origin", "")
    expected = f"{request.url.scheme}://{request.url.netloc}"
    if origin != expected:
        raise HTTPException(status_code=403, detail="Same-origin request required")


router = APIRouter(prefix="/admin", dependencies=[Depends(_require_loopback)])


@router.get("/config", response_class=HTMLResponse)
async def admin_config_page() -> HTMLResponse:
    global _admin_html_cache
    if _admin_html_cache is None:
        resource = importlib.resources.files("feishubot").joinpath("admin/index.html")
        _admin_html_cache = resource.read_text(encoding="utf-8")
    return HTMLResponse(_admin_html_cache)


@router.get("/api/status")
async def admin_status() -> dict[str, Any]:
    env_path = resolve_env_file().resolve()
    try:
        version = importlib.metadata.version("MA-Agent")
    except importlib.metadata.PackageNotFoundError:
        version = "0.2.0"
    workspace = settings.qoder_workspace_root.strip()
    return {
        "version": version,
        "config_file": str(env_path),
        "config_file_exists": env_path.exists(),
        "backend": settings.feishu_agent_backend,
        "restart_required_after_save": True,
        "connections": {
            "feishu": bool(settings.feishu_app_id and settings.feishu_app_secret),
            "qoder": bool(settings.qoder_personal_access_token.get_secret_value()),
            "workspace": bool(workspace and Path(workspace).is_dir()),
        },
    }


@router.get("/api/config")
async def get_admin_config() -> dict[str, Any]:
    managed = read_managed_config(resolve_env_file().resolve())
    return {
        "values": managed.values,
        "secrets": {
            key: {"configured": configured}
            for key, configured in managed.secrets.items()
        },
        "revision": managed.revision,
    }


@router.put("/api/config")
async def update_admin_config(
    payload: ConfigUpdateRequest, request: Request
) -> dict[str, Any]:
    _require_same_origin(request)
    try:
        managed = save_managed_config(
            resolve_env_file().resolve(),
            expected_revision=payload.revision,
            values=payload.values,
            secret_updates={
                key: update.model_dump() for key, update in payload.secrets.items()
            },
        )
    except ConfigConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConfigValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info("Local configuration updated; gateway and events restart required")
    return {
        "revision": managed.revision,
        "secrets": {
            key: {"configured": configured}
            for key, configured in managed.secrets.items()
        },
        "restart_required": True,
    }


def resolve_mcp_file() -> Path | None:
    raw = settings.mcp_servers_config_path.strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


@router.get("/api/mcp")
async def get_admin_mcp_config() -> dict[str, Any]:
    path = resolve_mcp_file()
    if path is None:
        return {
            "config_path": "",
            "config_file_exists": False,
            "parse_error": None,
            "servers": None,
            "revision": "",
        }
    managed = read_mcp_config(path)
    return {
        "config_path": str(path),
        "config_file_exists": path.is_file(),
        "parse_error": managed.parse_error,
        "servers": managed.servers,
        "revision": managed.revision,
    }


@router.put("/api/mcp")
async def update_admin_mcp_config(
    payload: McpConfigUpdateRequest, request: Request
) -> dict[str, Any]:
    _require_same_origin(request)
    path = resolve_mcp_file()
    if path is None:
        raise HTTPException(status_code=503, detail="MCP config path is not configured")
    try:
        managed = save_mcp_config(
            path,
            expected_revision=payload.revision,
            dms=payload.dms.model_dump(),
            observability=payload.observability.model_dump(),
        )
    except ConfigConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConfigValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info("Local MCP configuration updated; effective for new agent sessions")
    return {
        "revision": managed.revision,
        "servers": managed.servers,
        "effective_immediately": True,
    }


@router.post("/api/mcp/observability/authorize")
async def authorize_observability(request: Request) -> dict[str, Any]:
    _require_same_origin(request)
    try:
        return await _get_authorizer().start()
    except ObservabilityAuthError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/api/mcp/observability/status")
async def observability_status() -> dict[str, Any]:
    try:
        return await _get_authorizer().status()
    except ObservabilityAuthError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/api/mcp/observability/clear")
async def clear_observability_authorization(request: Request) -> dict[str, Any]:
    _require_same_origin(request)
    try:
        await _get_authorizer().clear()
    except ObservabilityAuthError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"cleared": True}


def install_admin(app: FastAPI) -> None:
    app.add_middleware(AdminSecurityHeadersMiddleware)
    app.include_router(router)
