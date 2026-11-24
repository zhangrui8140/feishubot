from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from feishubot.ai.prompts import build_system_prompt


def resolve_env_file() -> Path:
    configured_env_file = os.getenv("FEISHUBOT_ENV_FILE", "").strip()
    if configured_env_file:
        return Path(configured_env_file).expanduser()

    cwd_env_file = Path.cwd() / ".env"
    if cwd_env_file.exists():
        return cwd_env_file

    return Path.home() / ".feishubot" / ".env"


@dataclass(frozen=True)
class ActiveLLMConfig:
    name: str
    provider: str
    base_url: str
    api_key: str
    model: str
    chat_path: str
    timeout_seconds: float
    system_prompt: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=resolve_env_file(), env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "dev"
    log_level: str = "INFO"

    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_verification_token: str = ""
    feishu_encrypt_key: str = ""
    gateway_internal_api_key: str = ""
    default_channel: str = "feishu"

    llm_provider: str = "echo"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_chat_path: str = "/v1/chat/completions"
    llm_timeout_seconds: float = 60.0
    llm_system_prompt: str = "You are a helpful assistant."
    llm_active_model: str = ""
    llm_models_config_path: str = ""
    ai_tools_config_path: str = ""
    llm_image_max_bytes: int = 5 * 1024 * 1024
    image_local_dir: str = ""
    image_local_max_count: int = 30
    mcp_servers_config_path: str = ""

    feishu_agent_backend: str = "legacy"
    qoder_personal_access_token: SecretStr = SecretStr("")
    qoder_workspace_root: str = r"D:\workspace\projects\qoderwork\prod"
    qoder_model: str = "auto"
    # "readonly" (prod analysis) or "dev" (dev worktree development tasks).
    qoder_agent_mode: str = "readonly"
    qoder_session_store_dir: str = str(Path.home() / ".feishubot" / "qoder-sessions")
    qoder_run_timeout_seconds: float = 600.0
    qoder_cancel_grace_seconds: float = 5.0
    qoder_shutdown_grace_seconds: float = 30.0
    qoder_max_concurrent_runs: int = 2
    qoder_max_live_sessions: int = 4
    qoder_max_queued_per_session: int = 3
    qoder_session_idle_ttl_seconds: float = 1800.0
    qoder_dedup_ttl_seconds: float = 86400.0
    qoder_max_buffer_size_bytes: int = 32 * 1024 * 1024
    # Empty disables recording and reuse of domain module knowledge.
    qoder_domain_knowledge_dir: str = str(Path.home() / ".feishubot" / "domain-knowledge")
    # Empty disables handing in-flight analyses over to a newly started version.
    qoder_handoff_dir: str = str(Path.home() / ".feishubot" / "handoff")
    qoder_diagnosis_cards_enabled: bool = True
    qoder_diagnosis_store_dir: str = str(Path.home() / ".feishubot" / "diagnoses")

    def _load_models_from_toml(self, config_path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
        if not config_path.exists():
            raise ValueError(f"LLM models config not found: {config_path}")

        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("LLM models config must be a TOML table")

        default_model = str(raw.get("default_model", "")).strip()
        models_raw = raw.get("models")
        if not isinstance(models_raw, dict) or not models_raw:
            raise ValueError("LLM models config must include a non-empty [models] table")

        models: dict[str, dict[str, Any]] = {}
        for model_name, model_config in models_raw.items():
            if not isinstance(model_name, str) or not model_name.strip():
                continue
            if not isinstance(model_config, dict):
                raise ValueError(f"model '{model_name}' must be a TOML table")
            models[model_name.strip()] = dict(model_config)

        if not models:
            raise ValueError("LLM models config does not define any valid models")

        return default_model, models

    def _resolve_from_model_map(self) -> ActiveLLMConfig | None:
        global _models_cache
        models_config_path = self.llm_models_config_path.strip()
        if not models_config_path:
            return None

        resolved_path = Path(models_config_path).expanduser().resolve()
        try:
            current_mtime = resolved_path.stat().st_mtime
        except OSError:
            current_mtime = 0.0

        if _models_cache is not None:
            cached_mtime, cached_config = _models_cache
            if cached_mtime == current_mtime and current_mtime > 0:
                return cached_config

        default_model, models = self._load_models_from_toml(resolved_path)

        active_name = self.llm_active_model.strip() or default_model or next(iter(models))
        active_config = models.get(active_name)
        if not isinstance(active_config, dict):
            raise ValueError(f"active model '{active_name}' not found in model config")

        provider = str(active_config.get("provider", "openai_compatible"))
        base_url = str(active_config.get("base_url", ""))
        api_key = str(active_config.get("api_key", ""))
        model = str(active_config.get("model", ""))
        chat_path = str(active_config.get("chat_path", "/v1/chat/completions"))
        timeout_seconds_raw = active_config.get("timeout_seconds", self.llm_timeout_seconds)
        system_prompt = build_system_prompt(
            str(active_config.get("system_prompt", self.llm_system_prompt))
        )

        try:
            timeout_seconds = float(timeout_seconds_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"timeout_seconds for active model '{active_name}' must be a number"
            ) from exc

        result = ActiveLLMConfig(
            name=active_name,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            chat_path=chat_path,
            timeout_seconds=timeout_seconds,
            system_prompt=system_prompt,
        )
        _models_cache = (current_mtime, result)
        return result

    def active_llm_config(self) -> ActiveLLMConfig:
        model_map_config = self._resolve_from_model_map()
        if model_map_config is not None:
            return model_map_config

        return ActiveLLMConfig(
            name=self.llm_active_model.strip() or self.llm_model,
            provider=self.llm_provider,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
            model=self.llm_model,
            chat_path=self.llm_chat_path,
            timeout_seconds=self.llm_timeout_seconds,
            system_prompt=build_system_prompt(self.llm_system_prompt),
        )


_models_cache: tuple[float, ActiveLLMConfig] | None = None

settings = Settings()
