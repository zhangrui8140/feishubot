from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

from feishubot.qoder_agent.projects import PROJECT_SUFFIX, list_projects

logger = logging.getLogger(__name__)

RECORD_MARKER = "【领域记录】"
_MAX_MODULE_CHARS = 40
_MAX_SUMMARY_CHARS = 200
_MAX_NOTES_CHARS = 2000
_MAX_KEYWORDS = 12
_MAX_KEYWORD_CHARS = 30
_MAX_FILES = 20
_MAX_MATCHES = 2
_MAX_LISTED_MODULES = 40
_LINE_SUFFIX = re.compile(r"(?::\d+(?:-\d+)?|#L\d+(?:-L?\d+)?)$")
_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass(frozen=True, slots=True)
class DomainMatch:
    project: str
    module: str
    summary: str
    notes: str
    files: tuple[str, ...]
    changed_files: tuple[str, ...]


def split_record(text: str) -> tuple[str, dict[str, Any] | None]:
    """Remove the trailing domain record from an answer and return it parsed."""
    index = text.rfind(RECORD_MARKER)
    if index < 0:
        return text, None
    answer = text[:index].rstrip()
    raw = _JSON_FENCE.sub("", text[index + len(RECORD_MARKER):].strip())
    start, end = raw.find("{"), raw.rfind("}")
    payload: Any = None
    if 0 <= start < end:
        try:
            payload = json.loads(raw[start:end + 1], strict=False)
        except json.JSONDecodeError:
            payload = None
    if not isinstance(payload, dict):
        logger.warning("Ignored an unparsable domain record")
        payload = None
    return answer, payload


class DomainKnowledgeStore:
    """Per-project module notes from past analyses, keyed to source fingerprints."""

    def __init__(self, directory: str | Path, workspace_root: str | Path) -> None:
        self._directory = Path(directory).expanduser().resolve()
        self._workspace_root = Path(workspace_root).expanduser().resolve()
        self._lock = RLock()

    def record(self, payload: dict[str, Any]) -> str | None:
        project = payload.get("project")
        module = _clean(payload.get("module"), _MAX_MODULE_CHARS)
        if project not in list_projects(self._workspace_root) or not module:
            return None
        project_dir = self._workspace_root / project
        files = {}
        for raw_path in _str_list(payload.get("files"))[:_MAX_FILES]:
            relative = self._normalize_path(project_dir, project, raw_path)
            if relative is not None:
                files[relative] = _fingerprint(project_dir / relative)
        keywords = [
            keyword
            for keyword in (_clean(item, _MAX_KEYWORD_CHARS) for item in _str_list(
                payload.get("keywords")
            ))
            if keyword
        ]
        with self._lock:
            modules = self._load(project)
            previous = modules.get(module, {})
            merged_keywords = list(dict.fromkeys(keywords + previous.get("keywords", [])))
            merged_files = dict(files)
            # The module was just re-analysed, so older files are re-baselined rather than
            # left flagged as changed forever; deleted ones drop out.
            for path in previous.get("files", {}):
                if len(merged_files) >= _MAX_FILES:
                    break
                if path not in merged_files and (project_dir / path).is_file():
                    merged_files[path] = _fingerprint(project_dir / path)
            modules[module] = {
                "summary": _clean(payload.get("summary"), _MAX_SUMMARY_CHARS)
                or previous.get("summary", ""),
                "notes": _clean(payload.get("notes"), _MAX_NOTES_CHARS)
                or previous.get("notes", ""),
                "keywords": merged_keywords[:_MAX_KEYWORDS],
                "files": merged_files,
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            self._save(project, modules)
        logger.info(
            "Recorded domain module, project=%s, module=%s, files=%s",
            project,
            module,
            len(merged_files),
        )
        return module

    def match(self, question: str, projects: tuple[str, ...]) -> list[DomainMatch]:
        lowered = question.lower()
        scored: list[tuple[int, str, str, str, dict[str, Any]]] = []
        for project in projects:
            for module, entry in self._load(project).items():
                terms = {module, *entry.get("keywords", [])}
                score = sum(1 for term in terms if term and term.lower() in lowered)
                if score:
                    scored.append((score, entry.get("updated_at", ""), project, module, entry))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [
            self._to_match(project, module, entry)
            for _score, _updated, project, module, entry in scored[:_MAX_MATCHES]
        ]

    def module_names(self, projects: tuple[str, ...]) -> dict[str, list[str]]:
        names: dict[str, list[str]] = {}
        remaining = _MAX_LISTED_MODULES
        for project in projects:
            modules = sorted(self._load(project))[:remaining]
            if modules:
                names[project] = modules
                remaining -= len(modules)
        return names

    def build_context(self, question: str, projects: tuple[str, ...]) -> str:
        matches = self.match(question, projects)
        names = self.module_names(projects)
        if not matches and not names:
            return ""
        lines = ["【领域知识（来自既往分析）】"]
        for index, match in enumerate(matches, start=1):
            if match.changed_files:
                status = (
                    f"代码已变更：{'、'.join(match.changed_files)}，"
                    "必须先重读这些文件，再结合下述记录下结论"
                )
            else:
                status = "代码未变更：可直接采用下述记录，无需重读这些文件，只补查记录未覆盖的部分"
            lines.append(f"{index}. {match.project} / {match.module}（{status}）")
            lines.append(f"   摘要：{match.summary}")
            if match.notes:
                lines.append(f"   要点：{match.notes}")
            if match.files:
                lines.append(f"   关键文件：{'、'.join(match.files)}")
        if not matches:
            lines.append("未命中既往记录的模块，请按常规方式分析。")
        if names:
            listed = "；".join(f"{project}：{'、'.join(mods)}" for project, mods in names.items())
            lines.append(f"已有领域模块（记录时优先复用同名模块）：{listed}")
        return "\n".join(lines)

    def _to_match(self, project: str, module: str, entry: dict[str, Any]) -> DomainMatch:
        project_dir = self._workspace_root / project
        files = entry.get("files", {})
        changed = tuple(
            path
            for path, fingerprint in files.items()
            if not _unchanged(project_dir / path, fingerprint)
        )
        return DomainMatch(
            project=project,
            module=module,
            summary=entry.get("summary", ""),
            notes=entry.get("notes", ""),
            files=tuple(f"{project}/{path}" for path in files),
            changed_files=tuple(f"{project}/{path}" for path in changed),
        )

    def _normalize_path(self, project_dir: Path, project: str, raw_path: str) -> str | None:
        cleaned = _LINE_SUFFIX.sub("", raw_path.strip().strip("`").replace("\\", "/"))
        cleaned = cleaned.removeprefix("./")
        for prefix in (f"{project}/", f"{project.removesuffix(PROJECT_SUFFIX)}/"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        try:
            resolved = (project_dir / cleaned).resolve()
            relative = resolved.relative_to(project_dir.resolve())
        except (OSError, ValueError):
            return None
        if not resolved.is_file() or resolved.name.lower().startswith(".env"):
            return None
        return relative.as_posix()

    def _path(self, project: str) -> Path:
        return self._directory / f"{project}.json"

    def _load(self, project: str) -> dict[str, Any]:
        path = self._path(project)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        modules = payload.get("modules") if isinstance(payload, dict) else None
        return modules if isinstance(modules, dict) else {}

    def _save(self, project: str, modules: dict[str, Any]) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        temp_path = self._directory / f"{project}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temp_path.write_text(
            json.dumps({"modules": modules}, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self._path(project))


def _clean(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


_HASH_CHUNK_SIZE = 65536  # 64 KB


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return {
        "sha256": hasher.hexdigest(),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _unchanged(path: Path, fingerprint: dict[str, Any]) -> bool:
    try:
        stat = path.stat()
        if stat.st_size != fingerprint.get("size"):
            return False
        if stat.st_mtime_ns == fingerprint.get("mtime_ns"):
            return True
        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest() == fingerprint.get("sha256")
    except OSError:
        return False


async def _fingerprint_async(path: Path) -> dict[str, Any]:
    return await asyncio.to_thread(_fingerprint, path)


async def _unchanged_async(path: Path, fingerprint: dict[str, Any]) -> bool:
    return await asyncio.to_thread(_unchanged, path, fingerprint)
