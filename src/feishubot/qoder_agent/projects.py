from __future__ import annotations

import re
from pathlib import Path

PROJECT_SUFFIX = "-prod"
DEV_MARKER = "-dev"
_NAME_CHARS = "A-Za-z0-9_-"


def list_projects(workspace_root: Path) -> tuple[str, ...]:
    try:
        entries = list(workspace_root.iterdir())
    except OSError:
        return ()
    return tuple(
        sorted(
            entry.name
            for entry in entries
            if entry.is_dir() and entry.name.endswith(PROJECT_SUFFIX)
        )
    )


def list_dev_projects(workspace_root: Path) -> tuple[str, ...]:
    """Dev worktrees are named like erp-purchase-dev or erp-purchase-dev-<feature>."""
    try:
        entries = list(workspace_root.iterdir())
    except OSError:
        return ()
    return tuple(
        sorted(
            entry.name
            for entry in entries
            if entry.is_dir()
            and DEV_MARKER in entry.name
            and not entry.name.startswith(".")
            and not entry.name.endswith(".code-workspace")
        )
    )


def mentioned_projects(text: str, projects: tuple[str, ...]) -> tuple[str, ...]:
    """Match full names or names without the suffix, e.g. erp-mall for erp-mall-prod."""
    lowered = text.lower()
    found = []
    for project in projects:
        aliases = {project.lower(), project.lower().removesuffix(PROJECT_SUFFIX)}
        for alias in aliases:
            pattern = rf"(?<![{_NAME_CHARS}]){re.escape(alias)}(?![{_NAME_CHARS}])"
            if re.search(pattern, lowered):
                found.append(project)
                break
    return tuple(found)


def mentioned_dev_projects(text: str, projects: tuple[str, ...]) -> tuple[str, ...]:
    """Match full worktree names or the feature suffix after "-dev".

    e.g. both "大仓端堆活动" and "erp-purchase-dev-大仓端堆活动" match
    erp-purchase-dev-大仓端堆活动; "erp-purchase-dev" matches the bare worktree.
    """
    lowered = text.lower()
    found = []
    for project in projects:
        lowered_name = project.lower()
        feature = lowered_name.split(DEV_MARKER, 1)[1].lstrip("-")
        aliases = {lowered_name, feature} if feature else {lowered_name}
        for alias in aliases:
            if alias and alias in lowered:
                found.append(project)
                break
    return tuple(found)
