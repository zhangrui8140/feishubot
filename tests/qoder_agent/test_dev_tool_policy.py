from __future__ import annotations

from pathlib import Path

import pytest

from feishubot.qoder_agent.projects import list_dev_projects, mentioned_dev_projects
from feishubot.qoder_agent.tool_policy import DevToolPolicy


@pytest.fixture()
def policy(tmp_path: Path) -> DevToolPolicy:
    return DevToolPolicy(tmp_path)


@pytest.mark.parametrize(
    "command",
    [
        "git push origin HEAD:prod",
        "git push origin master",
        "git push origin main",
        "git push origin refs/heads/prod",
        "git push origin prod feature-x",
        "git push origin/prod",
        "git push --force origin feature-x",
        "git push --force-with-lease origin feature-x",
        "git push -f origin feature-x",
        "git push",
    ],
)
def test_bash_denies_unsafe_push(policy: DevToolPolicy, command: str) -> None:
    assert policy.denial_reason("Bash", {"command": command}) is not None


@pytest.mark.parametrize(
    "command",
    [
        "git push origin HEAD:feature-erp-erp-purchase-R7154951404-1109",
        "git push -u origin feature-x",
        'git add . && git commit -m "fix" && git push origin HEAD:feature-x',
        "mvn -q -pl erp-mall -am test",
        "git status",
    ],
)
def test_bash_allows_feature_push_and_builds(policy: DevToolPolicy, command: str) -> None:
    assert policy.denial_reason("Bash", {"command": command}) is None


@pytest.mark.parametrize(
    "command",
    ["rm -rf /", "shutdown /s", "format C:", "Remove-Item -Recurse -Force ."],
)
def test_bash_denies_destructive_commands(policy: DevToolPolicy, command: str) -> None:
    assert policy.denial_reason("Bash", {"command": command}) is not None


def test_write_inside_workspace_allowed(policy: DevToolPolicy, tmp_path: Path) -> None:
    target = tmp_path / "src" / "A.java"
    assert policy.denial_reason("Write", {"file_path": str(target)}) is None
    assert policy.denial_reason("Edit", {"file_path": "src/A.java"}) is None


def test_write_outside_workspace_denied(policy: DevToolPolicy) -> None:
    assert (
        policy.denial_reason("Write", {"file_path": r"C:\Windows\system32\evil.bat"})
        is not None
    )


def test_write_git_internals_denied(policy: DevToolPolicy, tmp_path: Path) -> None:
    assert (
        policy.denial_reason("Write", {"file_path": str(tmp_path / "proj" / ".git" / "config")})
        is not None
    )


def test_write_secret_file_denied(policy: DevToolPolicy, tmp_path: Path) -> None:
    assert policy.denial_reason("Write", {"file_path": str(tmp_path / "proj" / ".env")}) is not None


def test_read_still_guards_workspace(policy: DevToolPolicy) -> None:
    assert policy.denial_reason("Read", {"file_path": r"C:\other\file.txt"}) is not None


def test_dev_local_tools_enabled(policy: DevToolPolicy) -> None:
    for tool in ("Write", "Edit", "Bash"):
        assert tool in policy.allowed_tools


def test_dev_projects_listing(tmp_path: Path) -> None:
    (tmp_path / "erp-purchase-dev").mkdir()
    (tmp_path / "erp-purchase-dev-大仓端堆活动").mkdir()
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "dev-workspace.code-workspace").write_text("{}", encoding="utf-8")
    (tmp_path / ".cache").mkdir()
    assert list_dev_projects(tmp_path) == (
        "erp-purchase-dev",
        "erp-purchase-dev-大仓端堆活动",
    )


def test_mentioned_dev_projects() -> None:
    projects = ("erp-purchase-dev", "erp-purchase-dev-大仓端堆活动")
    assert mentioned_dev_projects("修复 大仓端堆活动 的校验", projects) == (
        "erp-purchase-dev-大仓端堆活动",
    )
    assert mentioned_dev_projects("erp-purchase-dev 修复xxx", projects) == (
        "erp-purchase-dev",
    )
    assert mentioned_dev_projects("和项目无关的消息", projects) == ()


def test_dev_knowledge_base_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "dev"
    workspace.mkdir()
    knowledge = tmp_path / "rui"
    knowledge.mkdir()
    policy = DevToolPolicy(workspace, knowledge_dir=knowledge)

    assert policy.denial_reason(
        "Read", {"file_path": str(knowledge / "scm-dev" / "index.md")}
    ) is None
    assert policy.denial_reason(
        "Write", {"file_path": str(knowledge / "scm-dev" / "modules" / "x.md")}
    ) is None
    assert policy.denial_reason(
        "Write", {"file_path": str(knowledge / "scm-dev" / "x.java")}
    ) is not None
    # Workspace writes keep their original freedom.
    assert policy.denial_reason(
        "Write", {"file_path": str(workspace / "erp-purchase-dev" / "A.java")}
    ) is None
