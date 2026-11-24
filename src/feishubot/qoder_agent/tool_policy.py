from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any

from feishubot.qoder_agent.sql_guard import read_only_sql_denial_reason

DMS_SERVER = "aliyun-dms-mcp-server"
OBSERVABILITY_SERVER = "observability"
OBSERVABILITY_SERVERS = ("observability", "observability-prod")

LOCAL_READ_ONLY_TOOLS = ("Read", "Glob", "Grep")
MCP_META_TOOLS = ("mcp_list", "mcp_get", "mcp_call")
BLOCKED_TOOLS = ("Write", "Edit", "Bash")

# Dev-task mode: local tools allowed inside the dev workspace, guarded below.
DEV_LOCAL_TOOLS = ("Read", "Glob", "Grep", "Write", "Edit", "Bash")
# Dev full-capability mode: user-level MCP servers merged from ~/.qoder/mcp.json.
MYSQL_SERVERS = ("mysql-erp-dev", "mysql-scm-dev", "bi")
CLAUDE_MEM_SERVER = "claude-mem"
_MYSQL_TOOLS = ("execute_sql", "get_schema_info", "get_table_sample")
_DEV_READ_ROOTS = (
    Path.home() / ".qoder" / "rules",
    Path.home() / ".qoder" / "skills",
    Path.home() / ".agents" / "skills",
)
_PUSH_PROTECTED_BRANCHES = ("prod", "master", "main")
# Read-only mode with a shared knowledge base: Bash is limited to single
# read-only git commands (RUI-02 consistency checks); writes are limited to
# .md documents inside the knowledge base (RUI-03/04).
_READONLY_GIT_SUBCOMMANDS = ("log", "diff", "show", "status", "rev-parse", "ls-files")
_READONLY_GIT_PATTERN = re.compile(
    r"^git(?:\s+-C\s+(?P<cpath>\S+))?\s+"
    r"(?P<sub>log|diff|show|status|rev-parse|ls-files)(?:\s+.*)?$"
)
_SHELL_METACHARACTERS = ";&|`$><\n\r"
_DEV_FORBIDDEN_COMMAND_SUBSTRINGS = (
    "rm -rf",
    "rm -fr",
    "rd /s",
    "rmdir /s",
    "remove-item -recurse",
    "format ",
    "shutdown",
    "diskpart",
    "mkfs",
)

_DMS_PREFIX = f"mcp__{DMS_SERVER}__"
# tool -> (required fields, optional fields). Write/order tools
# (createDataChangeOrder, submitOrderApproval, approveOrder, getOrderInfo) and
# askDatabase are deliberately excluded so they fall through to the read-only
# denial; askDatabase runs server-generated SQL that cannot be guarded here.
_DMS_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "listTables": (frozenset({"database_id"}), frozenset({"search_name", "page_number", "page_size"})),
    "getTableDetailInfo": (frozenset({"table_guid"}), frozenset()),
    "executeScript": (frozenset({"database_id", "script"}), frozenset()),
    "answerSqlSyntax": (frozenset({"database_id", "question"}), frozenset({"model"})),
    "fixSql": (frozenset({"database_id", "sql", "error"}), frozenset({"question", "model"})),
    "optimizeSql": (frozenset({"database_id", "sql"}), frozenset({"question", "model"})),
}
_DMS_TOOLS = tuple(f"{_DMS_PREFIX}{name}" for name in _DMS_FIELDS)

_OBSERVABILITY_PREFIX = f"mcp__{OBSERVABILITY_SERVER}__"
# tool -> (required fields, optional fields); "health" is deliberately excluded.
_OBSERVABILITY_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "victorialogs_query": (frozenset({"query", "start", "end", "limit"}), frozenset()),
    "victorialogs_stats": (frozenset({"query", "start", "end"}), frozenset()),
    "victoriatraces_query": (frozenset({"query", "start", "end", "limit"}), frozenset()),
    "promql_query": (frozenset({"query"}), frozenset({"time"})),
    "promql_range": (frozenset({"query", "start", "end", "step"}), frozenset()),
    "list_skills": (frozenset(), frozenset()),
    "read_skill": (frozenset({"name"}), frozenset()),
    "read_skill_reference": (frozenset({"name", "ref"}), frozenset()),
    "read_skill_document": (frozenset({"name", "path"}), frozenset()),
    "search_skills": (frozenset({"query"}), frozenset({"max_results"})),
}
_OBSERVABILITY_TOOLS = tuple(f"{_OBSERVABILITY_PREFIX}{name}" for name in _OBSERVABILITY_FIELDS)

_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$"
)
_STEP_PATTERN = re.compile(r"^[1-9]\d{0,4}[smhd]$")
_SKILL_PATH_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]{0,199}$")
_MAX_QUERY_LENGTH = 2000
_MAX_TIME_SPAN_SECONDS = 24 * 60 * 60
_MAX_RESULT_LIMIT = 200
_MAX_SKILL_RESULTS = 20
_MAX_DMS_TEXT_LENGTH = 20000
_MAX_DMS_PAGE_SIZE = 200
_MAX_DMS_PAGE_NUMBER = 1_000_000
_DMS_DATABASE_ID_PATTERN = re.compile(r"^[1-9]\d{0,19}$")


def _bounded_int_denial(value: Any, field: str, upper: int) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
        return f"{field} must be an integer from 1 to {upper}"
    return None


def _parse_rfc3339(value: Any) -> datetime | None:
    if not isinstance(value, str) or not _RFC3339_PATTERN.fullmatch(value):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _query_text_denial(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "query must be a non-empty string"
    if len(value) > _MAX_QUERY_LENGTH:
        return f"query must not exceed {_MAX_QUERY_LENGTH} characters"
    if any(ord(character) < 32 and character not in "\t\r\n" for character in value):
        return "query contains a control character"
    return None


def _time_range_denial(tool_input: dict[str, Any]) -> str | None:
    start = _parse_rfc3339(tool_input["start"])
    end = _parse_rfc3339(tool_input["end"])
    if start is None or end is None:
        return "start and end must be RFC3339 timestamps with a timezone"
    span = (end - start).total_seconds()
    if span < 0:
        return "end must not be earlier than start"
    if span > _MAX_TIME_SPAN_SECONDS:
        return "time range must not exceed 24 hours"
    return None


def _skill_path_denial(value: Any, field: str) -> str | None:
    if (
        not isinstance(value, str)
        or not _SKILL_PATH_PATTERN.fullmatch(value)
        or ".." in PurePath(value).parts
    ):
        return f"{field} must be a relative skill path without '..'"
    return None


def _observability_denial_reason(tool: str, tool_input: dict[str, Any]) -> str | None:
    required, optional = _OBSERVABILITY_FIELDS[tool]
    fields = set(tool_input)
    if unknown := fields - required - optional:
        return f"unsupported {tool} fields: {', '.join(sorted(unknown))}"
    if missing := required - fields:
        hint = " (always pass limit <= 200)" if "limit" in missing else ""
        return f"{tool} requires fields: {', '.join(sorted(missing))}{hint}"

    if "query" in tool_input and (denial := _query_text_denial(tool_input["query"])):
        return denial
    if "start" in tool_input and (denial := _time_range_denial(tool_input)):
        return denial
    if "limit" in tool_input and (
        denial := _bounded_int_denial(tool_input["limit"], "limit", _MAX_RESULT_LIMIT)
    ):
        return denial
    if "time" in tool_input and _parse_rfc3339(tool_input["time"]) is None:
        return "time must be an RFC3339 timestamp with a timezone"
    if "step" in tool_input and not (
        isinstance(tool_input["step"], str) and _STEP_PATTERN.fullmatch(tool_input["step"])
    ):
        return "step must look like 30s, 5m, 1h, or 1d"
    if "max_results" in tool_input and (
        denial := _bounded_int_denial(
            tool_input["max_results"], "max_results", _MAX_SKILL_RESULTS
        )
    ):
        return denial
    for field in ("name", "ref", "path"):
        if field in tool_input and (denial := _skill_path_denial(tool_input[field], field)):
            return denial
    return None


def _dms_text_denial(value: Any, field: str) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return f"{field} must be a non-empty string"
    if len(value) > _MAX_DMS_TEXT_LENGTH:
        return f"{field} must not exceed {_MAX_DMS_TEXT_LENGTH} characters"
    if any(ord(character) < 32 and character not in "\t\r\n" for character in value):
        return f"{field} contains a control character"
    return None


def _dms_database_id_denial(value: Any) -> str | None:
    if not isinstance(value, str) or not _DMS_DATABASE_ID_PATTERN.fullmatch(value):
        return "database_id must be a positive numeric string"
    return None


def _dms_denial_reason(tool: str, tool_input: dict[str, Any]) -> str | None:
    required, optional = _DMS_FIELDS[tool]
    fields = set(tool_input)
    if unknown := fields - required - optional:
        return f"unsupported {tool} fields: {', '.join(sorted(unknown))}"
    if missing := required - fields:
        return f"{tool} requires fields: {', '.join(sorted(missing))}"
    if "database_id" in tool_input and (denial := _dms_database_id_denial(tool_input["database_id"])):
        return denial

    if tool == "executeScript":
        script = tool_input.get("script")
        if not isinstance(script, str):
            return "executeScript script must be a string"
        return read_only_sql_denial_reason(script)

    for field in ("question", "sql", "error", "table_guid", "model"):
        if field in tool_input and (denial := _dms_text_denial(tool_input[field], field)):
            return denial
    if "search_name" in tool_input:
        search_name = tool_input["search_name"]
        if search_name is not None and (
            denial := _dms_text_denial(search_name, "search_name")
        ):
            return denial
    if "page_number" in tool_input and (
        denial := _bounded_int_denial(
            tool_input["page_number"], "page_number", _MAX_DMS_PAGE_NUMBER
        )
    ):
        return denial
    if "page_size" in tool_input and (
        denial := _bounded_int_denial(
            tool_input["page_size"], "page_size", _MAX_DMS_PAGE_SIZE
        )
    ):
        return denial
    return None


class ToolPolicy:
    def __init__(
        self,
        workspace_root: Path,
        mcp_server_names: Iterable[str] = (DMS_SERVER,),
        *,
        knowledge_dir: Path | None = None,
    ) -> None:
        self._workspace_root = workspace_root
        self._knowledge_dir = knowledge_dir.resolve() if knowledge_dir else None
        self._extra_read_roots: tuple[Path, ...] = (
            (self._knowledge_dir,) if self._knowledge_dir else ()
        )
        names = tuple(mcp_server_names)
        self.mcp_server_names = tuple(
            name
            for name in (DMS_SERVER, *OBSERVABILITY_SERVERS)
            if name in names
        )
        self._observability_server = next(
            (name for name in OBSERVABILITY_SERVERS if name in self.mcp_server_names),
            None,
        )
        mcp_tools: tuple[str, ...] = ()
        if DMS_SERVER in self.mcp_server_names:
            mcp_tools += _DMS_TOOLS
        if self.observability_enabled:
            for obs_name in OBSERVABILITY_SERVERS:
                if obs_name in self.mcp_server_names:
                    prefix = f"mcp__{obs_name}__"
                    mcp_tools += tuple(f"{prefix}{name}" for name in _OBSERVABILITY_FIELDS)
        self._mcp_tools = mcp_tools
        self.allowed_tools = LOCAL_READ_ONLY_TOOLS + mcp_tools + MCP_META_TOOLS
        if self._knowledge_dir is not None:
            self.allowed_tools += ("Write", "Edit", "Bash")

    @property
    def observability_enabled(self) -> bool:
        return self._observability_server is not None

    def denial_reason(self, tool_name: str, tool_input: Any) -> str | None:
        if tool_name in LOCAL_READ_ONLY_TOOLS:
            return self._local_tool_denial_reason(tool_name, tool_input)
        if tool_name in ("Write", "Edit"):
            if self._knowledge_dir is None:
                return f"tool is not allowed in read-only mode: {tool_name}"
            return self._knowledge_write_denial_reason(tool_name, tool_input)
        if tool_name == "Bash":
            if self._knowledge_dir is None:
                return f"tool is not allowed in read-only mode: {tool_name}"
            return _readonly_git_denial_reason(
                tool_input, self._workspace_root, self._knowledge_dir
            )
        if tool_name in MCP_META_TOOLS:
            return self._mcp_meta_tool_denial_reason(tool_name, tool_input)
        if tool_name not in self._mcp_tools:
            return f"tool is not allowed in read-only mode: {tool_name}"
        if not isinstance(tool_input, dict):
            return "tool input must be an object"
        if tool_name.startswith(_DMS_PREFIX):
            return _dms_denial_reason(tool_name.removeprefix(_DMS_PREFIX), tool_input)
        for obs_name in OBSERVABILITY_SERVERS:
            prefix = f"mcp__{obs_name}__"
            if tool_name.startswith(prefix):
                return _observability_denial_reason(
                    tool_name.removeprefix(prefix),
                    tool_input,
                )
        return f"tool is not allowed in read-only mode: {tool_name}"

    def _mcp_meta_tool_denial_reason(self, tool_name: str, tool_input: Any) -> str | None:
        if not isinstance(tool_input, dict):
            return "tool input must be an object"
        if tool_name == "mcp_list":
            if unknown_keys := set(tool_input) - {"keyword", "includeDescription"}:
                return f"unsupported mcp_list fields: {', '.join(sorted(unknown_keys))}"
            keyword = tool_input.get("keyword")
            include_description = tool_input.get("includeDescription")
            if keyword is not None and not isinstance(keyword, str):
                return "mcp_list keyword must be a string"
            if include_description is not None and not isinstance(include_description, bool):
                return "mcp_list includeDescription must be a boolean"
            return None

        if tool_name == "mcp_get":
            if set(tool_input) != {"toolName"}:
                return "mcp_get requires only the toolName field"
            if tool_input.get("toolName") not in self._mcp_tools:
                return "mcp_get target is not an approved MCP tool"
            return None

        if set(tool_input) != {"toolName", "arguments"}:
            return "mcp_call requires only toolName and arguments"
        target_tool = tool_input.get("toolName")
        if target_tool not in self._mcp_tools:
            return "mcp_call target is not an approved MCP tool"
        return self.denial_reason(str(target_tool), tool_input.get("arguments"))

    def _local_tool_denial_reason(self, tool_name: str, tool_input: Any) -> str | None:
        if not isinstance(tool_input, dict):
            return "tool input must be an object"

        for key in ("file_path", "path", "directory", "root"):
            value = tool_input.get(key)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for raw_path in values:
                if not isinstance(raw_path, str) or not (
                    self._is_inside_workspace(raw_path)
                    or _is_in_dev_read_roots(raw_path, self._extra_read_roots)
                ):
                    return f"path is outside the approved workspace: {raw_path}"
                if _is_secret_file(raw_path):
                    return f"reading environment secret files is not allowed: {raw_path}"

        if tool_name == "Glob":
            pattern = tool_input.get("pattern")
            if isinstance(pattern, str) and _unsafe_glob_pattern(pattern):
                if not (
                    PurePath(pattern).is_absolute()
                    and _is_in_dev_read_roots(pattern, self._extra_read_roots)
                ):
                    return "glob pattern may not escape the approved workspace"
        return None

    def _is_inside_workspace(self, raw_path: str) -> bool:
        if not raw_path.strip() or raw_path.startswith("~"):
            return False
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self._workspace_root / candidate
        try:
            candidate.resolve().relative_to(self._workspace_root)
        except (OSError, ValueError):
            return False
        return True

    def _knowledge_write_denial_reason(self, tool_name: str, tool_input: Any) -> str | None:
        if not isinstance(tool_input, dict):
            return "tool input must be an object"
        raw_path = next(
            (
                tool_input[key]
                for key in ("file_path", "path", "target_file", "target")
                if isinstance(tool_input.get(key), str)
            ),
            None,
        )
        if raw_path is None or not raw_path.strip():
            return f"{tool_name} requires a file path inside the knowledge base"
        if not _path_in_root(raw_path, self._knowledge_dir):
            return f"path is outside the knowledge base: {raw_path}"
        if PurePath(raw_path).suffix.lower() != ".md":
            return "only .md knowledge documents may be written in read-only mode"
        if _is_git_internal_path(raw_path):
            return "modifying .git internals is not allowed"
        if _is_secret_file(raw_path):
            return f"modifying environment secret files is not allowed: {raw_path}"
        return None


def _is_secret_file(raw_path: str) -> bool:
    name = PurePath(raw_path).name.lower()
    return name == ".env" or (name.startswith(".env.") and name != ".env.example")


def _unsafe_glob_pattern(pattern: str) -> bool:
    path = PurePath(pattern)
    return path.is_absolute() or ".." in path.parts or pattern.startswith("~")


def _is_in_dev_read_roots(raw_path: str, extra_roots: tuple[Path, ...] = ()) -> bool:
    if not raw_path.strip():
        return False
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        return False
    # The literal-path junction check below does not normalize '..', so reject
    # traversal parts up front; resolve() alone would hide them.
    if ".." in candidate.parts:
        return False
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    absolute = candidate.absolute()
    for root in (*_DEV_READ_ROOTS, *extra_roots):
        if resolved == root or root in resolved.parents:
            return True
        # Skill dirs may be junctions (e.g. ~/.qoder/skills/<name> ->
        # ~/.agents/skills/<name>): the literal path sits under a root even
        # when its resolved target does not.
        if absolute == root or root in absolute.parents:
            return True
    return False


def _path_in_root(raw_path: str, root: Path) -> bool:
    """Relative paths resolve against the root; '~' and escapes are rejected."""
    if not raw_path.strip() or raw_path.startswith("~"):
        return False
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate.resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _readonly_git_denial_reason(
    tool_input: Any, workspace_root: Path, knowledge_dir: Path
) -> str | None:
    if not isinstance(tool_input, dict):
        return "tool input must be an object"
    command = next(
        (
            tool_input[key]
            for key in ("command", "cmd", "script")
            if isinstance(tool_input.get(key), str)
        ),
        None,
    )
    if command is None or not command.strip():
        return "Bash requires a command string"
    if any(character in command for character in _SHELL_METACHARACTERS):
        return "read-only mode allows a single git command without shell operators"
    lowered = command.lower()
    for flag in ("--output", "--no-index", "--exec-path"):
        if flag in lowered:
            return f"{flag} is not allowed in read-only mode"
    if re.search(r"(^|\s)-c(\s|$)", command):
        return "git -c configuration overrides are not allowed in read-only mode"
    match = _READONLY_GIT_PATTERN.match(command.strip())
    if match is None:
        allowed = "/".join(_READONLY_GIT_SUBCOMMANDS)
        return f"read-only mode only allows git {allowed}"
    cpath = match.group("cpath")
    if cpath and not (
        _path_in_root(cpath, workspace_root) or _path_in_root(cpath, knowledge_dir)
    ):
        return f"git -C path is outside the approved workspace: {cpath}"
    return None


def _mysql_denial_reason(tool: str, tool_input: Any) -> str | None:
    if not isinstance(tool_input, dict):
        return "tool input must be an object"
    if tool == "execute_sql":
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            return "execute_sql requires a non-empty query string"
        return read_only_sql_denial_reason(query)
    if tool in ("get_schema_info", "get_table_sample"):
        return None
    return f"tool is not allowed in dev mode: {tool}"


class DevToolPolicy(ToolPolicy):
    """Dev-task policy: read/write/edit/bash inside the dev workspace with git guards.

    MCP tools (DMS / observability) stay read-only via the inherited checks.
    """

    def __init__(
        self,
        workspace_root: Path,
        mcp_server_names: Iterable[str] = (DMS_SERVER,),
        *,
        knowledge_dir: Path | None = None,
    ) -> None:
        names = tuple(mcp_server_names)
        base_names = tuple(
            name for name in names if name in (DMS_SERVER, *OBSERVABILITY_SERVERS)
        )
        super().__init__(workspace_root, base_names, knowledge_dir=knowledge_dir)
        self._extra_mcp_servers = tuple(
            name for name in names if name in (*MYSQL_SERVERS, CLAUDE_MEM_SERVER)
        )
        mysql_tools: tuple[str, ...] = ()
        for server in self._extra_mcp_servers:
            if server in MYSQL_SERVERS:
                mysql_tools += tuple(f"mcp__{server}__{tool}" for tool in _MYSQL_TOOLS)
        self._mcp_tools = self._mcp_tools + mysql_tools
        self.mcp_server_names = self.mcp_server_names + self._extra_mcp_servers
        self.allowed_tools = (
            DEV_LOCAL_TOOLS + ("Skill",) + self._mcp_tools + MCP_META_TOOLS
        )

    def denial_reason(self, tool_name: str, tool_input: Any) -> str | None:
        if tool_name in ("Write", "Edit"):
            return self._dev_write_denial_reason(tool_name, tool_input)
        if tool_name == "Bash":
            return _bash_denial_reason(tool_input)
        if tool_name in LOCAL_READ_ONLY_TOOLS:
            return self._dev_read_denial_reason(tool_name, tool_input)
        if tool_name in MCP_META_TOOLS:
            return self._mcp_meta_tool_denial_reason(tool_name, tool_input)
        if tool_name.startswith("mcp__"):
            for server in MYSQL_SERVERS:
                prefix = f"mcp__{server}__"
                if tool_name.startswith(prefix):
                    return _mysql_denial_reason(tool_name.removeprefix(prefix), tool_input)
            if tool_name.startswith(f"mcp__{CLAUDE_MEM_SERVER}__"):
                return None if isinstance(tool_input, dict) else "tool input must be an object"
            return super().denial_reason(tool_name, tool_input)
        # Full capability parity with the IDE: Skill / Task / WebSearch / TodoWrite
        # and any other non-MCP agent tool are allowed; the dangerous surfaces
        # above (writes, bash, MCP) stay guarded.
        return None

    def _dev_read_denial_reason(self, tool_name: str, tool_input: Any) -> str | None:
        if not isinstance(tool_input, dict):
            return "tool input must be an object"
        for key in ("file_path", "path", "directory", "root"):
            value = tool_input.get(key)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for raw_path in values:
                if not isinstance(raw_path, str):
                    return f"path is outside the approved workspace: {raw_path}"
                if _is_secret_file(raw_path):
                    return f"reading environment secret files is not allowed: {raw_path}"
                if not (
                    self._is_inside_workspace(raw_path)
                    or _is_in_dev_read_roots(raw_path, self._extra_read_roots)
                ):
                    return f"path is outside the approved workspace: {raw_path}"
        if tool_name == "Glob":
            pattern = tool_input.get("pattern")
            if isinstance(pattern, str) and _unsafe_glob_pattern(pattern):
                if not (
                    PurePath(pattern).is_absolute()
                    and _is_in_dev_read_roots(pattern, self._extra_read_roots)
                ):
                    return "glob pattern may not escape the approved workspace"
        return None

    def _dev_write_denial_reason(self, tool_name: str, tool_input: Any) -> str | None:
        if not isinstance(tool_input, dict):
            return "tool input must be an object"
        raw_path = next(
            (
                tool_input[key]
                for key in ("file_path", "path", "target_file", "target")
                if isinstance(tool_input.get(key), str)
            ),
            None,
        )
        if raw_path is None or not raw_path.strip():
            return f"{tool_name} requires a file path inside the dev workspace"
        in_knowledge = self._knowledge_dir is not None and _path_in_root(
            raw_path, self._knowledge_dir
        )
        if not self._is_inside_workspace(raw_path) and not in_knowledge:
            return f"path is outside the dev workspace: {raw_path}"
        if in_knowledge and not self._is_inside_workspace(raw_path):
            if PurePath(raw_path).suffix.lower() != ".md":
                return "only .md knowledge documents may be written outside the dev workspace"
        if _is_git_internal_path(raw_path):
            return "modifying .git internals is not allowed in dev mode"
        if _is_secret_file(raw_path):
            return f"modifying environment secret files is not allowed: {raw_path}"
        return None


def _is_git_internal_path(raw_path: str) -> bool:
    return ".git" in [part.lower() for part in PurePath(raw_path).parts]


def _bash_denial_reason(tool_input: Any) -> str | None:
    if not isinstance(tool_input, dict):
        return "tool input must be an object"
    command = next(
        (
            tool_input[key]
            for key in ("command", "cmd", "script")
            if isinstance(tool_input.get(key), str)
        ),
        None,
    )
    if command is None or not command.strip():
        return "Bash requires a command string"
    lowered = command.lower()
    for pattern in _DEV_FORBIDDEN_COMMAND_SUBSTRINGS:
        if pattern in lowered:
            return f"command is not allowed in dev mode: {pattern.strip()}"
    if "git push" in lowered:
        return _git_push_denial_reason(command)
    return None


def _git_push_denial_reason(command: str) -> str | None:
    match = re.search(r"\bgit\s+push\b(.*)", command, flags=re.IGNORECASE)
    if match is None:
        return None
    args = match.group(1)
    if re.search(r"(^|\s)(--force[a-z-]*|-f)(\s|$)", args):
        return "force push (incl. --force-with-lease) is not allowed in dev mode"
    # Remote/branch/refspec tokens: split on whitespace, comma, colon, and slash so
    # "HEAD:prod", "refs/heads/prod" and "origin/prod" all expose the branch name.
    tokens = (token for token in re.split(r"[\s,:/]+", args) if token)
    for token in tokens:
        if token.lstrip("-").lower() in _PUSH_PROTECTED_BRANCHES:
            return f"pushing to protected branch is not allowed: {token}"
    if not args.strip():
        return "git push requires an explicit remote and feature branch, e.g. git push origin HEAD:feature-..."
    return None
