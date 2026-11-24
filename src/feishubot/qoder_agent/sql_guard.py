from __future__ import annotations

_ALLOWED_SQL_STARTS = {"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC"}
_BLOCKED_SQL_TOKENS = {
    "ALTER",
    "BEGIN",
    "BENCHMARK",
    "CALL",
    "COMMIT",
    "CREATE",
    "DEALLOCATE",
    "DELETE",
    "DO",
    "DROP",
    "DUMPFILE",
    "EVENT",
    "EXECUTE",
    "FUNCTION",
    "GET_LOCK",
    "GRANT",
    "HANDLER",
    "INFILE",
    "INSERT",
    "INSTALL",
    "INTO",
    "LOAD",
    "LOAD_FILE",
    "LOCK",
    "MERGE",
    "PREPARE",
    "PROCEDURE",
    "RELEASE_LOCK",
    "RENAME",
    "REPLACE",
    "REVOKE",
    "ROLLBACK",
    "SAVEPOINT",
    "SET",
    "SLEEP",
    "START",
    "TRANSACTION",
    "TRIGGER",
    "TRUNCATE",
    "UNINSTALL",
    "UNLOCK",
    "UPDATE",
    "UPSERT",
    "USE",
}
_BLOCKED_SQL_PAIRS = {("FOR", "SHARE"), ("FOR", "UPDATE")}


def _skip_quoted_sql_value(query: str, start: int, quote: str) -> int | None:
    index = start + 1
    while index < len(query):
        character = query[index]
        if quote != "`" and character == "\\":
            index += 2
            continue
        if character == quote:
            if index + 1 < len(query) and query[index + 1] == quote:
                index += 2
                continue
            return index + 1
        index += 1
    return None


def _scan_sql_tokens(query: str) -> tuple[list[tuple[str, int]], str | None]:
    tokens: list[tuple[str, int]] = []
    depth = 0
    index = 0
    while index < len(query):
        character = query[index]
        if ord(character) < 32 and character not in "\t\r\n":
            return [], "SQL contains a control character"
        if character in "'\"`":
            next_index = _skip_quoted_sql_value(query, index, character)
            if next_index is None:
                return [], "SQL contains an unterminated quoted value"
            index = next_index
            continue
        if character == ";":
            return [], "multiple or terminated SQL statements are not allowed"
        if character == "#" or query.startswith("--", index) or query.startswith("/*", index):
            return [], "SQL comments are not allowed"
        if character == "@":
            return [], "SQL variables are not allowed"
        if character == "(":
            depth += 1
            index += 1
            continue
        if character == ")":
            depth -= 1
            if depth < 0:
                return [], "SQL parentheses are unbalanced"
            index += 1
            continue
        if character.isascii() and (character.isalnum() or character in "_$"):
            start = index
            index += 1
            while index < len(query):
                current = query[index]
                if not current.isascii() or not (current.isalnum() or current in "_$"):
                    break
                index += 1
            tokens.append((query[start:index].upper(), depth))
            continue
        index += 1

    if depth != 0:
        return [], "SQL parentheses are unbalanced"
    return tokens, None


def read_only_sql_denial_reason(query: str) -> str | None:
    if not query.strip():
        return "SQL query must not be empty"

    keywords, scan_error = _scan_sql_tokens(query)
    if scan_error:
        return scan_error
    if not keywords or keywords[0][1] != 0 or keywords[0][0] not in _ALLOWED_SQL_STARTS:
        return "only SELECT, WITH SELECT, SHOW, and DESCRIBE statements are allowed"

    first_keyword = keywords[0][0]
    if first_keyword == "WITH" and not any(
        keyword == "SELECT" and depth == 0 for keyword, depth in keywords[1:]
    ):
        return "WITH statements must end in a top-level SELECT"

    keyword_values = [keyword for keyword, _depth in keywords]
    blocked_keyword = next(
        (keyword for keyword in keyword_values if keyword in _BLOCKED_SQL_TOKENS),
        None,
    )
    if blocked_keyword:
        return f"SQL token is not allowed in read-only mode: {blocked_keyword}"
    keyword_pairs = zip(keyword_values, keyword_values[1:], strict=False)
    if any(pair in _BLOCKED_SQL_PAIRS for pair in keyword_pairs):
        return "locking SELECT statements are not allowed"
    return None
