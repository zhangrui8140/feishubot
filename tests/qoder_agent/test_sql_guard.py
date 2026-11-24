from __future__ import annotations

import pytest

from feishubot.qoder_agent.sql_guard import read_only_sql_denial_reason


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, name FROM users WHERE status = 1",
        "SELECT `id`, `user_name` FROM `mall_orders`",
        "SELECT `col``` FROM `table`",
        "WITH summary AS (SELECT dept, count(*) as cnt FROM emp GROUP BY dept) SELECT * FROM summary",
        "SHOW TABLES",
        "SHOW DATABASES",
        "DESCRIBE orders",
        "DESC mall_product",
    ],
)
def test_valid_read_only_sql_allowed(query: str):
    assert read_only_sql_denial_reason(query) is None


@pytest.mark.parametrize(
    ("query", "expected_fragment"),
    [
        ("", "must not be empty"),
        ("   ", "must not be empty"),
        ("INSERT INTO users VALUES (1, 'alice')", "only SELECT"),
        ("UPDATE users SET name = 'bob'", "only SELECT"),
        ("DELETE FROM users WHERE id = 1", "only SELECT"),
        ("DROP TABLE orders", "only SELECT"),
        ("ALTER TABLE orders ADD COLUMN x INT", "only SELECT"),
        ("SELECT * FROM orders; DROP TABLE users", "multiple or terminated"),
        ("SELECT * FROM orders -- comment", "comments are not allowed"),
        ("SELECT * FROM orders /* comment */", "comments are not allowed"),
        ("SELECT * FROM orders # comment", "comments are not allowed"),
        ("SELECT * FROM orders FOR UPDATE", "UPDATE"),
        ("SELECT * FROM orders FOR SHARE", "locking SELECT"),
        ("SELECT @user_var FROM orders", "variables are not allowed"),
        # 反引号转义注入测试：反斜杠不可转义反引号，INTO OUTFILE 必须被检测并拦截
        ("SELECT `col\\` INTO OUTFILE '/tmp/hack', `col2` FROM tbl", "INTO"),
        ("SELECT * INTO DUMPFILE '/tmp/hack' FROM tbl", "INTO"),
        ("WITH cte AS (SELECT 1) DELETE FROM cte", "WITH statements must end in a top-level SELECT"),
    ],
)
def test_blocked_or_unsafe_sql_denied(query: str, expected_fragment: str):
    denial = read_only_sql_denial_reason(query)
    assert denial is not None
    assert expected_fragment.lower() in denial.lower()
