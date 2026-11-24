from __future__ import annotations

import json
import os

import pytest

from feishubot.qoder_agent.domain_knowledge import DomainKnowledgeStore, split_record
from feishubot.qoder_agent.models import AgentRequest, AgentResult, SessionKey
from feishubot.qoder_agent.service import QoderAgentService, QoderServiceConfig

_PROJECT = "erp-mall-prod"


def _record_text(**overrides) -> str:
    payload = {
        "project": _PROJECT,
        "module": "门店订单审核",
        "summary": "门店订单审核流程",
        "keywords": ["storeorder", "门店订单"],
        "files": [f"{_PROJECT}/src/Order.java:12"],
        "notes": "- 审核前校验库存",
    }
    payload.update(overrides)
    return "【最终结论】库存不足\n\n【领域记录】\n" + json.dumps(payload, ensure_ascii=False)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    source = root / _PROJECT / "src"
    source.mkdir(parents=True)
    (source / "Order.java").write_text("class Order {}", encoding="utf-8")
    (root / _PROJECT / ".env").write_text("SECRET=1", encoding="utf-8")
    (root / "erp-mall-dev").mkdir()
    return root


def test_split_record_strips_block_and_parses_payload():
    answer, payload = split_record(_record_text())

    assert answer == "【最终结论】库存不足"
    assert payload["module"] == "门店订单审核"


def test_split_record_tolerates_fences_and_rejects_garbage():
    fenced = '结论\n【领域记录】\n```json\n{"module": "a", "notes": "x\ny"}\n```'
    assert split_record(fenced) == ("结论", {"module": "a", "notes": "x\ny"})
    assert split_record("结论\n【领域记录】\nnot json") == ("结论", None)
    assert split_record("纯结论") == ("纯结论", None)


def test_record_keeps_only_safe_files_inside_prod_project(workspace, tmp_path):
    store = DomainKnowledgeStore(tmp_path / "knowledge", workspace)
    _answer, payload = split_record(
        _record_text(files=[f"{_PROJECT}/src/Order.java:12", ".env", "../erp-mall-dev/x"])
    )

    assert store.record(payload) == "门店订单审核"
    assert store.record({**payload, "project": "erp-mall-dev"}) is None

    (match,) = store.match("门店订单审核失败", (_PROJECT,))
    assert match.files == (f"{_PROJECT}/src/Order.java",)
    assert match.changed_files == ()


def test_match_reports_changed_files_and_context(workspace, tmp_path):
    store = DomainKnowledgeStore(tmp_path / "knowledge", workspace)
    store.record(split_record(_record_text())[1])

    assert store.match("和商品无关", (_PROJECT,)) == []
    assert "代码未变更" in store.build_context("hxl.erp.storeorder.audit 报错", (_PROJECT,))

    source = workspace / _PROJECT / "src" / "Order.java"
    source.write_text("class Order { int changed; }", encoding="utf-8")
    context = store.build_context("storeorder 报错", (_PROJECT,))
    assert f"代码已变更：{_PROJECT}/src/Order.java" in context

    store.record(split_record(_record_text(files=[]))[1])
    assert store.match("storeorder", (_PROJECT,))[0].changed_files == ()


def test_touch_without_content_change_is_still_fresh(workspace, tmp_path):
    store = DomainKnowledgeStore(tmp_path / "knowledge", workspace)
    store.record(split_record(_record_text())[1])
    source = workspace / _PROJECT / "src" / "Order.java"
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))

    assert store.match("storeorder", (_PROJECT,))[0].changed_files == ()


def test_unmatched_question_still_lists_existing_modules(workspace, tmp_path):
    store = DomainKnowledgeStore(tmp_path / "knowledge", workspace)
    store.record(split_record(_record_text())[1])

    context = store.build_context("商品价格", (_PROJECT,))
    assert "未命中" in context
    assert f"{_PROJECT}：门店订单审核" in context
    assert store.build_context("商品价格", ("other-prod",)) == ""


class _Handle:
    session_id = "session-1"

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.prompts: list[str] = []

    async def run(self, prompt: str, *, images=(), on_progress=None) -> AgentResult:
        self.prompts.append(prompt)
        return AgentResult(self.answers.pop(0), "session-1", 1, 1, ())

    async def inject(self, text: str) -> bool:
        return False

    async def interrupt(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None


class _Adapter:
    observability_enabled = False

    def __init__(self, handle: _Handle) -> None:
        self.handle = handle

    def validate(self) -> None:
        return None

    async def open_session(self, resume_session_id=None) -> _Handle:
        return self.handle


@pytest.mark.asyncio
async def test_service_records_domain_and_reuses_it_next_time(workspace, tmp_path):
    handle = _Handle([_record_text(), "【最终结论】沿用领域记录"])
    service = QoderAgentService(
        QoderServiceConfig(
            workspace_root=workspace,
            model="auto",
            session_store_dir=tmp_path / "sessions",
            domain_knowledge_dir=tmp_path / "knowledge",
        ),
        adapter=_Adapter(handle),
    )
    await service.start()
    key = SessionKey("chat", "root", "user")
    try:
        first = await service.run(AgentRequest("m1", key, "erp-mall 门店订单审核失败"))
        second = await service.run(AgentRequest("m2", key, "storeorder 又报错了"))
    finally:
        await service.close()

    assert first.text == "【最终结论】库存不足"
    assert second.text == "【最终结论】沿用领域记录"
    assert handle.prompts[0] == "erp-mall 门店订单审核失败"
    assert handle.prompts[1].startswith("【领域知识（来自既往分析）】")
    assert "代码未变更" in handle.prompts[1]
    assert handle.prompts[1].endswith("【用户问题】\nstoreorder 又报错了")


@pytest.mark.asyncio
async def test_service_without_domain_dir_leaves_answer_untouched(workspace, tmp_path):
    handle = _Handle([_record_text()])
    service = QoderAgentService(
        QoderServiceConfig(
            workspace_root=workspace, model="auto", session_store_dir=tmp_path / "sessions"
        ),
        adapter=_Adapter(handle),
    )
    await service.start()
    try:
        result = await service.run(AgentRequest("m1", SessionKey("c", "r", "u"), "问题"))
    finally:
        await service.close()

    assert "【领域记录】" in result.text
