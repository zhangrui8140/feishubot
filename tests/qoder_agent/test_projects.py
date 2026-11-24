from __future__ import annotations

from feishubot.qoder_agent.projects import list_projects, mentioned_projects


def test_lists_only_prod_project_directories(tmp_path):
    for name in ("scm-prod", "erp-mall-prod", "rui", "feishubot"):
        (tmp_path / name).mkdir()
    (tmp_path / "notes-prod").write_text("not a directory", encoding="utf-8")

    assert list_projects(tmp_path) == ("erp-mall-prod", "scm-prod")


def test_missing_workspace_lists_no_projects(tmp_path):
    assert list_projects(tmp_path / "missing") == ()


def test_mentioned_projects_matches_name_without_suffix_but_not_substrings():
    projects = ("erp-distribute-prod", "som-erp-distribute-prod", "som-prod")

    assert mentioned_projects("ERP-Distribute 报错", projects) == ("erp-distribute-prod",)
    assert mentioned_projects("som-erp-distribute-prod 报错", projects) == (
        "som-erp-distribute-prod",
    )
    assert mentioned_projects("som的接口", projects) == ("som-prod",)
    assert mentioned_projects("门店订单保存报错", projects) == ()
