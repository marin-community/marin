# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the nightshift agent prompts.

Issue #3782: the scheduled nightshift prompts in `infra/scripts/nightshift_*.py`
used to instruct agents to write stylized PR/issue bodies (haiku epigraphs) that
conflict with the canonical plain-text PR style. These tests pin the prompts to
the post-fix contract:

- no "haiku" / "epigraph" instructions anywhere
- prompts that open PRs point at `.agents/skills/author-pr/SKILL.md`
- prompts that open PRs require an issue-linked PR body (`Fixes #NNNN` /
  `Part of #NNNN`)
- every prompt still `.format()`-renders without unbound placeholders, using the
  exact kwargs their caller passes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "infra" / "scripts"


def _load(name: str) -> ModuleType:
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rendered_prompts() -> dict[str, str]:
    """Render every nightshift prompt with the kwargs its caller actually uses."""
    doc_drift = _load("nightshift_doc_drift")
    cleanup = _load("nightshift_cleanup")
    ci_tests = _load("nightshift_ci_tests")

    return {
        "doc_drift": doc_drift.DOC_DRIFT_PROMPT.format(run_id="abc", run_attempt="1"),
        "scout": cleanup.SCOUT_PROMPT.format(
            subproject="lib/marin/src/marin", result_file="/tmp/x.json"
        ),
        "merge": cleanup.MERGE_PROMPT.format(
            scout_results="X", worktree_info="Y", date="20260530"
        ),
        "ci_tests": ci_tests.build_prompt(
            date="2026-05-30",
            candidate_file=Path("/tmp/c.json"),
            log_root=Path("/tmp/l"),
            repo="o/r",
            candidates=[],
        ),
    }


PR_AUTHORING_PROMPTS = ("doc_drift", "merge", "ci_tests")


@pytest.mark.parametrize("name", ["doc_drift", "scout", "merge", "ci_tests"])
def test_no_haiku_or_epigraph(rendered_prompts: dict[str, str], name: str) -> None:
    text = rendered_prompts[name].lower()
    assert "haiku" not in text
    assert "epigraph" not in text


@pytest.mark.parametrize("name", PR_AUTHORING_PROMPTS)
def test_points_at_author_pr_skill(rendered_prompts: dict[str, str], name: str) -> None:
    assert ".agents/skills/author-pr/SKILL.md" in rendered_prompts[name]


@pytest.mark.parametrize("name", PR_AUTHORING_PROMPTS)
def test_requires_issue_link(rendered_prompts: dict[str, str], name: str) -> None:
    text = rendered_prompts[name]
    assert "Fixes #NNNN" in text or "Part of #NNNN" in text
