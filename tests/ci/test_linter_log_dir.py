# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""`_review_log_dir` must give every review run a unique, branch-namespaced
directory. Many weaver worktrees run `pre-commit.py --review` concurrently on one
host into a shared log root; a flat one-second-resolution timestamp leaf clobbers
when two runs start in the same second and gives no way to tell whose run is whose."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "infra"))
import linter


def test_same_second_runs_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(linter, "LINT_REVIEW_LOG_ROOT", tmp_path)
    started = 1_700_000_000.0

    first = linter._review_log_dir("weaver/iris-federation-pr5", started)
    second = linter._review_log_dir("weaver/iris-federation-pr5", started)

    assert first != second
    assert first.is_dir() and second.is_dir()


def test_branch_namespaces_and_sanitizes_path(tmp_path, monkeypatch):
    monkeypatch.setattr(linter, "LINT_REVIEW_LOG_ROOT", tmp_path)

    log_dir = linter._review_log_dir("weaver/tokenizer-research", 1_700_000_000.0)

    # Slashes are sanitized so the branch becomes a single path component under the root.
    assert log_dir.parent == tmp_path / "weaver-tokenizer-research"


@pytest.mark.parametrize("branch", [None, ""])
def test_detached_head_still_gets_a_directory(tmp_path, monkeypatch, branch):
    monkeypatch.setattr(linter, "LINT_REVIEW_LOG_ROOT", tmp_path)

    log_dir = linter._review_log_dir(branch, 1_700_000_000.0)

    assert log_dir.parent == tmp_path / "detached"
    assert log_dir.is_dir()
