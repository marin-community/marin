# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest
from rigging.provenance import Provenance, username_segment


def test_str_clean_shows_commit_branch_user():
    p = Provenance(tree_hash="aaaa", base_commit="bbbb", dirty=False, branch="main", built_by="power")
    assert str(p) == "bbbb (main) (power)"


def test_str_dirty_shows_tree_off_of_base():
    p = Provenance(tree_hash="aaaa", base_commit="bbbb", dirty=True, branch="feat", built_by="power")
    assert str(p) == "aaaa (off of bbbb) (feat) (power)"


def test_str_omits_missing_branch_and_user():
    p = Provenance(tree_hash="aaaa", base_commit="bbbb", dirty=False, branch=None, built_by=None)
    assert str(p) == "bbbb"


def test_json_round_trip():
    p = Provenance(tree_hash="aaaa", base_commit="bbbb", dirty=True, branch=None, built_by="power")
    assert Provenance.from_json(p.to_json()) == p


def test_username_segment_sanitizes_email_to_a_clean_distinct_segment(monkeypatch):
    # Drops the domain but keeps the whole local name, so the segment is path-safe (no '@'/'.')
    # and two users who share a first name don't collapse onto one namespace.
    monkeypatch.setattr("rigging.provenance._getuser", lambda: "Russell.Power@gmail.com")
    assert username_segment() == "russell-power"


def test_username_segment_raises_when_unresolvable(monkeypatch):
    # Fail-fast is the contract: a username that does not resolve must not silently namespace
    # artifacts under a shared bucket.
    monkeypatch.setattr("rigging.provenance._getuser", lambda: None)
    with pytest.raises(RuntimeError):
        username_segment()


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def _init_repo(tmp_path: Path) -> Path:
    _run(["git", "init", "-b", "main"], tmp_path)
    _run(["git", "config", "user.email", "t@example.com"], tmp_path)
    _run(["git", "config", "user.name", "tester"], tmp_path)
    (tmp_path / "f.txt").write_text("hello\n")
    _run(["git", "add", "f.txt"], tmp_path)
    _run(["git", "commit", "-m", "init"], tmp_path)
    return tmp_path


def test_from_git_clean(tmp_path):
    repo = _init_repo(tmp_path)
    p = Provenance.from_git(repo)
    assert p.dirty is False
    assert p.branch == "main"
    assert p.tree_hash and p.base_commit
    # A clean tree's hash is HEAD's tree; dedup_key is that tree hash.
    head_tree = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD^{tree}"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert p.dedup_key == head_tree


def test_from_git_dirty_changes_tree_not_base(tmp_path):
    repo = _init_repo(tmp_path)
    clean = Provenance.from_git(repo)
    (repo / "f.txt").write_text("changed\n")
    dirty = Provenance.from_git(repo)
    assert dirty.dirty is True
    assert dirty.base_commit == clean.base_commit
    assert dirty.tree_hash != clean.tree_hash


def test_capture_is_best_effort_outside_a_repo(tmp_path):
    # from_git is strict (the tree hash is the point); capture degrades instead of raising,
    # still recording the launch context that has nothing to do with git.
    with pytest.raises(RuntimeError):
        Provenance.from_git(tmp_path)
    p = Provenance.capture(tmp_path)
    assert p.tree_hash == ""
    assert p.dirty is False
    assert p.command_line  # argv, captured regardless of git
    assert p.created_at  # timestamp, captured regardless of git


def test_capture_tolerates_missing_git_binary(tmp_path, monkeypatch):
    # A bundle image may lack git entirely: subprocess raises FileNotFoundError instead of
    # returning a nonzero exit. capture degrades like being outside a checkout.
    monkeypatch.setenv("PATH", "")
    p = Provenance.capture(tmp_path)
    assert p.tree_hash == ""
    assert p.base_commit == ""
    assert p.dirty is False
    assert p.command_line  # argv, captured regardless of git
    assert p.created_at  # timestamp, captured regardless of git


def test_json_round_trip_preserves_run_fields():
    p = Provenance(
        tree_hash="aaaa",
        base_commit="bbbb",
        dirty=False,
        branch="main",
        built_by="power",
        git_remote="git@github.com:o/r.git",
        created_at="2026-06-30T00:00:00",
        command_line=("python", "-m", "experiments.foo"),
    )
    assert Provenance.from_json(p.to_json()) == p
