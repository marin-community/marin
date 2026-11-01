# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin Belay guardrails: launches only from a clean, pushed commit of this week's branch, never reusing a run
name or passing the node cap, and results classified by the fixed win/noise/loss rule."""

import datetime
import subprocess
from pathlib import Path

import pytest

from experiments.belay.branch import cut_week_branch, git, require_clean_pushed_head, require_week_branch
from experiments.belay.config import CONFIG
from experiments.belay.results import Verdict, overall_verdict, verdict
from experiments.belay.submit import IrisJob, check_launch, parse_spec

MONDAY = datetime.date(2026, 11, 2)
SPEC = f"""experiment: X0001
launcher: {CONFIG.launcher}
runs:
  seed0: {{size: {CONFIG.rung}, seed: 0, variant: v, flags: [--opt-set, muonh_decay_power=0.5]}}
  control: {{size: {CONFIG.rung}, seed: 0, variant: c, control: true}}
"""


def commit_file(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "--quiet", "-m", name)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A clone whose origin has the base branch and last week's branch (base plus one commit)."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(origin)], check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "--quiet", str(origin), str(clone)], check=True)
    git(clone, "config", "user.email", "test@example.com")
    git(clone, "config", "user.name", "test")
    git(clone, "checkout", "--quiet", "-b", CONFIG.base_branch)
    commit_file(clone, "recipe.txt", "recipe\n")
    git(clone, "push", "--quiet", "origin", CONFIG.base_branch)
    last_week = CONFIG.week_branch(MONDAY - datetime.timedelta(days=7))
    git(clone, "checkout", "--quiet", "-b", last_week)
    commit_file(clone, "flag.txt", "last week's flag\n")
    git(clone, "push", "--quiet", "origin", last_week)
    return clone


def test_weekly_rollover_keeps_last_weeks_code_and_adds_the_base(repo: Path) -> None:
    git(repo, "checkout", "--quiet", CONFIG.base_branch)
    commit_file(repo, "recipe_fix.txt", "fix\n")
    git(repo, "push", "--quiet", "origin", CONFIG.base_branch)

    assert cut_week_branch(repo, MONDAY, fresh=False, dry_run=False)

    week = CONFIG.week_branch(MONDAY)
    assert git(repo, "ls-remote", "--heads", "origin", week)
    assert (repo / "flag.txt").exists() and (repo / "recipe_fix.txt").exists()
    assert require_week_branch(repo, MONDAY) == week


def test_launch_refused_off_this_weeks_branch(repo: Path) -> None:
    with pytest.raises(SystemExit, match="not this week's branch"):
        require_week_branch(repo, MONDAY)


def test_launch_refused_when_week_branch_lacks_the_base(repo: Path) -> None:
    git(repo, "checkout", "--quiet", "-b", CONFIG.week_branch(MONDAY))
    git(repo, "checkout", "--quiet", CONFIG.base_branch)
    commit_file(repo, "recipe_fix.txt", "fix\n")
    git(repo, "push", "--quiet", "origin", CONFIG.base_branch)
    git(repo, "checkout", "--quiet", CONFIG.week_branch(MONDAY))
    with pytest.raises(SystemExit, match="does not contain"):
        require_week_branch(repo, MONDAY)


def test_launch_refused_from_a_dirty_or_unpushed_tree(repo: Path) -> None:
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    assert require_clean_pushed_head(repo, branch) == git(repo, "rev-parse", "HEAD")

    (repo / "flag.txt").write_text("edited\n")
    with pytest.raises(SystemExit, match="differ from HEAD"):
        require_clean_pushed_head(repo, branch)

    git(repo, "commit", "--quiet", "-am", "unpushed")
    with pytest.raises(SystemExit, match="not pushed"):
        require_clean_pushed_head(repo, branch)


def test_launch_refused_for_a_reused_run_name() -> None:
    jobs = [IrisJob("/u/belay-X0001-seed0-coord", "succeeded", "2026-11-01T00:00")]
    with pytest.raises(SystemExit, match="already ran"):
        check_launch(parse_spec(SPEC), jobs)


def test_launch_held_at_the_node_cap() -> None:
    jobs = [IrisJob(f"/u/belay-X9{i:03d}-seed0-coord", "running", "") for i in range(CONFIG.node_cap - 1)]
    with pytest.raises(SystemExit, match="held"):
        check_launch(parse_spec(SPEC), jobs)
    check_launch(parse_spec(SPEC), jobs[:-1])  # two runs fit once one more node is free


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (-CONFIG.noise, Verdict.WIN),
        (-CONFIG.noise + 0.0001, Verdict.NOISE),
        (0.0058, Verdict.NOISE),
        (CONFIG.noise, Verdict.LOSS),
    ],
)
def test_verdict_rule(delta: float, expected: Verdict) -> None:
    assert verdict(delta) is expected


def test_overall_verdict() -> None:
    assert overall_verdict([Verdict.NOISE, Verdict.WIN]) is Verdict.WIN
    assert overall_verdict([Verdict.LOSS, Verdict.LOSS]) is Verdict.LOSS
    assert overall_verdict([Verdict.LOSS, Verdict.NOISE]) is Verdict.NOISE
