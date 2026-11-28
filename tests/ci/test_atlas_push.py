# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the atlas-push landing tool against a temporary bare remote."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "infra" / "ci" / "atlas-push"


def run(command: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=check)


@pytest.fixture(name="landing")
def landing_fixture(tmp_path: Path) -> dict[str, Path]:
    """One bare origin, one work repo carrying the script's own layout, one log path."""
    origin = tmp_path / "origin.git"
    run(["git", "init", "--bare", "-b", "atlas-env-audit", str(origin)], tmp_path)
    work = tmp_path / "work"
    run(["git", "clone", "-q", str(origin), str(work)], tmp_path)
    run(["git", "config", "user.email", "worker@example.com"], work)
    run(["git", "config", "user.name", "Worker"], work)
    (work / "infra").mkdir()
    stub = work / "infra" / "pre-commit.py"
    stub.write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n")
    stub.chmod(0o755)
    run(["git", "add", "-A"], work)
    run(["git", "commit", "-q", "-m", "seed the landing layout"], work)
    run(["git", "push", "-q", "origin", "HEAD:atlas-env-audit"], work)
    return {"origin": origin, "work": work, "log": tmp_path / "landings.jsonl", "lock": tmp_path / "lock"}


def push(
    landing: dict[str, Path], *, extra_env: dict[str, str] | None = None, proposal: str = ""
) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "ATLAS_PUSH_TARGET": "atlas-env-audit",
        "ATLAS_PUSH_LOG": str(landing["log"]),
        "ATLAS_PUSH_LOCK": str(landing["lock"]),
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--test",
            "true",
            "--who",
            "test-session",
            "--what",
            "test landing",
            "--registry-update",
            proposal,
        ],
        cwd=landing["work"],
        capture_output=True,
        text=True,
        env=env,
    )


def origin_tip(landing: dict[str, Path], ref: str = "atlas-env-audit") -> str:
    return run(["git", "--git-dir", str(landing["origin"]), "rev-parse", ref], landing["work"]).stdout.strip()


def merge_count(landing: dict[str, Path], base: str, tip: str) -> int:
    """Count merge commits between two explicit tips, resolved in the work repo."""
    out = run(["git", "rev-list", "--merges", f"{base}..{tip}"], landing["work"])
    return len(out.stdout.split())


def commit(work: Path, name: str, message: str) -> None:
    (work / name).write_text(f"{name}\n")
    run(["git", "add", "-A"], work)
    run(["git", "commit", "-q", "-m", message], work)


def test_linear_landing_rebases_and_pushes(landing) -> None:
    work = landing["work"]
    run(["git", "checkout", "-q", "-b", "worker/linear"], work)
    commit(work, "change.txt", "linear worker change")

    result = push(landing)
    assert result.returncode == 0, result.stderr
    tip = origin_tip(landing)
    assert tip == run(["git", "rev-parse", "HEAD"], work).stdout.strip()
    entry = json.loads(landing["log"].read_text().splitlines()[-1])
    assert entry["sha"] == tip and entry["who"] == "test-session"


def test_merge_commits_land_unchanged(landing) -> None:
    work = landing["work"]
    seed = run(["git", "rev-parse", "HEAD"], work).stdout.strip()
    run(["git", "checkout", "-q", "-b", "worker/merges"], work)
    commit(work, "side.txt", "side change")
    run(["git", "checkout", "-q", "-b", "mainline", seed], work)
    commit(work, "main.txt", "mainline change")
    run(["git", "push", "-q", "origin", "mainline:atlas-env-audit"], work)
    run(["git", "checkout", "-q", "worker/merges"], work)
    run(["git", "merge", "-q", "--no-edit", "mainline"], work)
    before = run(["git", "rev-parse", "HEAD"], work).stdout.strip()
    target_before = origin_tip(landing)
    assert merge_count(landing, target_before, before) == 1

    result = push(landing)
    assert result.returncode == 0, result.stderr
    tip = origin_tip(landing)
    # The merge commit reached the target unchanged: same tip, still one merge, no rebase.
    assert tip == before
    assert merge_count(landing, target_before, tip) == 1


def test_merge_landing_aborts_when_the_target_moved(landing) -> None:
    work = landing["work"]
    seed = run(["git", "rev-parse", "HEAD"], work).stdout.strip()
    run(["git", "checkout", "-q", "-b", "worker/moved"], work)
    commit(work, "side.txt", "side change")
    run(["git", "checkout", "-q", "-b", "mainline", seed], work)
    commit(work, "main.txt", "mainline change")
    run(["git", "push", "-q", "origin", "mainline:atlas-env-audit"], work)
    run(["git", "checkout", "-q", "worker/moved"], work)
    run(["git", "merge", "-q", "--no-edit", "mainline"], work)
    # Advance the target off the merged branch so the branch no longer contains the tip.
    run(["git", "checkout", "-q", "-b", "late", "mainline"], work)
    commit(work, "late.txt", "late target change")
    run(["git", "push", "-q", "origin", "HEAD:atlas-env-audit"], work)
    moved_tip = origin_tip(landing)
    run(["git", "checkout", "-q", "worker/moved"], work)

    result = push(landing)
    assert result.returncode == 3
    assert "merge the new origin" in result.stderr
    assert origin_tip(landing) == moved_tip


def test_linear_landing_rebases_onto_a_moved_target(landing) -> None:
    work = landing["work"]
    run(["git", "checkout", "-q", "-b", "worker/diverged"], work)
    commit(work, "worker.txt", "worker change")
    # Move the target behind the worker's back.
    run(["git", "checkout", "-q", "--detach", "atlas-env-audit"], work)
    commit(work, "target.txt", "target change on another machine")
    run(["git", "push", "-q", "origin", "HEAD:atlas-env-audit"], work)
    run(["git", "checkout", "-q", "worker/diverged"], work)

    result = push(landing)
    assert result.returncode == 0, result.stderr
    tip = origin_tip(landing)
    # The worker branch rebased onto the moved target: the target commit sits below it.
    assert (
        "worker change"
        in run(["git", "--git-dir", str(landing["origin"]), "log", "-1", "--format=%s", tip], work).stdout
    )
    assert (
        "target change on another machine"
        in run(["git", "--git-dir", str(landing["origin"]), "log", "-1", "--format=%s", f"{tip}~1"], work).stdout
    )


def test_shared_registry_change_without_a_proposal_cannot_reach_origin(landing) -> None:
    work = landing["work"]
    before = origin_tip(landing)
    run(["git", "checkout", "-q", "-b", "worker/registry"], work)
    path = work / "experiments/post_training/task_curation/source_fingerprint_index.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    run(["git", "add", "-A"], work)
    run(["git", "commit", "-q", "-m", "change the shared selection"], work)

    result = push(landing)

    assert result.returncode == 10, result.stderr
    assert origin_tip(landing) == before
    assert not landing["log"].exists()


def test_registry_compare_and_swap_rejects_a_moved_pointer_after_rebase(landing, tmp_path) -> None:
    work = landing["work"]
    package = work / "experiments/post_training/task_curation"
    package.mkdir(parents=True)
    old = {"repo": "fixture/registry", "revision": "a" * 40, "path": "old.json", "sha256": "b" * 64}
    for name in ("source_fingerprint_index.json", "source_population_locators.json"):
        (package / name).write_text(json.dumps(old))
    run(["git", "add", "-A"], work)
    run(["git", "commit", "-q", "-m", "seed the selected registry pointers"], work)
    run(["git", "push", "-q", "origin", "HEAD:atlas-env-audit"], work)
    run(["git", "checkout", "-q", "-b", "worker/registry-cas"], work)
    pair = {"fingerprints": old, "locators": old}
    (work / "update.json").write_text(json.dumps({"expected": pair, "proposed": pair}))
    run(["git", "add", "-A"], work)
    run(["git", "commit", "-q", "-m", "prepare a registry proposal"], work)

    run(["git", "checkout", "-q", "-b", "other-writer", "atlas-env-audit"], work)
    newer = {**old, "revision": "c" * 40}
    (package / "source_fingerprint_index.json").write_text(json.dumps(newer))
    run(["git", "add", "-A"], work)
    run(["git", "commit", "-q", "-m", "advance the shared registry pointer"], work)
    run(["git", "push", "-q", "origin", "HEAD:atlas-env-audit"], work)
    advanced = origin_tip(landing)
    run(["git", "checkout", "-q", "worker/registry-cas"], work)

    # Execute the real Python check in the temporary Git repository, using this
    # test environment in place of uv's project/environment selection boundary.
    binaries = tmp_path / "bin"
    binaries.mkdir()
    uv = binaries / "uv"
    uv.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "${{@:4}}"\n')
    uv.chmod(0o755)
    env = {"PATH": f"{binaries}:{os.environ['PATH']}", "PYTHONPATH": str(SCRIPT.parents[2])}

    result = push(landing, extra_env=env, proposal="update.json")

    assert result.returncode == 10, result.stderr
    assert origin_tip(landing) == advanced
    assert run(["git", "merge-base", "--is-ancestor", advanced, "HEAD"], work).returncode == 0
    assert not landing["log"].exists()
