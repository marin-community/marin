# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The mechanical checks on an idea's code change (runbook section 1b), made on the weekly branch itself.

Before it is committed to the weekly branch, a change must touch only the fast_track model code and its tests, add
at least one new config field whose default is off, leave the default model bit-for-bit identical to the commit
before it (loss and every gradient of a tiny CPU model), keep the model causal with the flag on, pass its own tests
and pass pre-commit. Losing ideas' flags stay as dead default-off code until a maintainer deletes them.
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.belay.branch import git

ALLOWED_PREFIXES = ("experiments/grug/fast_track/", "tests/grug/test_fast_track_")
TESTS_PREFIX = "tests/grug/test_fast_track_"
OFF_DEFAULTS = ("False", "None", "0", "0.0", "''", "'none'", "'off'")
# Past positions' hidden states may move by at most this much when future tokens change (float32 noise).
LEAK_TOLERANCE = 1e-6
PROBE_SCRIPT = Path(__file__).parent / "model_probe.py"


@dataclasses.dataclass(frozen=True)
class Probe:
    """What one checkout's code does: config defaults, a digest of the tiny model's loss and gradients, and the
    largest change in past hidden states when only future tokens change."""

    defaults: dict[str, dict[str, str]]
    digest: str
    leak: float


def changed_paths(repo: Path) -> list[str]:
    """Every uncommitted path in ``repo``: staged, unstaged and untracked."""
    tracked = git(repo, "diff", "--name-only", "HEAD").splitlines()
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    return sorted(set(tracked + untracked))


def refuse_outside_allowed(paths: list[str]) -> None:
    outside = [p for p in paths if not p.startswith(ALLOWED_PREFIXES)]
    if outside:
        raise SystemExit(
            f"refusing: the change touches {outside}; only {', '.join(ALLOWED_PREFIXES)}* may change.\n"
            "next: git checkout -- <those paths> (and delete new files there), then run code-change --finish again"
        )
    if not any(p.startswith(TESTS_PREFIX) for p in paths):
        raise SystemExit(f"refusing: add a test for the flag under {TESTS_PREFIX}*.py.\nnext: code-change --finish")


def new_flags(week: dict[str, dict[str, str]], idea: dict[str, dict[str, str]]) -> list[str]:
    """The idea's new config fields, after checking each defaults to off and no existing default changed."""
    added = []
    for cls, week_fields in week.items():
        idea_fields = idea[cls]
        changed = [k for k, v in week_fields.items() if idea_fields.get(k) != v]
        if changed:
            raise SystemExit(
                f"refusing: {cls} changes or removes existing fields {changed}.\n"
                "next: restore them and add a new default-off field instead, then run code-change --finish again"
            )
        for name, default in idea_fields.items():
            if name in week_fields:
                continue
            if default not in OFF_DEFAULTS:
                raise SystemExit(
                    f"refusing: new flag {cls}.{name} defaults to {default}; its default must be off.\n"
                    "next: make it default off (False, None, 0 or 0.0), then run code-change --finish again"
                )
            added.append(f"{cls}.{name}")
    if not added:
        raise SystemExit(
            "refusing: no new config flag.\nnext: put the change behind a new default-off field, then --finish again"
        )
    return added


def refuse_flag_off_change(week: Probe, idea: Probe) -> None:
    if week.digest != idea.digest:
        raise SystemExit(
            "refusing: with every flag off, the tiny model's loss or gradients differ from the commit before it.\n"
            "next: put every behavior change behind the new flag, then run code-change --finish again"
        )


def refuse_leak(flag_on: Probe) -> None:
    if not flag_on.leak <= LEAK_TOLERANCE:
        raise SystemExit(
            f"refusing: with the flag on, changing future tokens moves past hidden states by {flag_on.leak:.3g} "
            f"(limit {LEAK_TOLERANCE}); the change leaks future information.\n"
            "next: make the flag-on path causal and run code-change --finish again, or give up: git checkout -- . "
            "&& git clean -fd experiments/grug/fast_track tests/grug, then escalate"
        )


def checkout_python(checkout: Path) -> str:
    """The checkout's own environment when it has one (a Track branch with other library code), else this one."""
    own = checkout / ".venv/bin/python"
    return str(own) if own.exists() else sys.executable


def probe(checkout: Path, model_settings: list[str], python: str) -> Probe:
    """Run ``model_probe.py`` on CPU against the model code in ``checkout`` (any branch, with or without this
    directory): its source runs with ``python -c``, so ``experiments`` imports from ``checkout``."""
    out = subprocess.run(
        [python, "-c", PROBE_SCRIPT.read_text(), *model_settings],
        cwd=checkout,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return Probe(**json.loads(out.strip().splitlines()[-1]))


def run_tests(repo: Path, paths: list[str]) -> None:
    """The change's own tests, on CPU."""
    tests = [p for p in paths if p.startswith(TESTS_PREFIX) and p.endswith(".py")]
    command = [checkout_python(repo), "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", *tests]
    result = subprocess.run(
        command, cwd=repo, env={**os.environ, "JAX_PLATFORMS": "cpu"}, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(f"refusing: the change's tests fail:\n{result.stdout[-3000:]}\nnext: fix them, then --finish")


def run_precommit(repo: Path, paths: list[str]) -> None:
    files = [p for p in paths if (repo / p).exists()]
    result = subprocess.run(["./infra/pre-commit.py", *files], cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"refusing: pre-commit fails:\n{result.stdout[-3000:]}\n"
            f"next: ./infra/pre-commit.py --fix {' '.join(files)}, then --finish"
        )


def flag_commits(repo: Path, base_ref: str) -> list[str]:
    """The flag commits on this branch since ``base_ref``: ``X####: <flag> (default off)``."""
    return git(repo, "log", "--format=%h %s", "--grep=(default off)", f"{base_ref}..HEAD").splitlines()
