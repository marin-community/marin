# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weekly fork-ferry: refresh one unit of forks toward upstream under headless Claude.

The coordinator runs the ``refresh-fork`` skill for one unit per invocation. A unit is the set of
forks that must refresh together, read from the descriptor's ``group`` field: ``tpu-inference``
and the TPU ``vllm`` pin derive from a single base, so they stage, validate, and land in one Marin
PR; an ungrouped fork is its own unit. The skill selects the base from
``config/external/migration.toml``, stages the overlay on ``<branch>-next``, pins Marin at that
tip, runs the fork's declared e2e on the marin hub, and opens a draft Marin PR.

v1 is stop-at-PR: the run creates every durable ref it can with an ordinary token — the
``<branch>-next`` staging branch, a backup tag of the current stable tip, and a date tag on the
new tip — but does not force-move the protected stable ``<branch>``/``main``. That pointer move
needs a ruleset bypass this token lacks, so the PR flags it for an admin.

The agent runs under a scoped tool allowlist, not ``--dangerously-skip-permissions``: in headless
mode a command outside the allowlist is denied rather than auto-approved, matching the posture of
``ops-claude.yaml``.
"""

import argparse
import logging
import tomllib
from pathlib import Path

from scripts.ci.claude_runner import (
    NO_SELF_CREDIT_SETTINGS,
    ClaudeRunStatus,
    report_rate_limit,
    run_claude,
)

logger = logging.getLogger(__name__)

DESCRIPTOR_PATH = Path("config/external/migration.toml")

# Forks the ferry does not drive. vllm-gpu's wheel is built by the fork's own release CI, so a
# refresh must dispatch that build cross-repo and wait for the wheel rather than resolve it inline;
# MarinSkyRL needs a one-time re-fork (#7920) before an automated refresh means anything. Tracked
# so a newly added descriptor section cannot slip through unwired (tests/infra/test_fork_ferry.py).
EXCLUDED: frozenset[str] = frozenset({"vllm-gpu", "MarinSkyRL"})

# Scoped so a headless run can rebase overlays, run the e2e through Iris, and open the PR, but
# nothing outside these families. Unlisted commands are denied in headless mode.
ALLOWED_TOOLS = (
    "--allowedTools",
    "Read,Write,Edit,Glob,Grep,"
    "Bash(git:*),Bash(gh:*),Bash(uv:*),Bash(python:*),Bash(iris:*),Bash(./infra/pre-commit.py:*)",
)


def load_descriptor() -> dict:
    return tomllib.loads(DESCRIPTOR_PATH.read_text())


def drivable_units(descriptor: dict) -> dict[str, tuple[str, ...]]:
    """Group the drivable forks into units keyed by the descriptor's ``group`` field.

    Forks sharing a ``group`` refresh together under that group name, so a split run cannot pin
    them against different revisions of their shared base; an ungrouped fork is its own unit.
    Excluded forks are omitted.
    """
    members: dict[str, list[str]] = {}
    for name, section in descriptor.items():
        if name in EXCLUDED:
            continue
        members.setdefault(section.get("group", name), []).append(name)
    return {unit: tuple(forks) for unit, forks in members.items()}


def build_prompt(forks: tuple[str, ...]) -> str:
    """Compose the refresh instruction for one unit's forks."""
    names = " and ".join(f"`{fork}`" for fork in forks)
    return (
        f"Refresh {names} together using the refresh-fork skill "
        "(.claude/skills/refresh-fork/SKILL.md). Follow it exactly, with these v1 constraints:\n"
        "1. Select each new base from its config/external/migration.toml descriptor. If no newer "
        "base is selected and no pin metadata needs repair, exit with a no-op summary — no "
        "branches, tags, or PR.\n"
        "2. Stage the rebased overlay on `<branch>-next` and push it. Run the fork's declared e2e "
        "via Iris on the marin hub at interactive priority with bounded limits. Fix only failures "
        "that pass on the current pin and fail on the refreshed one; a baseline-broken e2e is not "
        "a gate.\n"
        "3. On a green e2e, create the durable refs (non-destructive new refs an ordinary token "
        "can push): back up the current stable `<branch>` tip as tag "
        "`<branch>-backup/<YYYYMMDD>/pre-<shortsha>`, and date-tag the new staged tip "
        "`<branch>-<YYYYMMDD>`.\n"
        "4. Do NOT force-move the protected stable `<branch>`/`main`; that pointer move needs a "
        "ruleset bypass this token lacks. Open ONE Marin draft PR for these forks pinning the "
        "staged tip, and state in the body that an admin must complete the `<branch>-next` -> "
        "`<branch>` promotion (the backup and date tags already exist).\n"
        "5. On a real external blocker (not a baseline failure), open no PR and file the "
        "can't-migrate issue per the skill.\n"
        "Never stop, restart, or bounce a cluster. Follow the commit skill for the PR."
    )


def refresh(unit: str) -> ClaudeRunStatus:
    """Run the refresh-fork skill for one unit, returning the agent's run status."""
    units = drivable_units(load_descriptor())
    if unit not in units:
        raise SystemExit(f"{unit}: not a drivable unit; choices are {sorted(units)}")
    result = run_claude(
        build_prompt(units[unit]),
        ["--model=opus", *ALLOWED_TOOLS, *NO_SELF_CREDIT_SETTINGS, "--max-turns", "600"],
    )
    logger.info("%s", result.output)
    return result.status


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True, help="descriptor group or ungrouped fork name to refresh")
    status = refresh(parser.parse_args().unit)
    if status == ClaudeRunStatus.RATE_LIMITED:
        report_rate_limit()
        return
    if status == ClaudeRunStatus.FAILED:
        raise SystemExit("fork-ferry: refresh failed")


if __name__ == "__main__":
    main()
