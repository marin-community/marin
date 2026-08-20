#!/usr/bin/env python3
"""Summarize post-merge Pulumi previews for a sticky PR comment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from assemble_comment import StackPreview, load_stacks


MARKER = "<!-- iac-deployment-check -->"


@dataclass(frozen=True)
class DeploymentStatus:
    pending_stacks: tuple[str, ...]
    error_stacks: tuple[str, ...]

    @property
    def needs_retry(self) -> bool:
        return bool(self.pending_stacks or self.error_stacks)


def deployment_status(stacks: list[StackPreview], expected_stacks: set[str]) -> DeploymentStatus:
    previews = {stack.stack: stack for stack in stacks}
    if len(previews) != len(stacks):
        raise ValueError("duplicate stack preview artifacts")

    unexpected = previews.keys() - expected_stacks
    if unexpected:
        raise ValueError(f"unexpected stack preview artifacts: {', '.join(sorted(unexpected))}")

    missing = expected_stacks - previews.keys()
    errors = missing | {stack.stack for stack in stacks if stack.severity == "error"}
    pending = {
        stack.stack
        for stack in stacks
        if stack.severity not in {"none", "error"}
    }
    return DeploymentStatus(
        pending_stacks=tuple(sorted(pending)),
        error_stacks=tuple(sorted(errors)),
    )


def _stack_list(stacks: tuple[str, ...]) -> str:
    return ", ".join(f"`{stack}`" for stack in stacks)


def render_comment(
    status: DeploymentStatus,
    *,
    attempt: int,
    check_delays_minutes: tuple[int, ...],
    merger: str,
    run_url: str,
) -> str:
    max_attempts = len(check_delays_minutes)
    lines = [MARKER, "## Pulumi deployment check", ""]
    if not status.needs_retry:
        lines.append(f"✅ Pulumi reports no pending changes after check {attempt} of {max_attempts}.")
    else:
        lines.append(f"@{merger}, run `pulumi up` from `main` for the changes merged by this PR.")
        if status.pending_stacks:
            lines.append(f"Pending changes: {_stack_list(status.pending_stacks)}.")
        if status.error_stacks:
            lines.append(f"Preview errors prevented verification: {_stack_list(status.error_stacks)}.")

    lines.extend(["", f"Check {attempt} of {max_attempts}: [workflow run]({run_url})."])
    if status.needs_retry and attempt < max_attempts:
        lines.append(f"The next check runs in {check_delays_minutes[attempt]} minutes.")
    elif status.needs_retry:
        lines.append("No further checks are scheduled.")
    return "\n".join(lines) + "\n"


def _write_outputs(path: Path, *, status: DeploymentStatus, attempt: int, max_attempts: int) -> None:
    should_comment = status.needs_retry or attempt > 1
    next_attempt = attempt + 1 if status.needs_retry and attempt < max_attempts else ""
    with path.open("a", encoding="utf-8") as output:
        output.write(f"needs_retry={str(status.needs_retry).lower()}\n")
        output.write(f"next_attempt={next_attempt}\n")
        output.write(f"should_comment={str(should_comment).lower()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previews-dir", type=Path, required=True)
    parser.add_argument("--preview-matrix", required=True)
    parser.add_argument("--check-delays-minutes", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--merger", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()

    preview_matrix = json.loads(args.preview_matrix)
    expected_stacks = {entry["stack"] for entry in preview_matrix["include"]}
    check_delays_minutes = tuple(json.loads(args.check_delays_minutes))
    if not 1 <= args.attempt <= len(check_delays_minutes):
        raise ValueError(f"attempt {args.attempt} is outside the retry policy")
    if not check_delays_minutes or any(delay <= 0 for delay in check_delays_minutes):
        raise ValueError("check delays must be positive")
    status = deployment_status(load_stacks(args.previews_dir), expected_stacks)
    args.out.write_text(
        render_comment(
            status,
            attempt=args.attempt,
            check_delays_minutes=check_delays_minutes,
            merger=args.merger,
            run_url=args.run_url,
        ),
        encoding="utf-8",
    )
    _write_outputs(
        args.github_output,
        status=status,
        attempt=args.attempt,
        max_attempts=len(check_delays_minutes),
    )


if __name__ == "__main__":
    main()
