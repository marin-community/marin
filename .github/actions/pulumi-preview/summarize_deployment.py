#!/usr/bin/env python3
"""Decide whether a merged Pulumi change has a later successful update."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from assemble_comment import METADATA_PREFIX, METADATA_SUFFIX


MARKER = "<!-- iac-deployment-check -->"
TRUSTED_COMMENT_AUTHOR = "github-actions[bot]"
_STACK_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")


@dataclass(frozen=True)
class PreviewStatus:
    affected_stacks: tuple[str, ...]
    ok: bool


@dataclass(frozen=True)
class DeploymentStatus:
    pending_stacks: tuple[str, ...]
    error_stacks: tuple[str, ...]

    @property
    def needs_retry(self) -> bool:
        return bool(self.pending_stacks or self.error_stacks)


def _timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError(f"timestamp lacks timezone: {value}")
    return timestamp


def _valid_stacks(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not all(isinstance(stack, str) for stack in value):
        return None
    stacks = tuple(sorted(value))
    if len(stacks) != len(set(stacks)) or any(_STACK_RE.fullmatch(stack) is None for stack in stacks):
        return None
    return stacks


def _comments(path: Path) -> list[dict[str, object]]:
    pages = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(pages, list) or not all(isinstance(page, list) for page in pages):
        raise ValueError("GitHub comments response is not a list of pages")
    return [comment for page in pages for comment in page if isinstance(comment, dict)]


def _trusted_comment_body(comment: dict[str, object]) -> str | None:
    author = comment.get("user")
    body = comment.get("body")
    if not isinstance(author, dict) or author.get("login") != TRUSTED_COMMENT_AUTHOR:
        return None
    return body if isinstance(body, str) else None


def preview_status(comment: str, head_sha: str) -> PreviewStatus:
    metadata_lines = [
        line
        for line in comment.splitlines()
        if line.startswith(METADATA_PREFIX) and line.endswith(METADATA_SUFFIX)
    ]
    if len(metadata_lines) != 1:
        return PreviewStatus(affected_stacks=(), ok=False)

    encoded = metadata_lines[0][len(METADATA_PREFIX) : -len(METADATA_SUFFIX)]
    try:
        metadata = json.loads(encoded)
    except json.JSONDecodeError:
        return PreviewStatus(affected_stacks=(), ok=False)
    stacks = _valid_stacks(metadata.get("affected_stacks")) if isinstance(metadata, dict) else None
    if (
        stacks is None
        or metadata.get("head_sha") != head_sha
        or metadata.get("ok") is not True
    ):
        return PreviewStatus(affected_stacks=(), ok=False)
    return PreviewStatus(affected_stacks=stacks, ok=True)


def latest_preview_status(comments: list[dict[str, object]], head_sha: str) -> PreviewStatus:
    previews = [
        comment
        for comment in comments
        if (body := _trusted_comment_body(comment)) is not None and METADATA_PREFIX in body
    ]
    if not previews:
        return PreviewStatus(affected_stacks=(), ok=False)
    latest = max(previews, key=lambda comment: str(comment.get("updated_at", "")))
    body = _trusted_comment_body(latest)
    assert body is not None
    return preview_status(body, head_sha)


def latest_reminder_id(comments: list[dict[str, object]]) -> int | None:
    reminders = [
        comment
        for comment in comments
        if (body := _trusted_comment_body(comment)) is not None and MARKER in body
    ]
    if not reminders:
        return None
    latest = max(reminders, key=lambda comment: str(comment.get("created_at", "")))
    comment_id = latest.get("id")
    return comment_id if isinstance(comment_id, int) else None


def _latest_successful_update_start(history_path: Path) -> datetime | None:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list):
        raise ValueError("Pulumi history is not a list")
    starts: list[datetime] = []
    for update in history:
        if not isinstance(update, dict):
            raise ValueError("Pulumi history entry is not an object")
        if update.get("kind") == "update" and update.get("result") == "succeeded":
            start_time = update.get("startTime")
            if not isinstance(start_time, str):
                raise ValueError("successful Pulumi update has no start time")
            starts.append(_timestamp(start_time))
    return max(starts, default=None)


def deployment_status(
    history_dir: Path,
    affected_stacks: tuple[str, ...],
    preview_ok: bool,
    merged_at: datetime,
) -> DeploymentStatus:
    if not preview_ok:
        return DeploymentStatus(pending_stacks=(), error_stacks=("IaC preview",))

    pending: list[str] = []
    errors: list[str] = []
    for stack in affected_stacks:
        history_path = history_dir / f"{stack}.json"
        try:
            last_update = _latest_successful_update_start(history_path)
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append(stack)
            continue
        if last_update is None or last_update <= merged_at:
            pending.append(stack)
    return DeploymentStatus(pending_stacks=tuple(pending), error_stacks=tuple(errors))


def _stack_list(stacks: tuple[str, ...]) -> str:
    return ", ".join(f"`{stack}`" for stack in stacks)


def render_comment(
    status: DeploymentStatus,
    *,
    attempt: int,
    max_attempts: int,
    merger: str,
    run_url: str,
) -> str:
    lines = [MARKER, "## Pulumi deployment check", ""]
    if not status.needs_retry:
        lines.append("✅ A successful `pulumi up` started after this PR merged for every affected stack.")
    elif status.pending_stacks:
        lines.append(f"@{merger}, run `pulumi up` from current `main` for this PR.")
        lines.append(f"Stacks without a later successful update: {_stack_list(status.pending_stacks)}.")
    elif attempt == max_attempts:
        lines.append(f"@{merger}, Pulumi update history could not be verified after {max_attempts} checks.")

    if status.error_stacks and attempt == max_attempts:
        lines.append(f"Verification errors: {_stack_list(status.error_stacks)}.")

    lines.extend(["", f"Check {attempt} of {max_attempts}: [workflow run]({run_url})."])
    if status.needs_retry and attempt < max_attempts:
        lines.append("Another check is scheduled.")
    elif status.needs_retry:
        lines.append("No further checks are scheduled.")
    return "\n".join(lines) + "\n"


def _comment_action(
    status: DeploymentStatus,
    attempt: int,
    max_attempts: int,
    reminder_id: int | None,
) -> str:
    if status.pending_stacks:
        return "create"
    if status.error_stacks:
        return "create" if attempt == max_attempts and reminder_id is None else "none"
    return "update" if reminder_id is not None else "none"


def _affected_stacks(args: argparse.Namespace) -> None:
    status = latest_preview_status(_comments(args.comments), args.head_sha)
    args.out.write_text(
        json.dumps(
            {"affected_stacks": status.affected_stacks, "preview_ok": status.ok},
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _check(args: argparse.Namespace) -> None:
    if not 1 <= args.attempt <= args.max_attempts:
        raise ValueError(f"attempt {args.attempt} is outside the retry policy")
    affected_stacks = _valid_stacks(json.loads(args.affected_stacks))
    if affected_stacks is None:
        raise ValueError("affected stacks are invalid")

    status = deployment_status(
        args.history_dir,
        affected_stacks,
        args.preview_ok == "true",
        _timestamp(args.merged_at),
    )
    reminder_id = latest_reminder_id(_comments(args.comments))
    args.out.write_text(
        render_comment(
            status,
            attempt=args.attempt,
            max_attempts=args.max_attempts,
            merger=args.merger,
            run_url=args.run_url,
        ),
        encoding="utf-8",
    )
    next_attempt = args.attempt + 1 if status.needs_retry and args.attempt < args.max_attempts else None
    comment_action = _comment_action(status, args.attempt, args.max_attempts, reminder_id)
    result: dict[str, object] = {
        "comment_action": comment_action,
        "needs_retry": status.needs_retry,
        "next_attempt": next_attempt,
    }
    if comment_action == "update":
        assert reminder_id is not None
        result["comment_id"] = reminder_id
    args.status_out.write_text(
        json.dumps(
            result,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    affected = commands.add_parser("affected-stacks", help="Read affected stacks from an IaC preview comment.")
    affected.add_argument("--comments", type=Path, required=True)
    affected.add_argument("--head-sha", required=True)
    affected.add_argument("--out", type=Path, required=True)
    affected.set_defaults(run=_affected_stacks)

    check = commands.add_parser("check", help="Compare successful stack updates with the PR merge time.")
    check.add_argument("--history-dir", type=Path, required=True)
    check.add_argument("--comments", type=Path, required=True)
    check.add_argument("--affected-stacks", required=True)
    check.add_argument("--preview-ok", required=True, choices=("true", "false"))
    check.add_argument("--merged-at", required=True)
    check.add_argument("--attempt", type=int, required=True)
    check.add_argument("--max-attempts", type=int, required=True)
    check.add_argument("--merger", required=True)
    check.add_argument("--run-url", required=True)
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--status-out", type=Path, required=True)
    check.set_defaults(run=_check)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
