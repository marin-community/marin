#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate and merge generated dependency update pull requests."""

import argparse
import json
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from scripts.ci.dependency_update_policy import (
    DEPENDENCY_UPDATE_POLICIES,
    REQUIRED_CHECKS,
    DependencyUpdate,
    PullRequestPolicy,
)


@dataclass(frozen=True)
class PullRequestSnapshot:
    author: str
    base_branch: str
    files: tuple[str, ...]
    head_branch: str
    head_sha: str
    state: str
    title: str
    url: str


@dataclass(frozen=True)
class RequiredCheckGate:
    failing: tuple[str, ...]
    missing: tuple[str, ...]
    pending: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failing and not self.missing and not self.pending


class MergeDecision(StrEnum):
    WAIT = "wait"
    FAIL = "fail"
    MERGE = "merge"
    DONE = "done"


@dataclass(frozen=True)
class CheckRow:
    name: str
    bucket: str

    @classmethod
    def from_json(cls, payload: object) -> "CheckRow":
        if not isinstance(payload, dict):
            raise ValueError(f"GitHub returned a non-object check row: {payload!r}")
        name = payload.get("name")
        bucket = payload.get("bucket")
        if not isinstance(name, str) or not isinstance(bucket, str):
            raise ValueError(f"GitHub returned an invalid check row: {payload!r}")
        return cls(name=name, bucket=bucket)


def validated_pull_request(
    pull_request: PullRequestSnapshot,
    *,
    policy: PullRequestPolicy,
    expected_app_slug: str,
    expected_head_sha: str,
) -> PullRequestSnapshot:
    """Return a pull request after verifying the generated update boundary."""
    expected_author = f"app/{expected_app_slug}"
    if pull_request.author != expected_author:
        raise ValueError(f"unexpected pull request author {pull_request.author!r}; expected {expected_author!r}")
    if pull_request.base_branch != policy.base_branch:
        raise ValueError(f"unexpected base branch {pull_request.base_branch!r}")
    if pull_request.head_branch != policy.head_branch:
        raise ValueError(f"unexpected head branch {pull_request.head_branch!r}")
    if pull_request.head_sha != expected_head_sha:
        raise ValueError(f"unexpected head SHA {pull_request.head_sha!r}; expected {expected_head_sha!r}")
    if pull_request.title != policy.title:
        raise ValueError(f"unexpected pull request title {pull_request.title!r}")
    if not pull_request.files:
        raise ValueError("pull request has no changed files")
    unexpected_files = sorted(set(pull_request.files) - policy.allowed_files)
    if unexpected_files:
        raise ValueError(f"pull request contains unexpected files: {unexpected_files}")
    return pull_request


def validate_changed_files(files: Iterable[str], *, policy: PullRequestPolicy) -> tuple[str, ...]:
    """Return sorted changed files after enforcing the generator allowlist."""
    changed = tuple(sorted(set(files)))
    unexpected = tuple(file for file in changed if file not in policy.allowed_files)
    if unexpected:
        raise ValueError(f"dependency update changed unexpected files: {list(unexpected)}")
    return changed


def evaluate_required_checks(rows: Iterable[CheckRow], *, required: tuple[str, ...]) -> RequiredCheckGate:
    """Classify the latest required GitHub check rows."""
    required_names = frozenset(required)
    check_buckets: dict[str, str] = {}
    for row in rows:
        if row.name not in required_names:
            continue
        if row.name in check_buckets:
            raise ValueError(f"GitHub returned duplicate check rows for {row.name!r}")
        check_buckets[row.name] = row.bucket
    missing = tuple(name for name in required if name not in check_buckets)
    pending = tuple(name for name in required if check_buckets.get(name) == "pending")
    failing = tuple(
        name for name in required if name in check_buckets and check_buckets[name] not in {"pass", "pending"}
    )
    return RequiredCheckGate(failing=failing, missing=missing, pending=pending)


def evaluate_merge(state: str, checks: RequiredCheckGate) -> MergeDecision:
    """Choose the next action from the pull-request and required-check state."""
    if state == "MERGED":
        return MergeDecision.DONE
    if state != "OPEN" or checks.failing:
        return MergeDecision.FAIL
    if checks.passed:
        return MergeDecision.MERGE
    return MergeDecision.WAIT


def _gh_json(*args: str) -> object:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def pull_request_snapshot(pr: str, repository: str) -> PullRequestSnapshot:
    """Read the identity and immutable merge boundary of one pull request."""
    payload = _gh_json(
        "pr",
        "view",
        pr,
        "--repo",
        repository,
        "--json",
        "author,baseRefName,files,headRefName,headRefOid,state,title,url",
    )
    assert isinstance(payload, dict)
    return PullRequestSnapshot(
        author=payload["author"]["login"],
        base_branch=payload["baseRefName"],
        files=tuple(sorted(file["path"] for file in payload["files"])),
        head_branch=payload["headRefName"],
        head_sha=payload["headRefOid"],
        state=payload["state"],
        title=payload["title"],
        url=payload["url"],
    )


def required_check_rows(pr: str, repository: str) -> tuple[CheckRow, ...]:
    """Read GitHub's latest rollup row for every check on a pull request."""
    result = subprocess.run(
        ["gh", "pr", "checks", pr, "--repo", repository, "--json", "name,bucket"],
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        return ()
    rows = json.loads(result.stdout)
    assert isinstance(rows, list)
    return tuple(CheckRow.from_json(row) for row in rows)


def changed_worktree_files() -> tuple[str, ...]:
    """Read files changed by a dependency generator."""
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(result.stdout.splitlines())


def merge_when_green(
    *,
    pr: str,
    repository: str,
    app_slug: str,
    policy: PullRequestPolicy,
    expected_head_sha: str,
    timeout: float,
    poll_interval: float,
) -> None:
    """Wait for the fixed required checks, then merge with the dedicated app token."""
    deadline = time.monotonic() + timeout
    while True:
        snapshot = validated_pull_request(
            pull_request_snapshot(pr, repository),
            policy=policy,
            expected_app_slug=app_slug,
            expected_head_sha=expected_head_sha,
        )
        checks = evaluate_required_checks(required_check_rows(pr, repository), required=REQUIRED_CHECKS)
        decision = evaluate_merge(snapshot.state, checks)
        if decision is MergeDecision.DONE:
            return
        if decision is MergeDecision.FAIL:
            raise RuntimeError(f"dependency update is blocked: state={snapshot.state}, failing={list(checks.failing)}")
        if decision is MergeDecision.MERGE:
            subprocess.run(
                ["gh", "pr", "merge", pr, "--repo", repository, "--squash"],
                check=True,
            )
            merged = pull_request_snapshot(pr, repository)
            if merged.state != "MERGED":
                raise RuntimeError(f"merge command completed but pull request is {merged.state}")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "required checks did not finish before the merge deadline: "
                f"missing={list(checks.missing)}, pending={list(checks.pending)}"
            )
        time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    changed_files = subparsers.add_parser("changed-files", help="validate and print generator changes")
    changed_files.add_argument("--kind", choices=DependencyUpdate, required=True)
    merge = subparsers.add_parser("merge", help="wait for required checks and merge")
    merge.add_argument("--kind", choices=DependencyUpdate, required=True)
    merge.add_argument("--pr", required=True)
    merge.add_argument("--repository", required=True)
    merge.add_argument("--app-slug", required=True)
    merge.add_argument("--expected-head-sha", required=True)
    merge.add_argument("--timeout", type=float, default=3600)
    merge.add_argument("--poll-interval", type=float, default=30)
    return parser


def main() -> None:
    args = _parser().parse_args()
    policy = DEPENDENCY_UPDATE_POLICIES[DependencyUpdate(args.kind)]
    if args.command == "changed-files":
        print("\n".join(validate_changed_files(changed_worktree_files(), policy=policy)))
        return
    merge_when_green(
        pr=args.pr,
        repository=args.repository,
        app_slug=args.app_slug,
        policy=policy,
        expected_head_sha=args.expected_head_sha,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
