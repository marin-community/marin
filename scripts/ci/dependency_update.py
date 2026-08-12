#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create, validate, and merge generated dependency update pull requests."""

import argparse
import json
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from scripts.ci.dependency_update_policy import (
    DEPENDENCY_UPDATE_POLICIES,
    REQUIRED_CHECKS,
    DependencyUpdate,
    PullRequestPolicy,
)
from scripts.ci.package_release import emit_github_output

MERGED_PULL_REQUEST_STATE = "MERGED"


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


class BranchPushMode(StrEnum):
    CREATE = "create"
    FORCE_WITH_LEASE = "force-with-lease"


@dataclass(frozen=True)
class UpdateBranch:
    expected_remote_sha: str
    pull_request_url: str
    push_mode: BranchPushMode


@dataclass(frozen=True)
class PublishedPullRequest:
    head_sha: str
    url: str


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
    if state == MERGED_PULL_REQUEST_STATE:
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
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(result.stdout.splitlines())


def prepare_update_branch(*, policy: PullRequestPolicy, repository: str) -> UpdateBranch:
    """Reset the update branch to its base and return the existing PR and push mode."""
    subprocess.run(["git", "fetch", "origin", policy.base_branch], check=True)
    pull_requests = _gh_json(
        "pr",
        "list",
        "--repo",
        repository,
        "--head",
        policy.head_branch,
        "--base",
        policy.base_branch,
        "--state",
        "open",
        "--json",
        "url",
    )
    if not isinstance(pull_requests, list) or any(not isinstance(pr, dict) for pr in pull_requests):
        raise ValueError(f"GitHub returned an invalid pull request list: {pull_requests!r}")
    if len(pull_requests) > 1:
        raise ValueError(f"found multiple open pull requests for {policy.head_branch!r}")
    pull_request_url = ""
    if pull_requests:
        url = pull_requests[0].get("url")
        if not isinstance(url, str):
            raise ValueError(f"GitHub returned an invalid pull request URL: {url!r}")
        pull_request_url = url

    remote_branch = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", policy.head_branch],
        capture_output=True,
        text=True,
    )
    if remote_branch.returncode == 0:
        fields = remote_branch.stdout.split()
        if len(fields) != 2 or fields[1] != f"refs/heads/{policy.head_branch}":
            raise ValueError(f"git returned an invalid remote branch: {remote_branch.stdout!r}")
        expected_remote_sha = fields[0]
        push_mode = BranchPushMode.FORCE_WITH_LEASE
    elif remote_branch.returncode == 2:
        expected_remote_sha = ""
        push_mode = BranchPushMode.CREATE
    else:
        remote_branch.check_returncode()
        raise AssertionError("unreachable")

    subprocess.run(
        ["git", "switch", "-C", policy.head_branch, f"origin/{policy.base_branch}"],
        check=True,
    )
    return UpdateBranch(
        expected_remote_sha=expected_remote_sha,
        pull_request_url=pull_request_url,
        push_mode=push_mode,
    )


def commit_update(policy: PullRequestPolicy) -> str:
    """Commit the allowlisted generator output and return its SHA."""
    changed_files = validate_changed_files(changed_worktree_files(), policy=policy)
    if not changed_files:
        raise ValueError("dependency update has no changed files to publish")
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
    subprocess.run(
        ["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
        check=True,
    )
    subprocess.run(["git", "add", "--", *changed_files], check=True)
    subprocess.run(["git", "commit", "-m", policy.title], check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def push_update_branch(
    *,
    policy: PullRequestPolicy,
    expected_remote_sha: str,
    push_mode: BranchPushMode,
) -> None:
    """Push the generated commit without overwriting a concurrent branch update."""
    push_args = ["git", "push"]
    if push_mode is BranchPushMode.FORCE_WITH_LEASE:
        if not expected_remote_sha:
            raise ValueError("force-with-lease requires the expected remote SHA")
        push_args.append(f"--force-with-lease=refs/heads/{policy.head_branch}:{expected_remote_sha}")
    push_args.extend(["origin", f"HEAD:{policy.head_branch}"])
    subprocess.run(push_args, check=True)


def upserted_pull_request_url(
    *,
    policy: PullRequestPolicy,
    repository: str,
    body_file: Path,
    pull_request_url: str,
) -> str:
    """Create or refresh the generated pull request and return its URL."""
    if pull_request_url:
        subprocess.run(
            [
                "gh",
                "pr",
                "edit",
                pull_request_url,
                "--repo",
                repository,
                "--title",
                policy.title,
                "--body-file",
                str(body_file),
                "--add-label",
                "agent-generated",
                "--add-label",
                "dependencies",
            ],
            check=True,
        )
        return pull_request_url

    subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            repository,
            "--base",
            policy.base_branch,
            "--head",
            policy.head_branch,
            "--title",
            policy.title,
            "--body-file",
            str(body_file),
            "--label",
            "agent-generated",
            "--label",
            "dependencies",
        ],
        check=True,
    )
    payload = _gh_json(
        "pr",
        "view",
        policy.head_branch,
        "--repo",
        repository,
        "--json",
        "url",
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("url"), str):
        raise ValueError(f"GitHub returned an invalid created pull request: {payload!r}")
    return payload["url"]


def publish_update(
    *,
    policy: PullRequestPolicy,
    repository: str,
    body_file: Path,
    expected_remote_sha: str,
    pull_request_url: str,
    push_mode: BranchPushMode,
) -> PublishedPullRequest:
    """Publish a generated commit and its pull request."""
    head_sha = commit_update(policy)
    push_update_branch(
        policy=policy,
        expected_remote_sha=expected_remote_sha,
        push_mode=push_mode,
    )
    pull_request_url = upserted_pull_request_url(
        policy=policy,
        repository=repository,
        body_file=body_file,
        pull_request_url=pull_request_url,
    )

    return PublishedPullRequest(head_sha=head_sha, url=pull_request_url)


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
            if merged.state != MERGED_PULL_REQUEST_STATE:
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
    prepare = subparsers.add_parser("prepare", help="reset the generated update branch")
    prepare.add_argument("--kind", choices=DependencyUpdate, required=True)
    prepare.add_argument("--repository", required=True)
    prepare.add_argument("--github-output", type=Path, required=True)
    publish = subparsers.add_parser("publish", help="commit and publish a generated update")
    publish.add_argument("--kind", choices=DependencyUpdate, required=True)
    publish.add_argument("--repository", required=True)
    publish.add_argument("--body-file", type=Path, required=True)
    publish.add_argument("--expected-remote-sha", default="")
    publish.add_argument("--pull-request-url", default="")
    publish.add_argument("--push-mode", choices=BranchPushMode, required=True)
    publish.add_argument("--github-output", type=Path, required=True)
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
    if args.command == "prepare":
        branch = prepare_update_branch(policy=policy, repository=args.repository)
        emit_github_output(
            args.github_output,
            expected_remote_sha=branch.expected_remote_sha,
            pr_url=branch.pull_request_url,
            push_mode=branch.push_mode,
        )
        return
    if args.command == "publish":
        pull_request = publish_update(
            policy=policy,
            repository=args.repository,
            body_file=args.body_file,
            expected_remote_sha=args.expected_remote_sha,
            pull_request_url=args.pull_request_url,
            push_mode=BranchPushMode(args.push_mode),
        )
        emit_github_output(
            args.github_output,
            head_sha=pull_request.head_sha,
            pr_url=pull_request.url,
        )
        print(f"Updated {pull_request.url}")
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
