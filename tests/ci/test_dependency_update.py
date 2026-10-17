# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

from scripts.ci.dependency_update import (
    BranchPushMode,
    CheckRow,
    MergeDecision,
    PullRequestSnapshot,
    evaluate_merge,
    evaluate_required_checks,
    prepare_update_branch,
    publish_update,
    required_check_rows,
    validate_changed_files,
    validated_pull_request,
)
from scripts.ci.dependency_update_policy import EXTERNAL_RUNTIME_POLICY, NATIVE_PACKAGE_POLICY, PullRequestPolicy
from scripts.ci.package_release import PACKAGES, requirement_paths_for_packages

EXPECTED_SHA = "a" * 40


def _pull_request(policy: PullRequestPolicy = EXTERNAL_RUNTIME_POLICY, **overrides) -> PullRequestSnapshot:
    values = {
        "author": "app/marin-external-runtime-updater",
        "base_branch": "main",
        "files": tuple(sorted(policy.allowed_files)),
        "head_branch": policy.head_branch,
        "head_sha": EXPECTED_SHA,
        "state": "OPEN",
        "title": policy.title,
        "url": "https://github.com/marin-community/marin/pull/123",
    }
    values.update(overrides)
    return PullRequestSnapshot(**values)


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_repository(tmp_path: Path) -> tuple[Path, Path, str]:
    remote = tmp_path / "remote.git"
    repository = tmp_path / "repository"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "-b", "main", str(repository)], check=True, capture_output=True)
    _git(repository, "config", "user.name", "Test User")
    _git(repository, "config", "user.email", "test@example.com")
    (repository / "uv.lock").write_text("initial\n")
    _git(repository, "add", "uv.lock")
    _git(repository, "commit", "-m", "initial")
    _git(repository, "remote", "add", "origin", str(remote))
    _git(repository, "push", "-u", "origin", "main")
    return repository, remote, _git(repository, "rev-parse", "HEAD")


@pytest.mark.parametrize("policy", [EXTERNAL_RUNTIME_POLICY, NATIVE_PACKAGE_POLICY])
def test_returns_the_dedicated_apps_exact_generated_pull_request(policy: PullRequestPolicy) -> None:
    pull_request = _pull_request(policy)

    validated = validated_pull_request(
        pull_request,
        policy=policy,
        expected_app_slug="marin-external-runtime-updater",
        expected_head_sha=EXPECTED_SHA,
    )

    assert validated == pull_request


@pytest.mark.parametrize(
    "override",
    [
        {"author": "octocat"},
        {"base_branch": "release"},
        {"head_branch": "feature/unrelated"},
        {"head_sha": "b" * 40},
        {"title": "Update dependencies"},
        {"files": (*tuple(sorted(EXTERNAL_RUNTIME_POLICY.allowed_files)), "src/backdoor.py")},
        {"files": ()},
    ],
    ids=["author", "base", "head", "sha", "title", "files", "empty"],
)
def test_rejects_a_pull_request_outside_the_generated_boundary(override: dict) -> None:
    with pytest.raises(ValueError):
        validated_pull_request(
            _pull_request(**override),
            policy=EXTERNAL_RUNTIME_POLICY,
            expected_app_slug="marin-external-runtime-updater",
            expected_head_sha=EXPECTED_SHA,
        )


def test_required_check_gate_distinguishes_missing_pending_and_failed_checks() -> None:
    required = ("marin-integration", "marin-lint", "rust-checks", "unit-tests")

    missing = evaluate_required_checks(
        [CheckRow(name="marin-lint", bucket="pass")],
        required=required,
    )
    pending = evaluate_required_checks(
        [
            CheckRow(name="marin-integration", bucket="pass"),
            CheckRow(name="marin-lint", bucket="pass"),
            CheckRow(name="rust-checks", bucket="pending"),
            CheckRow(name="unit-tests", bucket="pass"),
        ],
        required=required,
    )
    failed = evaluate_required_checks(
        [
            CheckRow(name="marin-integration", bucket="pass"),
            CheckRow(name="marin-lint", bucket="fail"),
            CheckRow(name="rust-checks", bucket="pass"),
            CheckRow(name="unit-tests", bucket="pass"),
        ],
        required=required,
    )

    assert missing.missing == ("marin-integration", "rust-checks", "unit-tests")
    assert pending.pending == ("rust-checks",)
    assert failed.failing == ("marin-lint",)
    assert evaluate_merge("OPEN", missing) is MergeDecision.WAIT
    assert evaluate_merge("OPEN", pending) is MergeDecision.WAIT
    assert evaluate_merge("OPEN", failed) is MergeDecision.FAIL


def test_required_check_gate_ignores_duplicate_unrelated_checks() -> None:
    gate = evaluate_required_checks(
        [
            CheckRow(name="changes", bucket="pass"),
            CheckRow(name="changes", bucket="pass"),
            CheckRow(name="marin-lint", bucket="pass"),
        ],
        required=("marin-lint",),
    )

    assert gate.passed


def test_merge_gate_only_releases_an_open_pull_request_after_all_required_checks_pass() -> None:
    checks = evaluate_required_checks(
        [
            CheckRow(name=name, bucket="pass")
            for name in ("marin-integration", "marin-lint", "rust-checks", "unit-tests")
        ],
        required=("marin-integration", "marin-lint", "rust-checks", "unit-tests"),
    )

    assert evaluate_merge("OPEN", checks) is MergeDecision.MERGE
    assert evaluate_merge("MERGED", checks) is MergeDecision.DONE
    assert evaluate_merge("CLOSED", checks) is MergeDecision.FAIL


def test_no_registered_github_checks_is_a_missing_gate_not_a_cli_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.ci.dependency_update.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="no checks"),
    )

    assert required_check_rows("123", "marin-community/marin") == ()


def test_changed_files_are_sorted_and_restricted_to_the_policy() -> None:
    changed = validate_changed_files(
        ["uv.lock", "config/external/harbor/uv.lock", "uv.lock"],
        policy=EXTERNAL_RUNTIME_POLICY,
    )

    assert changed == ("config/external/harbor/uv.lock", "uv.lock")
    with pytest.raises(ValueError):
        validate_changed_files(["uv.lock", "src/backdoor.py"], policy=EXTERNAL_RUNTIME_POLICY)


def test_prepare_update_branch_resets_main_with_a_lease_for_an_existing_branch(monkeypatch, tmp_path: Path) -> None:
    repository, _remote, main_sha = _git_repository(tmp_path)
    _git(repository, "switch", "-c", NATIVE_PACKAGE_POLICY.head_branch)
    (repository / "uv.lock").write_text("stale update\n")
    _git(repository, "commit", "-am", "stale update")
    _git(repository, "push", "origin", NATIVE_PACKAGE_POLICY.head_branch)
    remote_sha = _git(repository, "rev-parse", "HEAD")
    _git(repository, "switch", "main")
    monkeypatch.chdir(repository)
    monkeypatch.setattr(
        "scripts.ci.dependency_update._gh_json",
        lambda *_args: [{"url": "https://github.com/marin-community/marin/pull/123"}],
    )

    branch = prepare_update_branch(policy=NATIVE_PACKAGE_POLICY, repository="marin-community/marin")

    assert branch.expected_remote_sha == remote_sha
    assert branch.pull_request_url == "https://github.com/marin-community/marin/pull/123"
    assert branch.push_mode is BranchPushMode.FORCE_WITH_LEASE
    assert _git(repository, "branch", "--show-current") == NATIVE_PACKAGE_POLICY.head_branch
    assert _git(repository, "rev-parse", "HEAD") == main_sha


def test_publish_update_stages_the_allowlist_and_creates_an_app_pull_request(monkeypatch, tmp_path: Path) -> None:
    repository, remote, _main_sha = _git_repository(tmp_path)
    _git(repository, "switch", "-c", NATIVE_PACKAGE_POLICY.head_branch)
    (repository / "uv.lock").write_text("published update\n")
    body_file = repository / "body.md"
    body_file.write_text("Update native packages.\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        "if sys.argv[1:3] == ['pr', 'view']:\n"
        "    print(json.dumps({'url': 'https://github.com/marin-community/marin/pull/123'}))\n"
    )
    fake_gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.chdir(repository)

    published = publish_update(
        policy=NATIVE_PACKAGE_POLICY,
        repository="marin-community/marin",
        body_file=body_file,
        expected_remote_sha="",
        pull_request_url="",
        push_mode=BranchPushMode.CREATE,
    )

    assert published.url == "https://github.com/marin-community/marin/pull/123"
    remote_sha = subprocess.run(
        ["git", "--git-dir", str(remote), "rev-parse", NATIVE_PACKAGE_POLICY.head_branch],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert published.head_sha == remote_sha
    assert _git(repository, "show", f"{remote_sha}:uv.lock") == "published update"


def test_native_package_policy_matches_every_compatibility_floor() -> None:
    compatibility_floors = {path.as_posix() for path in requirement_paths_for_packages(PACKAGES)}

    assert NATIVE_PACKAGE_POLICY.allowed_files == {"uv.lock", *compatibility_floors}
