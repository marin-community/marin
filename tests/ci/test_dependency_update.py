# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest

from scripts.ci.dependency_update import (
    CheckRow,
    MergeDecision,
    PullRequestSnapshot,
    evaluate_merge,
    evaluate_required_checks,
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


def test_native_package_policy_matches_every_compatibility_floor() -> None:
    compatibility_floors = {path.as_posix() for path in requirement_paths_for_packages(PACKAGES)}

    assert NATIVE_PACKAGE_POLICY.allowed_files == {"uv.lock", *compatibility_floors}
