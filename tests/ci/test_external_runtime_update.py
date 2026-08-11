# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest

from scripts.ci.external_runtime_update import (
    EXPECTED_FILES,
    MergeDecision,
    PullRequestSnapshot,
    evaluate_merge,
    evaluate_required_checks,
    required_check_rows,
    validate_pull_request,
)

EXPECTED_SHA = "a" * 40


def _pull_request(**overrides) -> PullRequestSnapshot:
    values = {
        "author": "app/marin-external-runtime-updater",
        "base_branch": "main",
        "files": tuple(sorted(EXPECTED_FILES)),
        "head_branch": "automation/external-dependencies",
        "head_sha": EXPECTED_SHA,
        "state": "OPEN",
        "title": "[dependencies] Advance external runtimes",
        "url": "https://github.com/marin-community/marin/pull/123",
    }
    values.update(overrides)
    return PullRequestSnapshot(**values)


def test_accepts_the_dedicated_apps_exact_generated_pull_request() -> None:
    validate_pull_request(
        _pull_request(),
        expected_app_slug="marin-external-runtime-updater",
        expected_head_sha=EXPECTED_SHA,
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"author": "octocat"}, "author"),
        ({"base_branch": "release"}, "base branch"),
        ({"head_branch": "feature/unrelated"}, "head branch"),
        ({"head_sha": "b" * 40}, "head SHA"),
        ({"title": "Update dependencies"}, "title"),
        ({"files": (*tuple(sorted(EXPECTED_FILES)), "src/backdoor.py")}, "unexpected files"),
        ({"files": ()}, "no changed files"),
    ],
)
def test_rejects_a_pull_request_outside_the_generated_boundary(override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_pull_request(
            _pull_request(**override),
            expected_app_slug="marin-external-runtime-updater",
            expected_head_sha=EXPECTED_SHA,
        )


def test_required_check_gate_distinguishes_missing_pending_and_failed_checks() -> None:
    required = ("marin-integration", "marin-lint", "rust-checks", "unit-tests")

    missing = evaluate_required_checks(
        [{"name": "marin-lint", "bucket": "pass"}],
        required=required,
    )
    pending = evaluate_required_checks(
        [
            {"name": "marin-integration", "bucket": "pass"},
            {"name": "marin-lint", "bucket": "pass"},
            {"name": "rust-checks", "bucket": "pending"},
            {"name": "unit-tests", "bucket": "pass"},
        ],
        required=required,
    )
    failed = evaluate_required_checks(
        [
            {"name": "marin-integration", "bucket": "pass"},
            {"name": "marin-lint", "bucket": "fail"},
            {"name": "rust-checks", "bucket": "pass"},
            {"name": "unit-tests", "bucket": "pass"},
        ],
        required=required,
    )

    assert missing.missing == ("marin-integration", "rust-checks", "unit-tests")
    assert pending.pending == ("rust-checks",)
    assert failed.failing == ("marin-lint",)
    assert evaluate_merge("OPEN", missing) is MergeDecision.WAIT
    assert evaluate_merge("OPEN", pending) is MergeDecision.WAIT
    assert evaluate_merge("OPEN", failed) is MergeDecision.FAIL


def test_merge_gate_only_releases_an_open_pull_request_after_all_required_checks_pass() -> None:
    checks = evaluate_required_checks(
        [
            {"name": name, "bucket": "pass"}
            for name in ("marin-integration", "marin-lint", "rust-checks", "unit-tests")
        ],
        required=("marin-integration", "marin-lint", "rust-checks", "unit-tests"),
    )

    assert evaluate_merge("OPEN", checks) is MergeDecision.MERGE
    assert evaluate_merge("MERGED", checks) is MergeDecision.DONE
    assert evaluate_merge("CLOSED", checks) is MergeDecision.FAIL


def test_no_registered_github_checks_is_a_missing_gate_not_a_cli_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.ci.external_runtime_update.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="no checks"),
    )

    assert required_check_rows("123", "marin-community/marin") == []
