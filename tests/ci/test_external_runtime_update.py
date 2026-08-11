# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest

from scripts.ci.external_runtime_policy import EXPECTED_FILES
from scripts.ci.external_runtime_update import (
    CheckRow,
    MergeDecision,
    PullRequestSnapshot,
    evaluate_merge,
    evaluate_required_checks,
    required_check_rows,
    validate_changed_files,
    validated_pull_request,
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


def test_returns_the_dedicated_apps_exact_generated_pull_request() -> None:
    pull_request = _pull_request()

    validated = validated_pull_request(
        pull_request,
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
        {"files": (*tuple(sorted(EXPECTED_FILES)), "src/backdoor.py")},
        {"files": ()},
    ],
    ids=["author", "base", "head", "sha", "title", "files", "empty"],
)
def test_rejects_a_pull_request_outside_the_generated_boundary(override: dict) -> None:
    with pytest.raises(ValueError):
        validated_pull_request(
            _pull_request(**override),
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
        "scripts.ci.external_runtime_update.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="no checks"),
    )

    assert required_check_rows("123", "marin-community/marin") == ()


def test_changed_files_are_sorted_and_restricted_to_the_policy() -> None:
    changed = validate_changed_files(
        ["uv.lock", "config/external/harbor/uv.lock", "uv.lock"],
    )

    assert changed == ("config/external/harbor/uv.lock", "uv.lock")
    with pytest.raises(ValueError):
        validate_changed_files(["uv.lock", "src/backdoor.py"])
