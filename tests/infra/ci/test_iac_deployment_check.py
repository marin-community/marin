# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts for post-merge Pulumi deployment checks."""

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

ASSEMBLE_SCRIPT = Path(".github/actions/pulumi-preview/assemble_comment.py")
SUMMARY_SCRIPT = Path(".github/actions/pulumi-preview/summarize_deployment.py")
HEAD_SHA = "a" * 40
MERGED_AT = "2026-08-20T12:00:00Z"


def _write_preview(root: Path, stack: str, severity: str) -> None:
    artifact = root / stack
    artifact.mkdir(parents=True)
    (artifact / "meta.json").write_text(
        json.dumps({"stack": stack, "severity": severity}),
        encoding="utf-8",
    )


def _preview_outputs(tmp_path: Path, previews: dict[str, str], *, head_sha: str = HEAD_SHA) -> dict[str, object]:
    previews_dir = tmp_path / "previews"
    for stack, severity in previews.items():
        _write_preview(previews_dir, stack, severity)

    comment = tmp_path / "preview-comment.md"
    subprocess.run(
        [
            sys.executable,
            str(ASSEMBLE_SCRIPT),
            "--previews-dir",
            str(previews_dir),
            "--head-sha",
            head_sha,
            "--workflow-ok",
            "true",
            "--out",
            str(comment),
        ],
        check=True,
    )

    comments = tmp_path / "comments.json"
    comments.write_text(
        json.dumps(
            [
                [
                    {
                        "body": comment.read_text(),
                        "created_at": "2026-08-20T11:00:00Z",
                        "id": 1,
                        "updated_at": "2026-08-20T11:00:00Z",
                        "user": {"login": "github-actions[bot]"},
                    }
                ]
            ]
        ),
        encoding="utf-8",
    )
    status = tmp_path / "preview-status.json"
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "affected-stacks",
            "--comments",
            str(comments),
            "--head-sha",
            HEAD_SHA,
            "--out",
            str(status),
        ],
        check=True,
    )
    return json.loads(status.read_text())


def _write_history(root: Path, stack: str, updates: list[dict[str, str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{stack}.json").write_text(json.dumps(updates), encoding="utf-8")


def _check_outputs(
    tmp_path: Path,
    *,
    attempt: int,
    affected_stacks: tuple[str, ...] = ("marin",),
    histories: dict[str, list[dict[str, str]]] | None = None,
    preview_ok: bool = True,
    merger: str = "operator",
    reminder_id: int | None = None,
) -> tuple[str, dict[str, object]]:
    histories_dir = tmp_path / "histories"
    for stack, updates in (histories or {}).items():
        _write_history(histories_dir, stack, updates)

    comment = tmp_path / "deployment-comment.md"
    comments = tmp_path / "comments.json"
    reminder = []
    if reminder_id is not None:
        reminder.append(
            {
                "body": "<!-- iac-deployment-check -->\nPending update",
                "created_at": "2026-08-20T12:30:00Z",
                "id": reminder_id,
                "updated_at": "2026-08-20T12:30:00Z",
                "user": {"login": "github-actions[bot]"},
            }
        )
    comments.write_text(json.dumps([reminder]), encoding="utf-8")
    status = tmp_path / "deployment-status.json"
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "check",
            "--history-dir",
            str(histories_dir),
            "--comments",
            str(comments),
            "--affected-stacks",
            json.dumps(affected_stacks),
            "--preview-ok",
            str(preview_ok).lower(),
            "--merged-at",
            MERGED_AT,
            "--attempt",
            str(attempt),
            "--max-attempts",
            "3",
            "--merger",
            merger,
            "--run-url",
            "https://github.test/runs/1",
            "--out",
            str(comment),
            "--status-out",
            str(status),
        ],
        check=True,
    )
    return comment.read_text(), json.loads(status.read_text())


def _successful_update(start_time: str, end_time: str = "2026-08-20T13:05:00Z") -> dict[str, str]:
    return {
        "kind": "update",
        "result": "succeeded",
        "startTime": start_time,
        "endTime": end_time,
    }


def test_deployment_check_requires_updates_only_for_preview_changes(tmp_path: Path) -> None:
    preview = _preview_outputs(tmp_path, {"cw-rno2a": "none", "marin": "change"})
    assert preview == {"affected_stacks": ["marin"], "preview_ok": True}
    _, outputs = _check_outputs(
        tmp_path,
        attempt=1,
        affected_stacks=tuple(cast(list[str], preview["affected_stacks"])),
        histories={"marin": [_successful_update("2026-08-20T13:00:00Z")]},
    )

    assert outputs == {"comment_action": "none", "needs_retry": False, "next_attempt": None}


@pytest.mark.parametrize(
    "previews,head_sha",
    [
        ({"marin": "error"}, HEAD_SHA),
        ({"marin": "change"}, "b" * 40),
    ],
)
def test_preview_comment_rejects_failed_or_stale_preview(
    tmp_path: Path, previews: dict[str, str], head_sha: str
) -> None:
    outputs = _preview_outputs(tmp_path, previews, head_sha=head_sha)

    assert outputs == {"affected_stacks": [], "preview_ok": False}


@pytest.mark.parametrize(
    "attempt,start_time,reminder_id,expected",
    [
        (1, "2026-08-20T11:00:00Z", None, {"comment_action": "create", "needs_retry": True, "next_attempt": 2}),
        (2, "2026-08-20T11:00:00Z", 7, {"comment_action": "create", "needs_retry": True, "next_attempt": 3}),
        (1, "2026-08-20T13:00:00Z", None, {"comment_action": "none", "needs_retry": False, "next_attempt": None}),
        (
            1,
            "2026-08-20T13:00:00Z",
            7,
            {"comment_action": "update", "comment_id": 7, "needs_retry": False, "next_attempt": None},
        ),
    ],
)
def test_deployment_check_selects_comment_action(
    tmp_path: Path,
    attempt: int,
    start_time: str,
    reminder_id: int | None,
    expected: dict[str, object],
) -> None:
    _, outputs = _check_outputs(
        tmp_path,
        attempt=attempt,
        histories={"marin": [_successful_update(start_time)]},
        reminder_id=reminder_id,
    )

    assert outputs == expected


def test_deployment_check_uses_update_start_time(tmp_path: Path) -> None:
    comment, outputs = _check_outputs(
        tmp_path,
        attempt=1,
        histories={"marin": [_successful_update("2026-08-20T11:55:00Z", "2026-08-20T12:05:00Z")]},
    )

    assert outputs == {"comment_action": "create", "needs_retry": True, "next_attempt": 2}
    assert "@operator" in comment
    assert "`marin`" in comment


@pytest.mark.parametrize(
    "attempt,reminder_id,expected",
    [
        (1, None, {"comment_action": "none", "needs_retry": True, "next_attempt": 2}),
        (3, None, {"comment_action": "create", "needs_retry": True, "next_attempt": None}),
        (3, 7, {"comment_action": "none", "needs_retry": True, "next_attempt": None}),
    ],
)
def test_deployment_check_reports_errors_only_after_final_attempt(
    tmp_path: Path, attempt: int, reminder_id: int | None, expected: dict[str, object]
) -> None:
    comment, outputs = _check_outputs(
        tmp_path,
        attempt=attempt,
        preview_ok=False,
        reminder_id=reminder_id,
    )

    assert outputs == expected
    assert "`pulumi up`" not in comment
