# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts for post-merge Pulumi deployment checks."""

import json
import subprocess
import sys
from pathlib import Path

SUMMARY_SCRIPT = Path(".github/actions/pulumi-preview/summarize_deployment.py")
FORMAT_SCRIPT = Path(".github/actions/pulumi-preview/format_preview.py")
MERGED_AT = "2026-08-20T12:00:00Z"


def _write_preview(root: Path, stack: str, severity: str, *, last_successful_update: str | None = None) -> None:
    artifact = root / stack
    artifact.mkdir(parents=True)
    meta = {"stack": stack, "severity": severity}
    if last_successful_update is not None:
        meta["last_successful_update"] = last_successful_update
    (artifact / "meta.json").write_text(
        json.dumps(meta),
        encoding="utf-8",
    )


def _summarize(tmp_path: Path, *, attempt: int, merger: str = "operator") -> tuple[str, dict[str, str]]:
    previews = tmp_path / "previews"
    comment = tmp_path / "comment.md"
    github_output = tmp_path / "github-output"
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--previews-dir",
            str(previews),
            "--preview-matrix",
            '{"include":[{"stack":"marin"},{"stack":"cw-rno2a"}]}',
            "--check-delays-minutes",
            "[30,60,120]",
            "--merged-at",
            MERGED_AT,
            "--attempt",
            str(attempt),
            "--merger",
            merger,
            "--run-url",
            "https://github.test/runs/1",
            "--out",
            str(comment),
            "--github-output",
            str(github_output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    outputs = dict(line.split("=", 1) for line in github_output.read_text().splitlines())
    return comment.read_text(), outputs


def test_preview_metadata_records_last_successful_update(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("Resources:\n    1 to update\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(FORMAT_SCRIPT),
            "--stack",
            "marin",
            "--ok",
            "true",
            "--history-ok",
            "true",
            "--last-successful-update",
            "2026-08-20T13:00:00Z",
            "--input",
            str(raw),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
    )

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["severity"] == "change"
    assert meta["last_successful_update"] == "2026-08-20T13:00:00Z"


def test_deployment_summary_requests_retry_for_changes_and_missing_stacks(tmp_path: Path) -> None:
    _write_preview(tmp_path / "previews", "marin", "change")

    comment, outputs = _summarize(tmp_path, attempt=1)

    assert outputs == {"needs_retry": "true", "next_attempt": "2", "should_comment": "true"}
    assert "@operator" in comment
    assert "`marin`" in comment
    assert "`cw-rno2a`" in comment


def test_deployment_summary_clears_existing_reminder_after_retry(tmp_path: Path) -> None:
    _write_preview(tmp_path / "previews", "marin", "none")
    _write_preview(tmp_path / "previews", "cw-rno2a", "none")

    _, outputs = _summarize(tmp_path, attempt=2)

    assert outputs == {"needs_retry": "false", "next_attempt": "", "should_comment": "true"}


def test_deployment_summary_does_not_comment_on_initial_clean_check(tmp_path: Path) -> None:
    _write_preview(tmp_path / "previews", "marin", "none")
    _write_preview(tmp_path / "previews", "cw-rno2a", "none")

    _, outputs = _summarize(tmp_path, attempt=1)

    assert outputs == {"needs_retry": "false", "next_attempt": "", "should_comment": "false"}


def test_deployment_summary_ignores_historical_diff_after_later_update(tmp_path: Path) -> None:
    _write_preview(
        tmp_path / "previews",
        "marin",
        "change",
        last_successful_update="2026-08-20T13:00:00Z",
    )
    _write_preview(tmp_path / "previews", "cw-rno2a", "none")

    _, outputs = _summarize(tmp_path, attempt=1)

    assert outputs == {"needs_retry": "false", "next_attempt": "", "should_comment": "false"}


def test_deployment_summary_retries_diff_when_last_update_predates_merge(tmp_path: Path) -> None:
    _write_preview(
        tmp_path / "previews",
        "marin",
        "change",
        last_successful_update="2026-08-20T11:00:00Z",
    )
    _write_preview(tmp_path / "previews", "cw-rno2a", "none")

    _, outputs = _summarize(tmp_path, attempt=1)

    assert outputs == {"needs_retry": "true", "next_attempt": "2", "should_comment": "true"}


def test_deployment_summary_error_only_result_requests_investigation(tmp_path: Path) -> None:
    _write_preview(tmp_path / "previews", "marin", "error")
    _write_preview(tmp_path / "previews", "cw-rno2a", "error")

    comment, outputs = _summarize(tmp_path, attempt=1, merger="deployment-app[bot]")

    assert outputs == {"needs_retry": "true", "next_attempt": "2", "should_comment": "true"}
    assert "@deployment-app[bot]" in comment
    assert "`pulumi up`" not in comment


def test_deployment_summary_stops_after_third_pending_check(tmp_path: Path) -> None:
    _write_preview(tmp_path / "previews", "marin", "change")
    _write_preview(tmp_path / "previews", "cw-rno2a", "none")

    _, outputs = _summarize(tmp_path, attempt=3)

    assert outputs == {"needs_retry": "true", "next_attempt": "", "should_comment": "true"}
