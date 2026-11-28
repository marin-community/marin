# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for per-stage preflight receipts (DESIGN.md §6) and their guard hook."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from experiments.post_training.task_curation.release_stages import preflight
from experiments.post_training.task_curation.release_stages.guard import assert_submission_allowed


def _init_repo(root: Path) -> str:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    root.mkdir(parents=True, exist_ok=True)
    git("init", "--quiet")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (root / "file.txt").write_text("one\n")
    git("add", "file.txt")
    git("commit", "--quiet", "-m", "one")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_run_preflight_seals_a_fresh_receipt(tmp_path, monkeypatch):
    repo_head = _init_repo(tmp_path / "repo")
    checked = []
    monkeypatch.setattr(preflight, "type_check_module", lambda module, repo: checked.append(module) or "ok")
    receipt = preflight.run_preflight(
        "census",
        "experiments/post_training/task_curation/release_stages/census.py",
        {"release_id": "testrelease"},
        output_root=str(tmp_path / "release"),
        repo_root=tmp_path / "repo",
        slice_runner=lambda: {"tasks": 2, "only": "Marin:fixture"},
    )
    assert receipt.ok and receipt.repo_sha == repo_head
    assert checked == ["experiments/post_training/task_curation/release_stages/census.py"]
    assert receipt.slice_evidence["tasks"] == 2
    assert preflight.read_receipt(str(tmp_path / "release"), "census", repo_head, receipt.input_digests) == receipt
    assert preflight.receipt_is_fresh(receipt, repo_head, receipt.input_digests)


def test_run_preflight_records_failure_and_refuses(tmp_path, monkeypatch):
    repo_head = _init_repo(tmp_path / "repo")

    def failing_type_check(module, repo_root):
        raise ValueError("type check failed")

    monkeypatch.setattr(preflight, "type_check_module", failing_type_check)
    with pytest.raises(ValueError, match="preflight failed for census"):
        preflight.run_preflight(
            "census",
            "no-such-module.py",
            {},
            output_root=str(tmp_path / "release"),
            repo_root=tmp_path / "repo",
            slice_runner=lambda: {},
        )
    stored = preflight.read_receipt(str(tmp_path / "release"), "census", repo_head, preflight.input_digests({}))
    assert stored is not None and not stored.ok and "type check failed" in stored.error


def test_receipt_digest_binds_the_document():
    receipt = preflight.PreflightReceipt(
        stage="census",
        mode="preflight",
        repo_sha="a" * 40,
        input_digests={"inputs": "d" * 64},
        ok=True,
        checked_at="2026-11-23T00:00:00+00:00",
    )
    assert receipt.digest() == hashlib.sha256(receipt.model_dump_json().encode()).hexdigest()


def test_guard_refuses_a_stage_whose_preflight_is_stale_or_missing(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo_head = _init_repo(repo)
    package = repo / "experiments" / "post_training" / "task_curation" / "release_stages"
    package.mkdir(parents=True)
    landing_log = tmp_path / "landings.jsonl"
    landing_log.write_text(json.dumps({"sha": repo_head}) + "\n")
    monkeypatch.setenv("ATLAS_PUSH_LOG", str(landing_log))
    digests = {"inputs": "d" * 64}
    # Missing receipt: refusal.
    with pytest.raises(ValueError, match="no fresh preflight receipt"):
        assert_submission_allowed(
            repo,
            package,
            preflight_output_root=str(tmp_path / "release"),
            preflight_digests={"census": digests},
        )
    receipt = preflight.PreflightReceipt(
        stage="census",
        mode="preflight",
        repo_sha=repo_head,
        input_digests=digests,
        ok=True,
        checked_at="2026-11-23T00:00:00+00:00",
    )
    preflight.write_receipt(str(tmp_path / "release"), receipt)
    # Landed + fresh: passes.
    assert_submission_allowed(
        repo,
        package,
        preflight_output_root=str(tmp_path / "release"),
        preflight_digests={"census": digests},
    )
    # Stale inputs: refusal.
    with pytest.raises(ValueError, match="no fresh preflight receipt"):
        assert_submission_allowed(
            repo,
            package,
            preflight_output_root=str(tmp_path / "release"),
            preflight_digests={"census": {"inputs": "e" * 64}},
        )


def test_failed_preflight_can_recover_without_changing_earlier_evidence(tmp_path):
    receipt = preflight.PreflightReceipt(
        stage="snapshot_publication",
        mode="upstream",
        repo_sha="a" * 40,
        input_digests={"contract": "b" * 64},
        ok=False,
        error="not ready",
        checked_at="2026-11-28T12:00:00+00:00",
    )
    root = str(tmp_path)
    failed_ref = preflight.write_receipt(root, receipt)
    original = Path(failed_ref.uri).read_bytes()
    with pytest.raises(ValueError, match="failed"):
        preflight.require_fresh_preflight(root, receipt.stage, receipt.repo_sha, receipt.input_digests)
    recovered = receipt.model_copy(update={"ok": True, "error": None, "checked_at": "2026-11-28T12:01:00+00:00"})
    recovered_ref = preflight.write_receipt(root, recovered)
    assert recovered_ref != failed_ref
    assert preflight.require_fresh_preflight(root, receipt.stage, receipt.repo_sha, receipt.input_digests) == recovered
    assert Path(failed_ref.uri).read_bytes() == original
