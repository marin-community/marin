# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The no-oracle qualification checks what they claim to check."""

import gzip
import io
import tarfile

import pytest

from experiments.post_training.task_curation.no_oracle_qualification import (
    PreservationResult,
    QualificationReport,
    SetupExecution,
    VerifierResult,
    check_preservation,
    classify_verifier_run,
    copied_destinations,
    copied_paths,
    summarize,
)


def _tar(members: dict[str, tuple[bytes, int]], gz: bool = True) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, (content, mode) in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = mode
            archive.addfile(info, io.BytesIO(content))
    raw = buffer.getvalue()
    return gzip.compress(raw, mtime=0) if gz else raw


def _native(files: dict[str, bytes] | None = None, recipe: str | None = None) -> bytes:
    recipe = recipe or "FROM ubuntu:24.04\nCOPY files/ /app/files/\n"
    members = {f"environment/files/{name}": (content, 0o644) for name, content in (files or {"a.csv": b"1\n"}).items()}
    members["environment/Dockerfile"] = (recipe.encode(), 0o644)
    members["task.toml"] = (b'[task]\nid = "x"\n', 0o644)
    return _tar(members)


def _rebuilt(payload_files: dict[str, tuple[bytes, int]] | None = None) -> bytes:
    files = payload_files or {"files/a.csv": (b"1\n", 0o644)}
    payload = _tar(files, gz=False)
    members = {
        "task.toml": (b'[task]\nid = "x"\n', 0o644),
        "setup_files/calibforge-fixtures.tar.gz": (payload, 0o644),
        "setup_files/calibforge-initialize.sh": (b"#!/bin/sh\n", 0o755),
    }
    return _tar(members)


def test_copied_paths_lists_context_sources_and_rejects_add():
    assert copied_paths("FROM ubuntu:24.04\nCOPY files/ /app/files/\n") == ["files"]
    assert copied_paths("FROM x\nCOPY --chown=root a.txt b.txt /app/\n") == ["a.txt", "b.txt"]
    with pytest.raises(ValueError, match="ADD"):
        copied_paths("FROM x\nADD archive.tar /app/\n")


def test_preservation_passes_when_the_payload_keeps_bytes_and_modes():
    result = check_preservation("task", _native({"a.csv": b"1\n"}), _rebuilt({"files/a.csv": (b"1\n", 0o644)}))
    assert result.passed and result.expected


def test_preservation_flags_changed_missing_and_remoded_files():
    changed = check_preservation("task", _native({"a.csv": b"1\n"}), _rebuilt({"files/a.csv": (b"2\n", 0o644)}))
    assert not changed.passed and changed.changed == ("environment/files/a.csv",)

    missing = check_preservation("task", _native({"a.csv": b"1\n"}), _rebuilt({"files/other.csv": (b"1\n", 0o644)}))
    assert not missing.passed and missing.missing == ("environment/files/a.csv",)

    remoded = check_preservation("task", _native({"a.csv": b"1\n"}), _rebuilt({"files/a.csv": (b"1\n", 0o755)}))
    assert not remoded.passed and remoded.remoded == ("environment/files/a.csv",)


def test_verifier_classification_separates_a_real_run_from_infrastructure():
    assert classify_verifier_run(exit_code=0, timed_out=False, stdout="0.0") == "negative_verdict"
    assert classify_verifier_run(exit_code=0, timed_out=False, stdout="tests failed\nreward: 0") == "negative_verdict"
    assert classify_verifier_run(exit_code=0, timed_out=False, stdout="1.0") == "positive_reward"
    assert classify_verifier_run(exit_code=1, timed_out=False, stdout="0.0") == "infra_error"
    assert classify_verifier_run(exit_code=None, timed_out=True, stdout="") == "timeout"
    assert classify_verifier_run(exit_code=0, timed_out=False, stdout="") == "infra_error"
    assert classify_verifier_run(exit_code=0, timed_out=False, stdout="Traceback (most recent call last)") == "malformed"


def test_verdict_needs_every_phase_and_every_sample():
    preservation = [PreservationResult("a", expected=())]
    execution = [SetupExecution("a", "executed", written=1)]
    only_offline = QualificationReport(preservation=preservation)
    assert only_offline.verdict() == "skipped"  # no setup execution or verifier run yet

    no_execution = QualificationReport(preservation=preservation, verifier=[VerifierResult("a", "negative_verdict", 0)])
    assert no_execution.verdict() == "skipped"  # preservation and verifier alone are not enough

    clean = QualificationReport(
        preservation=preservation, execution=execution, verifier=[VerifierResult("a", "negative_verdict", 0)]
    )
    assert clean.verdict() == "verifier_executed"

    failed = QualificationReport(
        preservation=preservation, execution=execution, verifier=[VerifierResult("a", "malformed", 0)]
    )
    assert failed.verdict() == "skipped"

    summary = summarize(clean)
    assert summary["oracle_validation"] == "verifier_executed"
    assert summary["preservation_rate"] == 1.0 and summary["execution_rate"] == 1.0


def test_copied_destinations_resolve_against_the_native_workdir():
    """Files copied outside the workdir are scanned, not silently skipped."""
    recipe = "FROM ubuntu:24.04\nWORKDIR /app\nCOPY files/data /data\nCOPY files/local relative\n"
    assert copied_destinations(recipe, "/app") == ("/data", "/app/relative")


def test_setup_execution_requires_every_expected_file():
    """An executed script that misses a copied file is not a pass."""
    assert SetupExecution("task", "executed", written=2).passed
    assert not SetupExecution("task", "executed", written=1, missing=("abc:0644",)).passed
    assert not SetupExecution("task", "script_error", detail="pip3: not found").passed
