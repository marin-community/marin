# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``LocalLeanSandbox`` answers like the NeMo Skills sandbox server for ``language="lean4"``."""

import os
import subprocess
from pathlib import Path

import pytest

from experiments.post_training.task_curation.datasets.nemotron_ultra.scorers.local_lean_sandbox import (
    LEAN_PROJECT_DIR,
    SCRATCH_DIR_NAME,
    LocalLeanSandbox,
)

# Stands in for `lake env --dir PROJECT lean FILE`: the proof file's first line is a verdict. `ok`
# prints a line to stdout and exits 0; `fail` reports a Lean-style error at the file's path and
# exits 1; `hang` signals the FIFO on the second line after writing partial output, then sleeps;
# `loud` prints more than any limit.
FAKE_LAKE = """#!/bin/sh
file="$5"
verdict="$(head -n 1 "$file")"
case "$verdict" in
  ok) echo "compiled $file"; exit 0 ;;
  fail) echo "$file:2:0: error: unknown identifier 'nope'" >&2; exit 1 ;;
  hang) echo "partial"; echo ready > "$(sed -n '2p' "$file")"; sleep 30 ;;
  loud) printf '%0.sx' $(seq 1 3000); exit 0 ;;
esac
"""

MATHLIB_PROOF = "import Mathlib\ntheorem check (n : Nat) : n + 0 = n := by simp\n"
BROKEN_PROOF = "import Mathlib\ntheorem check (n : Nat) : n + 0 = n := by nope\n"


@pytest.fixture
def sandbox(tmp_path: Path) -> LocalLeanSandbox:
    project = tmp_path / "project"
    (project / SCRATCH_DIR_NAME).mkdir(parents=True)
    lake = tmp_path / "bin" / "lake"
    lake.parent.mkdir()
    lake.write_text(FAKE_LAKE)
    lake.chmod(0o755)
    return LocalLeanSandbox(user=None, project_dir=project, lake=lake, elan_home=tmp_path / "elan")


def test_a_compiling_proof_completes_with_its_output(sandbox: LocalLeanSandbox):
    result = sandbox.execute("ok\n", language="lean4", timeout_seconds=5)
    assert result["process_status"] == "completed"
    assert result["stdout"].startswith(f"compiled {sandbox.project_dir / SCRATCH_DIR_NAME}/")
    assert result["stderr"] == ""
    assert "output_truncated" not in result


def test_a_rejected_proof_fails_with_the_diagnostic_at_the_project_path(sandbox: LocalLeanSandbox):
    result = sandbox.execute("fail\n", language="lean4", timeout_seconds=5)
    assert result["process_status"] == "failed"
    assert result["stderr"].startswith(f"{sandbox.project_dir / SCRATCH_DIR_NAME}/")
    assert ": error: unknown identifier 'nope'" in result["stderr"]


def test_a_proof_past_the_deadline_times_out_with_partial_output(sandbox: LocalLeanSandbox, tmp_path, monkeypatch):
    ready = tmp_path / "output-ready"
    os.mkfifo(ready)
    wait = subprocess.Popen.wait

    def wait_after_output(process, timeout=None):
        if timeout is not None:
            # Child startup can exceed the deadline on a busy host. Require output before
            # starting the real timed wait so this checks preservation of existing stdout.
            with ready.open() as notification:
                assert notification.readline() == "ready\n"
        return wait(process, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait_after_output)
    result = sandbox.execute(f"hang\n{ready}\n", language="lean4", timeout_seconds=0.5)
    assert result["process_status"] == "timeout"
    assert result["stdout"] == "partial\n"
    assert "timed out" in result["stderr"]


def test_output_past_the_limit_is_cut_and_flagged(sandbox: LocalLeanSandbox):
    result = sandbox.execute("loud\n", language="lean4", timeout_seconds=5, max_output_characters=100)
    assert result["process_status"] == "completed"
    assert result["stdout"] == "x" * 100 + "<output cut>"
    assert result["output_truncated"] is True


def test_the_proof_file_does_not_outlive_the_compile(sandbox: LocalLeanSandbox):
    sandbox.execute("ok\n", language="lean4", timeout_seconds=5)
    assert list((sandbox.project_dir / SCRATCH_DIR_NAME).iterdir()) == []


@pytest.mark.integration
@pytest.mark.timeout(300)
@pytest.mark.skipif(not (LEAN_PROJECT_DIR / SCRATCH_DIR_NAME).is_dir(), reason="needs the Lean grader image's Mathlib")
@pytest.mark.parametrize(
    ("proof", "status"), [(MATHLIB_PROOF, "completed"), (BROKEN_PROOF, "failed")], ids=["proves", "rejects"]
)
def test_mathlib_compiles_proofs_in_the_lean_image(proof: str, status: str):
    result = LocalLeanSandbox(user=None).execute(proof, language="lean4", timeout_seconds=240)
    assert result["process_status"] == status, result
