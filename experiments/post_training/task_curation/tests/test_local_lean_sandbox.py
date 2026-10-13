# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``LocalLeanSandbox`` answers like the NeMo Skills sandbox server for ``language="lean4"``."""

from pathlib import Path

import pytest

from experiments.post_training.task_curation.datasets.environments import LEAN_GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.nemotron_ultra.scorers.local_lean_sandbox import (
    ELAN_HOME,
    LEAN_PROJECT_DIR,
    SCRATCH_DIR_NAME,
    LocalLeanSandbox,
)

# Stands in for `lake env --dir PROJECT lean FILE`: the proof file's first line is a verdict. `ok`
# prints a line to stdout and exits 0; `fail` reports a Lean-style error at the file's path and
# exits 1; `hang` sleeps past any deadline after writing partial output; `loud` prints more than any limit.
FAKE_LAKE = """#!/bin/sh
file="$5"
verdict="$(head -n 1 "$file")"
case "$verdict" in
  ok) echo "compiled $file"; exit 0 ;;
  fail) echo "$file:2:0: error: unknown identifier 'nope'" >&2; exit 1 ;;
  hang) echo "partial"; sleep 30 ;;
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


def test_a_proof_past_the_deadline_times_out_with_partial_output(sandbox: LocalLeanSandbox):
    result = sandbox.execute("hang\n", language="lean4", timeout_seconds=0.5)
    assert result["process_status"] == "timeout"
    assert result["stdout"] == "partial\n"
    assert "timed out" in result["stderr"]


def test_output_past_the_limit_is_cut_and_flagged(sandbox: LocalLeanSandbox):
    result = sandbox.execute("loud\n", language="lean4", timeout_seconds=5, max_output_characters=100)
    assert result["process_status"] == "completed"
    assert result["stdout"] == "x" * 100 + "<output cut>"
    assert result["output_truncated"] is True


def test_the_sandbox_paths_are_the_ones_the_lean_image_builds():
    assert LEAN_GRADER_PACKAGES.dockerfile is not None
    fragment = LEAN_GRADER_PACKAGES.dockerfile.read_text()
    assert f"ELAN_HOME={ELAN_HOME}" in fragment
    assert f"mkdir -p {LEAN_PROJECT_DIR.parent} && cd {LEAN_PROJECT_DIR.parent}" in fragment
    assert f"lake new {LEAN_PROJECT_DIR.name}" in fragment
    assert f"mkdir -m 1777 {SCRATCH_DIR_NAME}" in fragment


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
