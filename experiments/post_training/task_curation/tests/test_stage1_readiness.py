# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repository readiness accepts equivalent URLs without accepting wrong checkouts."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from experiments.post_training.task_curation.swesmith_staged import Stage1Contract, stage1_program


@pytest.mark.parametrize("remote", ["https://github.com/swesmith/example", "https://github.com/swesmith/example.git/"])
def test_readiness_accepts_equivalent_fork_urls_and_rejects_wrong_state(tmp_path, remote):
    repository = tmp_path / "checkout"
    repository.mkdir()
    module = repository / "readiness_fixture_module.py"
    module.write_text("ready = True\n")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()

    git("init", "-b", "task-branch")
    git("add", module.name)
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "Fixture")
    git("remote", "add", "origin", remote)
    contract = Stage1Contract(
        workspace=str(repository),
        origin="https://github.com/swesmith/example",
        branch="task-branch",
        commit=git("rev-parse", "HEAD"),
        tracked_sha256={module.name: hashlib.sha256(module.read_bytes()).hexdigest()},
        import_name=module.stem,
        distribution_name="readiness-fixture",
        symlink_targets={},
    )

    def grade():
        result = subprocess.check_output(
            [sys.executable, "-c", stage1_program(contract)],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(repository)},
            text=True,
        )
        return json.loads(result)

    assert grade()["passed"] is True
    git("remote", "set-url", "origin", "https://github.com/upstream/example.git")
    assert grade()["checks"]["origin"] is False
    git("remote", "set-url", "origin", remote)
    git("checkout", "--detach", contract.commit)
    result = grade()
    assert result["checks"]["commit"] is True
    assert result["checks"]["branch"] is False
    assert result["passed"] is False
