# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for TaskTrove's natural-language shell importer."""

import hashlib
import json
from pathlib import Path

import pytest
from tasktrove_verify.spec import Mode

from taskcompendium.importers.tasktrove import read_archive
from taskcompendium.importers.tasktrove_shell import import_task
from taskcompendium.models import Capability, ContainerRuntime, Rejected, ResourceRole

FIXTURES = Path(__file__).parent / "fixtures/shell"
IMAGE = "ubuntu@sha256:" + "0" * 64


def _archive(row: str):
    return read_archive((FIXTURES / f"script-row-{row}.tar.gz").read_bytes(), row, "shell-cmd")


@pytest.mark.parametrize("row", ["16158", "16159", "16160"])
def test_shell_importer_preserves_real_source_archive_and_semantic_paths(row):
    archive = _archive(row)
    sidecar = json.loads((FIXTURES / f"script-row-{row}.json").read_text())
    assert hashlib.sha256((FIXTURES / f"script-row-{row}.tar.gz").read_bytes()).hexdigest() == sidecar["archive_sha256"]

    result = import_task(archive, verifier_runtime=ContainerRuntime(IMAGE))

    assert not isinstance(result, Rejected)
    assert "verifier" not in result.steps[0].instructions.lower()
    assert "Do not delete any helper files generated during execution." in result.steps[0].instructions
    assert Capability.SHELL in result.requirements.capabilities
    assert result.requirements.state.workdir == "/workspace"
    assert result.requirements.state.setup_commands == ("cp -a /workspace/setup_files /setup_files",)
    assert result.steps[0].verifier.mode is Mode.SCRIPT
    assert result.steps[0].verifier.parameters["path"] == "nl2bash_check.py"
    assert result.steps[0].verifier.parameters["args"] == ("/output/command_capture.txt",)
    assert "workspace" not in result.steps[0].verifier.parameters
    assert {r.path for r in result.resources if ResourceRole.AGENT in r.roles} >= {"setup_files/setup_seeds.sh"}
    assert {r.path for r in result.resources if ResourceRole.VERIFIER in r.roles} >= {
        "nl2bash_check.py",
        "nl2bash_expected.json",
    }


def test_shell_importer_requires_pinned_verifier_image():
    result = import_task(_archive("16158"))

    assert isinstance(result, Rejected)
    assert result.reason.value == "unsupported_environment"


def test_shell_importer_rejects_wrong_converter():
    archive = _archive("16158")
    archive.files["task.toml"] = archive.files["task.toml"].replace(b'converter = "nl2bash"', b'converter = "other"')

    result = import_task(archive, verifier_runtime=ContainerRuntime(IMAGE))

    assert isinstance(result, Rejected)
    assert result.reason.value == "unsupported_environment"
