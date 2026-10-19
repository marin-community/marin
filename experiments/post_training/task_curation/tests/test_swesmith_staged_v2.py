# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import subprocess
import sys

import pytest

from experiments.post_training.task_curation.swesmith_staged_v2_runtime import handoff_program


def test_handoff_materializes_source_without_touching_protected_files(tmp_path):
    source = tmp_path / "implementation.py"
    source.write_text("value = 1\n")
    protected = tmp_path / "test_implementation.py"
    protected.write_text("assert value == 1\n")
    files = {"implementation.py": base64.b64encode(b"value = 2\n").decode()}
    program = handoff_program(files, ["implementation.py"], tmp_path)
    subprocess.run([sys.executable, "-c", program], check=True)
    assert source.read_text() == "value = 2\n"
    assert protected.read_text() == "assert value == 1\n"


@pytest.mark.parametrize("name", ["../outside.py", "test_implementation.py", "pkg/tests/test_private.py"])
def test_handoff_rejects_test_leak_and_workspace_escape(tmp_path, name):
    with pytest.raises(ValueError, match="outside permitted source"):
        handoff_program({name: base64.b64encode(b"secret test").decode()}, [name], tmp_path)


def test_handoff_rejects_symlink_into_private_files(tmp_path):
    private = tmp_path / "private.py"
    private.write_text("private data")
    (tmp_path / "implementation.py").symlink_to(private)
    program = handoff_program(
        {"implementation.py": base64.b64encode(b"replacement").decode()}, ["implementation.py"], tmp_path
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True)
    assert result.returncode != 0
    assert private.read_text() == "private data"
