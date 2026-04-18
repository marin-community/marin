# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for workspace bundling."""

import os
import re
from unittest.mock import patch

import pytest

from iris.cluster.client.bundle import MAX_BUNDLE_SIZE_BYTES, collect_workspace_files, create_workspace_zip


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"cached")
    return tmp_path


def _mock_git_none():
    return patch("iris.cluster.client.bundle._get_git_non_ignored_files", return_value=None)


def _mock_git_files(workspace, *rel_paths):
    files = {workspace / p for p in rel_paths}
    return patch("iris.cluster.client.bundle._get_git_non_ignored_files", return_value=files)


def test_collect_falls_back_when_git_unavailable(workspace):
    with _mock_git_none():
        files = collect_workspace_files(workspace)
    rel = {str(f.relative_to(workspace)) for f in files}
    assert "pyproject.toml" in rel
    assert "src/main.py" in rel
    assert not any("__pycache__" in r for r in rel)


def test_collect_includes_generated_proto_files(workspace):
    rpc_dir = workspace / "src" / "iris" / "rpc"
    rpc_dir.mkdir(parents=True)
    (rpc_dir / "job_pb2.py").write_text("# generated")
    (rpc_dir / "job_pb2.pyi").write_text("# generated")
    (rpc_dir / "controller_connect.py").write_text("# generated")

    with _mock_git_files(workspace, "pyproject.toml", "src/main.py"):
        files = collect_workspace_files(workspace)
    rel = {str(f.relative_to(workspace)) for f in files}
    assert "src/iris/rpc/job_pb2.py" in rel
    assert "src/iris/rpc/job_pb2.pyi" in rel
    assert "src/iris/rpc/controller_connect.py" in rel


def test_collect_respects_extra_exclude(workspace):
    (workspace / "data").mkdir()
    (workspace / "data" / "big.csv").write_text("1,2,3")

    with _mock_git_none():
        files = collect_workspace_files(workspace, exclude=re.compile(r"^data(/|$)"))
    rel = {str(f.relative_to(workspace)) for f in files}
    assert "data/big.csv" not in rel
    assert "pyproject.toml" in rel


def test_zip_rejects_oversized_bundles(workspace):
    large_file = workspace / "large_file.bin"
    large_file.write_bytes(os.urandom(MAX_BUNDLE_SIZE_BYTES + 1024 * 1024))

    with _mock_git_none():
        with pytest.raises(ValueError, match=r"Bundle size .* exceeds maximum"):
            create_workspace_zip(workspace)
