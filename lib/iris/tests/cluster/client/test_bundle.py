# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for BundleCreator."""

import io
import zipfile
from unittest.mock import patch

import pytest

from iris.cluster.client.bundle import MAX_BUNDLE_SIZE_BYTES, BundleCreator


@pytest.fixture
def workspace(tmp_path):
    """Create a test workspace."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"cached")
    return tmp_path


def test_bundle_creator_uses_fallback_when_git_unavailable(workspace):
    with patch("iris.cluster.client.bundle._get_git_non_ignored_files", return_value=None):
        creator = BundleCreator(workspace)
        bundle_bytes = creator.create_bundle()

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = zf.namelist()
        assert "pyproject.toml" in names
        assert "src/main.py" in names
        assert not any("__pycache__" in n for n in names)


def test_bundle_creator_uses_git_files_when_available(workspace):
    git_files = {workspace / "pyproject.toml", workspace / "src" / "main.py"}
    with patch("iris.cluster.client.bundle._get_git_non_ignored_files", return_value=git_files):
        creator = BundleCreator(workspace)
        bundle_bytes = creator.create_bundle()

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = zf.namelist()
        assert "pyproject.toml" in names
        assert "src/main.py" in names
        assert not any("__pycache__" in n for n in names)


def test_bundle_includes_generated_proto_files(workspace):
    """Generated protobuf files (gitignored) are included in the bundle."""
    rpc_dir = workspace / "src" / "iris" / "rpc"
    rpc_dir.mkdir(parents=True)
    (rpc_dir / "cluster_pb2.py").write_text("# generated")
    (rpc_dir / "cluster_pb2.pyi").write_text("# generated")
    (rpc_dir / "cluster_connect.py").write_text("# generated")

    # Simulate git ls-files returning only tracked files (not the generated ones)
    git_files = {workspace / "pyproject.toml", workspace / "src" / "main.py"}
    with patch("iris.cluster.client.bundle._get_git_non_ignored_files") as mock_git:
        # Call the real function's logic but with controlled git output,
        # then verify generated files are added via _include_generated_build_artifacts.
        mock_git.return_value = git_files
        creator = BundleCreator(workspace)
        bundle_bytes = creator.create_bundle()

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = zf.namelist()
        assert "src/iris/rpc/cluster_pb2.py" in names
        assert "src/iris/rpc/cluster_pb2.pyi" in names
        assert "src/iris/rpc/cluster_connect.py" in names


def test_bundle_creator_rejects_oversized_bundles(workspace):
    """Test that bundles exceeding MAX_BUNDLE_SIZE_BYTES are rejected."""
    # Create a large file with random data that won't compress well
    import os

    large_file = workspace / "large_file.bin"
    # Use urandom to create incompressible data
    large_file.write_bytes(os.urandom(MAX_BUNDLE_SIZE_BYTES + 1024 * 1024))

    with patch("iris.cluster.client.bundle._get_git_non_ignored_files", return_value=None):
        creator = BundleCreator(workspace)
        with pytest.raises(ValueError, match=r"Bundle size .* exceeds maximum"):
            creator.create_bundle()
