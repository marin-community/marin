# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for BundleCreator and create_workspace_zip."""

import io
import os
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from iris.cluster.client.bundle import MAX_BUNDLE_SIZE_BYTES, BundleCreator, create_workspace_zip


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
    with patch("iris.cluster.client.bundle.get_git_non_ignored_files", return_value=None):
        creator = BundleCreator(workspace)
        bundle_bytes = creator.create_bundle()

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as zf:
        names = zf.namelist()
        assert "pyproject.toml" in names
        assert "src/main.py" in names
        assert not any("__pycache__" in n for n in names)


def test_bundle_creator_uses_git_files_when_available(workspace):
    git_files = {workspace / "pyproject.toml", workspace / "src" / "main.py"}
    with patch("iris.cluster.client.bundle.get_git_non_ignored_files", return_value=git_files):
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
    with patch("iris.cluster.client.bundle.get_git_non_ignored_files") as mock_git:
        # Call the real function's logic but with controlled git output,
        # then verify generated files are added via include_generated_build_artifacts.
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
    large_file = workspace / "large_file.bin"
    # Use urandom to create incompressible data
    large_file.write_bytes(os.urandom(MAX_BUNDLE_SIZE_BYTES + 1024 * 1024))

    with patch("iris.cluster.client.bundle.get_git_non_ignored_files", return_value=None):
        creator = BundleCreator(workspace)
        with pytest.raises(ValueError, match=r"Bundle size .* exceeds maximum"):
            creator.create_bundle()


def test_create_workspace_zip_returns_path_with_expected_files(workspace):
    """create_workspace_zip returns a zip path containing workspace files, excluding __pycache__."""
    with patch("iris.cluster.client.bundle.get_git_non_ignored_files", return_value=None):
        zip_path = create_workspace_zip(workspace)

    assert Path(zip_path).exists()
    assert zip_path.endswith(".zip")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "pyproject.toml" in names
        assert "src/main.py" in names
        assert not any("__pycache__" in n for n in names)
        assert not any(n.endswith(".pyc") for n in names)


def test_create_workspace_zip_with_custom_exclude_dirs(workspace):
    """create_workspace_zip respects custom exclude_dirs in addition to the defaults."""
    (workspace / "experiments").mkdir()
    (workspace / "experiments" / "run.py").write_text("# experiment")
    (workspace / "src" / "core.py").write_text("# core")

    with patch("iris.cluster.client.bundle.get_git_non_ignored_files", return_value=None):
        zip_path = create_workspace_zip(workspace, exclude_dirs={"experiments"})

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "src/core.py" in names
        assert "pyproject.toml" in names
        assert not any("experiments" in n for n in names)


def test_custom_exclude_dirs_only_matches_top_level(workspace):
    """Custom exclude_dirs should only exclude top-level directories, not nested ones."""
    # Top-level tests/ should be excluded
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_foo.py").write_text("# top-level test")
    # Nested lib/pkg/tests/ should NOT be excluded
    (workspace / "lib").mkdir()
    (workspace / "lib" / "pkg").mkdir()
    (workspace / "lib" / "pkg" / "tests").mkdir()
    (workspace / "lib" / "pkg" / "tests" / "test_bar.py").write_text("# nested test")

    with patch("iris.cluster.client.bundle.get_git_non_ignored_files", return_value=None):
        zip_path = create_workspace_zip(workspace, exclude_dirs={"tests"})

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert not any(n.startswith("tests/") for n in names), "top-level tests/ should be excluded"
        assert "lib/pkg/tests/test_bar.py" in names, "nested tests/ should be included"
