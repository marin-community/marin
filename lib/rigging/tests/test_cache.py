# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from rigging.cache import combined_content_hash, directory_content_hash, file_content_hash, workspace_lock_hash


def test_directory_content_hash_is_independent_of_checkout_location(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "kernel.py").write_text("source")
    (second / "kernel.py").write_text("source")
    assert directory_content_hash(first) == directory_content_hash(second)


def test_directory_content_hash_changes_with_path_and_content(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    kernel = source / "kernel.py"
    kernel.write_text("first")
    original = directory_content_hash(source)
    kernel.write_text("second")
    changed_content = directory_content_hash(source)
    kernel.rename(source / "renamed.py")
    assert len({original, changed_content, directory_content_hash(source)}) == 3


def test_directory_content_hash_ignores_derived_bytecode(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.py").write_text("source")
    original = directory_content_hash(source)
    bytecode = source / "__pycache__"
    bytecode.mkdir()
    (bytecode / "kernel.cpython-312.pyc").write_bytes(b"derived")
    assert directory_content_hash(source) == original


def test_directory_content_hash_rejects_missing_paths_and_symlinks(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("source")
    symlink = tmp_path / "linked.py"
    symlink.symlink_to(source)
    with pytest.raises(ValueError, match="does not exist"):
        directory_content_hash(tmp_path / "missing")
    with pytest.raises(ValueError, match="cannot be symlinks"):
        directory_content_hash(symlink)


def test_file_content_hash_changes_with_content(tmp_path):
    source = tmp_path / "uv.lock"
    source.write_text("first")
    original = file_content_hash(source)
    source.write_text("second")
    assert file_content_hash(source) != original


def test_combined_content_hash_frames_and_orders_components():
    assert combined_content_hash(["ab", "c"]) != combined_content_hash(["a", "bc"])
    assert combined_content_hash(["first", "second"]) != combined_content_hash(["second", "first"])


def test_workspace_lock_hash_finds_and_hashes_the_marin_lockfile(tmp_path):
    workspace = tmp_path / "workspace"
    nested = workspace / "lib" / "levanter"
    nested.mkdir(parents=True)
    (workspace / "pyproject.toml").write_text("[tool.uv.workspace]\nmembers = []\n")
    lockfile = workspace / "uv.lock"
    lockfile.write_text("revision = 1")
    original = workspace_lock_hash(nested)
    lockfile.write_text("revision = 2")
    assert original != workspace_lock_hash(nested)
