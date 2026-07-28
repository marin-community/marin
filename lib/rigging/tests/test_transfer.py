# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


import pytest
import rigging.filesystem.factory as filesystem_factory
from rigging.filesystem import StoragePath, TreeTransferMode, copy_tree
from rigging.testing import memory_filesystem_and_resolver


def test_copy_tree_preserves_nested_files_and_empty_directories(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "empty").mkdir()
    (source / "root.txt").write_text("root")
    (source / "nested" / "data.bin").write_bytes(b"\x00\x01")

    result = copy_tree(
        StoragePath(str(source)), StoragePath(str(tmp_path / "destination")), mode=TreeTransferMode.RESUME
    )

    destination = tmp_path / "destination"
    assert result.copied_files == 2
    assert result.skipped_files == 0
    assert result.copied_bytes == 6
    assert (destination / "root.txt").read_text() == "root"
    assert (destination / "nested" / "data.bin").read_bytes() == b"\x00\x01"
    assert (destination / "empty").is_dir()


@pytest.mark.parametrize("protocol", ["gs", "s3"])
def test_copy_tree_uses_the_filesystem_from_each_uri(protocol, tmp_path, monkeypatch):
    _remote_fs, resolve = memory_filesystem_and_resolver(protocol, filesystem_factory.url_to_fs)
    monkeypatch.setattr("rigging.filesystem.factory.url_to_fs", resolve)

    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "task.toml").write_text("version = '1.0'")
    remote = StoragePath(f"{protocol}://regional-cache/benchmark")

    uploaded = copy_tree(StoragePath(str(source)), remote, mode=TreeTransferMode.RESUME)
    restored = copy_tree(remote, StoragePath(str(tmp_path / "restored")), mode=TreeTransferMode.RESUME)

    assert uploaded.copied_files == 1
    assert restored.copied_files == 1
    assert (tmp_path / "restored" / "nested" / "task.toml").read_text() == "version = '1.0'"


def test_copy_tree_resume_and_overwrite_are_explicit(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "same.txt").write_text("new")
    (destination / "same.txt").write_text("old")
    (source / "changed.txt").write_text("longer")
    (destination / "changed.txt").write_text("x")
    (destination / "destination-only.txt").write_text("keep")

    resumed = copy_tree(
        StoragePath(str(source)),
        StoragePath(str(destination)),
        mode=TreeTransferMode.RESUME,
    )

    assert resumed.copied_files == 1
    assert resumed.skipped_files == 1
    assert (destination / "same.txt").read_text() == "old"
    assert (destination / "changed.txt").read_text() == "longer"
    assert (destination / "destination-only.txt").read_text() == "keep"

    overwritten = copy_tree(
        StoragePath(str(source)),
        StoragePath(str(destination)),
        mode=TreeTransferMode.OVERWRITE,
    )

    assert overwritten.copied_files == 2
    assert overwritten.skipped_files == 0
    assert (destination / "same.txt").read_text() == "new"


def test_copy_tree_rejects_a_destination_inside_the_source(tmp_path):
    source_path = tmp_path / "source"
    source_path.mkdir()
    source = StoragePath(str(source_path))

    with pytest.raises(ValueError, match="inside source"):
        copy_tree(source, source / "nested", mode=TreeTransferMode.RESUME)


def test_copy_tree_normalizes_doubled_walk_separators_without_escaping_destination(monkeypatch):
    remote_fs, resolve = memory_filesystem_and_resolver("s3", filesystem_factory.url_to_fs)
    monkeypatch.setattr("rigging.filesystem.factory.url_to_fs", resolve)
    remote_fs.makedirs("regional-cache/source/nested")
    walk_entries = list(remote_fs.walk("regional-cache/source"))

    def walk_with_doubled_separator(_path):
        for directory, subdirectories, files in walk_entries:
            if directory.endswith("/nested"):
                directory = directory.replace("/nested", "//nested")
            yield directory, subdirectories, files

    monkeypatch.setattr(remote_fs, "walk", walk_with_doubled_separator)

    copy_tree(
        StoragePath("s3://regional-cache/source"),
        StoragePath("s3://regional-cache/destination"),
        mode=TreeTransferMode.RESUME,
    )

    assert remote_fs.isdir("regional-cache/destination/nested")
    assert not remote_fs.isdir("nested")
