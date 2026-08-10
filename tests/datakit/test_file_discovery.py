# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared datakit data-file walk."""

from pathlib import Path

from marin.datakit.file_discovery import walk_data_files


def _write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def test_walk_reports_sizes_and_skips_sidecars(tmp_path: Path):
    """Discovery yields only readable data files, each carrying its true byte size.

    Sizes come from the walk's listing rather than a per-file stat, so a filesystem or
    fsspec change that stops reporting them must fail here rather than silently size a
    normalize input at zero bytes (which would collapse it to a single shard).
    """
    _write(tmp_path / "a.jsonl", b"x" * 10)
    _write(tmp_path / "nested" / "b.parquet", b"y" * 25)
    _write(tmp_path / "README.md", b"unsupported extension")
    _write(tmp_path / "_SUCCESS", b"")
    _write(tmp_path / ".provenance.json", b"{}")
    _write(tmp_path / ".metrics" / "stats.json", b"{}")

    discovered = {file.path: file.size for file in walk_data_files(str(tmp_path))}

    assert discovered == {
        str(tmp_path / "a.jsonl"): 10,
        str(tmp_path / "nested" / "b.parquet"): 25,
    }


def test_walk_excludes_named_parent_directories(tmp_path: Path):
    """``exclude_dir_names`` drops a task directory without touching its siblings."""
    _write(tmp_path / "test" / "keep_me" / "data.jsonl", b"{}")
    _write(tmp_path / "test" / "drop_me" / "data.jsonl", b"{}")

    discovered = [file.path for file in walk_data_files(str(tmp_path), exclude_dir_names=frozenset({"drop_me"}))]

    assert discovered == [str(tmp_path / "test" / "keep_me" / "data.jsonl")]


def test_walk_honors_extension_filter(tmp_path: Path):
    """A caller-supplied extension tuple narrows discovery to that format."""
    _write(tmp_path / "a.jsonl", b"{}")
    _write(tmp_path / "b.parquet", b"")

    discovered = [file.path for file in walk_data_files(str(tmp_path), extensions=(".parquet",))]

    assert discovered == [str(tmp_path / "b.parquet")]
