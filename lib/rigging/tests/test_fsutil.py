# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for fsutil's copy semantics and file rendering, over local paths — which
``filesystem_for`` routes exactly as it routes an object store."""

import bz2
import gzip
import json
import lzma
import threading

import pytest
import rigging.fsutil.cli as cli_module
from click.testing import CliRunner
from rigging.fsutil import listing
from rigging.fsutil.cli import cli
from rigging.fsutil.listing import MAX_PREVIEW_BYTES, read_decompressed_preview
from rigging.fsutil.render import file_lines


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "b.txt").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("nested")
    return tmp_path


def test_cp_handles_the_awkward_destination_shapes(tree, tmp_path, monkeypatch):
    """A prefix copy mirrors the tree; the shapes around it must not silently misfire.

    A directory without -r is an error rather than a partial copy, a missing source is
    an error rather than "0 objects", a -r whose source is a single object keeps that
    object's name, and a destination with no directory component is written as a file
    rather than created as a directory.
    """
    run = CliRunner().invoke

    result = run(cli, ["cp", "-r", str(tree / "sub"), str(tmp_path / "tree-out")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tree-out" / "c.txt").read_text() == "nested"

    result = run(cli, ["cp", str(tree / "sub"), str(tmp_path / "no-flag")])
    assert result.exit_code != 0 and "-r" in result.output

    result = run(cli, ["cp", "-r", str(tree / "absent"), str(tmp_path / "missing")])
    assert result.exit_code != 0 and "does not exist" in result.output

    result = run(cli, ["cp", "-r", str(tree / "b.txt"), str(tmp_path / "file-out")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "file-out" / "b.txt").read_text() == "hello"

    monkeypatch.chdir(tmp_path)
    result = run(cli, ["cp", str(tree / "b.txt"), "bare.txt"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "bare.txt").read_text() == "hello"


def test_rm_requires_recursive_for_directories(tree):
    run = CliRunner().invoke

    result = run(cli, ["rm", str(tree / "sub")])
    assert result.exit_code != 0 and "-r" in result.output
    assert (tree / "sub" / "c.txt").exists()

    result = run(cli, ["rm", "-R", str(tree / "sub")])
    assert result.exit_code == 0, result.output
    assert not (tree / "sub").exists()
    assert "Removing 1 objects" in result.output

    result = run(cli, ["rm", str(tree / "b.txt")])
    assert result.exit_code == 0, result.output
    assert not (tree / "b.txt").exists()


def test_rm_uses_s3_bulk_delete_batches(monkeypatch):
    class RecordingS3FileSystem:
        def __init__(self):
            self.requests = []

        def isdir(self, _path):
            return True

        def find(self, path, *, detail):
            assert path == "bucket/prefix"
            assert detail is True
            return {
                f"bucket/prefix/{index}": {"name": f"bucket/prefix/{index}", "size": 1, "type": "file"}
                for index in range(1001)
            }

        def split_path(self, path):
            bucket, key = path.split("/", 1)
            return bucket, key, None

        def call_s3(self, method, **kwargs):
            assert method == "delete_objects"
            self.requests.append(kwargs)
            return {}

        def invalidate_cache(self):
            pass

    fs = RecordingS3FileSystem()
    monkeypatch.setattr(cli_module, "S3FileSystem", RecordingS3FileSystem)
    monkeypatch.setattr(cli_module, "filesystem_for", lambda _url: (fs, "bucket/prefix"))

    result = CliRunner().invoke(cli, ["rm", "-R", "s3://bucket/prefix"])

    assert result.exit_code == 0, result.output
    batches = [request["Delete"]["Objects"] for request in fs.requests]
    assert sorted(map(len, batches)) == [1, 1000]
    assert {item["Key"] for batch in batches for item in batch} == {f"prefix/{index}" for index in range(1001)}


def test_json_previews_render_as_tables_and_degrade_safely():
    """JSONL records become a column per key; a truncated record and binary data still
    produce something readable instead of an error."""
    table = file_lines("metrics.jsonl", b'{"step": 1, "loss": 3.5}\n{"step": 2, "loss": 2.0}\n')
    assert table[0].split() == ["step", "loss"]
    assert table[2].split() == ["1", "3.5"]

    assert file_lines("metrics.jsonl", b'{"step": 1}\n{"step": 2, trunc') == ['{"step": 1}', '{"step": 2, trunc']
    assert file_lines("nested.json", json.dumps({"a": {"b": 1}}).encode())[2].split()[0] == "a"
    assert file_lines("model.bin", b"\x00\xff\xfe") == ["[binary file, 3 bytes]"]


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [
        (".gz", gzip.compress),
        (".bz2", bz2.compress),
        (".xz", lzma.compress),
        (".lzma", lambda data: lzma.compress(data, format=lzma.FORMAT_ALONE)),
    ],
)
def test_compressed_json_preview_decompresses_and_renders_as_a_table(tmp_path, suffix, compress):
    payload = b'{"step": 1, "loss": 3.5}\n{"step": 2, "loss": 2.0}\n'
    path = tmp_path / f"metrics.jsonl{suffix}"
    path.write_bytes(compress(payload))

    preview = read_decompressed_preview(str(path))

    assert preview.truncated is False
    assert file_lines(path.name, preview.data)[2].split() == ["1", "3.5"]


def test_compressed_preview_applies_limit_to_decompressed_data(tmp_path):
    path = tmp_path / "large.txt.gz"
    path.write_bytes(gzip.compress(b"x" * (MAX_PREVIEW_BYTES + 1)))

    preview = read_decompressed_preview(str(path))

    assert preview.truncated is True
    assert preview.data == b"x" * MAX_PREVIEW_BYTES


def test_cat_raw_keeps_compressed_bytes(tmp_path):
    compressed = gzip.compress(b'{"step": 1}\n')
    path = tmp_path / "metrics.jsonl.gz"
    path.write_bytes(compressed)

    result = CliRunner().invoke(cli, ["cat", "--raw", str(path)])

    assert result.exit_code == 0, result.output
    assert result.stdout_bytes == compressed


@pytest.mark.parametrize("command", [["cat"], ["head", "-n", "3"]])
def test_formatted_cli_commands_decompress_json(tmp_path, command):
    path = tmp_path / "metrics.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"step": 1, "loss": 3.5}\n'))

    result = CliRunner().invoke(cli, [*command, str(path)])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[2].split() == ["1", "3.5"]


def test_cat_reports_full_size_when_uncompressed_preview_is_truncated(tmp_path, monkeypatch):
    path = tmp_path / "large.txt"
    path.write_text("abcdef")
    monkeypatch.setattr(listing, "MAX_PREVIEW_BYTES", 4)

    result = CliRunner().invoke(cli, ["cat", str(path)])

    assert result.exit_code == 0, result.output
    assert result.stderr == "[truncated: read 4 B of 6 B]\n"


def test_ls_long_renders_local_directory(tree):
    result = CliRunner().invoke(cli, ["ls", "-l", str(tree)])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split() == ["size", "modified", "name"]
    assert any(line.endswith("b.txt") for line in lines)
    assert any(line.endswith("sub/") for line in lines)


def test_ls_glob_renders_matches_with_listing_metadata(monkeypatch):
    class GlobFileSystem:
        def glob(self, path, *, detail):
            assert path == "bucket/*/foo"
            assert detail is True
            return {
                "bucket/first/foo": {"name": "bucket/first/foo", "size": 3, "type": "file"},
                "bucket/second/foo": {"name": "bucket/second/foo", "size": 5, "type": "file"},
            }

    monkeypatch.setattr(listing, "filesystem_for", lambda _url: (GlobFileSystem(), "bucket/*/foo"))

    result = CliRunner().invoke(cli, ["ls", "s3://bucket/*/foo"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["first/foo", "second/foo"]


def test_du_scans_directories_in_parallel_using_listing_metadata(monkeypatch):
    class ParallelListingFileSystem:
        def __init__(self):
            self.child_listings_started = threading.Barrier(2)

        def ls(self, path, *, detail):
            assert detail is True
            if path == "bucket/root":
                return [
                    {"name": "bucket/root/recon", "size": 0, "type": "directory"},
                    {"name": "bucket/root/top", "size": 5, "type": "file"},
                ]
            if path in ("bucket/root/recon/a", "bucket/root/recon/b"):
                self.child_listings_started.wait(timeout=2)
            return {
                "bucket/root/recon": [
                    {"name": "bucket/root/recon/a", "size": 0, "type": "directory"},
                    {"name": "bucket/root/recon/b", "size": 0, "type": "directory"},
                ],
                "bucket/root/recon/a": [
                    {"name": "bucket/root/recon/a/nested", "size": 0, "type": "directory"},
                    {"name": "bucket/root/recon/a/one", "size": 7, "type": "file"},
                ],
                "bucket/root/recon/b": [{"name": "bucket/root/recon/b/two", "size": 11, "type": "file"}],
                "bucket/root/recon/a/nested": [{"name": "bucket/root/recon/a/nested/three", "size": 13, "type": "file"}],
            }[path]

    monkeypatch.setattr(listing, "filesystem_for", lambda _url: (ParallelListingFileSystem(), "bucket/root"))

    assert listing.total_size("s3://bucket/root") == (36, 4)
