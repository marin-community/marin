# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for fsutil copy operations and file previews on local paths."""

import bz2
import gzip
import json
import lzma

import pytest
from click.testing import CliRunner
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
