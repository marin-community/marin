# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fsutil listing model, file rendering, and CLI verbs.

Exercised over the local filesystem, which ``filesystem_for`` routes exactly as it
routes an object store, so navigation and copying are covered without a bucket.
"""

import json

import pytest
from click.testing import CliRunner
from rigging.fsutil.cli import cli
from rigging.fsutil.listing import ROOT, list_entries, parent_url, total_size
from rigging.fsutil.render import file_lines, format_size


@pytest.fixture
def tree(tmp_path):
    """A small directory tree: two files and a subdirectory holding one more."""
    (tmp_path / "b.txt").write_text("hello")
    (tmp_path / "a.json").write_text(json.dumps({"key": "value"}))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("nested")
    return tmp_path


# --- listing ---------------------------------------------------------------


def test_listing_puts_directories_first_and_excludes_the_listed_path(tree):
    entries = list_entries(str(tree))
    assert [(e.name, e.is_dir) for e in entries] == [("sub", True), ("a.json", False), ("b.txt", False)]
    assert all(e.url.startswith(str(tree)) for e in entries)


def test_listed_entries_carry_sizes_for_files_only(tree):
    sizes = {e.name: e.size for e in list_entries(str(tree))}
    assert sizes["b.txt"] == len("hello")
    assert sizes["sub"] is None


def test_parent_walks_up_to_the_bucket_list():
    assert parent_url("gs://marin-us-central2/a/b") == "gs://marin-us-central2/a"
    assert parent_url("gs://marin-us-central2/a") == "gs://marin-us-central2"
    assert parent_url("gs://marin-us-central2") == ROOT
    assert parent_url(ROOT) == ROOT


def test_root_lists_the_declared_buckets():
    """The browser's top level spans backends, so both schemes appear."""
    urls = {e.url for e in list_entries(ROOT)}
    assert "gs://marin-us-central2" in urls
    assert "s3://marin-us-east-02a" in urls


def test_total_size_sums_the_whole_tree(tree):
    size, count = total_size(str(tree))
    assert count == 3
    assert size == len("hello") + len(json.dumps({"key": "value"})) + len("nested")


# --- rendering -------------------------------------------------------------


def test_jsonl_renders_as_a_table_with_a_column_per_key():
    raw = b'{"step": 1, "loss": 3.5}\n{"step": 2, "loss": 2.0}\n'
    lines = file_lines("metrics.jsonl", raw)
    assert lines[0].split() == ["step", "loss"]
    assert lines[2].split() == ["1", "3.5"]


def test_malformed_jsonl_falls_back_to_raw_lines():
    """A partially written record must still be readable, not an error."""
    raw = b'{"step": 1}\n{"step": 2, trunc'
    assert file_lines("metrics.jsonl", raw) == ['{"step": 1}', '{"step": 2, trunc']


def test_non_tabular_json_renders_as_indented_json():
    lines = file_lines("nested.json", b'{"a": {"b": [1, 2]}}')
    assert lines[0].split() == ["key", "value"]
    assert lines[2].split() == ["a", '{"b":', "[1,", "2]}"]


def test_binary_content_reports_its_size_instead_of_garbage():
    assert file_lines("model.bin", b"\x00\xff\xfe") == ["[binary file, 3 bytes]"]


def test_size_formatting_switches_units():
    assert format_size(512) == "512 B"
    assert format_size(2048) == "2.0 KB"
    assert format_size(None) == "-"


# --- CLI -------------------------------------------------------------------


def test_cat_renders_a_json_file(tree):
    result = CliRunner().invoke(cli, ["cat", str(tree / "a.json")])
    assert result.exit_code == 0
    assert "value" in result.output


def test_cat_raw_writes_the_bytes_unchanged(tree):
    result = CliRunner().invoke(cli, ["cat", "--raw", str(tree / "b.txt")])
    assert result.exit_code == 0
    assert result.output == "hello"


def test_cp_refuses_a_directory_without_recursive(tree, tmp_path):
    result = CliRunner().invoke(cli, ["cp", str(tree / "sub"), str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "-r" in result.output


def test_cp_recursive_reproduces_the_tree(tree, tmp_path):
    destination = tmp_path / "out"
    result = CliRunner().invoke(cli, ["cp", "-r", str(tree / "sub"), str(destination)])
    assert result.exit_code == 0, result.output
    assert (destination / "c.txt").read_text() == "nested"


def test_cp_recursive_on_a_single_file_keeps_its_name(tree, tmp_path):
    """A -r whose source resolves to one object has nothing to mirror below the source."""
    destination = tmp_path / "out"
    result = CliRunner().invoke(cli, ["cp", "-r", str(tree / "b.txt"), str(destination)])
    assert result.exit_code == 0, result.output
    assert (destination / "b.txt").read_text() == "hello"


def test_cp_to_a_bare_relative_name_writes_a_file(tree, tmp_path, monkeypatch):
    """A destination with no directory component must not be created as a directory."""
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["cp", str(tree / "b.txt"), "copied.txt"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "copied.txt").read_text() == "hello"


def test_cp_reports_a_missing_source_rather_than_copying_nothing(tree, tmp_path):
    """A silent "0 objects" on a typo'd prefix reads as success; it must not."""
    result = CliRunner().invoke(cli, ["cp", "-r", str(tree / "absent"), str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_du_reports_bytes_and_object_count(tree):
    result = CliRunner().invoke(cli, ["du", str(tree)])
    assert result.exit_code == 0
    assert "3 objects" in result.output
