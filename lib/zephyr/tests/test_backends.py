# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for backend implementations."""

import gzip
import io
import json

from zephyr.dataset import format_shard_path
from zephyr.writers import write_jsonl_file


def test_format_shard_path_basic():
    """Test basic path formatting with placeholders."""
    pattern = "output/data-{shard:05d}-of-{total:05d}.jsonl"
    result = format_shard_path(pattern, 0, 10)
    assert result == "output/data-00000-of-00010.jsonl"


def test_format_shard_path_normalizes_double_slashes():
    """Test that double slashes are normalized in paths."""
    # Simulate what happens when output_path has trailing slash
    pattern = "gs://bucket/path//data-{shard:05d}-of-{total:05d}.jsonl"
    result = format_shard_path(pattern, 0, 10)
    assert result == "gs://bucket/path/data-00000-of-00010.jsonl"
    assert "//" not in result.replace("://", "")  # No double slashes except in protocol


def test_format_shard_path_preserves_protocol():
    """Test that protocol prefixes (gs://, s3://, etc.) are preserved."""
    protocols = ["gs://", "s3://", "http://", "https://"]

    for protocol in protocols:
        pattern = f"{protocol}bucket/path//data-{{shard:05d}}.jsonl"
        result = format_shard_path(pattern, 5, 100)

        # Protocol should be preserved
        assert result.startswith(protocol)

        # No double slashes in the path part
        path_part = result[len(protocol) :]
        assert "//" not in path_part


def test_format_shard_path_multiple_consecutive_slashes():
    """Test normalization of multiple consecutive slashes."""
    pattern = "gs://bucket///path////data-{shard:05d}.jsonl"
    result = format_shard_path(pattern, 0, 1)
    assert result == "gs://bucket/path/data-00000.jsonl"


def test_format_shard_path_local_paths():
    """Test that local paths also get normalized."""
    pattern = "/tmp//output//data-{shard:05d}.jsonl"
    result = format_shard_path(pattern, 0, 1)
    assert result == "/tmp/output/data-00000.jsonl"


def test_format_shard_path_basename_placeholder():
    """Test that {basename} placeholder works."""
    pattern = "output/{basename}-{shard:05d}.jsonl"
    result = format_shard_path(pattern, 3, 10)
    assert result == "output/shard_3-00003.jsonl"


def test_write_jsonl_infers_compression_from_gz_extension(tmp_path):
    """Test that .gz extension triggers gzip compression."""
    records = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    output_path = str(tmp_path / "test.jsonl.gz")

    result = write_jsonl_file(records, output_path)
    assert result["path"] == output_path
    assert result["count"] == 2

    # Verify file was created and is gzip compressed
    with gzip.open(output_path, "rt") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1, "text": "hello"}
        assert json.loads(lines[1]) == {"id": 2, "text": "world"}


def test_write_jsonl_no_compression_without_gz_extension(tmp_path):
    """Test that files without .gz extension are not compressed."""
    records = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    output_path = str(tmp_path / "test.jsonl")

    result = write_jsonl_file(records, output_path)
    assert result["path"] == output_path
    assert result["count"] == 2

    # Verify file was created and is NOT compressed
    with open(output_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1, "text": "hello"}
        assert json.loads(lines[1]) == {"id": 2, "text": "world"}


def test_write_jsonl_infers_compression_from_zst_extension(tmp_path):
    """Test that .zst extension triggers zstd compression."""
    records = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    output_path = str(tmp_path / "test.jsonl.zst")

    result = write_jsonl_file(records, output_path)
    assert result["path"] == output_path
    assert result["count"] == 2

    # Verify file was created and is zstd compressed
    import zstandard as zstd

    dctx = zstd.ZstdDecompressor()
    with open(output_path, "rb") as raw_f:
        with dctx.stream_reader(raw_f) as reader:
            text_f = io.TextIOWrapper(reader, encoding="utf-8")
            lines = text_f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0]) == {"id": 1, "text": "hello"}
            assert json.loads(lines[1]) == {"id": 2, "text": "world"}
