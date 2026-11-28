# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import click
import pytest

from scripts.ops.storage.copy_prefix import (
    CopiedObject,
    CopyShard,
    Progress,
    SourceFile,
    pack_shards,
    progress_line,
    shard_id,
    shard_is_copied,
    storage_prefix,
)


def test_shards_cover_every_file_once_in_key_order_and_close_at_the_byte_target() -> None:
    files = [SourceFile("d", 4, "e4"), SourceFile("a", 6, "e1"), SourceFile("c", 5, "e3"), SourceFile("b", 1, "e2")]

    shards = pack_shards(files, target_bytes=7, max_files=10)

    assert [[f.path for f in shard] for shard in shards] == [["a", "b"], ["c", "d"]]


def test_splitting_a_run_of_tiny_files_keeps_the_later_shards_unchanged() -> None:
    tiny = [SourceFile(f"a-{i}", 1, f"t{i}") for i in range(7)]
    large = [SourceFile(f"b-{i}", 5, f"l{i}") for i in range(4)]

    capped = pack_shards(tiny + large, target_bytes=10, max_files=3)
    uncapped = pack_shards(tiny + large, target_bytes=10, max_files=len(tiny) + len(large))

    assert [[f.path for f in shard] for shard in capped] == [
        ["a-0", "a-1", "a-2"],
        ["a-3", "a-4", "a-5"],
        ["a-6", "b-0"],
        ["b-1", "b-2"],
        ["b-3"],
    ]
    assert capped[-2:] == uncapped[-2:]


def test_a_run_of_tiny_files_splits_at_the_file_cap() -> None:
    files = [SourceFile("big", 100, "e0")] + [SourceFile(f"tiny-{i}", 1, f"e{i}") for i in range(1, 6)]

    shards = pack_shards(files, target_bytes=100, max_files=2)

    assert [[f.path for f in shard] for shard in shards] == [
        ["big"],
        ["tiny-1", "tiny-2"],
        ["tiny-3", "tiny-4"],
        ["tiny-5"],
    ]


def test_a_rewritten_source_file_renames_its_shard_so_a_rerun_recopies_it() -> None:
    shard = (SourceFile("a", 10, "ETag=1"), SourceFile("b", 20, "ETag=2"))
    relisted = (SourceFile("a", 10, "ETag=1"), SourceFile("b", 20, "ETag=2"))
    rewritten = (SourceFile("a", 10, "ETag=1"), SourceFile("b", 20, "ETag=3"))

    assert shard_id(relisted) == shard_id(shard)
    assert shard_id(rewritten) != shard_id(shard)


def test_a_recorded_shard_is_copied_again_when_a_destination_object_no_longer_matches() -> None:
    files = (SourceFile("a", 10, "ETag=src-a"), SourceFile("b", 20, "ETag=src-b"))
    shard = CopyShard(shard_id(files), "s3://src/p", "s3://dst/p", files)
    record = [CopiedObject("a", 10, "ETag=src-a", "etag=dst-a"), CopiedObject("b", 20, "ETag=src-b", "etag=dst-b")]
    destination = {"a": {"size": 10, "ETag": '"dst-a"'}, "b": {"size": 20, "ETag": '"dst-b"'}}
    overwritten = {**destination, "b": {"size": 20, "ETag": '"other"'}}
    expired = {"a": destination["a"]}

    assert shard_is_copied(shard, record, destination)
    assert not shard_is_copied(shard, record, overwritten)
    assert not shard_is_copied(shard, record, expired)
    assert not shard_is_copied(shard, None, destination)


def test_progress_line_reports_share_done_and_time_left() -> None:
    progress = Progress(
        done_bytes=20 * 2**40,
        total_bytes=80 * 2**40,
        done_files=246512,
        total_files=985945,
        done_shards=500,
        total_shards=2000,
    )

    line = progress_line(progress, rate=2**30)

    assert line == ("progress: 20.0 TB / 80.0 TB (25.0%), 246512/985945 files, 500/2000 shards, 1.0 GB/s, ETA 17h04m")


def test_progress_line_for_a_source_of_only_empty_files_counts_files_and_shards() -> None:
    progress = Progress(done_bytes=0, total_bytes=0, done_files=3, total_files=10, done_shards=1, total_shards=4)

    assert progress_line(progress, rate=0) == "progress: 0 B / 0 B, 3/10 files, 1/4 shards"


def test_a_prefix_with_an_empty_path_segment_is_rejected_and_a_trailing_slash_is_dropped() -> None:
    with pytest.raises(click.BadParameter, match="s3://marin-us-east-02a/a//b"):
        storage_prefix("s3://marin-us-east-02a/a//b")

    assert storage_prefix("s3://marin-us-east-02a/a/b/") == "s3://marin-us-east-02a/a/b"
