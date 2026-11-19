# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.ops.storage.copy_prefix import CopiedObject, CopyShard, SourceFile, pack_shards, shard_id, shard_is_copied


def test_shards_cover_every_file_once_in_key_order_and_close_at_the_byte_target() -> None:
    files = [SourceFile("d", 4, "e4"), SourceFile("a", 6, "e1"), SourceFile("c", 5, "e3"), SourceFile("b", 1, "e2")]

    shards = pack_shards(files, target_bytes=7)

    assert [[f.path for f in shard] for shard in shards] == [["a", "b"], ["c", "d"]]


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
