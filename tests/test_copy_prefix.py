# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.ops.storage.copy_prefix import SourceFile, pack_shards, shard_id


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
