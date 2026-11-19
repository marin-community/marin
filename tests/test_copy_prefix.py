# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.ops.storage.copy_prefix import pack_shards, pending_files


def test_rerun_copies_only_missing_and_resized_files() -> None:
    source = {"a/0.parquet": 10, "a/1.parquet": 20, "b/0.parquet": 30}
    destination = {"a/0.parquet": 10, "a/1.parquet": 7, "stale.parquet": 5}

    assert pending_files(source, destination) == {"a/1.parquet": 20, "b/0.parquet": 30}


def test_shards_cover_every_file_once_and_close_at_the_byte_target() -> None:
    files = {"d.parquet": 4, "a.parquet": 6, "c.parquet": 5, "b.parquet": 1}

    assert pack_shards(files, target_bytes=7) == [("a.parquet", "b.parquet"), ("c.parquet", "d.parquet")]
