# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.datakit import partition_filename
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams
from marin.processing.classification.deduplication.repack_fuzzy_dups import repack_fuzzy_dups_source
from zephyr.execution import ZephyrContext
from zephyr.shard_keys import deterministic_hash
from zephyr.writers import write_parquet_file

TEST_MINHASH_PARAMS = MinHashParams(num_perms=8, num_bands=4, ngram_size=5, seed=0)


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    client = LocalClient()
    try:
        with set_current_client(client):
            yield
    finally:
        client.shutdown()


def _id_for_shard(shard: int, total: int) -> str:
    index = 0
    while True:
        doc_id = f"document-{shard}-{index}"
        if deterministic_hash(doc_id) % total == shard:
            return doc_id
        index += 1


def test_repack_routes_candidates_to_new_normalized_shards(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    shard_count = 3
    normalized_dir = tmp_path / "normalized" / "outputs" / "main"
    normalized_dir.mkdir(parents=True)
    for shard in range(shard_count):
        write_parquet_file([], str(normalized_dir / partition_filename(shard, shard_count)))

    normalized = NormalizedData(
        main_output_dir=str(normalized_dir),
        dup_output_dir=str(tmp_path / "normalized" / "outputs" / "dups"),
        counters={},
    )
    legacy_source_key = "normalized/legacy-focus/outputs/main"
    other_source_key = "normalized/other/outputs/main"
    candidate_dir = tmp_path / "candidates" / "outputs" / "source_000"
    candidate_dir.mkdir(parents=True)
    rows = [
        {"id": _id_for_shard(0, shard_count), "dup_cluster_id": "cluster-a", "is_cluster_canonical": True},
        {"id": _id_for_shard(2, shard_count), "dup_cluster_id": "cluster-a", "is_cluster_canonical": False},
    ]
    write_parquet_file(rows, str(candidate_dir / "part-00000.parquet"))
    write_parquet_file([rows[0]], str(candidate_dir / "part-00001.parquet"))
    other_attr_dir = tmp_path / "candidates" / "outputs" / "source_001"
    candidates = FuzzyDupsAttrData(
        params=TEST_MINHASH_PARAMS,
        sources={
            legacy_source_key: FuzzyDupsPerSource(attr_dir=str(candidate_dir)),
            other_source_key: FuzzyDupsPerSource(attr_dir=str(other_attr_dir)),
        },
        counters={"dedup/fuzzy/document/cluster_members": 2},
    )

    output_path = str(tmp_path / "repacked")
    result = repack_fuzzy_dups_source(
        candidates=candidates,
        legacy_source_key=legacy_source_key,
        normalized=normalized,
        output_path=output_path,
        zephyr_context=ZephyrContext(name="test-repack", max_workers=3),
    )

    new_source_key = datakit_source_key(normalized.main_output_dir)
    assert legacy_source_key not in result.sources
    assert result.sources[other_source_key].attr_dir == str(other_attr_dir)
    assert result.counters["dedup/fuzzy/document/cluster_members"] == 2

    repacked_dir = Path(result.sources[new_source_key].attr_dir)
    output_files = sorted(repacked_dir.glob("*.parquet"))
    assert [path.name for path in output_files] == [
        partition_filename(shard, shard_count) for shard in range(shard_count)
    ]
    assert pq.read_table(output_files[0]).to_pylist() == [rows[0]]
    assert pq.read_table(output_files[1]).to_pylist() == []
    assert pq.read_table(output_files[2]).to_pylist() == [rows[1]]
    assert (Path(output_path) / ".source_manifest.json").is_file()
