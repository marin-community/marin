# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fray import ResourceConfig
from fray.local_backend import LocalClient
from zephyr import Dataset, ZephyrContext

import scripts.analysis.nemotron_math_val_full_scan as scan
from scripts.analysis.nemotron_math_val_full_scan import (
    JOIN_SOURCE_CORPUS,
    JOIN_SOURCE_PAIR,
    PAIR_SCHEMA,
    _first_pair,
    _join_record_key,
    _pair_key,
    _verify_join_group,
    _verify_join_records,
)


def _read_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(pq.read_table(path, columns=["val_id", "other_id"]).to_pylist())
    return rows


def test_pair_dedup_reducer_writes_parquet_records(tmp_path: Path) -> None:
    client = LocalClient()
    ctx = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="test-nemotron-pair-dedup",
    )
    try:
        output_pattern = str(tmp_path / "pairs" / "part-{shard:05d}-of-{total:05d}.parquet")
        outcome = ctx.execute(
            Dataset.from_list(
                [
                    {"val_id": "v1", "other_id": "o1"},
                    {"val_id": "v1", "other_id": "o1"},
                    {"val_id": "v1", "other_id": "o2"},
                ]
            )
            .group_by(_pair_key, reducer=_first_pair)
            .write_parquet(output_pattern, schema=PAIR_SCHEMA)
        )
    finally:
        ctx.shutdown()
        client.shutdown(wait=True)

    rows = sorted(_read_rows(outcome.results), key=lambda r: (r["val_id"], r["other_id"]))
    assert rows == [{"val_id": "v1", "other_id": "o1"}, {"val_id": "v1", "other_id": "o2"}]


def test_key_join_verifier_matches_pairs_to_corpus_text(tmp_path: Path) -> None:
    val_docs = tmp_path / "val_docs"
    val_docs.mkdir()
    pq.write_table(pa.table({"id": ["v1"], "text": ["abcdef"]}), val_docs / "val.parquet")

    pair_path = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({"val_id": ["v1", "v1"], "other_id": ["o1", "o2"]}), pair_path)

    corpus_path = tmp_path / "corpus.parquet"
    pq.write_table(pa.table({"id": ["o1", "o2"], "text": ["abcdef", "zzzzzz"]}), corpus_path)

    scan._VAL_TEXT = None
    records = [
        *_verify_join_records({"kind": JOIN_SOURCE_PAIR, "path": str(pair_path)}),
        *_verify_join_records({"kind": JOIN_SOURCE_CORPUS, "path": str(corpus_path)}),
    ]
    by_key: dict[str, list[dict]] = {}
    for record in records:
        by_key.setdefault(_join_record_key(record), []).append(record)

    verified = []
    for key, items in by_key.items():
        verified.extend(_verify_join_group(key, iter(items), str(val_docs)))

    assert verified == [{"val_id": "v1", "other_id": "o1", "jaccard": 1.0}]
