# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test: noop dedup followed by consolidate keeps every doc."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marin.datakit.normalize import NormalizedData
from marin.processing.classification.consolidate import (
    FilterConfig,
    FilterType,
    consolidate,
)

from experiments.datakit_testbed.noop_dedup import compute_noop_dedup_attrs


def _write_doc_shard(path: Path, ids: list[str], texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"id": ids, "text": texts, "source_id": ids})
    pq.write_table(table, path)


def _make_normalized(tmp_path: Path, name: str, shard_specs: dict[str, list[str]]) -> NormalizedData:
    main_dir = tmp_path / name / "outputs" / "main"
    dup_dir = tmp_path / name / "outputs" / "dup"
    dup_dir.mkdir(parents=True, exist_ok=True)
    for shard_rel, ids in shard_specs.items():
        _write_doc_shard(main_dir / shard_rel, ids, [f"text of {i}" for i in ids])
    return NormalizedData(
        main_output_dir=str(main_dir),
        dup_output_dir=str(dup_dir),
        counters={},
    )


def test_noop_dedup_writes_empty_attr_per_shard(tmp_path: Path):
    source_a = _make_normalized(
        tmp_path,
        "A",
        {"part-0000.parquet": ["a0", "a1"], "sub/part-0001.parquet": ["a2"]},
    )
    source_b = _make_normalized(tmp_path, "B", {"part-0000.parquet": ["b0"]})

    attrs = compute_noop_dedup_attrs(
        inputs=[source_a, source_b],
        output_path=str(tmp_path / "noop-out"),
    )

    assert set(attrs.sources.keys()) == {source_a.main_output_dir, source_b.main_output_dir}
    assert attrs.counters["noop_dedup/empty_shards"] == 3

    # Every source's attr_dir mirrors its normalized shard tree with 0-row parquets.
    for src in (source_a, source_b):
        per_source = attrs.sources[src.main_output_dir]
        input_shards = sorted(Path(src.main_output_dir).rglob("*.parquet"))
        for shard in input_shards:
            rel = shard.relative_to(src.main_output_dir)
            attr_shard = Path(per_source.attr_dir) / rel
            assert attr_shard.exists(), f"missing attr shard {attr_shard}"
            table = pq.read_table(attr_shard)
            assert table.num_rows == 0
            assert set(table.schema.names) == {"id", "attributes"}


def test_noop_dedup_then_consolidate_keeps_every_doc(tmp_path: Path):
    """Full pipeline: noop_dedup → consolidate(keep_if_missing=True) preserves all rows."""
    source = _make_normalized(
        tmp_path,
        "only",
        {
            "part-0000.parquet": ["x0", "x1", "x2"],
            "sub/part-0001.parquet": ["y0", "y1"],
        },
    )

    attrs = compute_noop_dedup_attrs(inputs=[source], output_path=str(tmp_path / "noop-out"))
    per_source = attrs.sources[source.main_output_dir]

    consolidated_dir = tmp_path / "consolidated"
    consolidate(
        input_path=source.main_output_dir,
        output_path=str(consolidated_dir),
        filters=[
            FilterConfig(
                type=FilterType.KEEP_DOC,
                attribute_path=per_source.attr_dir,
                name="is_cluster_canonical",
                attribute_filetype="parquet",
                keep_if_missing=True,
            )
        ],
        filetype="parquet",
        max_workers=1,
    )

    out_tables = [pq.read_table(p) for p in sorted(consolidated_dir.rglob("*.parquet"))]
    assert out_tables, "consolidate produced no output files"
    out_ids = {r for t in out_tables for r in t.column("id").to_pylist()}
    assert out_ids == {"x0", "x1", "x2", "y0", "y1"}


def test_noop_dedup_rejects_empty_inputs():
    with pytest.raises(ValueError, match="at least one input"):
        compute_noop_dedup_attrs(inputs=[], output_path="/tmp/unused")


def test_noop_dedup_rejects_source_with_no_shards(tmp_path: Path):
    empty_main = tmp_path / "empty" / "main"
    empty_main.mkdir(parents=True)
    bad = NormalizedData(main_output_dir=str(empty_main), dup_output_dir=str(tmp_path / "x"), counters={})
    with pytest.raises(FileNotFoundError, match="No parquet shards"):
        compute_noop_dedup_attrs(inputs=[bad], output_path=str(tmp_path / "out"))
