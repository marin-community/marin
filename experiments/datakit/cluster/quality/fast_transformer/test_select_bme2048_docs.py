# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the bme2048 document selection."""

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.datakit.cluster.quality.fast_transformer.select_bme2048_docs import (
    scaleup_types,
    select_docs,
)


def _pool(ids: list[str], types: list[str], pools: list[str] | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ids,
            "content_type": types,
            "source": ["src"] * len(ids),
            "shard": ["s3://shard.parquet"] * len(ids),
            "pool": pools or ["legacy88k"] * len(ids),
        }
    )


def test_holdout_documents_are_never_selected():
    ids = [f"doc{i}" for i in range(50)]
    selected, _ = select_docs(_pool(ids, ["prose"] * 50), holdout={"doc3", "doc7"}, docs_per_type=50, seed=0)
    assert selected.height == 48
    assert not {"doc3", "doc7"} & set(selected.get_column("id").to_list())


def test_a_document_in_both_pools_is_selected_once_from_the_first():
    pools = _pool(["dup", "other", "dup"], ["prose"] * 3, ["legacy88k", "scaleup", "scaleup"])
    selected, _ = select_docs(pools, holdout=set(), docs_per_type=10, seed=0)
    assert selected.get_column("id").to_list().count("dup") == 1
    assert selected.filter(pl.col("id") == "dup").get_column("pool").to_list() == ["legacy88k"]


def test_the_draw_is_a_hash_order_so_row_order_does_not_change_it():
    ids = [f"doc{i}" for i in range(200)]
    forward, _ = select_docs(_pool(ids, ["code"] * 200), holdout=set(), docs_per_type=20, seed=0)
    reverse, _ = select_docs(_pool(ids[::-1], ["code"] * 200), holdout=set(), docs_per_type=20, seed=0)
    assert forward.get_column("id").to_list() == reverse.get_column("id").to_list()


def test_a_larger_quota_extends_the_same_draw():
    """Hash order, not a reshuffle: growing the target adds documents rather than
    replacing the ones a smaller run already graded."""
    ids = [f"doc{i}" for i in range(200)]
    small, _ = select_docs(_pool(ids, ["math"] * 200), holdout=set(), docs_per_type=20, seed=0)
    large, _ = select_docs(_pool(ids, ["math"] * 200), holdout=set(), docs_per_type=60, seed=0)
    assert set(small.get_column("id").to_list()) <= set(large.get_column("id").to_list())


def test_a_short_type_reports_a_shortfall_instead_of_borrowing_from_another():
    pools = _pool(["a", "b", "c", "d"], ["prose", "prose", "prose", "agentic"])
    selected, shortfalls = select_docs(pools, holdout=set(), docs_per_type=3, seed=0)
    by_type = dict(zip(*[selected.get_column(c).to_list() for c in ("id", "content_type")], strict=True))
    assert sum(1 for t in by_type.values() if t == "agentic") == 1
    agentic = next(s for s in shortfalls if s.content_type == "agentic")
    assert (agentic.wanted, agentic.available) == (3, 1)
    assert not any(s.content_type == "prose" for s in shortfalls)


def test_scaleup_type_prefers_the_begin_grade_over_the_window_majority(tmp_path):
    path = tmp_path / "windows.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["long", "long", "long", "middle_only", "middle_only"],
                "window": ["begin", "middle", "end", "middle", "end"],
                "content_type": ["prose", "code", "code", "math", "math"],
            }
        ),
        path,
    )
    assert scaleup_types(str(path)) == {"long": "prose", "middle_only": "math"}
