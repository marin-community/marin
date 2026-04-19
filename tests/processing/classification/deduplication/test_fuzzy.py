# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fray.v1.job import create_job_ctx, fray_default_job_ctx

from marin.datakit.normalize import NormalizedData, generate_id, normalize_to_parquet
from marin.processing.classification.deduplication.fuzzy_dups import compute_fuzzy_dups_attrs
from marin.processing.classification.deduplication.fuzzy_minhash import (
    compute_minhash_attrs,
)


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with fray_default_job_ctx(create_job_ctx("sync")):
        yield


def _normalize(input_dir: str, output_dir: str) -> NormalizedData:
    """Normalize a fox-corpus shard directory into a NormalizedData dataset."""
    return normalize_to_parquet(input_path=input_dir, output_path=output_dir)


def _read_main_records(source: NormalizedData) -> dict[str, dict]:
    """Return a mapping from generated ``id`` to the full main-output record."""
    out: dict[str, dict] = {}
    for pf in sorted(Path(source.main_output_dir).glob("*.parquet")):
        for record in pq.read_table(str(pf)).to_pylist():
            out[record["id"]] = record
    return out


def _read_cluster_attrs(attr_dir: str) -> list[dict]:
    """Return every cluster-member row (flat list) under *attr_dir*."""
    rows: list[dict] = []
    for pf in sorted(Path(attr_dir).glob("*.parquet")):
        rows.extend(pq.read_table(str(pf)).to_pylist())
    return rows


def test_minhash_attrs_co_partitioned_with_source(fox_corpus):
    """Each source shard produces a same-named MinHash attr parquet with {id, buckets}."""
    norm_dir = os.path.join(fox_corpus["output_dir"], "normalized")
    minhash_dir = os.path.join(fox_corpus["output_dir"], "minhash")

    source = _normalize(fox_corpus["test_dir"], norm_dir)
    minhash = compute_minhash_attrs(source=source, output_path=minhash_dir)

    assert minhash.source_main_dir == source.main_output_dir
    assert minhash.params.num_perms == 286
    assert minhash.params.num_bands == 26

    source_basenames = {p.name for p in Path(source.main_output_dir).glob("*.parquet")}
    attr_basenames = {p.name for p in Path(minhash.attr_dir).glob("*.parquet")}
    assert source_basenames == attr_basenames
    assert source_basenames  # non-empty

    # At least one non-empty shard exists with the expected {id, buckets} schema.
    # Empty source shards produce empty attr parquets with no schema, which we skip.
    seen_non_empty = False
    for pf in Path(minhash.attr_dir).glob("*.parquet"):
        rows = pq.read_table(str(pf)).to_pylist()
        if not rows:
            continue
        seen_non_empty = True
        rec = rows[0]
        assert isinstance(rec["id"], str)
        assert isinstance(rec["buckets"], list)
        assert all(isinstance(b, str) for b in rec["buckets"])
    assert seen_non_empty, "expected at least one non-empty MinHash attr shard"
    assert minhash.counters["minhash/documents"] >= 1


def test_fuzzy_dups_single_source_schema_and_pair(fox_corpus):
    """Attr rows cover every cluster member; singletons get no row.

    Uses the test corpus's fuzzy pair (``test_contaminated_1`` ~
    ``test_high_overlap``, one-word diff) to exercise a real cluster of 2
    and verifies:
    * both members get an attr row,
    * rows carry ``dup_cluster_id`` and ``is_cluster_canonical``,
    * all rows for one cluster share the same ``dup_cluster_id``,
    * exactly one member per cluster is canonical,
    * unique docs have no attr row.
    """
    norm_dir = os.path.join(fox_corpus["output_dir"], "normalized")
    minhash_dir = os.path.join(fox_corpus["output_dir"], "minhash")
    dups_dir = os.path.join(fox_corpus["output_dir"], "fuzzy_dups")

    source = _normalize(fox_corpus["test_dir"], norm_dir)
    minhash = compute_minhash_attrs(source=source, output_path=minhash_dir)
    dups = compute_fuzzy_dups_attrs(inputs=[minhash], output_path=dups_dir, max_parallelism=4)

    assert dups.params == minhash.params
    per_source = dups.sources[source.main_output_dir]

    by_id = _read_main_records(source)
    rows = _read_cluster_attrs(per_source.attr_dir)
    by_source_id = {by_id[r["id"]]["source_id"]: r for r in rows if r["id"] in by_id}

    # The fuzzy pair is a cluster of 2: both members must have an attr row,
    # sharing one dup_cluster_id, with exactly one canonical.
    pair = {"test_contaminated_1", "test_high_overlap"}
    assert pair <= by_source_id.keys(), f"missing attr rows for pair: {pair - by_source_id.keys()}"
    cluster_ids = {by_source_id[s]["attributes"]["dup_cluster_id"] for s in pair}
    assert len(cluster_ids) == 1, f"pair should share a dup_cluster_id; got {cluster_ids}"
    canonicals = [s for s in pair if by_source_id[s]["attributes"]["is_cluster_canonical"]]
    assert len(canonicals) == 1, f"exactly one canonical expected; got {canonicals}"

    # Unique docs never have attr rows (no cluster → no annotation).
    assert "test_unique_1" not in by_source_id
    assert "test_unique_2" not in by_source_id


def test_fuzzy_dups_multi_source_per_source_attr_trees(fox_corpus):
    """Two MinHashAttrData inputs produce two per-source attr trees.

    Cross-source exact-text duplicates (e.g. ``train_arctic_1`` ==
    ``test_contaminated_1`` byte-identical → same normalized id in both
    datasets) must be detected as a 2-member cluster rather than collapsing
    into a single node. Each side independently carries its own attr row for
    the shared content hash, with a shared ``dup_cluster_id`` and exactly one
    canonical across the pair.
    """
    train_norm = _normalize(fox_corpus["train_dir"], os.path.join(fox_corpus["output_dir"], "norm_train"))
    test_norm = _normalize(fox_corpus["test_dir"], os.path.join(fox_corpus["output_dir"], "norm_test"))

    train_mh = compute_minhash_attrs(source=train_norm, output_path=os.path.join(fox_corpus["output_dir"], "mh_train"))
    test_mh = compute_minhash_attrs(source=test_norm, output_path=os.path.join(fox_corpus["output_dir"], "mh_test"))

    dups = compute_fuzzy_dups_attrs(
        inputs=[train_mh, test_mh],
        output_path=os.path.join(fox_corpus["output_dir"], "fuzzy_dups"),
        max_parallelism=4,
    )

    assert set(dups.sources.keys()) == {train_norm.main_output_dir, test_norm.main_output_dir}
    for per_source in dups.sources.values():
        assert per_source.attr_dir.rsplit("/", 1)[-1].startswith("source_"), per_source.attr_dir
        assert Path(per_source.attr_dir).exists()

    def rows_by_id(main_dir: str) -> dict[str, dict]:
        return {r["id"]: r for r in _read_cluster_attrs(dups.sources[main_dir].attr_dir)}

    train_rows = rows_by_id(train_norm.main_output_dir)
    test_rows = rows_by_id(test_norm.main_output_dir)

    # Each cross-source byte-identical text must appear as an attr row on both
    # sides (keyed by the same content hash), share a dup_cluster_id, and have
    # exactly one canonical across the pair.
    for shared_text in (
        "Arctic predators have superior auditory capabilities for hunting beneath snow.",
        "Red canids inhabit northern territories worldwide.",
    ):
        content_id = generate_id(shared_text)
        assert content_id in train_rows, f"missing train attr row for {shared_text!r}"
        assert content_id in test_rows, f"missing test attr row for {shared_text!r}"
        a, b = train_rows[content_id]["attributes"], test_rows[content_id]["attributes"]
        assert a["dup_cluster_id"] == b["dup_cluster_id"], f"{shared_text!r}: dup_cluster_id mismatch"
        assert (
            a["is_cluster_canonical"] != b["is_cluster_canonical"]
        ), f"{shared_text!r}: exactly one canonical expected across pair"


def test_fuzzy_dups_rejects_param_mismatch(fox_corpus):
    """Inputs with mismatched MinHash params must be rejected up front."""
    source = _normalize(fox_corpus["test_dir"], os.path.join(fox_corpus["output_dir"], "norm"))
    a = compute_minhash_attrs(source=source, output_path=os.path.join(fox_corpus["output_dir"], "mh_a"))
    # Same num_perms, different num_bands → still divisible, but params differ.
    b = compute_minhash_attrs(
        source=source,
        output_path=os.path.join(fox_corpus["output_dir"], "mh_b"),
        num_bands=22,  # 286 % 22 == 0
    )

    with pytest.raises(ValueError, match=r"identical MinHash params"):
        compute_fuzzy_dups_attrs(
            inputs=[a, b],
            output_path=os.path.join(fox_corpus["output_dir"], "fuzzy_dups"),
            max_parallelism=4,
        )


def test_fuzzy_dups_rejects_duplicate_source(fox_corpus):
    """Two inputs pointing to the same ``source_main_dir`` must be rejected to avoid output clobbering."""
    source = _normalize(fox_corpus["test_dir"], os.path.join(fox_corpus["output_dir"], "norm"))
    mh = compute_minhash_attrs(source=source, output_path=os.path.join(fox_corpus["output_dir"], "mh"))

    with pytest.raises(ValueError, match=r"Duplicate source_main_dir"):
        compute_fuzzy_dups_attrs(
            inputs=[mh, mh],
            output_path=os.path.join(fox_corpus["output_dir"], "fuzzy_dups"),
            max_parallelism=4,
        )
