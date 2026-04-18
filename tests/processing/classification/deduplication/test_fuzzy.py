# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fray.v1.job import create_job_ctx, fray_default_job_ctx

from marin.datakit.normalize import NormalizedData, normalize_to_parquet
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


def _read_dup_attr_ids(attr_dir: str) -> set[str]:
    """Return the set of generated ``id``s flagged as ``dup_doc=True`` under *attr_dir*."""
    ids: set[str] = set()
    for pf in sorted(Path(attr_dir).glob("*.parquet")):
        for record in pq.read_table(str(pf)).to_pylist():
            assert record["attributes"]["dup_doc"] is True
            ids.add(record["id"])
    return ids


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


def test_fuzzy_dups_single_source_finds_fuzzy_pair(fox_corpus):
    """Within the test corpus, ``contaminated_1`` and ``high_overlap`` are LSH-similar."""
    norm_dir = os.path.join(fox_corpus["output_dir"], "normalized")
    minhash_dir = os.path.join(fox_corpus["output_dir"], "minhash")
    dups_dir = os.path.join(fox_corpus["output_dir"], "fuzzy_dups")

    source = _normalize(fox_corpus["test_dir"], norm_dir)
    minhash = compute_minhash_attrs(source=source, output_path=minhash_dir)
    dups = compute_fuzzy_dups_attrs(inputs=[minhash], output_path=dups_dir, max_parallelism=4)

    assert dups.params == minhash.params
    assert source.main_output_dir in dups.sources
    per_source = dups.sources[source.main_output_dir]

    by_id = _read_main_records(source)
    flagged_ids = _read_dup_attr_ids(per_source.attr_dir)
    flagged_source_ids = {by_id[i]["source_id"] for i in flagged_ids if i in by_id}

    # The fuzzy pair: test_contaminated_1 ~ test_high_overlap (one-word diff).
    # Exactly one of them should be marked as the duplicate; the other is canonical.
    arctic_pair = {"test_contaminated_1", "test_high_overlap"}
    assert (
        len(arctic_pair & flagged_source_ids) == 1
    ), f"expected exactly one of {arctic_pair} to be flagged; got {flagged_source_ids}"

    # Unique docs should never be flagged.
    assert "test_unique_1" not in flagged_source_ids
    assert "test_unique_2" not in flagged_source_ids


def test_fuzzy_dups_multi_source_per_source_attr_trees(fox_corpus):
    """Two MinHashAttrData inputs produce two co-partitioned attr trees keyed by source_main_dir."""
    train_norm = _normalize(fox_corpus["train_dir"], os.path.join(fox_corpus["output_dir"], "norm_train"))
    test_norm = _normalize(fox_corpus["test_dir"], os.path.join(fox_corpus["output_dir"], "norm_test"))

    train_mh = compute_minhash_attrs(source=train_norm, output_path=os.path.join(fox_corpus["output_dir"], "mh_train"))
    test_mh = compute_minhash_attrs(source=test_norm, output_path=os.path.join(fox_corpus["output_dir"], "mh_test"))

    dups = compute_fuzzy_dups_attrs(
        inputs=[train_mh, test_mh],
        output_path=os.path.join(fox_corpus["output_dir"], "fuzzy_dups"),
        max_parallelism=4,
    )

    # One per-source entry per input; both attr_dirs exist on disk.
    assert set(dups.sources.keys()) == {train_norm.main_output_dir, test_norm.main_output_dir}
    for src_dir, per_source in dups.sources.items():
        assert per_source.attr_dir.endswith("/outputs/source_000") or per_source.attr_dir.endswith(
            "/outputs/source_001"
        ), per_source.attr_dir
        assert Path(per_source.attr_dir).exists(), f"missing attr dir for {src_dir}"

    # At least one document is marked across both source trees: test_high_overlap
    # (in test) is fuzzy-similar to train_arctic_1 (in train), so the cross-dataset
    # cluster forces one of them to be flagged in its respective attr tree.
    train_flagged = _read_dup_attr_ids(dups.sources[train_norm.main_output_dir].attr_dir)
    test_flagged = _read_dup_attr_ids(dups.sources[test_norm.main_output_dir].attr_dir)
    assert (len(train_flagged) + len(test_flagged)) >= 1


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
