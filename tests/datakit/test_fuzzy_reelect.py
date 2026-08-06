# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the quality-based fuzzy-canonical re-election.

Hand-built attr and corpus fixtures (following tests/processing/classification/deduplication/
test_fuzzy.py's hand-built datasets) exercise the election policy: highest ``edu_max`` wins,
ties fall to the smallest id, unscored members lose to scored ones, singletons stay untouched,
and the rewritten tree keeps the co-partitioning invariants ``consolidate`` joins on.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import write_artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData, FuzzyDupsPerSource
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashParams

from experiments.build_pdf_source.fuzzy_ocr_all import (
    consolidate_fuzzy_clean,
    reelect_cluster_canonicals,
    reelect_fuzzy_canonicals,
)

_PARAMS = MinHashParams(num_perms=286, num_bands=26, ngram_size=5, seed=42)

_CORPUS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("edu_max", pa.float32()),
    ]
)

_SHARDS = (
    "part-00000-of-00003.parquet",
    "part-00001-of-00003.parquet",
    "part-00002-of-00003.parquet",
)

# Three clusters spread across two shards, plus singletons. Old canonicals are the ones the
# election must overturn: the low-scored, the larger-id tie member, and the unscored member.
_ATTR_ROWS: dict[str, list[dict]] = {
    _SHARDS[0]: [
        {"id": "a-low", "dup_cluster_id": "cl-alpha", "is_cluster_canonical": True},
        {"id": "s-scored", "dup_cluster_id": "cl-missing", "is_cluster_canonical": False},
        {"id": "t-b", "dup_cluster_id": "cl-tie", "is_cluster_canonical": True},
    ],
    _SHARDS[1]: [
        {"id": "a-high", "dup_cluster_id": "cl-alpha", "is_cluster_canonical": False},
        {"id": "t-a", "dup_cluster_id": "cl-tie", "is_cluster_canonical": False},
        {"id": "u-none", "dup_cluster_id": "cl-missing", "is_cluster_canonical": True},
    ],
    _SHARDS[2]: [],
}

# "u-none" carries a null score (unscored member); "solo" and "z-solo2" are singletons with no
# attr row anywhere.
_CORPUS_ROWS: dict[str, list[dict]] = {
    _SHARDS[0]: [
        {"id": "a-low", "text": "alpha low", "edu_max": 1.0},
        {"id": "s-scored", "text": "missing scored", "edu_max": 0.5},
        {"id": "solo", "text": "singleton", "edu_max": 4.0},
        {"id": "t-b", "text": "tie b", "edu_max": 2.5},
    ],
    _SHARDS[1]: [
        {"id": "a-high", "text": "alpha high", "edu_max": 3.0},
        {"id": "t-a", "text": "tie a", "edu_max": 2.5},
        {"id": "u-none", "text": "missing unscored", "edu_max": None},
    ],
    _SHARDS[2]: [
        {"id": "z-solo2", "text": "singleton two", "edu_max": 2.0},
    ],
}

_EXPECTED_CANONICALS = {"a-high", "t-a", "s-scored"}


def _build_fixtures(tmp_path: Path) -> tuple[FuzzyDupsAttrData, NormalizedData]:
    main_dir = tmp_path / "quality" / "outputs" / "main"
    main_dir.mkdir(parents=True)
    for basename, rows in _CORPUS_ROWS.items():
        pq.write_table(pa.Table.from_pylist(rows, schema=_CORPUS_SCHEMA), main_dir / basename)
    quality = NormalizedData(
        main_output_dir=str(main_dir),
        dup_output_dir=str(tmp_path / "quality" / "outputs" / "dups"),
        counters={},
    )

    attr_dir = tmp_path / "fuzzy_dups" / "outputs" / "source_000"
    attr_dir.mkdir(parents=True)
    for basename, rows in _ATTR_ROWS.items():
        # The library writes member-less shards schema-less (write_parquet_file with no rows
        # and no schema); mirror that so the empty-shard path is the production one.
        table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([], schema=pa.schema([]))
        pq.write_table(table, attr_dir / basename)
    fuzzy = FuzzyDupsAttrData(
        params=_PARAMS,
        sources={datakit_source_key(str(main_dir)): FuzzyDupsPerSource(attr_dir=str(attr_dir))},
        counters={},
    )
    return fuzzy, quality


def _reelect(tmp_path: Path) -> tuple[FuzzyDupsAttrData, NormalizedData]:
    fuzzy, quality = _build_fixtures(tmp_path)
    reelected = reelect_cluster_canonicals(fuzzy=fuzzy, quality=quality, output_path=str(tmp_path / "reelect"))
    return reelected, quality


def _read_rows_by_shard(attr_dir: str) -> dict[str, list[dict]]:
    return {path.name: pq.read_table(path).to_pylist() for path in sorted(Path(attr_dir).glob("*.parquet"))}


def _assignment(attr_dir: str) -> dict[str, tuple[str, bool]]:
    return {
        row["id"]: (row["dup_cluster_id"], row["is_cluster_canonical"])
        for rows in _read_rows_by_shard(attr_dir).values()
        for row in rows
    }


def _new_attr_dir(reelected: FuzzyDupsAttrData, quality: NormalizedData) -> str:
    return reelected.attr_dir_for_source(quality.main_output_dir)


def test_reelect_highest_edu_max_wins_within_cluster(tmp_path):
    reelected, quality = _reelect(tmp_path)
    assignment = _assignment(_new_attr_dir(reelected, quality))

    assert assignment["a-high"] == ("cl-alpha", True)
    assert assignment["a-low"] == ("cl-alpha", False)


def test_reelect_tie_on_score_falls_to_smallest_id(tmp_path):
    reelected, quality = _reelect(tmp_path)
    assignment = _assignment(_new_attr_dir(reelected, quality))

    assert assignment["t-a"] == ("cl-tie", True)
    assert assignment["t-b"] == ("cl-tie", False)


def test_reelect_unscored_member_loses_to_scored_member(tmp_path):
    """ "u-none" has a null edu_max, so even the low-scored "s-scored" (0.5) must beat it."""
    reelected, quality = _reelect(tmp_path)
    assignment = _assignment(_new_attr_dir(reelected, quality))

    assert assignment["s-scored"] == ("cl-missing", True)
    assert assignment["u-none"] == ("cl-missing", False)


def test_reelect_exactly_one_canonical_per_cluster_and_cluster_ids_preserved(tmp_path):
    reelected, quality = _reelect(tmp_path)
    assignment = _assignment(_new_attr_dir(reelected, quality))

    input_clusters = {row["id"]: row["dup_cluster_id"] for rows in _ATTR_ROWS.values() for row in rows}
    assert {doc_id: cluster for doc_id, (cluster, _) in assignment.items()} == input_clusters

    canonicals_per_cluster: dict[str, int] = {}
    for cluster_id, is_canonical in assignment.values():
        canonicals_per_cluster[cluster_id] = canonicals_per_cluster.get(cluster_id, 0) + int(is_canonical)
    assert canonicals_per_cluster == {"cl-alpha": 1, "cl-tie": 1, "cl-missing": 1}


def test_reelect_preserves_shard_basenames_and_id_sort_order(tmp_path):
    """The rewritten tree must keep consolidate's join invariants: 1:1 basenames, sorted rows."""
    reelected, quality = _reelect(tmp_path)
    rows_by_shard = _read_rows_by_shard(_new_attr_dir(reelected, quality))

    assert set(rows_by_shard) == set(_SHARDS)
    for basename, rows in rows_by_shard.items():
        ids = [row["id"] for row in rows]
        assert ids == sorted(ids), f"{basename} lost its id sort"
        assert ids == [row["id"] for row in _ATTR_ROWS[basename]], f"{basename} changed membership"


def test_reelect_two_runs_produce_identical_assignment(tmp_path):
    """Ties are the common case (byte-identical texts score identically), so the election must
    not depend on dict/set ordering or shard visit order — mirror
    test_fuzzy_dups_canonical_selection_is_deterministic's invariants."""
    first_reelected, first_quality = _reelect(tmp_path / "run_a")
    second_reelected, second_quality = _reelect(tmp_path / "run_b")

    first = _assignment(_new_attr_dir(first_reelected, first_quality))
    second = _assignment(_new_attr_dir(second_reelected, second_quality))

    assert first == second
    assert first, "expected at least one cluster member row"


def test_consolidate_keeps_elected_canonicals_and_singletons(tmp_path):
    """Round trip through the step fns: re-elect, then consolidate with keep_if_missing.

    Survivors must be exactly the elected canonicals plus the singletons (no attr row); the old
    canonicals the election overturned must be gone.
    """
    fuzzy, quality = _build_fixtures(tmp_path)
    quality_step_dir = tmp_path / "quality"
    fuzzy_step_dir = tmp_path / "fuzzy_dups"
    write_artifact(quality, str(quality_step_dir))
    write_artifact(fuzzy, str(fuzzy_step_dir))

    reelect_dir = tmp_path / "reelect"
    reelected = reelect_fuzzy_canonicals(str(reelect_dir), str(fuzzy_step_dir), str(quality_step_dir))
    write_artifact(reelected, str(reelect_dir))

    clean_dir = tmp_path / "fuzzy_clean"
    clean = consolidate_fuzzy_clean(str(clean_dir), str(quality_step_dir), str(reelect_dir))

    survivors = {
        row["id"]
        for path in sorted(Path(clean.main_output_dir).glob("*.parquet"))
        for row in pq.read_table(path).to_pylist()
    }
    assert survivors == _EXPECTED_CANONICALS | {"solo", "z-solo2"}
