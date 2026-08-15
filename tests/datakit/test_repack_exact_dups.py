# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the exact-duplicate repack.

Builds a legacy source and its marks on the local filesystem, runs the repack
through a local Zephyr context, and checks which marks survive. The interesting
cases are the ones a shard-renaming repack gets wrong: the legacy source held
repeated ids, ``DedupMode.EXACT`` removed those copies, and whether the mark
still applies depends on whether the canonical copy was in this source or
another one.
"""

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key

from experiments.datakit.global_exact_dedup import ExactDupsPerSource, GlobalExactDedupData
from experiments.datakit.repack_exact_dups import (
    MARKS_DROPPED,
    MARKS_KEPT,
    REPACK_COUNTER_PREFIX,
    _decide,
    repack_exact_dups_source,
)

CURRENT_SHARDS = 3


def _tallies(occurrences: int, marks: int) -> list[dict]:
    return [{"id": "x", "occurrences": 1, "marks": 0}] * occurrences + [
        {"id": "x", "occurrences": 0, "marks": 1}
    ] * marks


def test_a_source_that_held_the_canonical_copy_keeps_it():
    """Three copies, two marked: the unmarked one was canonical and still is."""
    assert list(_decide("x", iter(_tallies(occurrences=3, marks=2)))) == []


def test_a_source_whose_every_copy_was_a_duplicate_stays_marked():
    """Three copies, three marked: the canonical lives in another source."""
    assert list(_decide("x", iter(_tallies(occurrences=3, marks=3)))) == [{"id": "x", "dup_doc": True}]


def test_an_unduplicated_id_is_not_marked():
    assert list(_decide("x", iter(_tallies(occurrences=1, marks=0)))) == []


def test_a_single_copy_that_duplicates_another_source_stays_marked():
    assert list(_decide("x", iter(_tallies(occurrences=1, marks=1)))) == [{"id": "x", "dup_doc": True}]


def test_a_mark_for_an_absent_id_is_rejected():
    with pytest.raises(ValueError, match="does not contain"):
        list(_decide("x", iter(_tallies(occurrences=0, marks=1))))


def test_more_marks_than_copies_is_rejected():
    """Global exact dedup never marks more occurrences than a source has."""
    with pytest.raises(ValueError, match="keeps exactly"):
        list(_decide("x", iter(_tallies(occurrences=2, marks=4))))


def _write(path: str, ids: list[str], marks: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns: dict[str, pa.Array] = {"id": pa.array(ids, type=pa.string())}
    if marks:
        columns["dup_doc"] = pa.array([True] * len(ids))
    pq.write_table(pa.table(columns), path)


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    """A legacy source with repeated ids, and the marks a global run left on it.

    ``keep`` is duplicated inside the source and holds the canonical copy, so its
    two other copies are marked. ``drop`` matches a document in some other source,
    so every copy of it is marked. ``solo`` is not duplicated at all.
    """
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    source_dir = tmp_path / "legacy/outputs/main"
    attr_dir = tmp_path / "legacy_attrs"
    _write(str(source_dir / "part-00000-of-00002.parquet"), ["keep", "keep", "drop", "solo"])
    _write(str(source_dir / "part-00001-of-00002.parquet"), ["keep", "drop"])
    _write(str(attr_dir / "part-00000-of-00002.parquet"), ["keep", "drop"], marks=True)
    _write(str(attr_dir / "part-00001-of-00002.parquet"), ["keep", "drop"], marks=True)
    return source_dir, attr_dir


def test_repack_keeps_cross_source_marks_and_drops_intra_source_ones(tmp_path, legacy):
    """The whole point: an id whose copies normalize removed must not stay marked.

    ``keep`` carried two marks because the legacy source held three copies of it.
    Normalize leaves one, and that one is the canonical copy, so a repack that
    only renamed shards would delete a document the corpus is supposed to have.
    """
    source_dir, attr_dir = legacy
    current_dir = tmp_path / "current/outputs/main"
    for shard in range(CURRENT_SHARDS):
        _write(str(current_dir / f"part-0000{shard}-of-0000{CURRENT_SHARDS}.parquet"), [])

    output_path = str(tmp_path / "repacked")
    repacked = repack_exact_dups_source(
        exact_dups=GlobalExactDedupData(
            sources={
                datakit_source_key(str(source_dir)): ExactDupsPerSource(attr_dir=str(attr_dir)),
                "other/outputs/main": ExactDupsPerSource(attr_dir=str(tmp_path / "other_attrs")),
            },
            counters={},
        ),
        legacy_source_key=datakit_source_key(str(source_dir)),
        normalized=NormalizedData(
            main_output_dir=str(current_dir), dup_output_dir=str(tmp_path / "current/outputs/dups"), counters={}
        ),
        output_path=output_path,
        max_workers=2,
    )

    current_key = datakit_source_key(str(current_dir))
    assert set(repacked.sources) == {current_key, "other/outputs/main"}

    attr_files = sorted(str(p) for p in (tmp_path / "repacked/outputs/repacked_source").glob("*.parquet"))
    # One attribute file per current shard, including the ones with no marks:
    # consumers resolve every input shard to an attribute path before reading.
    assert len(attr_files) == CURRENT_SHARDS
    marked = {row for path in attr_files for row in pq.read_table(path).column("id").to_pylist()}
    assert marked == {"drop"}

    # The counter a real run is checked against. "keep" contributes its two stale
    # marks to the dropped side; "drop" contributes the one mark that survives.
    assert repacked.counters[f"{REPACK_COUNTER_PREFIX}/{MARKS_KEPT}"] == 1
    assert repacked.counters[f"{REPACK_COUNTER_PREFIX}/{MARKS_DROPPED}"] == 2


def test_repack_refuses_a_source_key_that_did_not_change(tmp_path, legacy):
    source_dir, attr_dir = legacy
    key = datakit_source_key(str(source_dir))

    with pytest.raises(ValueError, match="did not change"):
        repack_exact_dups_source(
            exact_dups=GlobalExactDedupData(sources={key: ExactDupsPerSource(attr_dir=str(attr_dir))}, counters={}),
            legacy_source_key=key,
            normalized=NormalizedData(
                main_output_dir=str(source_dir), dup_output_dir=str(tmp_path / "legacy/outputs/dups"), counters={}
            ),
            output_path=str(tmp_path / "repacked"),
        )
