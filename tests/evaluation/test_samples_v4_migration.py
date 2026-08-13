# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The v4 samples migration preserves stale data before removing its logical table."""

from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest
from finestore.admin import set_table_metadata
from finestore.reader import ReadView
from marin.evaluation.archive import ARCHIVE_SAMPLES_TABLE, EvalSample, EvaluationStore, Grading, SampleKind
from rigging.filesystem import StoragePath

from experiments.evaluation.migrations.samples_v4 import (
    copy_table,
    drop_table,
    replace_stale_samples,
    replace_table,
)


def _archive(results, samples: list[EvalSample]) -> None:
    store = EvaluationStore.open(str(results), writer_id="evalchemy")
    for sample in samples:
        store.add_sample(sample)
    store.seal()
    store.close()


def _generation(doc_id: str) -> EvalSample:
    return EvalSample(task="gsm8k", doc_id=doc_id, kind=SampleKind.GENERATION, output="4")


def _agentic(doc_id: str) -> EvalSample:
    return EvalSample(
        task="tb2",
        doc_id=doc_id,
        kind=SampleKind.AGENTIC,
        trajectory_uri="finestore://blobs/t1/trajectory.json",
        grading=Grading(method="harbor:verifier", metric="reward", score=1.0, passed=True),
        correct=True,
    )


def _stamp_schema_version(results, version: int) -> None:
    reader = ReadView(str(results))
    metadata = reader.table_metadata(ARCHIVE_SAMPLES_TABLE).model_copy(update={"schema_version": version})
    set_table_metadata(str(results), ARCHIVE_SAMPLES_TABLE, metadata)


def test_replacing_a_table_preserves_it_outside_the_run_first(tmp_path):
    # A contract change is the one moment the archive holds rows nothing else does. They go to the
    # 30-day bucket before the drop, so a bad rebuild is recoverable rather than terminal.
    results = tmp_path / "run" / "results"
    _archive(results, [_generation("1"), _generation("2")])
    destination = tmp_path / "backup-v3"

    replace_table(str(results), ARCHIVE_SAMPLES_TABLE, str(destination))

    assert ReadView(str(results)).scan(ARCHIVE_SAMPLES_TABLE) is None
    assert (destination / "_schema.json").exists()
    assert list(destination.rglob("*.parquet"))


def test_a_preserved_table_still_holds_the_dropped_rows(tmp_path):
    # The point of the snapshot is the data, not the directory: every row the drop removed has to be
    # readable from the copy.
    results = tmp_path / "run" / "results"
    _archive(results, [_generation("1"), _generation("2")])
    destination = tmp_path / "backup-v3"

    replace_table(str(results), ARCHIVE_SAMPLES_TABLE, str(destination))

    preserved = pq.read_table(sorted(destination.rglob("*.parquet")))
    assert sorted(preserved["doc_id"].to_pylist()) == ["1", "2"]


def test_a_mismatched_existing_snapshot_aborts_before_drop(tmp_path):
    results = tmp_path / "run" / "results"
    _archive(results, [_generation("1")])
    destination = tmp_path / "backup-v3"
    destination.mkdir()
    (destination / "_schema.json").write_text('{"sentinel": true}')

    with pytest.raises(ValueError, match="metadata backup"):
        replace_table(str(results), ARCHIVE_SAMPLES_TABLE, str(destination))

    assert json.loads((destination / "_schema.json").read_text()) == {"sentinel": True}
    assert ReadView(str(results)).scan(ARCHIVE_SAMPLES_TABLE) is not None


def test_dropping_a_table_only_changes_logical_visibility(tmp_path):
    results = tmp_path / "run" / "results"
    _archive(results, [_generation("1")])

    shard_path = ReadView(str(results)).list_shards(ARCHIVE_SAMPLES_TABLE)[0].path

    assert drop_table(str(results), ARCHIVE_SAMPLES_TABLE) == 1
    assert ReadView(str(results)).scan(ARCHIVE_SAMPLES_TABLE) is None
    assert StoragePath(shard_path).exists()


def test_copying_a_table_that_was_never_written_is_not_an_error(tmp_path):
    # A sweep visits archives that never held the table; that is a no-op, not a failure.
    results = tmp_path / "run" / "results"
    _archive(results, [_generation("1")])

    assert copy_table(str(results), "trajectories", str(tmp_path / "backup")) == 0


def test_replace_refuses_a_table_holding_samples_it_cannot_regenerate(tmp_path):
    # Agentic rows come from Harbor, which writes no lm-eval jsonl. The migration must keep the table
    # visible when it cannot regenerate all of it.
    results = tmp_path / "run" / "results"
    _archive(results, [_agentic("1")])
    _stamp_schema_version(results, 3)

    with pytest.raises(ValueError, match="agentic"):
        replace_stale_samples(str(results))
    assert ReadView(str(results)).scan(ARCHIVE_SAMPLES_TABLE).num_rows == 1


def test_an_archive_already_at_the_current_contract_is_left_alone(tmp_path):
    results = tmp_path / "run" / "results"
    _archive(results, [_generation("1")])

    assert replace_stale_samples(str(results)) is None
    assert ReadView(str(results)).scan(ARCHIVE_SAMPLES_TABLE).num_rows == 1
