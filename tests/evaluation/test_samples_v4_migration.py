# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The v4 samples migration: the only eval code that deletes, and what it preserves before it does."""

from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest
from finestore.eval import ARCHIVE_SAMPLES_TABLE, EvalSample, EvaluationStore, Grading, SampleKind
from finestore.reader import CompositeReader

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
    schema_path = results / ARCHIVE_SAMPLES_TABLE / "_schema.json"
    meta = json.loads(schema_path.read_text())
    meta["schema_version"] = version
    schema_path.write_text(json.dumps(meta))


def test_replacing_a_table_preserves_it_outside_the_run_first(tmp_path):
    # A contract change is the one moment the archive holds rows nothing else does. They go to the
    # 30-day bucket before the drop, so a bad rebuild is recoverable rather than terminal.
    results = tmp_path / "run" / "results"
    _archive(results, [_generation("1"), _generation("2")])
    destination = tmp_path / "backup-v3"

    replace_table(str(results), ARCHIVE_SAMPLES_TABLE, str(destination))

    assert CompositeReader(str(results)).scan(ARCHIVE_SAMPLES_TABLE) is None
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


def test_an_existing_snapshot_is_never_overwritten(tmp_path):
    # The first snapshot is the pristine one. A resumed or repeated replace must not put a degraded
    # table on top of it.
    results = tmp_path / "run" / "results"
    _archive(results, [_generation("1")])
    destination = tmp_path / "backup-v3"
    destination.mkdir()
    (destination / "_schema.json").write_text('{"sentinel": true}')

    copy_table(str(results), ARCHIVE_SAMPLES_TABLE, str(destination))

    assert json.loads((destination / "_schema.json").read_text()) == {"sentinel": True}


def test_dropping_a_table_leaves_its_schema_for_the_next_writer(tmp_path):
    # The shards go; the declaration of what the table is stays, so the writer that rewrites the rows
    # does not have to re-derive the primary key.
    results = tmp_path / "run" / "results"
    _archive(results, [_generation("1")])

    assert drop_table(str(results), ARCHIVE_SAMPLES_TABLE) == 1

    assert (results / ARCHIVE_SAMPLES_TABLE / "_schema.json").exists()
    assert not list((results / ARCHIVE_SAMPLES_TABLE).rglob("*.parquet"))


def test_copying_a_table_that_was_never_written_is_not_an_error(tmp_path):
    # A sweep visits archives that never held the table; that is a no-op, not a failure.
    results = tmp_path / "run" / "results"
    _archive(results, [_generation("1")])

    assert copy_table(str(results), "trajectories", str(tmp_path / "backup")) == 0


def test_replace_refuses_a_table_holding_samples_it_cannot_regenerate(tmp_path):
    # Agentic rows come from Harbor, which writes no lm-eval jsonl. Dropping them would leave the
    # table missing a half nothing can put back, so the migration stops before it deletes anything.
    results = tmp_path / "run" / "results"
    _archive(results, [_agentic("1")])
    _stamp_schema_version(results, 3)

    with pytest.raises(ValueError, match="agentic"):
        replace_stale_samples(str(results))
    assert CompositeReader(str(results)).scan(ARCHIVE_SAMPLES_TABLE).num_rows == 1


def test_an_archive_already_at_the_current_contract_is_left_alone(tmp_path):
    results = tmp_path / "run" / "results"
    _archive(results, [_generation("1")])

    assert replace_stale_samples(str(results)) is None
    assert CompositeReader(str(results)).scan(ARCHIVE_SAMPLES_TABLE).num_rows == 1
