"""FineStore result and sample tables for Evalchemy evaluations."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa
from finestore.eval import (
    ARCHIVE_SAMPLES_TABLE,
    SAMPLES_MERGE_KEY,
    SCHEMA_VERSION,
    sample_to_archive_row,
    samples_schema,
)
from finestore.layout import OnConflict
from finestore.reader import ReadView
from finestore.store import DataStore

from eval.contracts.lm_eval_normalization import samples_from_lm_eval
from eval.contracts.result_serialization import handle_non_serializable_extended

RESULTS_TABLE = "evalchemy_results"
DEFAULT_SOURCE_PREFIX = "run"
_SOURCE_PREFIX = "source_prefix"
_DOCUMENT = "document"
_SCORED = "scored"
_SAMPLE_BUFFER_BYTES = 64 * 1024 * 1024
_RESULTS_SCHEMA = pa.schema(
    [pa.field(_SOURCE_PREFIX, pa.string()), pa.field(_DOCUMENT, pa.large_string()), pa.field(_SCORED, pa.bool_())]
)


def _scored_results(results: Mapping[str, Any]) -> bool:
    outcomes = results.get("task_outcomes", {})
    return bool(results.get("results")) and all(
        outcome.get("status") in ("succeeded", "exported") for outcome in outcomes.values()
    )


def _result_row(root: str, source_prefix: str) -> dict[str, Any] | None:
    rows = list(ReadView(root).iter_rows(RESULTS_TABLE, where=[(_SOURCE_PREFIX, "==", source_prefix)]))
    return rows[0] if rows else None


def read_finestore_output(root: str, source_prefix: str) -> dict[str, Any] | None:
    """Read one Evalchemy result document from the FineStore results table."""
    row = _result_row(root, source_prefix)
    return None if row is None else json.loads(row[_DOCUMENT])


def completed_finestore_output(root: str, source_prefix: str) -> bool:
    """Whether this task has a committed successful result in FineStore."""
    row = _result_row(root, source_prefix)
    return row is not None and row[_SCORED]


class FineStoreOutput:
    """Stage scored samples during evaluation and publish the result last."""

    def __init__(self, root: str, source_prefix: str, task_names: Sequence[str]) -> None:
        self.root = root
        self.source_prefix = source_prefix
        self.task_names = tuple(task_names)
        self._store = DataStore.open(root, writer_id=f"evalchemy-{uuid.uuid4().hex}")
        self._sample_schema = samples_schema()
        self._sample_bytes = 0
        self._samples = self._store.table(
            ARCHIVE_SAMPLES_TABLE,
            primary_key=SAMPLES_MERGE_KEY,
            schema=self._sample_schema,
            schema_version=SCHEMA_VERSION,
            on_conflict=OnConflict.SUPERSEDE,
        )
        self._results = self._store.table(
            RESULTS_TABLE,
            primary_key=(_SOURCE_PREFIX,),
            schema=_RESULTS_SCHEMA,
            on_conflict=OnConflict.SUPERSEDE,
        )

    def write_sample(self, task_name: str, record: Mapping[str, Any]) -> None:
        """Append one canonical record and its filter variants to FineStore."""
        normalized_task = (
            self.source_prefix
            if self.task_names == (task_name,)
            else f"{self.source_prefix}/{task_name}"
        )
        repeat = record.get("sample_repeat")
        trial_id = "" if repeat is None else str(repeat)
        for sample in samples_from_lm_eval(normalized_task, dict(record)):
            row = sample_to_archive_row(sample, trial_id=trial_id)
            row_bytes = pa.Table.from_pylist([row], schema=self._sample_schema).nbytes
            self._samples.append(row)
            self._sample_bytes += row_bytes
            if self._sample_bytes >= _SAMPLE_BUFFER_BYTES:
                # Background flush requests do not bound admission while a
                # previous write or compaction owns the storage commit lock.
                self._samples.flush()
                self._sample_bytes = 0

    @property
    def store(self) -> DataStore:
        """Share this archive's writer; this output context owns its lifetime."""
        return self._store

    def finish_task(self) -> None:
        """Make the task's staged samples readable before starting another task."""
        self._store.flush()
        self._sample_bytes = 0

    def finish(self, results: Mapping[str, Any]) -> None:
        """Publish success only after all samples and task outcomes are complete."""
        self._store.flush()
        self._sample_bytes = 0
        self._results.append(
            {
                _SOURCE_PREFIX: self.source_prefix,
                _DOCUMENT: json.dumps(results, default=handle_non_serializable_extended, ensure_ascii=False),
                _SCORED: _scored_results(results),
            }
        )
        self._store.flush()

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> FineStoreOutput:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
