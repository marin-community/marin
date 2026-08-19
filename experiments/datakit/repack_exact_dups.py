# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repack exact-duplicate marks for a source that has been normalized since.

The pinned global exact dedup ran over the Focus Crawl's jusText extraction tree,
before #8111 put that tree through ``normalize_step``. Its marks are addressed by
the extraction's shard layout, so the store looks for them under the current
shard names and finds nothing -- and because duplicate attributes are sparse,
"nothing" reads as "no duplicates" rather than as an error.

Moving them is not the shard-renaming that :mod:`repack_fuzzy_dups` does, because
exact marks count occurrences and normalize removed most of them.
``normalize_step`` defaults to ``DedupMode.EXACT``, which drops rows that repeat
an ``id`` within a shard, and hash partitioning puts every occurrence of an
``id`` in one shard, so each ``id`` now appears exactly once. On the Focus Crawl
that removed 14,316,834 of 36,327,068 rows, or 39.4%. Marks written against the
copies normalize removed no longer describe anything, and carrying them over
unchanged deletes the copy that remains.

Which marks still apply follows from counting, and the counts settle it exactly.
Global exact dedup keeps one occurrence of an ``id`` across all sources and marks
the rest, so for an ``id`` with ``f`` occurrences in this source:

* ``marks == f - 1`` -- this source held the canonical occurrence, and it holds
  it still. The surviving copy must stay. Emit nothing.
* ``marks == f`` -- every copy here was a duplicate of one in another source,
  which is still true of the copy that remains. Emit one mark.

Reading the extraction's ``id`` column is what makes ``f`` available, so this
repack reads the old source and :mod:`repack_fuzzy_dups` does not. It is one pass
over one column.

The Focus Crawl's arithmetic closes on those two counts and says what the run
must produce. Summing ``f - 1`` over every ``id`` is what normalize already
reported as ``duplicate_records_out``, 14,316,834, and the source carries
14,316,837 marks, so exactly three of its documents duplicate something outside
it. ``marks_kept`` must come out at 3. A rename would have carried all 14,316,837
across and deleted 65.0% of the 22,010,234 documents the source has left.

The dedup decision itself is not recomputed. Which source holds the canonical
copy of a cross-source duplicate is a choice the pinned run already made, and any
choice is correct as long as exactly one copy survives globally.
"""

import logging
import os
from collections.abc import Iterator
from typing import TypedDict

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit import partition_filename
from marin.datakit.copartitioned import write_copartitioned_source_manifest
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key, datakit_source_path
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.writers import write_parquet_file

from experiments.datakit.global_exact_dedup import ExactDupsPerSource, GlobalExactDedupData

logger = logging.getLogger(__name__)

REPACK_COUNTER_PREFIX = "repack_exact_dups"

# Pipeline-side counter names. Bare, because the artifact re-namespaces every
# outcome counter under REPACK_COUNTER_PREFIX on the way out.
MARKS_KEPT = "marks_kept"
MARKS_DROPPED = "marks_dropped"

_ATTR_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field("dup_doc", pa.bool_(), nullable=False),
    ]
)


class _Tally(TypedDict):
    """One ``id``, with how many times it occurs and how many marks it carries."""

    id: str
    occurrences: int
    marks: int


def _parquet_paths(directory: str) -> list[str]:
    return sorted(str(path) for path in StoragePath(prefix_join(directory, "*.parquet")).glob())


def _normalized_shard_basenames(normalized: NormalizedData) -> list[str]:
    """Return the current shard names, checked against the Datakit partition layout."""
    paths = _parquet_paths(normalized.main_output_dir)
    if not paths:
        raise FileNotFoundError(f"No Parquet files found under {normalized.main_output_dir}")

    basenames = [os.path.basename(path) for path in paths]
    expected = [partition_filename(shard, len(paths)) for shard in range(len(paths))]
    if basenames != expected:
        raise ValueError(
            f"Normalized shard names do not use the Datakit partition layout. Expected {expected}, got {basenames}"
        )
    return basenames


def _read_ids(path: str, marked: bool) -> Iterator[_Tally]:
    """Yield one tally per row of ``path``, counted on the side ``marked`` selects."""
    with StoragePath(path).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(columns=["id"]):
            for record_id in batch.column("id").to_pylist():
                yield {"id": record_id, "occurrences": 0 if marked else 1, "marks": 1 if marked else 0}


def _combine(doc_id: str, tallies: Iterator[_Tally]) -> Iterator[_Tally]:
    """Sum a group's tallies. Associative, so it runs map-side as well."""
    occurrences = marks = 0
    for tally in tallies:
        occurrences += tally["occurrences"]
        marks += tally["marks"]
    yield {"id": doc_id, "occurrences": occurrences, "marks": marks}


def _decide(doc_id: str, tallies: Iterator[_Tally]) -> Iterator[dict[str, str | bool]]:
    """Emit a mark for ``doc_id`` only when the surviving copy is still a duplicate.

    Counts what it keeps and what it discards. ``marks_kept`` is the number the
    source's own arithmetic predicts, so it is how a run says whether it did what
    the module docstring claims rather than merely finishing.
    """
    occurrences = marks = 0
    for tally in tallies:
        occurrences += tally["occurrences"]
        marks += tally["marks"]

    if occurrences == 0:
        raise ValueError(f"exact-duplicate marks name id={doc_id!r}, which the legacy source does not contain")
    if marks == occurrences:
        counters.pipeline.update_counter(MARKS_KEPT, 1)
        yield {"id": doc_id, "dup_doc": True}
        return
    if marks != occurrences - 1:
        raise ValueError(
            f"id={doc_id!r} has {occurrences} occurrences and {marks} marks; global exact dedup keeps exactly "
            "one occurrence, so a source's marks must equal its occurrences or one fewer"
        )
    if marks:
        counters.pipeline.update_counter(MARKS_DROPPED, marks)


def repack_exact_dups_source(
    *,
    exact_dups: GlobalExactDedupData,
    legacy_source_key: str,
    normalized: NormalizedData,
    output_path: str,
    max_workers: int = 64,
    worker_resources: ResourceConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
    zephyr_context: ZephyrContext | None = None,
) -> GlobalExactDedupData:
    """Move one source's exact-duplicate marks onto its current normalize.

    Returns the same artifact with that source re-keyed to the current source key
    and pointed at the repacked attribute directory. Every other source is passed
    through untouched.
    """
    legacy_source = exact_dups.sources.get(legacy_source_key)
    if legacy_source is None:
        raise KeyError(f"Exact duplicate data has no source_key={legacy_source_key!r}")

    source_key = datakit_source_key(normalized.main_output_dir)
    if source_key == legacy_source_key:
        raise ValueError("The source key did not change")
    if source_key in exact_dups.sources:
        raise ValueError(f"Exact duplicate data already has source_key={source_key!r}")

    legacy_source_paths = _parquet_paths(datakit_source_path(legacy_source_key))
    if not legacy_source_paths:
        raise FileNotFoundError(f"No Parquet files found under the legacy source {legacy_source_key}")
    mark_paths = _parquet_paths(legacy_source.attr_dir)
    if not mark_paths:
        raise FileNotFoundError(f"No Parquet files found under {legacy_source.attr_dir}")

    basenames = _normalized_shard_basenames(normalized)
    attr_dir = prefix_join(output_path, "outputs/repacked_source")
    logger.info(
        "repacking %d mark shards against %d legacy source shards into %d current shards",
        len(mark_paths),
        len(legacy_source_paths),
        len(basenames),
    )

    def output_path_for_shard(shard: int, total: int) -> str:
        if total != len(basenames):
            raise ValueError(f"Expected {len(basenames)} output shards, got {total}")
        return prefix_join(attr_dir, basenames[shard])

    inputs = [{"path": path, "marked": False} for path in legacy_source_paths]
    inputs += [{"path": path, "marked": True} for path in mark_paths]
    pipeline = (
        Dataset.from_list(inputs)
        .flat_map(lambda row: _read_ids(row["path"], row["marked"]))
        .group_by(
            key=lambda tally: tally["id"],
            reducer=_decide,
            combiner=_combine,
            # Normalize partitions on ``id`` with this same rule, so grouping here
            # lands each mark in the shard that now holds its document.
            num_output_shards=len(basenames),
        )
        .write_parquet(output_path_for_shard, schema=_ATTR_SCHEMA)
    )

    context_args: dict = {"name": "repack-exact-dups", "max_workers": max_workers}
    if worker_resources is not None:
        context_args["resources"] = worker_resources
    if coordinator_resources is not None:
        context_args["coordinator_resources"] = coordinator_resources
    context = zephyr_context or ZephyrContext(**context_args)
    outcome = context.execute(
        pipeline,
        verbose=True,
        map_task_resources=map_task_resources or worker_resources,
        reduce_task_resources=reduce_task_resources or worker_resources,
    )
    written = _fill_empty_shards(attr_dir, basenames)

    sources = dict(exact_dups.sources)
    del sources[legacy_source_key]
    sources[source_key] = ExactDupsPerSource(attr_dir=attr_dir)
    write_copartitioned_source_manifest(
        output_path=output_path,
        attr_dirs={key: source.attr_dir for key, source in sources.items()},
    )
    counters = dict(exact_dups.counters)
    counters[f"{REPACK_COUNTER_PREFIX}/shards_with_marks"] = written
    counters[f"{REPACK_COUNTER_PREFIX}/shards_total"] = len(basenames)
    counters.update({f"{REPACK_COUNTER_PREFIX}/{key}": value for key, value in outcome.counters.items()})
    return GlobalExactDedupData(sources=sources, counters=counters)


def _fill_empty_shards(attr_dir: str, basenames: list[str]) -> int:
    """Write an empty attribute file for every shard the repack had no marks for.

    Global exact dedup writes a file per shard whether or not it holds a
    duplicate, because consumers resolve every input shard to an attribute path
    before reading. A shuffle only writes the shards it has rows for, so the
    shards it skipped are filled in here. Returns how many did have marks.
    """
    present = {os.path.basename(path) for path in _parquet_paths(attr_dir)}
    for basename in basenames:
        if basename not in present:
            write_parquet_file(iter(()), output_path=prefix_join(attr_dir, basename), schema=_ATTR_SCHEMA)
    return len(present)
