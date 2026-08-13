# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge co-partitioned Harrier document sets in normalized row order."""

import argparse
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import read_artifact, read_record
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from rigging.filesystem import StoragePath, marin_prefix, marin_temp_bucket, prefix_join
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import iter_parquet_row_groups
from zephyr.runners import InlineRunner

from experiments.datakit.embeddings.harrier.pipeline import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_SCHEMA,
    HARRIER_DIM,
    HARRIER_REPO,
    HARRIER_REVISION,
    QUANT_RANGE,
    QUANT_SCALE,
    EmbeddingAttrData,
    EmbeddingDocumentSet,
)
from experiments.datakit.reference_pipeline import select_sources

DEDUPLICATED_PREFIX = "datakit/embed/harrier"
FUZZY_DUPLICATE_PREFIX = "datakit/embed/harrier-fuzzy-duplicates"
MERGED_PREFIX = "datakit/embed/harrier-all"
MERGE_VERSION = 1
WORKERS_PER_SOURCE = 32
MAX_CONCURRENT = 8


@dataclass(frozen=True)
class EmbeddingShardPair:
    """The three input files for one merged output shard."""

    normalized_path: str
    deduplicated_path: str
    fuzzy_duplicate_path: str
    basename: str


@dataclass(frozen=True)
class EmbeddingSourcePair:
    """The normalized path and two embedding artifacts for one source."""

    source_name: str
    source_key: str
    normalized_path: str
    deduplicated_path: str
    fuzzy_duplicate_path: str


def _resolved_prefix(path: str) -> str:
    if StoragePath(path).scheme or path.startswith("/"):
        return path.rstrip("/")
    return prefix_join(marin_prefix(), path).rstrip("/")


def _artifact_paths_by_source(prefix: str, source_names: list[str]) -> dict[str, list[str]]:
    paths = {}
    for source_name in source_names:
        matches = []
        for artifact_path in StoragePath(f"{prefix}/{source_name}_????????/.artifact.json").glob():
            artifact_path_text = str(artifact_path)
            output_path = artifact_path_text[: -len("/.artifact.json")]
            if StatusFile(output_path, worker_id="harrier-merge-discovery").status != STATUS_SUCCESS:
                continue
            matches.append(output_path)
        if matches:
            paths[source_name] = matches
    return paths


def _one_artifact_path(paths: dict[str, list[str]], source_name: str, prefix: str) -> str:
    matches = paths.get(source_name, [])
    if not matches:
        raise FileNotFoundError(f"No completed embedding artifact for {source_name} under {prefix}")
    if len(matches) != 1:
        raise ValueError(f"Multiple completed embedding artifacts for {source_name} under {prefix}: {sorted(matches)}")
    return matches[0]


def _validate_deduplicated_artifact(path: str) -> EmbeddingAttrData:
    artifact = read_artifact(path, EmbeddingAttrData)
    expected = {
        "model_name": HARRIER_REPO,
        "model_revision": HARRIER_REVISION,
        "embedding_dim": HARRIER_DIM,
        "quantization_scale": QUANT_SCALE,
        "quantization_range": QUANT_RANGE,
        "batch_size": DEFAULT_BATCH_SIZE,
    }
    actual = {key: getattr(artifact, key) for key in expected}
    if actual != expected:
        raise ValueError(f"Incompatible deduplicated Harrier artifact at {path}: {actual}")
    return artifact


def _fuzzy_duplicate_source(path: str) -> tuple[str, str]:
    record = read_record(path)
    if record is None:
        raise FileNotFoundError(f"No artifact record at {path}")

    expected_config = {
        "model": HARRIER_REPO,
        "revision": HARRIER_REVISION,
        "batch_size": DEFAULT_BATCH_SIZE,
        "document_set": EmbeddingDocumentSet.FUZZY_DUPLICATES.value,
    }
    config = record.config or {}
    actual_config = {key: config.get(key) for key in expected_config}
    if actual_config != expected_config:
        raise ValueError(f"Incompatible fuzzy-duplicate Harrier artifact at {path}: {actual_config}")
    if len(record.dep_paths) != 1:
        raise ValueError(f"Fuzzy-duplicate artifact at {path} must have one normalized dependency")

    normalized = read_artifact(record.dep_paths[0], NormalizedData)
    return datakit_source_key(normalized.main_output_dir), normalized.main_output_dir


def discover_source_pairs(
    *,
    deduplicated_prefix: str,
    fuzzy_duplicate_prefix: str,
    source_names: list[str],
) -> list[EmbeddingSourcePair]:
    """Find and validate one completed input artifact pair per source."""
    deduplicated_prefix = _resolved_prefix(deduplicated_prefix)
    fuzzy_duplicate_prefix = _resolved_prefix(fuzzy_duplicate_prefix)
    deduplicated_paths = _artifact_paths_by_source(deduplicated_prefix, source_names)
    fuzzy_duplicate_paths = _artifact_paths_by_source(fuzzy_duplicate_prefix, source_names)

    pairs = []
    for source_name in source_names:
        deduplicated_path = _one_artifact_path(deduplicated_paths, source_name, deduplicated_prefix)
        fuzzy_duplicate_path = _one_artifact_path(fuzzy_duplicate_paths, source_name, fuzzy_duplicate_prefix)
        deduplicated = _validate_deduplicated_artifact(deduplicated_path)
        fuzzy_source_key, normalized_path = _fuzzy_duplicate_source(fuzzy_duplicate_path)
        if deduplicated.source_key != fuzzy_source_key:
            raise ValueError(
                f"Embedding source mismatch for {source_name}: "
                f"{deduplicated.source_key!r} != {fuzzy_source_key!r}"
            )
        pairs.append(
            EmbeddingSourcePair(
                source_name=source_name,
                source_key=deduplicated.source_key,
                normalized_path=normalized_path,
                deduplicated_path=deduplicated_path,
                fuzzy_duplicate_path=fuzzy_duplicate_path,
            )
        )
    return pairs


def _parquet_paths_by_basename(path: str) -> dict[str, str]:
    paths = sorted(str(item) for item in StoragePath(f"{path.rstrip('/')}/**/*.parquet").glob())
    if not paths:
        raise FileNotFoundError(f"No Parquet shards under {path}")
    by_basename = {}
    for item in paths:
        basename = os.path.basename(item)
        if basename in by_basename:
            raise ValueError(f"Duplicate Parquet basename {basename} under {path}")
        by_basename[basename] = item
    return by_basename


def _embedding_shard_pairs(
    normalized_path: str,
    deduplicated_path: str,
    fuzzy_duplicate_path: str,
) -> list[EmbeddingShardPair]:
    normalized = _parquet_paths_by_basename(normalized_path)
    deduplicated = _parquet_paths_by_basename(deduplicated_path)
    fuzzy_duplicates = _parquet_paths_by_basename(fuzzy_duplicate_path)
    if normalized.keys() != deduplicated.keys() or normalized.keys() != fuzzy_duplicates.keys():
        missing_normalized = sorted((deduplicated.keys() | fuzzy_duplicates.keys()) - normalized.keys())
        missing_deduplicated = sorted((normalized.keys() | fuzzy_duplicates.keys()) - deduplicated.keys())
        missing_fuzzy = sorted((normalized.keys() | deduplicated.keys()) - fuzzy_duplicates.keys())
        raise ValueError(
            "Harrier input shards are not co-partitioned: "
            f"missing normalized shards={missing_normalized}, "
            f"missing deduplicated shards={missing_deduplicated}, "
            f"missing fuzzy-duplicate shards={missing_fuzzy}"
        )
    return [
        EmbeddingShardPair(
            normalized_path=normalized[basename],
            deduplicated_path=deduplicated[basename],
            fuzzy_duplicate_path=fuzzy_duplicates[basename],
            basename=basename,
        )
        for basename in sorted(deduplicated)
    ]


def _validate_id_order(ids: pa.ChunkedArray, path: str, previous_id: str | None) -> str | None:
    if ids.null_count:
        raise ValueError(f"Null Harrier ID in {path}")
    if len(ids) == 0:
        return previous_id

    first_id = ids[0].as_py()
    last_id = ids[-1].as_py()
    if previous_id is not None and first_id < previous_id:
        raise ValueError(f"Harrier IDs are not sorted in {path}")
    if len(ids) > 1 and pc.any(pc.greater(ids.slice(0, len(ids) - 1), ids.slice(1))).as_py():
        raise ValueError(f"Harrier IDs are not sorted in {path}")
    return last_id


def _validated_tables(path: str, counter_name: str) -> Iterator[pa.Table]:
    previous_id = None
    for table in iter_parquet_row_groups(path):
        if not table.schema.equals(EMBEDDING_SCHEMA, check_metadata=False):
            raise ValueError(f"Unexpected Harrier embedding schema in {path}: {table.schema}")
        # Parquet uses `element` for the list child name. Restore the canonical schema before concatenation.
        table = table.cast(EMBEDDING_SCHEMA)
        previous_id = _validate_id_order(table.column("id"), path, previous_id)
        counters.pipeline.update_counter(counter_name, len(table))
        if len(table):
            yield table


def _prefix_equal(table: pa.Table, document_id: str) -> int:
    mask = pc.equal(table.column("id"), pa.scalar(document_id))
    return int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)


def _consume_prefix(table: pa.Table, length: int, tables: Iterator[pa.Table]) -> tuple[pa.Table, pa.Table | None]:
    prefix = table.slice(0, length)
    remainder = table.slice(length)
    return prefix, remainder if len(remainder) else next(tables, None)


def _table_groups(tables: Iterator[pa.Table]) -> Iterator[tuple[str, list[pa.Table]]]:
    table = next(tables, None)
    while table is not None:
        document_id = table.column("id")[0].as_py()
        pieces = []
        while table is not None and table.column("id")[0].as_py() == document_id:
            length = _prefix_equal(table, document_id)
            piece, table = _consume_prefix(table, length, tables)
            pieces.append(piece)
        yield document_id, pieces


def _row_count(tables: list[pa.Table]) -> int:
    return sum(len(table) for table in tables)


def _batches_from_prefix(tables: list[pa.Table], length: int) -> Iterator[pa.RecordBatch]:
    remaining = length
    for table in tables:
        if remaining == 0:
            return
        piece = table.slice(0, min(remaining, len(table)))
        remaining -= len(piece)
        yield from piece.to_batches(max_chunksize=DEFAULT_BATCH_SIZE)
    if remaining:
        raise ValueError(f"Harrier input group is {remaining} rows shorter than expected")


def _merge_in_normalized_order(
    deduplicated_tables: Iterator[pa.Table],
    fuzzy_duplicate_tables: Iterator[pa.Table],
    normalized_tables: Iterator[pa.Table],
) -> Iterator[pa.RecordBatch]:
    deduplicated_groups = _table_groups(deduplicated_tables)
    fuzzy_duplicate_groups = _table_groups(fuzzy_duplicate_tables)
    deduplicated = next(deduplicated_groups, None)
    fuzzy_duplicates = next(fuzzy_duplicate_groups, None)

    for normalized_id, normalized_pieces in _table_groups(normalized_tables):
        if deduplicated is not None and deduplicated[0] < normalized_id:
            raise ValueError(f"Deduplicated Harrier input has extra ID {deduplicated[0]!r}")
        if fuzzy_duplicates is not None and fuzzy_duplicates[0] < normalized_id:
            raise ValueError(f"Fuzzy-duplicate Harrier input has extra ID {fuzzy_duplicates[0]!r}")

        deduplicated_pieces = []
        if deduplicated is not None and deduplicated[0] == normalized_id:
            deduplicated_pieces = deduplicated[1]
            deduplicated = next(deduplicated_groups, None)

        fuzzy_duplicate_pieces = []
        if fuzzy_duplicates is not None and fuzzy_duplicates[0] == normalized_id:
            fuzzy_duplicate_pieces = fuzzy_duplicates[1]
            fuzzy_duplicates = next(fuzzy_duplicate_groups, None)

        normalized_count = _row_count(normalized_pieces)
        deduplicated_count = _row_count(deduplicated_pieces)
        fuzzy_duplicate_count = _row_count(fuzzy_duplicate_pieces)
        if deduplicated_count > normalized_count:
            raise ValueError(
                f"Deduplicated Harrier input has too many rows for ID {normalized_id!r}: "
                f"{deduplicated_count} > {normalized_count}"
            )
        if deduplicated_count + fuzzy_duplicate_count < normalized_count:
            raise ValueError(
                f"Harrier inputs have too few rows for ID {normalized_id!r}: "
                f"{deduplicated_count} + {fuzzy_duplicate_count} < {normalized_count}"
            )

        yield from _batches_from_prefix(deduplicated_pieces, deduplicated_count)
        fuzzy_rows_needed = normalized_count - deduplicated_count
        yield from _batches_from_prefix(fuzzy_duplicate_pieces, fuzzy_rows_needed)
        overlap_count = fuzzy_duplicate_count - fuzzy_rows_needed
        if overlap_count:
            counters.pipeline.update_counter("merge/overlapping_docs", overlap_count)

    if deduplicated is not None:
        raise ValueError(f"Deduplicated Harrier input has extra ID {deduplicated[0]!r}")
    if fuzzy_duplicates is not None:
        raise ValueError(f"Fuzzy-duplicate Harrier input has extra ID {fuzzy_duplicates[0]!r}")


def _id_batches(path: str, counter_name: str | None = None) -> Iterator[pa.RecordBatch]:
    for table in _id_tables(path, counter_name):
        yield from table.to_batches(max_chunksize=DEFAULT_BATCH_SIZE)


def _id_tables(path: str, counter_name: str | None = None) -> Iterator[pa.Table]:
    previous_id = None
    for table in iter_parquet_row_groups(path, columns=["id"]):
        if table.column("id").type != pa.string():
            raise ValueError(f"Unexpected ID type in {path}: {table.column('id').type}")
        previous_id = _validate_id_order(table.column("id"), path, previous_id)
        if counter_name is not None:
            counters.pipeline.update_counter(counter_name, len(table))
        if len(table):
            yield table


def _verify_normalized_order(
    batches: Iterator[pa.RecordBatch],
    normalized_path: str,
    counter_name: str | None,
) -> Iterator[pa.RecordBatch]:
    normalized_batches = _id_batches(normalized_path, counter_name)
    normalized = next(normalized_batches, None)
    normalized_offset = 0

    for batch in batches:
        merged_ids = batch.column("id")
        merged_offset = 0
        while merged_offset < len(merged_ids):
            if normalized is None:
                raise ValueError(f"Merged Harrier shard has extra IDs compared with {normalized_path}")
            compare_length = min(len(merged_ids) - merged_offset, len(normalized) - normalized_offset)
            normalized_ids = normalized.column("id")
            if not merged_ids.slice(merged_offset, compare_length).equals(
                normalized_ids.slice(normalized_offset, compare_length)
            ):
                raise ValueError(f"Merged Harrier ID order does not match {normalized_path}")
            merged_offset += compare_length
            normalized_offset += compare_length
            if normalized_offset == len(normalized):
                normalized = next(normalized_batches, None)
                normalized_offset = 0
        yield batch

    if normalized is not None:
        raise ValueError(f"Merged Harrier shard has missing IDs compared with {normalized_path}")


def _parquet_metadata(path: str) -> tuple[int, pa.Schema]:
    with StoragePath(path).open("rb") as file:
        parquet = pq.ParquetFile(file)
        return parquet.metadata.num_rows, parquet.schema_arrow


def _verify_output_shard(paths: tuple[str, str, str]) -> int:
    basename, output_shard, normalized_shard = paths
    output_rows, output_schema = _parquet_metadata(output_shard)
    normalized_rows, _ = _parquet_metadata(normalized_shard)
    if not output_schema.equals(EMBEDDING_SCHEMA, check_metadata=False):
        raise ValueError(f"Unexpected Harrier embedding schema in {output_shard}: {output_schema}")
    if output_rows != normalized_rows:
        raise ValueError(
            f"Merged Harrier row count does not match normalized data for {basename}: "
            f"{output_rows} != {normalized_rows}"
        )
    for _ in _verify_normalized_order(_id_batches(output_shard), normalized_shard, None):
        pass
    return output_rows


def verify_merged_output(output_path: str, normalized_path: str) -> tuple[int, int]:
    """Verify output shard names, schemas, row counts, and normalized ID order."""
    output_shards = _parquet_paths_by_basename(output_path)
    normalized_shards = _parquet_paths_by_basename(normalized_path)
    if output_shards.keys() != normalized_shards.keys():
        missing = sorted(normalized_shards.keys() - output_shards.keys())
        extra = sorted(output_shards.keys() - normalized_shards.keys())
        raise ValueError(f"Merged Harrier output shards do not match normalized data: missing={missing}, extra={extra}")

    shard_paths = [
        (basename, output_shards[basename], normalized_shards[basename]) for basename in sorted(normalized_shards)
    ]
    with ThreadPoolExecutor(max_workers=min(WORKERS_PER_SOURCE, len(shard_paths))) as executor:
        total_rows = sum(executor.map(_verify_output_shard, shard_paths))
    return len(output_shards), total_rows


def _merge_embedding_shard(pairs: Iterator[EmbeddingShardPair], shard: ShardInfo) -> Iterator[pa.RecordBatch]:
    try:
        pair = next(pairs)
    except StopIteration:
        raise ValueError(f"Merge shard {shard.shard_idx} has no input pair") from None
    try:
        next(pairs)
    except StopIteration:
        pass
    else:
        raise ValueError(f"Merge shard {shard.shard_idx} has more than one input pair")

    yield from _merge_in_normalized_order(
        _validated_tables(pair.deduplicated_path, "merge/deduplicated_docs"),
        _validated_tables(pair.fuzzy_duplicate_path, "merge/fuzzy_duplicate_docs"),
        _id_tables(pair.normalized_path, "merge/normalized_docs"),
    )


def merge_embedding_source(
    *,
    output_path: str,
    source_name: str,
    source_key: str,
    normalized_path: str,
    deduplicated_path: str,
    fuzzy_duplicate_path: str,
    max_workers: int = WORKERS_PER_SOURCE,
    chunk_storage_prefix: str | None = None,
) -> EmbeddingAttrData:
    """Stream one source and verify that its output follows normalized row order."""
    shard_pairs = _embedding_shard_pairs(normalized_path, deduplicated_path, fuzzy_duplicate_path)
    output_basenames = tuple(pair.basename for pair in shard_pairs)

    def _output_path(shard_index: int, _total: int, basenames: tuple[str, ...] = output_basenames) -> str:
        return str(StoragePath(output_path) / basenames[shard_index])

    dataset = (
        Dataset.from_list(shard_pairs)
        .map_shard(_merge_embedding_shard)
        .write_parquet(_output_path, schema=EMBEDDING_SCHEMA, skip_existing=True)
    )
    context = ZephyrContext(
        resources=ResourceConfig.with_cpu(cpu=1, ram="4g", disk="8g"),
        coordinator_resources=ResourceConfig(cpu=1, ram="4g", preemptible=False),
        max_workers=min(max_workers, len(shard_pairs)),
        chunk_storage_prefix=chunk_storage_prefix
        or marin_temp_bucket(ttl_days=1, prefix="zephyr", source_prefix=output_path),
        name=f"merge-harrier-{os.path.basename(source_name)[:32]}",
        stage_runner_factory=InlineRunner,
    )
    outcome = context.execute(dataset, verbose=True)
    verified_shards, verified_rows = verify_merged_output(output_path, normalized_path)
    outcome_counters = dict(outcome.counters)
    outcome_counters["merge/verified_shards"] = verified_shards
    outcome_counters["merge/verified_rows"] = verified_rows
    return EmbeddingAttrData(
        output_dir=output_path,
        source_key=source_key,
        model_name=HARRIER_REPO,
        model_revision=HARRIER_REVISION,
        embedding_dim=HARRIER_DIM,
        quantization_scale=QUANT_SCALE,
        quantization_range=QUANT_RANGE,
        batch_size=DEFAULT_BATCH_SIZE,
        counters=outcome_counters,
    )


def _merge_source(output_path: str, pair: EmbeddingSourcePair) -> EmbeddingAttrData:
    return merge_embedding_source(
        output_path=output_path,
        source_name=pair.source_name,
        source_key=pair.source_key,
        normalized_path=pair.normalized_path,
        deduplicated_path=pair.deduplicated_path,
        fuzzy_duplicate_path=pair.fuzzy_duplicate_path,
    )


def build_steps(
    *,
    deduplicated_prefix: str = DEDUPLICATED_PREFIX,
    fuzzy_duplicate_prefix: str = FUZZY_DUPLICATE_PREFIX,
    output_prefix: str | None = None,
    source_names: list[str] | None = None,
) -> list[StepSpec]:
    """Build one merge step per selected source."""
    if source_names is None:
        source_names = list(select_sources())
    pairs = discover_source_pairs(
        deduplicated_prefix=deduplicated_prefix,
        fuzzy_duplicate_prefix=fuzzy_duplicate_prefix,
        source_names=source_names,
    )
    resolved_output_prefix = _resolved_prefix(output_prefix) if output_prefix is not None else None
    steps = []
    for pair in pairs:
        step = StepSpec(
            name=f"{MERGED_PREFIX}/{pair.source_name}",
            hash_attrs={
                "normalized_path": pair.normalized_path,
                "deduplicated_path": pair.deduplicated_path,
                "fuzzy_duplicate_path": pair.fuzzy_duplicate_path,
                "model": HARRIER_REPO,
                "revision": HARRIER_REVISION,
                "v": MERGE_VERSION,
            },
            fn=remote(
                lambda output_path, source_pair=pair: _merge_source(output_path, source_pair),
                resources=ResourceConfig(cpu=2, ram="8g", disk="8g"),
                pip_dependency_groups=["datakit"],
            ),
        )
        if resolved_output_prefix is not None:
            step = replace(
                step,
                override_output_path=prefix_join(resolved_output_prefix, f"{pair.source_name}_{step.hash_id}"),
            )
        steps.append(step)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deduplicated-prefix", default=DEDUPLICATED_PREFIX)
    parser.add_argument("--fuzzy-duplicate-prefix", default=FUZZY_DUPLICATE_PREFIX)
    parser.add_argument("--output-prefix")
    parser.add_argument("--sources", help="Comma-separated source names. The default selects all sources.")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_names = None
    if args.sources:
        source_names = [source.strip() for source in args.sources.split(",") if source.strip()]

    configure_logging()
    StepRunner().run(
        build_steps(
            deduplicated_prefix=args.deduplicated_prefix,
            fuzzy_duplicate_prefix=args.fuzzy_duplicate_prefix,
            output_prefix=args.output_prefix,
            source_names=source_names,
        ),
        dry_run=args.dry_run,
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    main()
