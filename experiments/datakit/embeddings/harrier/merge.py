# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge co-partitioned Harrier document sets in normalized row order."""

import argparse
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace

import pyarrow as pa
import pyarrow.compute as pc
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
MERGE_VERSION = 2
WORKERS_PER_SOURCE = 32
MAX_CONCURRENT = 8
MAX_POOL_WORKERS = WORKERS_PER_SOURCE * MAX_CONCURRENT
MERGE_WORKER_RESOURCES = ResourceConfig.with_cpu(cpu=1, ram="4g", disk="8g")
MERGE_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="4g", preemptible=False)


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
                f"Embedding source mismatch for {source_name}: " f"{deduplicated.source_key!r} != {fuzzy_source_key!r}"
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


def _embeddings_by_id(tables: Iterator[pa.Table], duplicate_counter_name: str) -> Iterator[tuple[str, pa.Array]]:
    previous = None
    for table in tables:
        ids = table.column("id").combine_chunks()
        encoded_ids = pc.run_end_encode(ids)
        run_ends = encoded_ids.run_ends.to_pylist()
        first_indices = pa.array([0, *run_ends[:-1]], type=pa.int64())
        embeddings = table.column("embedding").combine_chunks().take(first_indices)
        run_start = 0
        for index, (document_id_scalar, run_end) in enumerate(zip(encoded_ids.values, run_ends, strict=True)):
            document_id = document_id_scalar.as_py()
            duplicate_count = run_end - run_start - 1
            run_start = run_end
            if previous is not None and document_id == previous[0]:
                counters.pipeline.update_counter(duplicate_counter_name, duplicate_count + 1)
                continue
            if previous is not None:
                yield previous
            if duplicate_count:
                counters.pipeline.update_counter(duplicate_counter_name, duplicate_count)
            previous = document_id, embeddings.slice(index, 1)

    if previous is not None:
        yield previous


def _selected_embeddings(
    deduplicated_tables: Iterator[pa.Table],
    fuzzy_duplicate_tables: Iterator[pa.Table],
) -> Iterator[tuple[str, pa.Array]]:
    deduplicated_embeddings = _embeddings_by_id(deduplicated_tables, "merge/deduplicated_duplicate_docs")
    fuzzy_duplicate_embeddings = _embeddings_by_id(fuzzy_duplicate_tables, "merge/fuzzy_duplicate_duplicate_docs")
    deduplicated = next(deduplicated_embeddings, None)
    fuzzy_duplicate = next(fuzzy_duplicate_embeddings, None)

    while deduplicated is not None and fuzzy_duplicate is not None:
        if deduplicated[0] <= fuzzy_duplicate[0]:
            selected = deduplicated
            deduplicated = next(deduplicated_embeddings, None)
            if selected[0] == fuzzy_duplicate[0]:
                counters.pipeline.update_counter("merge/overlapping_ids", 1)
                fuzzy_duplicate = next(fuzzy_duplicate_embeddings, None)
        else:
            selected = fuzzy_duplicate
            fuzzy_duplicate = next(fuzzy_duplicate_embeddings, None)
        yield selected

    if deduplicated is not None:
        yield deduplicated
        yield from deduplicated_embeddings
    elif fuzzy_duplicate is not None:
        yield fuzzy_duplicate
        yield from fuzzy_duplicate_embeddings


def _merge_in_normalized_order(
    deduplicated_tables: Iterator[pa.Table],
    fuzzy_duplicate_tables: Iterator[pa.Table],
    normalized_tables: Iterator[pa.Table],
) -> Iterator[pa.RecordBatch]:
    embeddings = _selected_embeddings(deduplicated_tables, fuzzy_duplicate_tables)
    embedding = next(embeddings, None)
    previous = None

    for normalized_table in normalized_tables:
        normalized_ids = normalized_table.column("id").combine_chunks()
        encoded_ids = pc.run_end_encode(normalized_ids)
        lookup_embeddings = []
        for normalized_id_scalar in encoded_ids.values:
            normalized_id = normalized_id_scalar.as_py()
            if previous is not None and normalized_id == previous[0]:
                lookup_embeddings.append(previous[1])
                continue
            if embedding is not None and embedding[0] < normalized_id:
                raise ValueError(f"Harrier inputs have extra ID {embedding[0]!r}")
            if embedding is None or embedding[0] > normalized_id:
                raise ValueError(f"Normalized ID {normalized_id!r} has no Harrier embedding")
            previous = embedding
            lookup_embeddings.append(embedding[1])
            embedding = next(embeddings, None)

        unique_embeddings = pa.concat_arrays(lookup_embeddings)
        run_indices = pa.RunEndEncodedArray.from_arrays(
            encoded_ids.run_ends,
            pa.array(range(len(encoded_ids.values)), type=pa.int32()),
        )
        output_table = pa.Table.from_arrays(
            [normalized_ids, pc.take(unique_embeddings, pc.run_end_decode(run_indices))],
            schema=EMBEDDING_SCHEMA,
        )
        yield from output_table.to_batches(max_chunksize=DEFAULT_BATCH_SIZE)

    if embedding is not None:
        raise ValueError(f"Harrier inputs have extra ID {embedding[0]!r}")


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
    source_key: str,
    normalized_path: str,
    deduplicated_path: str,
    fuzzy_duplicate_path: str,
    zephyr_context: ZephyrContext,
) -> EmbeddingAttrData:
    """Merge one source in normalized order, preferring deduplicated embeddings on ID overlap."""
    shard_pairs = _embedding_shard_pairs(normalized_path, deduplicated_path, fuzzy_duplicate_path)
    output_basenames = tuple(pair.basename for pair in shard_pairs)

    def _output_path(shard_index: int, _total: int, basenames: tuple[str, ...] = output_basenames) -> str:
        return str(StoragePath(output_path) / basenames[shard_index])

    dataset = (
        Dataset.from_list(shard_pairs)
        .map_shard(_merge_embedding_shard)
        .write_parquet(_output_path, schema=EMBEDDING_SCHEMA, skip_existing=True)
    )
    outcome = zephyr_context.execute(dataset, verbose=True)
    return EmbeddingAttrData(
        output_dir=output_path,
        source_key=source_key,
        model_name=HARRIER_REPO,
        model_revision=HARRIER_REVISION,
        embedding_dim=HARRIER_DIM,
        quantization_scale=QUANT_SCALE,
        quantization_range=QUANT_RANGE,
        batch_size=DEFAULT_BATCH_SIZE,
        counters=dict(outcome.counters),
    )


def _merge_source(
    output_path: str,
    pair: EmbeddingSourcePair,
    zephyr_context: ZephyrContext | None,
) -> EmbeddingAttrData:
    if zephyr_context is None:
        raise ValueError("A Zephyr context is required to run a Harrier merge source")
    return merge_embedding_source(
        output_path=output_path,
        source_key=pair.source_key,
        normalized_path=pair.normalized_path,
        deduplicated_path=pair.deduplicated_path,
        fuzzy_duplicate_path=pair.fuzzy_duplicate_path,
        zephyr_context=zephyr_context,
    )


def build_steps(
    *,
    deduplicated_prefix: str = DEDUPLICATED_PREFIX,
    fuzzy_duplicate_prefix: str = FUZZY_DUPLICATE_PREFIX,
    output_prefix: str | None = None,
    source_names: list[str] | None = None,
    zephyr_context: ZephyrContext | None = None,
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
                lambda output_path, source_pair=pair: _merge_source(output_path, source_pair, zephyr_context),
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
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help="Maximum source steps and shared-pool pipelines that can run at one time.",
    )
    parser.add_argument(
        "--pool-workers",
        type=int,
        help=f"Shared Zephyr workers. The default is {WORKERS_PER_SOURCE} per source, capped at {MAX_POOL_WORKERS}.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be at least 1")
    if args.pool_workers is not None and args.pool_workers < 1:
        parser.error("--pool-workers must be at least 1")

    if args.sources is None:
        source_names = list(select_sources())
    else:
        source_names = [source.strip() for source in args.sources.split(",") if source.strip()]
        if not source_names:
            parser.error("--sources must contain at least one source")

    configure_logging()
    if args.dry_run:
        steps = build_steps(
            deduplicated_prefix=args.deduplicated_prefix,
            fuzzy_duplicate_prefix=args.fuzzy_duplicate_prefix,
            output_prefix=args.output_prefix,
            source_names=source_names,
        )
        StepRunner().run(steps, dry_run=True, max_concurrent=args.max_concurrent)
        return

    pool_workers = args.pool_workers or min(MAX_POOL_WORKERS, WORKERS_PER_SOURCE * len(source_names))
    resolved_output_prefix = _resolved_prefix(args.output_prefix or MERGED_PREFIX)
    zephyr_context = ZephyrContext(
        name="merge-harrier",
        resources=MERGE_WORKER_RESOURCES,
        coordinator_resources=MERGE_COORDINATOR_RESOURCES,
        max_workers=pool_workers,
        max_concurrent_pipelines=args.max_concurrent * 2,
        chunk_storage_prefix=marin_temp_bucket(
            ttl_days=1,
            prefix="zephyr",
            source_prefix=resolved_output_prefix,
        ),
        stage_runner_factory=InlineRunner,
    )
    steps = build_steps(
        deduplicated_prefix=args.deduplicated_prefix,
        fuzzy_duplicate_prefix=args.fuzzy_duplicate_prefix,
        output_prefix=args.output_prefix,
        source_names=source_names,
        zephyr_context=zephyr_context,
    )
    with zephyr_context:
        StepRunner().run(steps, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
