# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join the Datakit 10% sample with Harrier, quality, and token attributes."""

import argparse
import json
import logging
import os
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import cache, partial

os.environ["MARIN_PREFIX"] = "s3://marin-us-east-02a/marin"

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact, write_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.attributes import TokenizedAttrData, tokenize_attributes_step
from pydantic import BaseModel
from rigging.filesystem import StoragePath, marin_temp_bucket, open_url
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.readers import InputFileSpec, load_file

from experiments.datakit.cluster.domain.v0.assign import _get_index
from experiments.datakit.cluster.quality.fast_transformer.artifact import QualityScores
from experiments.datakit.embeddings.harrier.pipeline import (
    HARRIER_DIM,
    QUANT_RANGE,
    QUANT_SCALE,
    EmbeddingAttrData,
    dequantize_to_fp32,
)
from experiments.datakit.embeddings.harrier.run import build_steps
from experiments.datakit.reference_pipeline import (
    DEFAULT_SCALE,
    TOKENIZER_BACKEND,
    select_sources,
)
from experiments.datakit.reference_pipeline import (
    TOKENIZER as MARIN_TOKENIZER,
)
from experiments.datakit.reference_pipeline import (
    TOKENIZER_REVISION as MARIN_TOKENIZER_REVISION,
)

MARIN_PREFIX = "s3://marin-us-east-02a/marin"
SAMPLE_ROOT = f"{MARIN_PREFIX}/datakit/sample_10pct_91269634"
MAIN_HARRIER_ROOT = f"{MARIN_PREFIX}/datakit/embed/harrier"
FUZZY_HARRIER_ROOT = f"{MARIN_PREFIX}/datakit/embed/harrier-fuzzy-duplicates"
OUTPUT_ROOT = f"{MARIN_PREFIX}/datakit/samples/sample_10pct_91269634-harrier-two-tokenizers-clusters-quality-v2"
PREFLIGHT_OUTPUT = f"{OUTPUT_ROOT}/preflight.json"
CLUSTER_ROOT = f"{MARIN_PREFIX}/datakit/cluster/domain/v1/harrier-all-sources-10m/train_fe81b456"
NEMOTRON_TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
NEMOTRON_TOKENIZER_REVISION = "624ba927cfbef0427354998700de3d51173c8c04"
NEMOTRON_TOKENIZED_ARTIFACT_VERSION = 3
MARIN_TOKENIZED_ARTIFACT_VERSION = 2
QUALITY_MODEL_VERSION = "pooled-junkgate2"
QUALITY_MODEL_SUFFIX = "/datakit/models/quality/pooled_junkgate2"
MAX_WORKERS = 2_048
OUTPUT_BATCH_ROWS = 65_536
METADATA_WORKERS = 64
WORKER_RESOURCES = ResourceConfig.with_cpu(cpu=4, ram="64g", disk="100g")
COORDINATOR_RESOURCES = ResourceConfig.with_cpu(cpu=2, ram="16g", disk="16g", preemptible=False)
APPENDED_FIELDS = (
    pa.field("embedding", pa.list_(pa.int8(), HARRIER_DIM), nullable=False),
    pa.field("cluster_5000", pa.int32(), nullable=False),
    pa.field("dist_5000", pa.float32(), nullable=False),
    pa.field("domain_id", pa.int32(), nullable=False),
    pa.field("quality_score_pooled_junkgate2", pa.float64()),
    pa.field("nemotron_input_ids", pa.list_(pa.int32())),
    pa.field("marin_input_ids", pa.list_(pa.int32())),
)

logger = logging.getLogger(__name__)


class HarrierTokenSampleData(BaseModel):
    version: str = "v2"
    output_dir: str
    sample_root: str
    main_harrier_root: str
    fuzzy_harrier_root: str
    cluster_root: str
    quality_model_version: str
    nemotron_tokenizer: str
    nemotron_tokenizer_revision: str
    marin_tokenizer: str
    marin_tokenizer_revision: str
    rows: int
    missing_nemotron_rows: int
    missing_marin_rows: int
    missing_quality_rows: int
    sources: int
    shards: int
    counters: dict[str, int | float]


@dataclass(frozen=True)
class SampleFile:
    source: str
    path: str


@dataclass(frozen=True)
class CoPartition:
    source: str
    basename: str
    sample_path: str
    schema_path: str
    main_path: str | None
    fuzzy_path: str | None
    nemotron_path: str | None
    marin_path: str | None
    quality_path: str | None


@dataclass(frozen=True)
class InputPlan:
    sample_files: tuple[SampleFile, ...]
    main_dirs: dict[str, str]
    fuzzy_dirs: dict[str, str]
    nemotron_dirs: dict[str, str]
    marin_dirs: dict[str, str]
    quality_dirs: dict[str, str]
    schema_paths: dict[str, str]


def _quality_output_dirs(sources: list[str], quality_prefix: str) -> dict[str, str]:
    normalized = select_sources(sources)
    return {
        source: (
            StepSpec(
                name=f"datakit/quality/{source}",
                output_path_prefix=quality_prefix,
                deps=[normalize_step],
                hash_attrs={"model_version": QUALITY_MODEL_VERSION, "v": 1},
            ).output_path
        )
        for source, normalize_step in normalized.items()
    }


def tokenized_attr_data(
    source_name: str,
    *,
    tokenizer: str,
    tokenizer_revision: str,
    artifact_version: int | None,
) -> TokenizedAttrData:
    normalize = select_sources([source_name])[source_name]
    step = tokenize_attributes_step(
        name=f"datakit/tokenize/{source_name}",
        train_normalize=normalize,
        tokenizer=tokenizer,
        tokenizer_backend=TOKENIZER_BACKEND,
        tokenizer_revision=tokenizer_revision,
        max_workers=DEFAULT_SCALE.pool.n_workers,
        worker_resources=DEFAULT_SCALE.pool.worker,
    )
    if artifact_version is not None:
        step = replace(step, hash_attrs={**step.hash_attrs, "artifact_version": artifact_version})
    return read_artifact(step.output_path, TokenizedAttrData)


def source_paths(root: str) -> dict[str, list[str]]:
    paths = sorted(str(path) for path in StoragePath(f"{root}/**/*.parquet").glob())
    grouped: dict[str, list[str]] = {}
    for path in paths:
        relative = path.removeprefix(f"{root}/")
        source = relative.split("/outputs/main/", 1)[0]
        grouped.setdefault(source, []).append(path)
    return grouped


def fuzzy_output_dirs() -> dict[str, str]:
    output_dirs = {}
    for path in StoragePath(f"{FUZZY_HARRIER_ROOT}/**/.artifact.json").glob():
        directory = str(path).removesuffix("/.artifact.json")
        relative = directory.removeprefix(f"{FUZZY_HARRIER_ROOT}/")
        source = re.sub(r"_[0-9a-f]{8}$", "", relative)
        if list(StoragePath(f"{directory}/*.parquet").glob()):
            output_dirs[source] = directory
    return output_dirs


def _token_dirs(sources: list[str], tokenizer: str, revision: str, artifact_version: int | None) -> dict[str, str]:
    def resolve(source: str) -> tuple[str, str | None]:
        try:
            artifact = tokenized_attr_data(
                source,
                tokenizer=tokenizer,
                tokenizer_revision=revision,
                artifact_version=artifact_version,
            )
        except FileNotFoundError:
            return source, None
        return source, artifact.output_dirs.get("train")

    with ThreadPoolExecutor(max_workers=32) as pool:
        return {source: directory for source, directory in pool.map(resolve, sources) if directory is not None}


def _quality_dirs(sources: list[str]) -> dict[str, str]:
    artifact_paths = _quality_output_dirs(sources, MARIN_PREFIX)

    def resolve(source: str) -> tuple[str, str | None]:
        try:
            artifact = read_artifact(artifact_paths[source], QualityScores)
        except FileNotFoundError:
            return source, None
        if not artifact.model_dir.rstrip("/").endswith(QUALITY_MODEL_SUFFIX):
            raise ValueError(f"Unexpected quality model for {source}: {artifact.model_dir}")
        return source, artifact.main_output_dir

    with ThreadPoolExecutor(max_workers=32) as pool:
        return {source: directory for source, directory in pool.map(resolve, sources) if directory is not None}


def input_plan(selected_sources: tuple[str, ...] = ()) -> InputPlan:
    sample_by_source = source_paths(SAMPLE_ROOT)
    if selected_sources:
        unknown = set(selected_sources) - set(sample_by_source)
        if unknown:
            raise ValueError(f"Unknown sample sources: {sorted(unknown)}")
        sample_by_source = {source: sample_by_source[source] for source in selected_sources}
    sources = sorted(sample_by_source)
    harrier_steps = {step.name.removeprefix("datakit/embed/harrier/"): step for step in build_steps("unused")}
    main_artifacts = {source: read_artifact(harrier_steps[source].output_path, EmbeddingAttrData) for source in sources}
    if set(main_artifacts) != set(sample_by_source):
        raise ValueError("Sample and main Harrier source sets differ")
    main_dirs = {source: artifact.output_dir for source, artifact in main_artifacts.items()}
    fuzzy_dirs = {source: directory for source, directory in fuzzy_output_dirs().items() if source in sample_by_source}
    unknown_fuzzy_sources = set(fuzzy_dirs) - set(sample_by_source)
    if unknown_fuzzy_sources:
        raise ValueError(f"Unknown fuzzy Harrier sources: {sorted(unknown_fuzzy_sources)}")
    nemotron_dirs = _token_dirs(
        sources,
        NEMOTRON_TOKENIZER,
        NEMOTRON_TOKENIZER_REVISION,
        NEMOTRON_TOKENIZED_ARTIFACT_VERSION,
    )
    marin_dirs = _token_dirs(
        sources,
        MARIN_TOKENIZER,
        MARIN_TOKENIZER_REVISION,
        MARIN_TOKENIZED_ARTIFACT_VERSION,
    )
    quality_dirs = _quality_dirs(sources)
    sample_files = tuple(
        SampleFile(source, path) for source, paths in sorted(sample_by_source.items()) for path in paths
    )
    return InputPlan(
        sample_files=sample_files,
        main_dirs=main_dirs,
        fuzzy_dirs=fuzzy_dirs,
        nemotron_dirs=nemotron_dirs,
        marin_dirs=marin_dirs,
        quality_dirs=quality_dirs,
        schema_paths={source: paths[0] for source, paths in sample_by_source.items()},
    )


def parquet_rows(path: str) -> tuple[str, int]:
    with StoragePath(path).open("rb") as file:
        return path, pq.ParquetFile(file).metadata.num_rows


def _basenames(directory: str | None) -> set[str]:
    if directory is None:
        return set()
    return {os.path.basename(str(path)) for path in StoragePath(f"{directory}/*.parquet").glob()}


def preflight() -> None:
    cluster_files = ("centroids_5000.npy", "lookup_5000_to_40.npy")
    missing_cluster_files = [
        name for name in cluster_files if not StoragePath(f"{CLUSTER_ROOT.rstrip('/')}/{name}").exists()
    ]
    if missing_cluster_files:
        raise FileNotFoundError(f"Missing cluster files at {CLUSTER_ROOT}: {missing_cluster_files}")
    plan = input_plan()
    partitions = co_partitions(plan)
    partitions_by_source: dict[str, list[CoPartition]] = {}
    for partition in partitions:
        partitions_by_source.setdefault(partition.source, []).append(partition)
    sample_by_source: dict[str, list[str]] = {}
    for item in plan.sample_files:
        sample_by_source.setdefault(item.source, []).append(item.path)
    with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as pool:
        sample_rows_by_path = dict(pool.map(parquet_rows, (item.path for item in plan.sample_files)))

    source_stats = []
    for source, sample_paths in sorted(sample_by_source.items()):
        main_basenames = _basenames(plan.main_dirs[source])
        fuzzy_basenames = _basenames(plan.fuzzy_dirs[source]) if source in plan.fuzzy_dirs else set()
        nemotron_basenames = _basenames(plan.nemotron_dirs.get(source))
        marin_basenames = _basenames(plan.marin_dirs.get(source))
        source_partitions = partitions_by_source[source]
        aligned_sample = [
            partition
            for partition in source_partitions
            if (partition.main_path is not None or partition.fuzzy_path is not None)
            and partition.nemotron_path is not None
            and partition.marin_path is not None
        ]
        source_stats.append(
            {
                "source": source,
                "sample_rows": sum(sample_rows_by_path[path] for path in sample_paths),
                "sample_shards": len(sample_paths),
                "main_shards": len(main_basenames),
                "fuzzy_shards": len(fuzzy_basenames),
                "nemotron_shards": len(nemotron_basenames),
                "marin_shards": len(marin_basenames),
                "current_shards_missing_nemotron": sum(
                    partition.nemotron_path is None for partition in source_partitions
                ),
                "current_shards_missing_marin": sum(partition.marin_path is None for partition in source_partitions),
                "current_shards_missing_quality": sum(partition.quality_path is None for partition in source_partitions),
                "aligned_sample_shards": len(aligned_sample),
                "aligned_sample_rows": sum(sample_rows_by_path[partition.sample_path] for partition in aligned_sample),
            }
        )
    sample_rows = sum(int(stat["sample_rows"]) for stat in source_stats)
    aligned_rows = sum(int(stat["aligned_sample_rows"]) for stat in source_stats)
    payload = {
        "sample_root": SAMPLE_ROOT,
        "main_harrier_root": MAIN_HARRIER_ROOT,
        "fuzzy_harrier_root": FUZZY_HARRIER_ROOT,
        "cluster_root": CLUSTER_ROOT,
        "quality_model_version": QUALITY_MODEL_VERSION,
        "output_root": OUTPUT_ROOT,
        "nemotron_tokenizer": NEMOTRON_TOKENIZER,
        "nemotron_tokenizer_revision": NEMOTRON_TOKENIZER_REVISION,
        "marin_tokenizer": MARIN_TOKENIZER,
        "marin_tokenizer_revision": MARIN_TOKENIZER_REVISION,
        "sample_rows": sample_rows,
        "sample_shards": len(plan.sample_files),
        "embedding_shards": sum(int(stat["main_shards"]) + int(stat["fuzzy_shards"]) for stat in source_stats),
        "source_count": len(source_stats),
        "fuzzy_source_count": len(plan.fuzzy_dirs),
        "nemotron_source_count": len(plan.nemotron_dirs),
        "marin_source_count": len(plan.marin_dirs),
        "quality_source_count": len(plan.quality_dirs),
        "aligned_sample_rows": aligned_rows,
        "aligned_sample_fraction": aligned_rows / sample_rows,
        "source_stats": source_stats,
    }
    StoragePath(OUTPUT_ROOT).mkdirs()
    with open_url(PREFLIGHT_OUTPUT, "w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    print(json.dumps({key: value for key, value in payload.items() if key != "source_stats"}, indent=2, sort_keys=True))


def _attribute_path(paths: tuple[str, ...], shard_index: int) -> str | None:
    if shard_index >= len(paths):
        return None
    return paths[shard_index]


def co_partitions(plan: InputPlan) -> tuple[CoPartition, ...]:
    inventories = {
        (source, kind): (
            tuple(sorted(str(path) for path in StoragePath(f"{directory}/*.parquet").glob()))
            if directory is not None
            else ()
        )
        for source in plan.schema_paths
        for kind, directory in (
            ("main", plan.main_dirs.get(source)),
            ("fuzzy", plan.fuzzy_dirs.get(source)),
            ("nemotron", plan.nemotron_dirs.get(source)),
            ("marin", plan.marin_dirs.get(source)),
            ("quality", plan.quality_dirs.get(source)),
        )
    }
    partitions = []
    sample_by_source: dict[str, list[SampleFile]] = {}
    for sample in plan.sample_files:
        sample_by_source.setdefault(sample.source, []).append(sample)
    for source, samples in sorted(sample_by_source.items()):
        for shard_index, sample in enumerate(sorted(samples, key=lambda item: item.path)):
            paths = {
                kind: _attribute_path(inventories[source, kind], shard_index)
                for kind in ("main", "fuzzy", "nemotron", "marin", "quality")
            }
            if paths["main"] is None and paths["fuzzy"] is None:
                raise ValueError(f"No co-partitioned Harrier shard for {source} shard {shard_index}")
            partitions.append(
                CoPartition(
                    source=source,
                    basename=os.path.basename(sample.path),
                    sample_path=sample.path,
                    schema_path=plan.schema_paths[source],
                    main_path=paths["main"],
                    fuzzy_path=paths["fuzzy"],
                    nemotron_path=paths["nemotron"],
                    marin_path=paths["marin"],
                    quality_path=paths["quality"],
                )
            )
    return tuple(partitions)


def _load_sample(partition: CoPartition) -> Iterator[dict]:
    for record in load_file(partition.sample_path):
        yield {"id": str(record["id"]), "record": record, "schema_path": partition.schema_path}


def _load_attribute(partition: CoPartition, path_field: str, columns: list[str], value_field: str) -> Iterator[dict]:
    path = getattr(partition, path_field)
    if path is None:
        return
    for record in load_file(InputFileSpec(path=path, columns=columns)):
        yield {"id": str(record["id"]), value_field: record[value_field]}


def _load_tokens(partition: CoPartition, path_field: str) -> Iterator[dict]:
    path = getattr(partition, path_field)
    if path is None:
        return
    yield from load_file(InputFileSpec(path=path, columns=["id", "chunk_index", "input_ids"]))


def _coalesce_tokens(items: Iterator[dict], _shard: ShardInfo) -> Iterator[dict]:
    current_id = None
    input_ids = []
    next_chunk = 0
    for item in items:
        document_id = str(item["id"])
        if document_id != current_id:
            if current_id is not None:
                yield {"id": current_id, "input_ids": input_ids}
            current_id = document_id
            input_ids = []
            next_chunk = 0
        if int(item.get("chunk_index", next_chunk)) != next_chunk:
            raise ValueError(f"Non-contiguous token chunks for {document_id}")
        input_ids.extend(item["input_ids"])
        next_chunk += 1
    if current_id is not None:
        yield {"id": current_id, "input_ids": input_ids}


def _attach_embedding(left: dict | None, right: dict | None, origin: str) -> dict:
    if left is None:
        raise ValueError("Embedding join received no sample row")
    if left.get("embedding") is None and right is not None:
        return {**left, "embedding": right["embedding"], "embedding_origin": origin}
    return left


def _attach_value(left: dict | None, right: dict | None, source_field: str, output_field: str) -> dict:
    if left is None:
        raise ValueError("Attribute join received no sample row")
    return {**left, output_field: None if right is None else right[source_field]}


def _cluster_columns(embeddings: pa.Array, cluster_root: str) -> tuple[pa.Array, pa.Array, pa.Array]:
    context = _get_index(
        f"{cluster_root.rstrip('/')}/centroids_5000.npy",
        {40: f"{cluster_root.rstrip('/')}/lookup_5000_to_40.npy"},
    )
    quantized = embeddings.values.to_numpy(zero_copy_only=False).reshape(-1, HARRIER_DIM)
    distances, fine_clusters = context["index"].search(dequantize_to_fp32(quantized), 1)
    fine = fine_clusters[:, 0].astype(np.int32, copy=False)
    distance = distances[:, 0].astype(np.float32, copy=False)
    domain = context["lookups"][40][fine]
    return pa.array(fine), pa.array(distance), pa.array(domain)


@cache
def _source_schema(path: str) -> pa.Schema:
    with StoragePath(path).open("rb") as file:
        return pq.ParquetFile(file).schema_arrow


def _output_schema(sample_schema: pa.Schema, cluster_root: str = CLUSTER_ROOT) -> pa.Schema:
    collisions = set(sample_schema.names) & {field.name for field in APPENDED_FIELDS}
    if collisions:
        raise ValueError(f"Sample schema already contains joined fields: {sorted(collisions)}")
    metadata = dict(sample_schema.metadata or {})
    metadata.update(
        {
            b"harrier_main_root": MAIN_HARRIER_ROOT.encode(),
            b"harrier_fuzzy_root": FUZZY_HARRIER_ROOT.encode(),
            b"harrier_quantization_range": str(QUANT_RANGE).encode(),
            b"harrier_quantization_scale": str(QUANT_SCALE).encode(),
            b"harrier_cluster_root": cluster_root.encode(),
            b"quality_model_version": QUALITY_MODEL_VERSION.encode(),
            b"nemotron_tokenizer": NEMOTRON_TOKENIZER.encode(),
            b"nemotron_tokenizer_revision": NEMOTRON_TOKENIZER_REVISION.encode(),
            b"marin_tokenizer": MARIN_TOKENIZER.encode(),
            b"marin_tokenizer_revision": MARIN_TOKENIZER_REVISION.encode(),
        }
    )
    return pa.schema([*sample_schema, *APPENDED_FIELDS], metadata=metadata)


def _assign_batch(items: list[dict], cluster_root: str) -> Iterator[pa.RecordBatch]:
    embeddings = pa.array([item["embedding"] for item in items], type=pa.list_(pa.int8(), HARRIER_DIM))
    cluster_5000, dist_5000, domain_id = _cluster_columns(embeddings, cluster_root)
    quality_scores = pa.array(
        [item["quality_score_pooled_junkgate2"] for item in items],
        type=pa.float64(),
    )
    nemotron_ids = pa.array([item["nemotron_input_ids"] for item in items], type=pa.list_(pa.int32()))
    marin_ids = pa.array([item["marin_input_ids"] for item in items], type=pa.list_(pa.int32()))
    sample_schema = _source_schema(items[0]["schema_path"])
    schema = _output_schema(sample_schema, cluster_root)
    sample_table = pa.Table.from_pylist([item["record"] for item in items], schema=sample_schema)
    table = pa.Table.from_arrays(
        [
            *sample_table.columns,
            embeddings,
            cluster_5000,
            dist_5000,
            domain_id,
            quality_scores,
            nemotron_ids,
            marin_ids,
        ],
        schema=schema,
    )
    origins = [item["embedding_origin"] for item in items]
    counters.pipeline.update_counter("harrier_tokens/rows", len(items))
    counters.pipeline.update_counter("harrier_tokens/main_embeddings", origins.count("main"))
    counters.pipeline.update_counter("harrier_tokens/fuzzy_embeddings", origins.count("fuzzy"))
    counters.pipeline.update_counter("harrier_tokens/missing_nemotron", nemotron_ids.null_count)
    counters.pipeline.update_counter("harrier_tokens/missing_marin", marin_ids.null_count)
    counters.pipeline.update_counter("harrier_tokens/missing_quality", quality_scores.null_count)
    yield from table.to_batches()


def _materialize_dataset(
    partitions: tuple[CoPartition, ...],
    output_root: str,
    cluster_root: str,
) -> Dataset[str]:
    source = Dataset.from_list(list(partitions))
    sample = source.flat_map(_load_sample)
    main = source.flat_map(
        partial(_load_attribute, path_field="main_path", columns=["id", "embedding"], value_field="embedding")
    )
    fuzzy = source.flat_map(
        partial(_load_attribute, path_field="fuzzy_path", columns=["id", "embedding"], value_field="embedding")
    )
    nemotron = source.flat_map(partial(_load_tokens, path_field="nemotron_path")).map_shard(_coalesce_tokens)
    marin = source.flat_map(partial(_load_tokens, path_field="marin_path")).map_shard(_coalesce_tokens)
    quality = source.flat_map(
        partial(_load_attribute, path_field="quality_path", columns=["id", "score"], value_field="score")
    )
    records = (
        sample.sorted_merge_join(
            main,
            left_key=lambda item: item["id"],
            right_key=lambda item: item["id"],
            combiner=partial(_attach_embedding, origin="main"),
            how="left",
        )
        .sorted_merge_join(
            fuzzy,
            left_key=lambda item: item["id"],
            right_key=lambda item: item["id"],
            combiner=partial(_attach_embedding, origin="fuzzy"),
            how="left",
        )
        .filter(lambda item: item.get("embedding") is not None)
    )
    records = (
        records.sorted_merge_join(
            quality,
            left_key=lambda item: item["id"],
            right_key=lambda item: item["id"],
            combiner=partial(
                _attach_value,
                source_field="score",
                output_field="quality_score_pooled_junkgate2",
            ),
            how="left",
        )
        .sorted_merge_join(
            nemotron,
            left_key=lambda item: item["id"],
            right_key=lambda item: item["id"],
            combiner=partial(_attach_value, source_field="input_ids", output_field="nemotron_input_ids"),
            how="left",
        )
        .sorted_merge_join(
            marin,
            left_key=lambda item: item["id"],
            right_key=lambda item: item["id"],
            combiner=partial(_attach_value, source_field="input_ids", output_field="marin_input_ids"),
            how="left",
        )
    )
    output_paths = tuple(f"{output_root.rstrip('/')}/{part.source}/{part.basename}" for part in partitions)

    def output_path(shard_idx: int, total_shards: int, paths: tuple[str, ...] = output_paths) -> str:
        if total_shards != len(paths):
            raise ValueError(f"Expected {len(paths)} co-partitions, got {total_shards}")
        return paths[shard_idx]

    dataset = (
        records.window(OUTPUT_BATCH_ROWS)
        .flat_map(partial(_assign_batch, cluster_root=cluster_root))
        .write_parquet(output_path, skip_existing=True)
    )
    return dataset


def materialize(
    output_root: str,
    max_workers: int,
    selected_sources: tuple[str, ...] = (),
) -> None:
    plan = input_plan(selected_sources)
    partitions = co_partitions(plan)
    dataset = _materialize_dataset(partitions, output_root, CLUSTER_ROOT)
    StoragePath(output_root).mkdirs()
    context = ZephyrContext(
        resources=WORKER_RESOURCES,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=min(max_workers, len(partitions)),
        chunk_storage_prefix=marin_temp_bucket(ttl_days=1, prefix="harrier-token-join", source_prefix=output_root),
        name="materialize-10pct-harrier-clusters-quality-nemotron",
    )
    outcome = context.execute(dataset, verbose=True, map_task_resources=WORKER_RESOURCES)
    artifact = HarrierTokenSampleData(
        output_dir=output_root,
        sample_root=SAMPLE_ROOT,
        main_harrier_root=MAIN_HARRIER_ROOT,
        fuzzy_harrier_root=FUZZY_HARRIER_ROOT,
        cluster_root=CLUSTER_ROOT,
        quality_model_version=QUALITY_MODEL_VERSION,
        nemotron_tokenizer=NEMOTRON_TOKENIZER,
        nemotron_tokenizer_revision=NEMOTRON_TOKENIZER_REVISION,
        marin_tokenizer=MARIN_TOKENIZER,
        marin_tokenizer_revision=MARIN_TOKENIZER_REVISION,
        rows=int(outcome.counters.get("harrier_tokens/rows", 0)),
        missing_nemotron_rows=int(outcome.counters.get("harrier_tokens/missing_nemotron", 0)),
        missing_marin_rows=int(outcome.counters.get("harrier_tokens/missing_marin", 0)),
        missing_quality_rows=int(outcome.counters.get("harrier_tokens/missing_quality", 0)),
        sources=len({partition.source for partition in partitions}),
        shards=len(outcome.results),
        counters=dict(outcome.counters),
    )
    write_artifact(artifact, output_root)
    print(artifact.model_dump_json(indent=2))


def validate(output_root: str) -> None:
    artifact = read_artifact(output_root, HarrierTokenSampleData)
    paths = sorted(str(path) for path in StoragePath(f"{output_root}/**/*.parquet").glob())

    def inspect(path: str) -> int:
        with StoragePath(path).open("rb") as file:
            parquet = pq.ParquetFile(file)
            fields = parquet.schema_arrow
            if fields.names[-len(APPENDED_FIELDS) :] != [field.name for field in APPENDED_FIELDS]:
                raise ValueError(f"Unexpected joined schema at {path}")
            if fields.field("embedding").type != APPENDED_FIELDS[0].type:
                raise ValueError(f"Unexpected embedding type at {path}")
            return parquet.metadata.num_rows

    with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as pool:
        rows = sum(pool.map(inspect, paths))
    if rows != artifact.rows or len(paths) != artifact.shards:
        raise ValueError(f"Output inventory differs: rows={rows}/{artifact.rows}, shards={len(paths)}/{artifact.shards}")
    payload = {"rows": rows, "shards": len(paths), "sources": artifact.sources}
    with open_url(f"{output_root.rstrip('/')}/validation_stats.json", "w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "materialize", "validate"), default="preflight")
    parser.add_argument("--output", default=OUTPUT_ROOT)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args()
    configure_logging(logging.INFO)
    if args.mode == "preflight":
        preflight()
    elif args.mode == "materialize":
        materialize(args.output, args.max_workers, tuple(args.source))
    else:
        validate(args.output)


if __name__ == "__main__":
    main()
