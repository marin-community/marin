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
from functools import cache

os.environ["MARIN_PREFIX"] = "s3://marin-us-east-02a/marin"

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import xxhash
from fray.types import ResourceConfig
from marin.execution.artifact import read_artifact, read_record, write_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.attributes import TokenizedAttrData, tokenize_attributes_step
from pydantic import BaseModel
from rigging.filesystem import StoragePath, marin_temp_bucket, open_url
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.writers import write_parquet_file

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
OUTPUT_ROOT = f"{MARIN_PREFIX}/datakit/samples/sample_10pct_91269634-harrier-two-tokenizers-clusters-quality-v1"
PREFLIGHT_OUTPUT = f"{OUTPUT_ROOT}/preflight.json"
CLUSTER_ROOT = f"{MARIN_PREFIX}/datakit/cluster/domain/v1/harrier-all-sources-10m/train_fe81b456"
NEMOTRON_TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
NEMOTRON_TOKENIZER_REVISION = "624ba927cfbef0427354998700de3d51173c8c04"
TOKENIZED_ARTIFACT_VERSION = 3
QUALITY_MODEL_VERSION = "pooled-junkgate2"
QUALITY_MODEL_SUFFIX = "/datakit/models/quality/pooled_junkgate2"
JOIN_SHARDS = 16_384
MATERIALIZE_SHARDS = 8_192
MAX_WORKERS = 2_048
READ_BATCH_ROWS = 1_024
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
    version: str = "v1"
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
class EmbeddingIdFile:
    source: str
    path: str


@dataclass(frozen=True)
class InputPlan:
    sample_files: tuple[SampleFile, ...]
    embedding_files: tuple[EmbeddingIdFile, ...]
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


def _sample_token_dirs(sources: list[str], tokenizer: str) -> dict[str, str]:
    candidates: dict[str, list[str]] = {source: [] for source in sources}
    sample_marker = "datakit/sample_10pct_91269634/"
    source_suffix = "/outputs/main"
    for record_path in StoragePath(f"{MARIN_PREFIX}/datakit/tokenize/**/.artifact.json").glob():
        directory = str(record_path).removesuffix("/.artifact.json")
        record = read_record(directory)
        if record is None or not isinstance(record.result, dict):
            continue
        payload = record.result
        output_dirs = payload.get("output_dirs")
        source_keys = payload.get("source_keys")
        if (
            payload.get("tokenizer") != tokenizer
            or not isinstance(output_dirs, dict)
            or "train" not in output_dirs
            or not isinstance(source_keys, dict)
        ):
            continue
        source_key = str(source_keys.get("train", ""))
        if sample_marker not in source_key or not source_key.endswith(source_suffix):
            continue
        source = source_key.split(sample_marker, 1)[1].removesuffix(source_suffix)
        if source in candidates:
            candidates[source].append(f"{directory}/train")
    duplicates = {source: paths for source, paths in candidates.items() if len(paths) > 1}
    if duplicates:
        raise ValueError(f"Multiple sample-specific {tokenizer} caches: {duplicates}")
    return {source: paths[0] for source, paths in candidates.items() if paths}


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
        TOKENIZED_ARTIFACT_VERSION,
    )
    marin_dirs = _sample_token_dirs(sources, MARIN_TOKENIZER)
    quality_dirs = _quality_dirs(sources)
    sample_files = tuple(
        SampleFile(source, path) for source, paths in sorted(sample_by_source.items()) for path in paths
    )
    embedding_files = tuple(
        EmbeddingIdFile(source, path)
        for source in sources
        for directory in (main_dirs[source], fuzzy_dirs.get(source))
        if directory is not None
        for path in sorted(str(item) for item in StoragePath(f"{directory}/*.parquet").glob())
    )
    return InputPlan(
        sample_files=sample_files,
        embedding_files=embedding_files,
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
    sample_by_source: dict[str, list[str]] = {}
    for item in plan.sample_files:
        sample_by_source.setdefault(item.source, []).append(item.path)
    with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as pool:
        sample_rows_by_path = dict(pool.map(parquet_rows, (item.path for item in plan.sample_files)))

    source_stats = []
    for source, sample_paths in sorted(sample_by_source.items()):
        sample_by_basename = {os.path.basename(path): path for path in sample_paths}
        main_basenames = _basenames(plan.main_dirs[source])
        fuzzy_basenames = _basenames(plan.fuzzy_dirs[source]) if source in plan.fuzzy_dirs else set()
        nemotron_basenames = _basenames(plan.nemotron_dirs.get(source))
        marin_basenames = _basenames(plan.marin_dirs.get(source))
        quality_basenames = _basenames(plan.quality_dirs.get(source))
        current_basenames = main_basenames | fuzzy_basenames
        aligned_sample = set(sample_by_basename) & current_basenames & nemotron_basenames & marin_basenames
        source_stats.append(
            {
                "source": source,
                "sample_rows": sum(sample_rows_by_path[path] for path in sample_paths),
                "sample_shards": len(sample_paths),
                "main_shards": len(main_basenames),
                "fuzzy_shards": len(fuzzy_basenames),
                "nemotron_shards": len(nemotron_basenames),
                "marin_shards": len(marin_basenames),
                "current_shards_missing_nemotron": len(current_basenames - nemotron_basenames),
                "current_shards_missing_marin": len(current_basenames - marin_basenames),
                "current_shards_missing_quality": len(current_basenames - quality_basenames),
                "aligned_sample_shards": len(aligned_sample),
                "aligned_sample_rows": sum(sample_rows_by_path[sample_by_basename[name]] for name in aligned_sample),
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
        "embedding_shards": len(plan.embedding_files),
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


def join_key(source: str, document_id: str) -> bytes:
    return xxhash.xxh3_128_digest(f"{source}\0{document_id}".encode())


def _sample_ids(spec: SampleFile) -> Iterator[dict]:
    row_index = 0
    with StoragePath(spec.path).open("rb") as file:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=8_192, columns=["id"]):
            for document_id in pc.cast(batch["id"], pa.string()).to_pylist():
                yield {
                    "join_key": join_key(spec.source, document_id),
                    "source": spec.source,
                    "id": document_id,
                    "sample_path": spec.path,
                    "row_index": row_index,
                }
                row_index += 1


def _embedding_ids(spec: EmbeddingIdFile) -> Iterator[dict]:
    basename = os.path.basename(spec.path)
    with StoragePath(spec.path).open("rb") as file:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=8_192, columns=["id"]):
            for document_id in pc.cast(batch["id"], pa.string()).to_pylist():
                yield {
                    "join_key": join_key(spec.source, document_id),
                    "source": spec.source,
                    "id": document_id,
                    "basename": basename,
                }


def _keep_all(_key: bytes, items: Iterator[dict]) -> Iterator[dict]:
    yield from items


def _keep_first(_key: bytes, items: Iterator[dict]) -> dict:
    return next(items)


def _combine_sample_embedding(sample: dict | None, embedding: dict | None) -> dict:
    if sample is None or embedding is None:
        raise ValueError("Inner join received a missing side")
    if sample["source"] != embedding["source"] or sample["id"] != embedding["id"]:
        raise ValueError("Hashed join key collision")
    return {
        "source": sample["source"],
        "id": sample["id"],
        "sample_path": sample["sample_path"],
        "row_index": sample["row_index"],
        "basename": embedding["basename"],
    }


def _attach_sample_records(
    key: tuple[str, str],
    items: Iterator[dict],
    *,
    marin_dirs: dict[str, str],
) -> Iterator[dict]:
    source, sample_path = key
    basename = os.path.basename(sample_path)
    marin_path = f"{marin_dirs[source].rstrip('/')}/{basename}" if source in marin_dirs else None
    if marin_path is not None and not StoragePath(marin_path).exists():
        marin_path = None
    marin = RowCursor(marin_path, "input_ids")
    targets = iter(items)
    target = next(targets, None)
    row_offset = 0
    try:
        with StoragePath(sample_path).open("rb") as file:
            parquet = pq.ParquetFile(file)
            for batch in parquet.iter_batches(batch_size=READ_BATCH_ROWS):
                records = batch.to_pylist()
                row_limit = row_offset + len(records)
                while target is not None and target["row_index"] < row_limit:
                    if target["row_index"] < row_offset:
                        raise ValueError(f"Unsorted sample row indices for {sample_path}")
                    if target["source"] != source or target["sample_path"] != sample_path:
                        raise ValueError(f"Sample locator differs from group key for {sample_path}")
                    record = records[target["row_index"] - row_offset]
                    if str(record["id"]) != target["id"]:
                        raise ValueError(f"Sample ID differs at row {target['row_index']} in {sample_path}")
                    token_id, input_ids = marin.value(target["row_index"])
                    if token_id is not None and token_id != target["id"]:
                        raise ValueError(f"Marin token ID differs at row {target['row_index']} in {marin_path}")
                    yield {
                        "source": source,
                        "id": target["id"],
                        "record": record,
                        "basename": target["basename"],
                        "marin_input_ids": input_ids,
                    }
                    target = next(targets, None)
                row_offset = row_limit
    finally:
        marin.close()
    if target is not None:
        raise ValueError(f"Sample row {target['row_index']} is outside {sample_path}")


def _parquet_batches(path: str, columns: list[str]) -> Iterator[pa.Table]:
    with StoragePath(path).open("rb") as file:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=READ_BATCH_ROWS, columns=columns):
            yield pa.Table.from_batches([batch])


class RowCursor:
    def __init__(self, path: str | None, column: str):
        self.batches = _parquet_batches(path, ["id", column]) if path is not None else iter(())
        self.column = column
        self.current: pa.Table | None = None
        self.offset = 0

    def value(self, row_index: int) -> tuple[str | None, list[int] | None]:
        while self.current is None or row_index >= self.offset + len(self.current):
            if self.current is not None:
                self.offset += len(self.current)
            self.current = next(self.batches, None)
            if self.current is None:
                return None, None
        if row_index < self.offset:
            raise ValueError(f"Row index {row_index} precedes cursor offset {self.offset}")
        local_index = row_index - self.offset
        document_id = pc.cast(self.current["id"].combine_chunks(), pa.string())[local_index].as_py()
        return document_id, self.current[self.column].combine_chunks()[local_index].as_py()

    def close(self) -> None:
        close = getattr(self.batches, "close", None)
        if close is not None:
            close()


class MatchCursor:
    def __init__(self, path: str | None, columns: list[str]):
        self.batches = _parquet_batches(path, columns) if path is not None else iter(())
        self.current: pa.Table | None = None

    def matches(self, target_ids: pa.Array) -> list[pa.Table]:
        minimum = target_ids[0].as_py()
        maximum = target_ids[-1].as_py()
        matches = []
        while True:
            if self.current is None:
                self.current = next(self.batches, None)
            if self.current is None:
                return matches
            ids = pc.cast(self.current["id"].combine_chunks(), pa.string())
            first = ids[0].as_py()
            last = ids[-1].as_py()
            if last < minimum:
                self.current = None
                continue
            if first > maximum:
                return matches
            mask = pc.is_in(ids, value_set=target_ids)
            if pc.any(mask).as_py():
                matches.append(self.current.filter(mask))
            if last <= maximum:
                self.current = None
                continue
            return matches

    def close(self) -> None:
        close = getattr(self.batches, "close", None)
        if close is not None:
            close()


def _tables_column(tables: list[pa.Table], target_ids: pa.Array, column: str, column_type: pa.DataType) -> pa.Array:
    if not tables:
        return pa.nulls(len(target_ids), type=column_type)
    table = pa.concat_tables(tables)
    ids = pc.cast(table["id"].combine_chunks(), pa.string())
    indices = pc.index_in(target_ids, value_set=ids)
    return pc.cast(pc.take(table[column].combine_chunks(), indices), column_type)


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


def _record_chunks(items: Iterator[dict]) -> Iterator[list[dict]]:
    chunk: list[dict] = []
    for item in items:
        if chunk and len(chunk) >= OUTPUT_BATCH_ROWS and item["id"] != chunk[-1]["id"]:
            yield chunk
            chunk = []
        chunk.append(item)
    if chunk:
        yield chunk


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


def _existing_result(path: str, source: str, basename: str) -> dict | None:
    output = StoragePath(path)
    if not output.exists():
        return None
    with output.open("rb") as file:
        parquet = pq.ParquetFile(file)
        rows = parquet.metadata.num_rows
        if parquet.schema_arrow.names[-len(APPENDED_FIELDS) :] != [field.name for field in APPENDED_FIELDS]:
            raise ValueError(f"Incomplete existing output at {path}")
        missing = {}
        for name in ("nemotron_input_ids", "marin_input_ids", "quality_score_pooled_junkgate2"):
            index = parquet.schema_arrow.get_field_index(name)
            missing[name] = sum(
                parquet.metadata.row_group(row_group).column(index).statistics.null_count
                for row_group in range(parquet.num_row_groups)
            )
    return {
        "source": source,
        "basename": basename,
        "rows": rows,
        "missing_nemotron": missing["nemotron_input_ids"],
        "missing_marin": missing["marin_input_ids"],
        "missing_quality": missing["quality_score_pooled_junkgate2"],
        "reused": True,
    }


def _materialize_group(
    key: tuple[str, str],
    items: Iterator[dict],
    *,
    main_dirs: dict[str, str],
    fuzzy_dirs: dict[str, str],
    nemotron_dirs: dict[str, str],
    quality_dirs: dict[str, str],
    schema_paths: dict[str, str],
    output_root: str,
    cluster_root: str = CLUSTER_ROOT,
) -> dict:
    source, basename = key
    output_path = f"{output_root.rstrip('/')}/{source}/{basename}"
    if existing := _existing_result(output_path, source, basename):
        return existing

    main_path = f"{main_dirs[source].rstrip('/')}/{basename}"
    if not StoragePath(main_path).exists():
        main_path = None
    main = MatchCursor(main_path, ["id", "embedding"])
    fuzzy_path = f"{fuzzy_dirs[source].rstrip('/')}/{basename}" if source in fuzzy_dirs else None
    if fuzzy_path is not None and not StoragePath(fuzzy_path).exists():
        fuzzy_path = None
    fuzzy = MatchCursor(fuzzy_path, ["id", "embedding"])
    nemotron_path = f"{nemotron_dirs[source].rstrip('/')}/{basename}" if source in nemotron_dirs else None
    if nemotron_path is not None and not StoragePath(nemotron_path).exists():
        nemotron_path = None
    nemotron = MatchCursor(nemotron_path, ["id", "input_ids"])
    quality_path = f"{quality_dirs[source].rstrip('/')}/{basename}" if source in quality_dirs else None
    if quality_path is not None and not StoragePath(quality_path).exists():
        quality_path = None
    quality = MatchCursor(quality_path, ["id", "score"])
    sample_schema = _source_schema(schema_paths[source])
    schema = _output_schema(sample_schema, cluster_root)
    stats = {
        "rows": 0,
        "main": 0,
        "fuzzy": 0,
        "missing_nemotron": 0,
        "missing_marin": 0,
        "missing_quality": 0,
    }

    def output_batches() -> Iterator[pa.RecordBatch]:
        try:
            for chunk in _record_chunks(items):
                target_ids = pa.array([item["id"] for item in chunk], type=pa.string())
                main_tables = main.matches(target_ids)
                fuzzy_tables = fuzzy.matches(target_ids)
                main_embeddings = _tables_column(
                    main_tables,
                    target_ids,
                    "embedding",
                    pa.list_(pa.int8(), HARRIER_DIM),
                )
                fuzzy_embeddings = _tables_column(
                    fuzzy_tables,
                    target_ids,
                    "embedding",
                    pa.list_(pa.int8(), HARRIER_DIM),
                )
                embeddings = pc.coalesce(main_embeddings, fuzzy_embeddings)
                if embeddings.null_count:
                    raise ValueError(f"Missing {embeddings.null_count} embeddings for {source}/{basename}")
                main_ids = (
                    pc.cast(pa.concat_tables(main_tables)["id"].combine_chunks(), pa.string())
                    if main_tables
                    else pa.array([], type=pa.string())
                )
                fuzzy_ids = (
                    pc.cast(pa.concat_tables(fuzzy_tables)["id"].combine_chunks(), pa.string())
                    if fuzzy_tables
                    else pa.array([], type=pa.string())
                )
                main_matches = pc.is_in(target_ids, value_set=main_ids)
                fuzzy_matches = pc.is_in(target_ids, value_set=fuzzy_ids)
                fuzzy_fallback_matches = pc.and_(pc.invert(main_matches), fuzzy_matches)
                cluster_5000, dist_5000, domain_id = _cluster_columns(embeddings, cluster_root)
                quality_scores = _tables_column(
                    quality.matches(target_ids),
                    target_ids,
                    "score",
                    pa.float64(),
                )
                nemotron_ids = _tables_column(
                    nemotron.matches(target_ids),
                    target_ids,
                    "input_ids",
                    pa.list_(pa.int32()),
                )
                marin_ids = pa.array([item["marin_input_ids"] for item in chunk], type=pa.list_(pa.int32()))
                sample_table = pa.Table.from_pylist([item["record"] for item in chunk], schema=sample_schema)
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
                stats["rows"] += len(chunk)
                stats["main"] += int(pc.sum(pc.cast(main_matches, pa.int64())).as_py())
                stats["fuzzy"] += int(pc.sum(pc.cast(fuzzy_fallback_matches, pa.int64())).as_py())
                stats["missing_nemotron"] += nemotron_ids.null_count
                stats["missing_marin"] += marin_ids.null_count
                stats["missing_quality"] += quality_scores.null_count
                yield from table.to_batches()
        finally:
            main.close()
            fuzzy.close()
            nemotron.close()
            quality.close()

    result = write_parquet_file(output_batches(), output_path, schema=schema)
    if int(result["count"]) != stats["rows"]:
        raise ValueError(f"Writer count differs for {output_path}")
    counters.pipeline.update_counter("harrier_tokens/rows", stats["rows"])
    counters.pipeline.update_counter("harrier_tokens/main_embeddings", stats["main"])
    counters.pipeline.update_counter("harrier_tokens/fuzzy_embeddings", stats["fuzzy"])
    counters.pipeline.update_counter("harrier_tokens/missing_nemotron", stats["missing_nemotron"])
    counters.pipeline.update_counter("harrier_tokens/missing_marin", stats["missing_marin"])
    counters.pipeline.update_counter("harrier_tokens/missing_quality", stats["missing_quality"])
    return {"source": source, "basename": basename, **stats, "reused": False}


def materialize(
    output_root: str,
    join_shards: int,
    materialize_shards: int,
    max_workers: int,
    selected_sources: tuple[str, ...] = (),
) -> None:
    plan = input_plan(selected_sources)
    sample = (
        Dataset.from_list(list(plan.sample_files))
        .flat_map(_sample_ids)
        .group_by(
            key=lambda item: item["join_key"],
            reducer=_keep_all,
            num_output_shards=join_shards,
        )
    )
    embeddings = (
        Dataset.from_list(list(plan.embedding_files))
        .flat_map(_embedding_ids)
        .group_by(
            key=lambda item: item["join_key"],
            reducer=_keep_first,
            num_output_shards=join_shards,
        )
    )
    matched = sample.sorted_merge_join(
        embeddings,
        left_key=lambda item: item["join_key"],
        right_key=lambda item: item["join_key"],
        combiner=_combine_sample_embedding,
        how="inner",
    )
    records = matched.group_by(
        key=lambda item: (item["source"], item["sample_path"]),
        sort_by=lambda item: item["row_index"],
        reducer=lambda key, items: _attach_sample_records(key, items, marin_dirs=plan.marin_dirs),
        num_output_shards=join_shards,
    )
    dataset = records.group_by(
        key=lambda item: (item["source"], item["basename"]),
        sort_by=lambda item: item["id"],
        reducer=lambda key, items: _materialize_group(
            key,
            items,
            main_dirs=plan.main_dirs,
            fuzzy_dirs=plan.fuzzy_dirs,
            nemotron_dirs=plan.nemotron_dirs,
            quality_dirs=plan.quality_dirs,
            schema_paths=plan.schema_paths,
            output_root=output_root,
        ),
        num_output_shards=materialize_shards,
    )
    StoragePath(output_root).mkdirs()
    context = ZephyrContext(
        resources=WORKER_RESOURCES,
        coordinator_resources=COORDINATOR_RESOURCES,
        max_workers=max_workers,
        chunk_storage_prefix=marin_temp_bucket(ttl_days=1, prefix="harrier-token-join", source_prefix=output_root),
        name="materialize-10pct-harrier-clusters-quality-nemotron",
    )
    outcome = context.execute(dataset, verbose=True, map_task_resources=WORKER_RESOURCES)
    reused = [result for result in outcome.results if result["reused"]]
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
        rows=sum(int(result["rows"]) for result in outcome.results),
        missing_nemotron_rows=sum(int(result["missing_nemotron"]) for result in outcome.results),
        missing_marin_rows=sum(int(result["missing_marin"]) for result in outcome.results),
        missing_quality_rows=sum(int(result["missing_quality"]) for result in outcome.results),
        sources=len({result["source"] for result in outcome.results}),
        shards=len(outcome.results),
        counters={**dict(outcome.counters), "reused_shards": len(reused)},
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
    parser.add_argument("--join-shards", type=int, default=JOIN_SHARDS)
    parser.add_argument("--materialize-shards", type=int, default=MATERIALIZE_SHARDS)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--source", action="append", default=[])
    args = parser.parse_args()
    configure_logging(logging.INFO)
    if args.mode == "preflight":
        preflight()
    elif args.mode == "materialize":
        materialize(args.output, args.join_shards, args.materialize_shards, args.max_workers, tuple(args.source))
    else:
        validate(args.output)


if __name__ == "__main__":
    main()
