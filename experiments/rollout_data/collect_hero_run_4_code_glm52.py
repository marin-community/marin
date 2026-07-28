# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect GLM-5.2 responses for mlfoundations-dev/hero_run_4_code."""

import argparse
import json
import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import pyarrow.parquet as pq
import requests
from iris.client import iris_ctx
from rigging.filesystem import StoragePath

from experiments.rollout_data.glm52_vllm import (
    GPU_MEMORY_UTILIZATION,
    MODEL,
    MODEL_CACHE_TTL_DAYS,
    MODEL_REVISION,
    Glm52LaunchConfig,
    ServerConfig,
    prepare_model_cache,
    submit_glm52,
    wait_for_endpoint_url,
)

logger = logging.getLogger(__name__)

DATASET = "mlfoundations-dev/hero_run_4_code"
DATASET_SPLIT = "train"
DATASET_REVISION = "18bef55db16de0a0f7416c200697a201e82e6ff5"
PARQUET_REVISION = "8982cc7d8d510777b546c5ee5d6196cefb97e25a"
PARQUET_FILE_ROWS = (
    19185,
    19185,
    19185,
    19185,
    19185,
    19185,
    19185,
    19185,
    19185,
    19185,
    19185,
    19185,
    19185,
    19185,
    19185,
    19185,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
    19184,
)
PARQUET_ROW_GROUP_SIZE = 1000
PARQUET_URL = (
    "https://huggingface.co/datasets/"
    f"{DATASET}/resolve/{PARQUET_REVISION}/default/{DATASET_SPLIT}/{{file_index:04d}}.parquet"
)
VLLM_ENDPOINT = "glm52-openai"
RAY_ENDPOINT = "glm52-ray"
DEFAULT_MAX_MODEL_LEN = 64 * 1024
DEFAULT_MAX_NUM_SEQS = 12
DEFAULT_MAX_TOKENS = 16 * 1024
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_CONCURRENCY = 16
DEFAULT_CHUNK_SIZE = 32
REQUEST_TIMEOUT = 3 * 3600
INPUT_COLUMNS = ("id", "instruction_seed", "__original_row_idx")
PROGRESS_RUNNING = "running"


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    max_tokens: int


@dataclass(frozen=True)
class CollectionConfig:
    run_id: str
    output_path: StoragePath
    shard_index: int
    num_shards: int
    max_records: int | None
    chunk_size: int
    concurrency: int


@dataclass(frozen=True)
class PartitionSpec:
    ordinal: int
    file_index: int
    row_group_index: int
    row_start: int
    num_rows: int

    @property
    def url(self) -> str:
        return PARQUET_URL.format(file_index=self.file_index)


@dataclass(frozen=True)
class PartitionSlice:
    partition: PartitionSpec
    num_rows: int


@dataclass(frozen=True)
class PromptRecord:
    dataset_index: int
    source_file_index: int
    source_row_group: int
    source_row_offset: int
    id: str | None
    original_row_index: int
    instruction_seed: str


@dataclass(frozen=True)
class CollectionDelta:
    complete_records: int
    generated_records: int
    skipped_records: int
    output_path: StoragePath | None


@dataclass(frozen=True)
class ResponseMessage:
    role: str
    content: str | None
    reasoning_content: str | None


@dataclass(frozen=True)
class CompletionRecord:
    dataset_index: int
    source_file_index: int
    source_row_group: int
    source_row_offset: int
    id: str | None
    original_row_index: int
    instruction_seed: str
    dataset: str
    dataset_split: str
    dataset_revision: str
    parquet_revision: str
    model: str
    model_revision: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    response: ResponseMessage
    finish_reason: str | None
    usage: dict[str, Any]
    response_id: str | None
    elapsed_seconds: float


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    output_path: str
    dataset: str
    dataset_split: str
    dataset_revision: str
    parquet_revision: str
    dataset_rows: int
    model: str
    model_revision: str
    num_shards: int
    max_records_per_shard: int | None
    chunk_size: int
    concurrency: int
    max_model_len: int
    max_num_seqs: int
    kv_cache_dtype: str
    decode_context_parallel_size: int
    gpu_memory_utilization: float
    temperature: float
    top_p: float
    max_tokens: int
    enable_thinking: bool


@dataclass(frozen=True)
class ProgressRecord:
    run_id: str
    state: str
    shard_index: int
    num_shards: int
    expected_records: int
    complete_records: int
    generated_records_this_attempt: int
    skipped_records_this_attempt: int
    elapsed_seconds: float
    updated_at: str


@dataclass(frozen=True)
class ModelCacheManifest:
    model: str
    model_revision: str
    weights: str
    cache_ttl_days: int
    prepared_at: str


def partition_specs() -> list[PartitionSpec]:
    partitions = []
    row_start = 0
    for file_index, file_rows in enumerate(PARQUET_FILE_ROWS):
        for row_group_index, group_start in enumerate(range(0, file_rows, PARQUET_ROW_GROUP_SIZE)):
            num_rows = min(PARQUET_ROW_GROUP_SIZE, file_rows - group_start)
            partitions.append(
                PartitionSpec(
                    ordinal=len(partitions),
                    file_index=file_index,
                    row_group_index=row_group_index,
                    row_start=row_start + group_start,
                    num_rows=num_rows,
                )
            )
        row_start += file_rows
    return partitions


def partition_slices(
    partitions: list[PartitionSpec],
    shard_index: int,
    num_shards: int,
    max_records: int | None,
) -> list[PartitionSlice]:
    selected = [partition for partition in partitions if partition.ordinal % num_shards == shard_index]
    if max_records is None:
        return [PartitionSlice(partition, partition.num_rows) for partition in selected]

    remaining = max_records
    slices = []
    for partition in selected:
        if remaining == 0:
            break
        num_rows = min(partition.num_rows, remaining)
        slices.append(PartitionSlice(partition, num_rows))
        remaining -= num_rows
    return slices


def _read_partition(partition_slice: PartitionSlice) -> list[PromptRecord]:
    partition = partition_slice.partition
    with StoragePath(partition.url).open("rb") as handle:
        table = pq.ParquetFile(handle).read_row_group(partition.row_group_index, columns=list(INPUT_COLUMNS))
    rows = table.slice(0, partition_slice.num_rows).to_pylist()
    records = []
    for row_offset, row in enumerate(rows):
        instruction_seed = row["instruction_seed"]
        if not isinstance(instruction_seed, str) or not instruction_seed:
            raise ValueError(f"Dataset row {partition.row_start + row_offset} has no instruction_seed")
        original_row_index = row["__original_row_idx"]
        if not isinstance(original_row_index, int):
            raise ValueError(f"Dataset row {partition.row_start + row_offset} has no original row index")
        records.append(
            PromptRecord(
                dataset_index=partition.row_start + row_offset,
                source_file_index=partition.file_index,
                source_row_group=partition.row_group_index,
                source_row_offset=row_offset,
                id=row["id"],
                original_row_index=original_row_index,
                instruction_seed=instruction_seed,
            )
        )
    return records


def _completion(vllm_url: str, prompt: PromptRecord, sampling: SamplingConfig) -> CompletionRecord:
    started = time.time()
    response = requests.post(
        f"{vllm_url}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt.instruction_seed}],
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "seed": prompt.dataset_index,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(f"vLLM returned {response.status_code}: {response.text[:2000]}")
    payload = response.json()
    choice = payload["choices"][0]
    message = choice["message"]
    return CompletionRecord(
        **asdict(prompt),
        dataset=DATASET,
        dataset_split=DATASET_SPLIT,
        dataset_revision=DATASET_REVISION,
        parquet_revision=PARQUET_REVISION,
        model=MODEL,
        model_revision=MODEL_REVISION,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
        seed=prompt.dataset_index,
        response=ResponseMessage(
            role=message.get("role", "assistant"),
            content=message.get("content"),
            reasoning_content=message.get("reasoning_content"),
        ),
        finish_reason=choice.get("finish_reason"),
        usage=payload.get("usage", {}),
        response_id=payload.get("id"),
        elapsed_seconds=time.time() - started,
    )


def _chunk_path(output_path: StoragePath, partition: PartitionSpec, row_start: int, row_end: int) -> StoragePath:
    return (
        output_path
        / "responses"
        / f"file-{partition.file_index:04d}"
        / f"row-group-{partition.row_group_index:02d}"
        / f"rows-{row_start:04d}-{row_end - 1:04d}.jsonl.gz"
    )


def _run_config(
    collection: CollectionConfig,
    server: ServerConfig,
    sampling: SamplingConfig,
) -> RunConfig:
    return RunConfig(
        run_id=collection.run_id,
        output_path=str(collection.output_path),
        dataset=DATASET,
        dataset_split=DATASET_SPLIT,
        dataset_revision=DATASET_REVISION,
        parquet_revision=PARQUET_REVISION,
        dataset_rows=sum(PARQUET_FILE_ROWS),
        model=MODEL,
        model_revision=MODEL_REVISION,
        num_shards=collection.num_shards,
        max_records_per_shard=collection.max_records,
        chunk_size=collection.chunk_size,
        concurrency=collection.concurrency,
        max_model_len=server.max_model_len,
        max_num_seqs=server.max_num_seqs,
        kv_cache_dtype=server.kv_cache_dtype,
        decode_context_parallel_size=server.decode_context_parallel_size,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
        enable_thinking=True,
    )


def _ensure_run_config(output_path: StoragePath, config: RunConfig) -> None:
    path = output_path / "run-config.json"
    expected = asdict(config)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != expected:
            raise ValueError(f"Output root contains a different run config: {path}")
        return
    path.write_text(json.dumps(expected, indent=2, sort_keys=True))


def _write_progress(
    collection: CollectionConfig,
    state: str,
    expected_records: int,
    complete_records: int,
    generated_records: int,
    skipped_records: int,
    started: float,
) -> None:
    progress = ProgressRecord(
        run_id=collection.run_id,
        state=state,
        shard_index=collection.shard_index,
        num_shards=collection.num_shards,
        expected_records=expected_records,
        complete_records=complete_records,
        generated_records_this_attempt=generated_records,
        skipped_records_this_attempt=skipped_records,
        elapsed_seconds=time.time() - started,
        updated_at=datetime.now(UTC).isoformat(),
    )
    (collection.output_path / "progress" / f"shard-{collection.shard_index:03d}.json").write_text(
        json.dumps(asdict(progress), indent=2, sort_keys=True)
    )


def _collect_partition(
    executor: ThreadPoolExecutor,
    vllm_url: str,
    collection: CollectionConfig,
    sampling: SamplingConfig,
    partition_slice: PartitionSlice,
) -> Iterator[CollectionDelta]:
    partition = partition_slice.partition
    pending_chunks = []
    skipped_records = 0
    for row_start in range(0, partition_slice.num_rows, collection.chunk_size):
        row_end = min(partition_slice.num_rows, row_start + collection.chunk_size)
        path = _chunk_path(collection.output_path, partition, row_start, row_end)
        if path.exists():
            skipped_records += row_end - row_start
            continue
        pending_chunks.append((row_start, row_end, path))
    if skipped_records:
        yield CollectionDelta(skipped_records, 0, skipped_records, None)
    if not pending_chunks:
        return

    records = _read_partition(partition_slice)
    for row_start, row_end, path in pending_chunks:
        chunk = records[row_start:row_end]
        outputs = list(executor.map(lambda prompt: _completion(vllm_url, prompt, sampling), chunk))
        path.write_text(
            "".join(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n" for record in outputs),
            compression="gzip",
        )
        yield CollectionDelta(len(outputs), len(outputs), 0, path)


def _run_collection(
    vllm_url: str,
    collection: CollectionConfig,
    server: ServerConfig,
    sampling: SamplingConfig,
) -> None:
    config = _run_config(collection, server, sampling)
    _ensure_run_config(collection.output_path, config)
    slices = partition_slices(
        partition_specs(),
        collection.shard_index,
        collection.num_shards,
        collection.max_records,
    )
    expected_records = sum(partition_slice.num_rows for partition_slice in slices)
    complete_records = 0
    generated_records = 0
    skipped_records = 0
    started = time.time()
    _write_progress(
        collection,
        PROGRESS_RUNNING,
        expected_records,
        complete_records,
        generated_records,
        skipped_records,
        started,
    )

    with ThreadPoolExecutor(max_workers=collection.concurrency) as executor:
        for partition_slice in slices:
            for delta in _collect_partition(executor, vllm_url, collection, sampling, partition_slice):
                complete_records += delta.complete_records
                generated_records += delta.generated_records
                skipped_records += delta.skipped_records
                _write_progress(
                    collection,
                    PROGRESS_RUNNING,
                    expected_records,
                    complete_records,
                    generated_records,
                    skipped_records,
                    started,
                )
                if delta.output_path is not None:
                    logger.info(
                        "Shard %d saved %d/%d records to %s",
                        collection.shard_index,
                        complete_records,
                        expected_records,
                        delta.output_path,
                    )

    _write_progress(
        collection,
        "complete",
        expected_records,
        complete_records,
        generated_records,
        skipped_records,
        started,
    )
    if complete_records != expected_records:
        raise RuntimeError(f"Shard completed {complete_records} of {expected_records} expected records")


def prepare_model(output_path: StoragePath) -> None:
    weights = prepare_model_cache()
    manifest = ModelCacheManifest(
        model=MODEL,
        model_revision=MODEL_REVISION,
        weights=weights,
        cache_ttl_days=MODEL_CACHE_TTL_DAYS,
        prepared_at=datetime.now(UTC).isoformat(),
    )
    (output_path / "model-cache.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    logger.info("Prepared model cache at %s", weights)


def run(
    collection: CollectionConfig,
    server: ServerConfig,
    sampling: SamplingConfig,
) -> None:
    ctx = iris_ctx()
    if ctx is None or ctx.client is None:
        raise RuntimeError("Collector must run inside an Iris job")
    endpoint_suffix = f"{collection.run_id}-s{collection.shard_index}"
    vllm_endpoint = f"{VLLM_ENDPOINT}-{endpoint_suffix}"
    ray_endpoint = f"{RAY_ENDPOINT}-{endpoint_suffix}"
    vllm_job = submit_glm52(ctx, Glm52LaunchConfig(vllm_endpoint, ray_endpoint, server))
    try:
        vllm_url = wait_for_endpoint_url(vllm_endpoint, vllm_job)
        logger.info("GLM-5.2 ready; writing responses to %s", collection.output_path)
        _run_collection(vllm_url, collection, server, sampling)
    finally:
        try:
            vllm_job.terminate()
        except Exception:
            logger.warning("Failed to terminate vLLM child job %s during collector cleanup", vllm_job, exc_info=True)


def _validate_positive(parser: argparse.ArgumentParser, name: str, value: int) -> None:
    if value < 1:
        parser.error(f"{name} must be positive")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-path", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--output-path", required=True)
    run_parser.add_argument("--shard-index", type=int, required=True)
    run_parser.add_argument("--num-shards", type=int, required=True)
    run_parser.add_argument("--max-records", type=int)
    run_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    run_parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    run_parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    run_parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    run_parser.add_argument("--kv-cache-dtype", default=ServerConfig.kv_cache_dtype)
    run_parser.add_argument(
        "--decode-context-parallel-size",
        type=int,
        default=ServerConfig.decode_context_parallel_size,
    )
    run_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    run_parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "prepare":
        prepare_model(StoragePath(args.output_path))
        return

    for name in (
        "num_shards",
        "chunk_size",
        "concurrency",
        "max_model_len",
        "max_num_seqs",
        "decode_context_parallel_size",
        "max_tokens",
    ):
        _validate_positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    if args.max_records is not None:
        _validate_positive(parser, "--max-records", args.max_records)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.temperature < 0:
        parser.error("--temperature must not be negative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")

    run(
        CollectionConfig(
            run_id=args.run_id,
            output_path=StoragePath(args.output_path),
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            max_records=args.max_records,
            chunk_size=args.chunk_size,
            concurrency=args.concurrency,
        ),
        ServerConfig(
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            kv_cache_dtype=args.kv_cache_dtype,
            decode_context_parallel_size=args.decode_context_parallel_size,
        ),
        SamplingConfig(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens),
    )


if __name__ == "__main__":
    main()
