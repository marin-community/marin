# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect GLM-5.2 responses for mlfoundations-dev/hero_run_4_code."""

import argparse
import json
import logging
import os
import socket
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import psutil
import pyarrow.parquet as pq
import requests
from iris.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.setup_scripts import default_setup_script
from iris.cluster.types import CoschedulingConfig, Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device, is_job_finished
from marin.inference.config import DEFAULT_CUDA_VLLM_VERSION
from marin.inference.model_preparation import resolve_model_path
from marin.inference.proxy import _reserve_port
from marin.inference.vllm_server import IsolatedCudaVllm
from rigging.filesystem import StoragePath
from rigging.timing import Duration

logger = logging.getLogger(__name__)

MODEL = "zai-org/GLM-5.2-FP8"
MODEL_REVISION = "ba978f7d347eaf65d22f1a86833408afdb953541"
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
CUDA_COMPILER_REQUIREMENT = "cuda-toolkit[cccl,crt,cudart,nvcc,nvvm]==13.0.2"
MODEL_CACHE_TTL_DAYS = 30
GPUS_PER_NODE = 4
VLLM_REPLICAS = 2
TENSOR_PARALLEL_SIZE = GPUS_PER_NODE * VLLM_REPLICAS
DEFAULT_MAX_MODEL_LEN = 64 * 1024
DEFAULT_MAX_NUM_SEQS = 12
DEFAULT_MAX_TOKENS = 16 * 1024
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_CONCURRENCY = 16
DEFAULT_CHUNK_SIZE = 32
GPU_MEMORY_UTILIZATION = 0.9
ENDPOINT_TIMEOUT = 3 * 3600
REQUEST_TIMEOUT = 3 * 3600
RUN_TIMEOUT_HOURS = 30 * 24
INPUT_COLUMNS = ("id", "instruction_seed", "__original_row_idx")


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    max_tokens: int


@dataclass(frozen=True)
class ServerConfig:
    max_model_len: int
    max_num_seqs: int


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


def _ray_worker_port_args(*excluded_ports: int) -> list[str]:
    for minimum in (20000, 30000, 40000, 50000):
        maximum = minimum + 9999
        if not any(minimum <= port <= maximum for port in excluded_ports):
            return [f"--min-worker-port={minimum}", f"--max-worker-port={maximum}"]
    raise ValueError(f"Could not select Ray worker ports excluding {excluded_ports}")


def _network_interface(host: str) -> str:
    for name, addresses in psutil.net_if_addrs().items():
        if any(address.family == socket.AF_INET and address.address == host for address in addresses):
            return name
    raise RuntimeError(f"No network interface owns advertised host IP {host}")


def _wait_for_http(url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Process exited with code {process.returncode} before {url} became ready")
        try:
            response = requests.get(url, timeout=10)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for {url}")


def _wait_for_endpoint(name: str, job=None, timeout: float = ENDPOINT_TIMEOUT) -> str:
    ctx = iris_ctx()
    assert ctx is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = ctx.resolver.resolve(name)
        if not result.is_empty:
            return result.first().url
        if job is not None and is_job_finished(job.state):
            raise RuntimeError(f"Job {job} finished before registering endpoint {name!r}")
        time.sleep(15)
    raise TimeoutError(f"Timed out waiting for endpoint {name!r}")


def _vllm_process_command() -> tuple[list[str], dict[str, str]]:
    launcher = IsolatedCudaVllm(version=DEFAULT_CUDA_VLLM_VERSION)
    command = launcher.command()
    python_index = command.index("--python")
    command[python_index:python_index] = [
        "--with",
        "ray[cgraph]>=2.55.1",
        "--with",
        CUDA_COMPILER_REQUIREMENT,
    ]
    return command, {**os.environ, **launcher.env()}


def _cuda_overlay(cuda_root: Path) -> str:
    cuda_home = Path(tempfile.mkdtemp(prefix="cuda-home-"))
    for directory in ("bin", "include", "nvvm"):
        (cuda_home / directory).symlink_to(cuda_root / directory, target_is_directory=True)
    lib64 = cuda_home / "lib64"
    lib64.mkdir()
    for library in (cuda_root / "lib").iterdir():
        (lib64 / library.name).symlink_to(library, target_is_directory=library.is_dir())
    cudart = next((cuda_root / "lib").glob("libcudart.so.*"))
    (lib64 / "libcudart.so").symlink_to(cudart)
    return str(cuda_home)


def _cuda_home(vllm_command: list[str], environment: dict[str, str]) -> str:
    script = """\
from pathlib import Path
import sys

nvcc = next(path for entry in sys.path for path in Path(entry).glob("nvidia/**/bin/nvcc"))
print(nvcc.parent.parent)
"""
    command = [*vllm_command[:-1], "python", "-c", script]
    cuda_root = Path(subprocess.check_output(command, env=environment, text=True).strip())
    return _cuda_overlay(cuda_root)


def _serve_glm52(vllm_endpoint: str, ray_endpoint: str, config: ServerConfig) -> None:
    info = get_job_info()
    ctx = iris_ctx()
    if info is None or ctx is None:
        raise RuntimeError("GLM-5.2 serving must run inside an Iris task")

    vllm_command, environment = _vllm_process_command()
    ray_command = [*vllm_command[:-1], "ray"]
    host = info.advertise_host
    environment["CUDA_HOME"] = _cuda_home(vllm_command, environment)
    environment["VLLM_HOST_IP"] = host
    environment["GLOO_SOCKET_IFNAME"] = _network_interface(host)
    if info.task_index == 0:
        weights = resolve_model_path(MODEL, MODEL_CACHE_TTL_DAYS, MODEL_REVISION)
        ray_port = _reserve_port(host, ctx.get_port("ray"))
        http_port = _reserve_port(host, ctx.get_port("http"))
        ray_address = f"{host}:{ray_port}"
        subprocess.run(
            [
                *ray_command,
                "start",
                "--head",
                f"--node-ip-address={host}",
                f"--port={ray_port}",
                *_ray_worker_port_args(ray_port, http_port),
                f"--num-gpus={GPUS_PER_NODE}",
                "--disable-usage-stats",
            ],
            check=True,
            env=environment,
        )
        ray_endpoint_id = ctx.registry.register(ray_endpoint, ray_address)
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                status = subprocess.run(
                    [*ray_command, "status", f"--address={ray_address}"],
                    env=environment,
                    text=True,
                    capture_output=True,
                )
                if status.returncode == 0 and f"/{TENSOR_PARALLEL_SIZE}.0 GPU" in status.stdout:
                    break
                time.sleep(10)
            else:
                raise TimeoutError(f"Ray cluster did not register all {TENSOR_PARALLEL_SIZE} GB200 GPUs")

            process = subprocess.Popen(
                [
                    *vllm_command,
                    "serve",
                    weights,
                    "--served-model-name",
                    MODEL,
                    "--host",
                    host,
                    "--port",
                    str(http_port),
                    "--tensor-parallel-size",
                    str(TENSOR_PARALLEL_SIZE),
                    "--distributed-executor-backend",
                    "ray",
                    "--enable-expert-parallel",
                    "--max-model-len",
                    str(config.max_model_len),
                    "--max-num-seqs",
                    str(config.max_num_seqs),
                    "--gpu-memory-utilization",
                    str(GPU_MEMORY_UTILIZATION),
                    "--trust-remote-code",
                ],
                env={**environment, "RAY_ADDRESS": ray_address},
            )
            try:
                base_url = f"http://{host}:{http_port}"
                _wait_for_http(f"{base_url}/health", process, ENDPOINT_TIMEOUT)
                endpoint_id = ctx.registry.register(vllm_endpoint, base_url)
                try:
                    return_code = process.wait()
                    raise RuntimeError(f"vLLM exited with code {return_code}")
                finally:
                    ctx.registry.unregister(endpoint_id)
            finally:
                if process.poll() is None:
                    process.terminate()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=30)
                if process.poll() is None:
                    process.kill()
        finally:
            ctx.registry.unregister(ray_endpoint_id)
            subprocess.run([*ray_command, "stop", "--force"], env=environment, check=False)
        return

    ray_address = _wait_for_endpoint(ray_endpoint, timeout=ENDPOINT_TIMEOUT)
    subprocess.run(
        [
            *ray_command,
            "start",
            f"--address={ray_address}",
            f"--node-ip-address={host}",
            *_ray_worker_port_args(),
            f"--num-gpus={GPUS_PER_NODE}",
            "--disable-usage-stats",
            "--block",
        ],
        check=True,
        env=environment,
    )


def _submit_vllm(ctx, vllm_endpoint: str, ray_endpoint: str, config: ServerConfig):
    return ctx.client.submit(
        entrypoint=Entrypoint.from_callable(_serve_glm52, vllm_endpoint, ray_endpoint, config),
        name="vllm",
        resources=ResourceSpec(
            cpu=120,
            memory="850g",
            disk="1000g",
            device=gpu_device("GB200", GPUS_PER_NODE),
        ),
        environment=EnvironmentSpec(
            setup_scripts=[default_setup_script(packages=["marin-core"])],
            env_vars={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        ),
        ports=["ray", "http"],
        coscheduling=CoschedulingConfig(group_by="nvlink.domain"),
        replicas=VLLM_REPLICAS,
        timeout=Duration.from_hours(RUN_TIMEOUT_HOURS),
        max_retries_failure=0,
    )


def _read_partition(partition_slice: PartitionSlice) -> list[PromptRecord]:
    partition = partition_slice.partition
    with fsspec.open(partition.url, "rb") as handle:
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


def _completion(vllm_url: str, prompt: PromptRecord, sampling: SamplingConfig) -> dict[str, Any]:
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
    return {
        **asdict(prompt),
        "dataset": DATASET,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "parquet_revision": PARQUET_REVISION,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": sampling.max_tokens,
        "seed": prompt.dataset_index,
        "response": {
            "role": message.get("role", "assistant"),
            "content": message.get("content"),
            "reasoning_content": message.get("reasoning_content"),
        },
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage", {}),
        "response_id": payload.get("id"),
        "elapsed_seconds": time.time() - started,
    }


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
) -> dict[str, Any]:
    return {
        "run_id": collection.run_id,
        "output_path": str(collection.output_path),
        "dataset": DATASET,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "parquet_revision": PARQUET_REVISION,
        "dataset_rows": sum(PARQUET_FILE_ROWS),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "num_shards": collection.num_shards,
        "max_records_per_shard": collection.max_records,
        "chunk_size": collection.chunk_size,
        "concurrency": collection.concurrency,
        "max_model_len": server.max_model_len,
        "max_num_seqs": server.max_num_seqs,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": sampling.max_tokens,
        "enable_thinking": True,
    }


def _ensure_run_config(output_path: StoragePath, config: dict[str, Any]) -> None:
    path = output_path / "run-config.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != config:
            raise ValueError(f"Output root contains a different run config: {path}")
        return
    path.write_text(json.dumps(config, indent=2, sort_keys=True))


def _write_progress(
    collection: CollectionConfig,
    state: str,
    expected_records: int,
    complete_records: int,
    generated_records: int,
    skipped_records: int,
    started: float,
) -> None:
    progress = {
        "run_id": collection.run_id,
        "state": state,
        "shard_index": collection.shard_index,
        "num_shards": collection.num_shards,
        "expected_records": expected_records,
        "complete_records": complete_records,
        "generated_records_this_attempt": generated_records,
        "skipped_records_this_attempt": skipped_records,
        "elapsed_seconds": time.time() - started,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    (collection.output_path / "progress" / f"shard-{collection.shard_index:03d}.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True)
    )


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
        "running",
        expected_records,
        complete_records,
        generated_records,
        skipped_records,
        started,
    )

    with ThreadPoolExecutor(max_workers=collection.concurrency) as executor:
        for partition_slice in slices:
            partition = partition_slice.partition
            pending_chunks = []
            for row_start in range(0, partition_slice.num_rows, collection.chunk_size):
                row_end = min(partition_slice.num_rows, row_start + collection.chunk_size)
                path = _chunk_path(collection.output_path, partition, row_start, row_end)
                if path.exists():
                    complete_records += row_end - row_start
                    skipped_records += row_end - row_start
                    continue
                pending_chunks.append((row_start, row_end, path))
            if not pending_chunks:
                continue

            records = _read_partition(partition_slice)
            for row_start, row_end, path in pending_chunks:
                chunk = records[row_start:row_end]
                outputs = list(executor.map(lambda prompt: _completion(vllm_url, prompt, sampling), chunk))
                path.write_text(
                    "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in outputs),
                    compression="gzip",
                )
                complete_records += len(outputs)
                generated_records += len(outputs)
                _write_progress(
                    collection,
                    "running",
                    expected_records,
                    complete_records,
                    generated_records,
                    skipped_records,
                    started,
                )
                logger.info(
                    "Shard %d saved %d/%d records to %s",
                    collection.shard_index,
                    complete_records,
                    expected_records,
                    path,
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
    weights = resolve_model_path(MODEL, MODEL_CACHE_TTL_DAYS, MODEL_REVISION)
    manifest = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "weights": weights,
        "cache_ttl_days": MODEL_CACHE_TTL_DAYS,
        "prepared_at": datetime.now(UTC).isoformat(),
    }
    (output_path / "model-cache.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
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
    vllm_job = _submit_vllm(ctx, vllm_endpoint, ray_endpoint, server)
    try:
        vllm_url = _wait_for_endpoint(vllm_endpoint, vllm_job)
        logger.info("GLM-5.2 ready; writing responses to %s", collection.output_path)
        _run_collection(vllm_url, collection, server, sampling)
    finally:
        try:
            vllm_job.terminate()
        except Exception:
            logger.warning("Failed to terminate vLLM child job %s during collector cleanup", vllm_job, exc_info=True)


def _positive(parser: argparse.ArgumentParser, name: str, value: int) -> None:
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
    run_parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    run_parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "prepare":
        prepare_model(StoragePath(args.output_path))
        return

    for name in ("num_shards", "chunk_size", "concurrency", "max_model_len", "max_num_seqs", "max_tokens"):
        _positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    if args.max_records is not None:
        _positive(parser, "--max-records", args.max_records)
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
        ServerConfig(max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs),
        SamplingConfig(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens),
    )


if __name__ == "__main__":
    main()
