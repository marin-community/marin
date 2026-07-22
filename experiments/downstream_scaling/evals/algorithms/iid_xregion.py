# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IID completion algorithm for cross-region Zephyr workers."""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext

from experiments.downstream_scaling.evals.framework.schema import (
    completions_file,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.framework.xregion import ledger
from experiments.downstream_scaling.evals.framework.xregion import pool as xregion_pool
from experiments.downstream_scaling.evals.framework.xregion.pool import WorkerPoolConfig
from experiments.downstream_scaling.evals.utils import discover_hf_checkpoints, localize_mirror_path, version_path

logger = logging.getLogger(__name__)

VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
    # Bound CPU staging when several engines load concurrently on one VM.
    "RUNAI_STREAMER_MEMORY_LIMIT": "4294967296",
}

DEFAULT_HEARTBEAT_TIMEOUT = 2 * 60
DEFAULT_LEDGER_PREFIX = "gs://marin-us-central2"
VLLM_CONSTRUCTOR_SEED = 0
DEFAULT_POLL_BACKOFF = 10.0


@dataclass(frozen=True)
class IIDSamplingConfig:
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    stop: tuple[str, ...] | None = None


@dataclass(frozen=True)
class IIDModelConfig:
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    enable_prefix_caching: bool | None = None
    # Required for Delphi-shaped models, but slower for standard model shapes.
    apply_rpa_block_size_patch: bool = False


@dataclass(frozen=True)
class IIDExecutionConfig:
    worker_pools: tuple[WorkerPoolConfig, ...]
    ledger_prefix: str = DEFAULT_LEDGER_PREFIX
    chunk_size: int = 512
    heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT
    poll_backoff: float = DEFAULT_POLL_BACKOFF
    tensor_parallel_size: int = 1
    aggregate_workers: int = 32

    def __post_init__(self) -> None:
        if self.aggregate_workers < 1:
            raise ValueError(f"aggregate_workers must be >= 1 (got {self.aggregate_workers})")


@dataclass(frozen=True)
class IIDConfig:
    n_samples: int
    seed: int
    sampling_configs: tuple[IIDSamplingConfig, ...]
    execution: IIDExecutionConfig
    model: IIDModelConfig = IIDModelConfig()

    def __post_init__(self) -> None:
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1 (got {self.n_samples})")
        if not self.sampling_configs:
            raise ValueError("sampling_configs must be non-empty")
        if len(set(self.sampling_configs)) != len(self.sampling_configs):
            raise ValueError(f"sampling_configs must be distinct (got {self.sampling_configs})")


@dataclass(frozen=True)
class IIDCompletionStepConfig:
    output_path: str
    model_path: str
    prompts_path: str
    n_samples: int
    seed: int
    sampling_configs: tuple[IIDSamplingConfig, ...]
    model: IIDModelConfig
    worker_pools: tuple[WorkerPoolConfig, ...]
    ledger_prefix: str
    chunk_size: int
    heartbeat_timeout: float
    poll_backoff: float
    tensor_parallel_size: int
    aggregate_workers: int


@dataclass(frozen=True)
class IIDChunkSpec:
    chunk_id: int
    sampling_config_index: int
    chunk_start: int
    chunk_end: int
    output_path: str


@dataclass(frozen=True)
class IIDLocalEngineWorkerConfig:
    model_path: str
    prompts_path: str
    n_samples: int
    seed: int
    sampling_configs: tuple[IIDSamplingConfig, ...]
    model: IIDModelConfig
    ledger_path: str
    poll_backoff: float
    tensor_parallel_size: int
    owner: str
    chip_group: tuple[int, ...]


@dataclass(frozen=True)
class IIDCompletionAlgorithm:
    config: IIDConfig

    def make_completions_step(
        self,
        *,
        name: str,
        model_path: str | InputName | MirroredValue,
        prompts_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return make_iid_completion_step(
            name=name,
            model_path=model_path,
            prompts_path=prompts_path,
            config=self.config,
        )


@functools.cache
def _load_vllm(model_path: str, tensor_parallel_size: int, model: IIDModelConfig):
    for key, value in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(key, value)

    if model.apply_rpa_block_size_patch:
        from joint_decode.tpu.worker import _patch_rpa_kernel_block_sizes  # noqa: PLC0415

        _patch_rpa_kernel_block_sizes()

    from vllm import LLM, SamplingParams  # noqa: PLC0415

    resolved_model_path = discover_hf_checkpoints(model_path)[-1]
    resolved_model_path = localize_mirror_path(resolved_model_path)
    logger.info("Resolved %s -> %s", model_path, resolved_model_path)

    model_kwargs: dict[str, Any] = {}
    if model.max_model_len is not None:
        model_kwargs["max_model_len"] = model.max_model_len
    if model.gpu_memory_utilization is not None:
        model_kwargs["gpu_memory_utilization"] = model.gpu_memory_utilization
    if model.enable_prefix_caching is not None:
        model_kwargs["enable_prefix_caching"] = model.enable_prefix_caching

    llm = LLM(
        model=resolved_model_path,
        trust_remote_code=True,
        load_format="runai_streamer",
        seed=VLLM_CONSTRUCTOR_SEED,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=1,
        **model_kwargs,
    )
    return llm, SamplingParams


@functools.cache
def _load_prompts(prompts_path: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    prompt_rows = tuple(read_prompt_rows(prompts_path))
    return (
        tuple(row["id"] for row in prompt_rows),
        tuple(row["prompt"] for row in prompt_rows),
    )


def _reseed_sampler(worker, seed: int) -> None:
    import jax  # noqa: PLC0415
    from flax import nnx  # noqa: PLC0415

    assert hasattr(
        worker.model_runner, "rng_params_for_sampling"
    ), "tpu_inference runner missing rng_params_for_sampling; upstream API may have changed"
    worker.model_runner.rng_params_for_sampling = nnx.Rngs(jax.random.key(seed)).params()


def make_iid_completion_step(
    *,
    name: str,
    model_path: str | InputName | MirroredValue,
    prompts_path: str | InputName | MirroredValue,
    config: IIDConfig,
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(
            run_iid_completion_chunks,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
            pip_dependency_groups=["vllm", "tpu"],
            env_vars=VLLM_TPU_ENV_VARS,
        ),
        config=IIDCompletionStepConfig(
            output_path=this_output_path(),
            model_path=version_path(model_path),  # type: ignore[arg-type]
            prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
            n_samples=versioned(config.n_samples),  # type: ignore[arg-type]
            seed=versioned(config.seed),  # type: ignore[arg-type]
            sampling_configs=versioned(config.sampling_configs),  # type: ignore[arg-type]
            # Only the context limit changes output semantics; the other fields tune execution.
            model=IIDModelConfig(
                max_model_len=versioned(config.model.max_model_len),  # type: ignore[arg-type]
                gpu_memory_utilization=config.model.gpu_memory_utilization,
                enable_prefix_caching=config.model.enable_prefix_caching,
                apply_rpa_block_size_patch=config.model.apply_rpa_block_size_patch,
            ),
            worker_pools=config.execution.worker_pools,
            ledger_prefix=config.execution.ledger_prefix,
            chunk_size=versioned(config.execution.chunk_size),  # type: ignore[arg-type]
            heartbeat_timeout=config.execution.heartbeat_timeout,
            poll_backoff=config.execution.poll_backoff,
            tensor_parallel_size=config.execution.tensor_parallel_size,
            aggregate_workers=config.execution.aggregate_workers,
        ),
    )


def _chunk_specs(
    chunks_dir: str,
    num_prompts: int,
    n_samples: int,
    num_sampling_configs: int,
    chunk_size: int,
) -> list[IIDChunkSpec]:
    total_requests = num_prompts * n_samples
    specs = []
    for sampling_config_index in range(num_sampling_configs):
        for start in range(0, total_requests, chunk_size):
            chunk_id = len(specs)
            specs.append(
                IIDChunkSpec(
                    chunk_id=chunk_id,
                    sampling_config_index=sampling_config_index,
                    chunk_start=start,
                    chunk_end=min(start + chunk_size, total_requests),
                    output_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz"),
                )
            )
    return specs


def _sampling_kwargs(sampling: IIDSamplingConfig) -> dict[str, Any]:
    return {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "max_tokens": sampling.max_tokens,
        "stop": list(sampling.stop) if sampling.stop is not None else None,
    }


def _run_iid_chunk(
    chunk: IIDChunkSpec,
    *,
    model_path: str,
    prompts_path: str,
    model: IIDModelConfig,
    sampling: IIDSamplingConfig,
    n_samples: int,
    seed: int,
    tensor_parallel_size: int,
) -> None:
    llm, SamplingParams = _load_vllm(model_path, tensor_parallel_size, model)
    prompt_ids, prompts = _load_prompts(prompts_path)
    sampling_params = SamplingParams(n=1, **_sampling_kwargs(sampling))

    # TPU vLLM ignores SamplingParams.seed, so resume-safety comes from
    # directly reseeding the sampler for each durable chunk.
    llm.collective_rpc(_reseed_sampler, args=(seed + chunk.chunk_id,))

    request_indices = range(chunk.chunk_start, chunk.chunk_end)
    chunk_prompt_ids = [prompt_ids[i // n_samples] for i in request_indices]
    completion_index_offset = chunk.sampling_config_index * n_samples
    chunk_completion_indices = [completion_index_offset + i % n_samples for i in request_indices]
    chunk_prompts = [prompts[i // n_samples] for i in request_indices]

    records = []
    outputs = llm.generate(chunk_prompts, sampling_params)
    for prompt_id, completion_index, output in zip(
        chunk_prompt_ids,
        chunk_completion_indices,
        outputs,
        strict=True,
    ):
        completion_output = output.outputs[0]
        records.append(
            {
                "id": prompt_id,
                "completion_index": completion_index,
                "completion": {
                    "text": completion_output.text,
                    "metadata": {
                        "finish_reason": getattr(completion_output, "finish_reason", None),
                        "sampling_config": asdict(sampling),
                    },
                },
            }
        )

    with fsspec.open(chunk.output_path, "wt", compression="gzip") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _num_prompts(prompts_path: str) -> int:
    return sum(1 for _ in read_prompt_rows(prompts_path))


def _child_config_from_file(path: str) -> IIDLocalEngineWorkerConfig:
    with open(path) as f:
        data = json.load(f)
    sampling_configs = tuple(
        IIDSamplingConfig(
            temperature=sampling_config["temperature"],
            top_p=sampling_config["top_p"],
            top_k=sampling_config["top_k"],
            max_tokens=sampling_config["max_tokens"],
            stop=tuple(sampling_config["stop"]) if sampling_config["stop"] is not None else None,
        )
        for sampling_config in data["sampling_configs"]
    )
    return IIDLocalEngineWorkerConfig(
        model_path=data["model_path"],
        prompts_path=data["prompts_path"],
        n_samples=data["n_samples"],
        seed=data["seed"],
        sampling_configs=sampling_configs,
        model=IIDModelConfig(**data["model"]),
        ledger_path=data["ledger_path"],
        poll_backoff=data["poll_backoff"],
        tensor_parallel_size=data["tensor_parallel_size"],
        owner=data["owner"],
        chip_group=tuple(data["chip_group"]),
    )


def _run_iid_local_engine_worker(config: IIDLocalEngineWorkerConfig) -> None:
    expected_visible_chips = ",".join(str(chip) for chip in config.chip_group)
    actual_visible_chips = os.environ.get("TPU_VISIBLE_CHIPS")
    if actual_visible_chips != expected_visible_chips:
        raise ValueError(f"TPU_VISIBLE_CHIPS={actual_visible_chips!r}, expected {expected_visible_chips!r}")

    for key, value in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(key, value)

    while True:
        with ledger.claim_next_chunk(config.ledger_path, config.owner) as claim:
            if claim is None:
                summary = ledger.summarize(config.ledger_path)
                if summary.done == summary.total:
                    return
                time.sleep(config.poll_backoff)
                continue

            chunk = IIDChunkSpec(**claim.chunk)
            _run_iid_chunk(
                chunk,
                model_path=config.model_path,
                prompts_path=config.prompts_path,
                model=config.model,
                sampling=config.sampling_configs[chunk.sampling_config_index],
                n_samples=config.n_samples,
                seed=config.seed,
                tensor_parallel_size=config.tensor_parallel_size,
            )
            ledger.mark_done(claim)


def _chip_groups(chips_per_vm: int, tensor_parallel_size: int) -> list[tuple[int, ...]]:
    if tensor_parallel_size <= 0:
        raise ValueError(f"tensor_parallel_size must be positive, got {tensor_parallel_size}")
    if chips_per_vm % tensor_parallel_size != 0:
        raise ValueError(f"chips_per_vm={chips_per_vm} must be divisible by tensor_parallel_size={tensor_parallel_size}")
    return [tuple(range(start, start + tensor_parallel_size)) for start in range(0, chips_per_vm, tensor_parallel_size)]


def _child_owner(pool_id: str, shard_idx: int, chip_group: tuple[int, ...]) -> str:
    chips = ",".join(str(chip) for chip in chip_group)
    return f"{pool_id}/shard-{shard_idx}/chips-{chips}"


def _write_child_config(tmpdir: Path, config: IIDLocalEngineWorkerConfig) -> Path:
    chips = "-".join(str(chip) for chip in config.chip_group)
    path = tmpdir / f"child_chips_{chips}.json"
    with open(path, "wt") as f:
        json.dump(asdict(config), f, sort_keys=True)
    return path


def _stream_child_output(proc: subprocess.Popen[str], *, label: str) -> list[threading.Thread]:
    threads = []

    def stream(pipe, stream_name: str) -> None:
        assert pipe is not None
        for line in pipe:
            logger.info("iid local worker %s %s: %s", label, stream_name, line.rstrip())

    for pipe, stream_name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
        thread = threading.Thread(target=stream, args=(pipe, stream_name), daemon=True)
        thread.start()
        threads.append(thread)
    return threads


def _spawn_child(
    *,
    tmpdir: Path,
    config: IIDCompletionStepConfig,
    ledger_path: str,
    pool_id: str,
    shard_idx: int,
    chip_group: tuple[int, ...],
) -> tuple[subprocess.Popen[str], list[threading.Thread]]:
    child_config = IIDLocalEngineWorkerConfig(
        model_path=config.model_path,
        prompts_path=config.prompts_path,
        n_samples=config.n_samples,
        seed=config.seed,
        sampling_configs=config.sampling_configs,
        model=config.model,
        ledger_path=ledger_path,
        poll_backoff=config.poll_backoff,
        tensor_parallel_size=config.tensor_parallel_size,
        owner=_child_owner(pool_id, shard_idx, chip_group),
        chip_group=chip_group,
    )
    config_path = _write_child_config(tmpdir, child_config)
    chip_label = ",".join(str(chip) for chip in chip_group)

    env = os.environ.copy()
    env["TPU_VISIBLE_CHIPS"] = chip_label
    env["TPU_PROCESS_BOUNDS"] = "1,1,1"
    env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = f"{config.tensor_parallel_size},1,1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["JAX_COMPILATION_CACHE_DIR"] = str(tmpdir / f"jax_cache_{chip_label.replace(',', '_')}")
    env["VLLM_ASSETS_CACHE"] = str(tmpdir / f"vllm_assets_{chip_label.replace(',', '_')}")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.downstream_scaling.evals.algorithms.iid_xregion",
        "--xregion-worker-child-config",
        str(config_path),
    ]
    logger.info("Launching IID local worker shard=%d chips=%s", shard_idx, chip_label)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    return proc, _stream_child_output(proc, label=chip_label)


def _terminate_children(procs: list[subprocess.Popen[str]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()


def _wait_for_children(procs: list[subprocess.Popen[str]], threads: list[threading.Thread], ledger_path: str) -> None:
    while True:
        summary = ledger.summarize(ledger_path)
        ledger_complete = summary.done == summary.total
        all_done = True

        for proc in procs:
            return_code = proc.poll()
            if return_code is None:
                all_done = False
                continue
            if return_code != 0:
                if ledger_complete:
                    logger.warning("IID local worker exited after ledger completion with rc=%d", return_code)
                    continue
                _terminate_children(procs)
                raise RuntimeError(
                    f"IID local worker failed with rc={return_code}; ledger is {summary.done}/{summary.total} done"
                )

        if ledger_complete and all_done:
            break
        if all_done:
            raise RuntimeError(f"IID local workers exited before completion: {summary.done}/{summary.total} chunks done")

        time.sleep(1.0)

    for thread in threads:
        thread.join(timeout=5)


def _supervise_iid_worker(
    _worker_ids: Iterator[int],
    shard_info: ShardInfo,
    *,
    config: IIDCompletionStepConfig,
    ledger_path: str,
    pool: WorkerPoolConfig,
) -> Iterator[dict[str, object]]:
    if os.environ.get("TPU_VISIBLE_CHIPS") is not None:
        raise ValueError("IID per-chip supervisor expects to own the full TPU VM; TPU_VISIBLE_CHIPS is already set")
    if pool.vm_count != 1:
        raise ValueError(f"IID per-chip workers support only single-VM TPU pools, got vm_count={pool.vm_count}")

    groups = _chip_groups(pool.chips_per_vm, config.tensor_parallel_size)
    logger.info(
        "Starting IID per-chip supervisor pool=%s shard=%d chips_per_vm=%d tensor_parallel_size=%d groups=%s",
        pool.pool_id,
        shard_info.shard_idx,
        pool.chips_per_vm,
        config.tensor_parallel_size,
        groups,
    )

    with tempfile.TemporaryDirectory(prefix="iid_xregion_local_workers_") as tmp:
        tmpdir = Path(tmp)
        procs: list[subprocess.Popen[str]] = []
        threads: list[threading.Thread] = []
        try:
            for group in groups:
                proc, proc_threads = _spawn_child(
                    tmpdir=tmpdir,
                    config=config,
                    ledger_path=ledger_path,
                    pool_id=pool.pool_id,
                    shard_idx=shard_info.shard_idx,
                    chip_group=group,
                )
                procs.append(proc)
                threads.extend(proc_threads)
            _wait_for_children(procs, threads, ledger_path)
        except Exception:
            _terminate_children(procs)
            raise

    yield {"status": "done", "pool_id": pool.pool_id, "shard_idx": shard_info.shard_idx}


def run_iid_completion_chunks(config: IIDCompletionStepConfig) -> None:
    if not config.worker_pools:
        raise ValueError("IID xregion requires at least one worker pool")

    chunks_dir = os.path.join(config.output_path, "chunks", f"chunk_size={config.chunk_size}")
    chunks = _chunk_specs(
        chunks_dir,
        _num_prompts(config.prompts_path),
        config.n_samples,
        len(config.sampling_configs),
        config.chunk_size,
    )
    ledger_path = ledger.convert_mirror_path(
        ledger_prefix=config.ledger_prefix,
        output_path=config.output_path,
    )
    ledger.ensure_manifest(ledger_path, chunks)

    def make_process_shard(pool: WorkerPoolConfig):
        return functools.partial(
            _supervise_iid_worker,
            config=config,
            ledger_path=ledger_path,
            pool=pool,
        )

    xregion_pool.run_worker_pools(
        worker_pools=config.worker_pools,
        ledger_path=ledger_path,
        make_process_shard=make_process_shard,
        poll_backoff=config.poll_backoff,
        heartbeat_timeout=config.heartbeat_timeout,
    )

    summary = ledger.summarize(ledger_path)
    if summary.done != summary.total:
        raise RuntimeError(f"IID xregion incomplete: {summary.done}/{summary.total} chunks done")

    path = completions_file(config.output_path)
    done_ids = set(ledger.done_chunk_ids(ledger_path))
    chunk_paths = [chunk.output_path for chunk in chunks if chunk.chunk_id in done_ids]
    aggregate_pipeline = (
        Dataset.from_list(chunk_paths)
        .load_jsonl()
        .group_by(
            key=lambda record: record["id"],
            reducer=lambda prompt_id, items: {
                "id": prompt_id,
                "completions": [item["completion"] for item in items],
                "metadata": {
                    "completion_algorithm": "iid_xregion",
                    "model_path": config.model_path,
                },
            },
            sort_by=lambda record: record["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    ZephyrContext(
        name="iid-xregion-completions-aggregate",
        max_workers=config.aggregate_workers,
        resources=ResourceConfig(cpu=1, ram="4g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(aggregate_pipeline)
    logger.info("Wrote IID xregion completion rows to %s", path)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xregion-worker-child-config", required=True)
    args = parser.parse_args()
    _run_iid_local_engine_worker(_child_config_from_file(args.xregion_worker_child_config))


if __name__ == "__main__":
    _main()
