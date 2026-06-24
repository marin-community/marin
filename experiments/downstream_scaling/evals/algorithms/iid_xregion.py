# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IID completion algorithm for cross-region Zephyr workers."""

from __future__ import annotations

import functools
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, InputName, MirroredValue
from marin.execution.remote import remote
from marin.execution.types import this_output_path, versioned
from zephyr import Dataset, ZephyrContext

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
}

DEFAULT_HEARTBEAT_TIMEOUT = 2 * 60
DEFAULT_LEDGER_PREFIX = "gs://marin-us-central2"
VLLM_CONSTRUCTOR_SEED = 0
DEFAULT_POLL_BACKOFF = 10.0


@dataclass(frozen=True)
class IIDSamplingConfig:
    n_samples: int
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    seed: int
    stop: tuple[str, ...] | None = None


@dataclass(frozen=True)
class IIDExecutionConfig:
    worker_pools: tuple[WorkerPoolConfig, ...]
    ledger_prefix: str = DEFAULT_LEDGER_PREFIX
    chunk_size: int = 512
    heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT
    poll_backoff: float = DEFAULT_POLL_BACKOFF


@dataclass(frozen=True)
class IIDConfig:
    sampling: IIDSamplingConfig
    execution: IIDExecutionConfig


@dataclass(frozen=True)
class IIDCompletionStepConfig:
    output_path: str
    model_path: str
    prompts_path: str
    sampling: IIDSamplingConfig
    worker_pools: tuple[WorkerPoolConfig, ...]
    ledger_prefix: str
    chunk_size: int
    heartbeat_timeout: float
    poll_backoff: float


@dataclass(frozen=True)
class IIDChunkSpec:
    chunk_id: int
    chunk_start: int
    chunk_end: int
    output_path: str
    success_path: str


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
def _load_vllm(model_path: str):
    for key, value in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(key, value)

    from vllm import LLM, SamplingParams  # noqa: PLC0415

    resolved_model_path = discover_hf_checkpoints(model_path)[-1]
    resolved_model_path = localize_mirror_path(resolved_model_path)
    logger.info("Resolved %s -> %s", model_path, resolved_model_path)

    with fsspec.open(f"{resolved_model_path}/config.json", "r") as f:
        n_heads = json.load(f)["num_attention_heads"]
    tp = 2 if n_heads % 2 == 0 else 1

    llm = LLM(
        model=resolved_model_path,
        trust_remote_code=True,
        load_format="runai_streamer",
        seed=VLLM_CONSTRUCTOR_SEED,
        tensor_parallel_size=tp,
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
            sampling=versioned(config.sampling),  # type: ignore[arg-type]
            worker_pools=config.execution.worker_pools,
            ledger_prefix=config.execution.ledger_prefix,
            chunk_size=versioned(config.execution.chunk_size),  # type: ignore[arg-type]
            heartbeat_timeout=config.execution.heartbeat_timeout,
            poll_backoff=config.execution.poll_backoff,
        ),
    )


def _chunk_specs(chunks_dir: str, num_prompts: int, n_samples: int, chunk_size: int) -> list[IIDChunkSpec]:
    total_requests = num_prompts * n_samples
    return [
        IIDChunkSpec(
            chunk_id=chunk_id,
            chunk_start=start,
            chunk_end=min(start + chunk_size, total_requests),
            output_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz"),
            success_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.SUCCESS"),
        )
        for chunk_id, start in enumerate(range(0, total_requests, chunk_size))
    ]


def _sampling_kwargs(sampling: IIDSamplingConfig) -> dict[str, Any]:
    return {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "max_tokens": sampling.max_tokens,
        "stop": list(sampling.stop) if sampling.stop is not None else None,
    }


def _run_iid_chunk(chunk: IIDChunkSpec, *, config: IIDCompletionStepConfig) -> None:
    llm, SamplingParams = _load_vllm(config.model_path)
    prompt_ids, prompts = _load_prompts(config.prompts_path)
    sampling_params = SamplingParams(n=1, **_sampling_kwargs(config.sampling))
    n_samples = config.sampling.n_samples

    # TPU vLLM ignores SamplingParams.seed, so resume-safety comes from
    # directly reseeding the sampler for each durable chunk.
    llm.collective_rpc(_reseed_sampler, args=(config.sampling.seed + chunk.chunk_id,))

    request_indices = range(chunk.chunk_start, chunk.chunk_end)
    chunk_prompt_ids = [prompt_ids[i // n_samples] for i in request_indices]
    chunk_completion_indices = [i % n_samples for i in request_indices]
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
                    },
                },
            }
        )

    with fsspec.open(chunk.output_path, "wt", compression="gzip") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _num_prompts(prompts_path: str) -> int:
    return sum(1 for _ in read_prompt_rows(prompts_path))


def run_iid_completion_chunks(config: IIDCompletionStepConfig) -> None:
    if not config.worker_pools:
        raise ValueError("IID xregion requires at least one worker pool")

    chunks_dir = os.path.join(config.output_path, "chunks", f"chunk_size={config.chunk_size}")
    chunks = _chunk_specs(chunks_dir, _num_prompts(config.prompts_path), config.sampling.n_samples, config.chunk_size)
    ledger_path = ledger.convert_mirror_path(
        ledger_prefix=config.ledger_prefix,
        output_path=config.output_path,
    )
    ledger.ensure_manifest(ledger_path, chunks)

    def process_chunk(chunk_record: dict[str, Any]) -> None:
        _run_iid_chunk(IIDChunkSpec(**chunk_record), config=config)

    xregion_pool.run_worker_pools(
        worker_pools=config.worker_pools,
        ledger_path=ledger_path,
        process_chunk=process_chunk,
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
                },
            },
            sort_by=lambda record: record["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    aggregate_workers = max(pool.num_workers for pool in config.worker_pools)
    ZephyrContext(
        name="iid-xregion-completions-aggregate",
        max_workers=aggregate_workers,
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(aggregate_pipeline)
    logger.info("Wrote IID xregion completion rows to %s", path)
