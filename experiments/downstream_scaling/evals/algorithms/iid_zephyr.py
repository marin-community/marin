# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IID completion algorithm for downstream-scaling evals."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.executor import ExecutorStep, InputName, MirroredValue, this_output_path, versioned
from marin.execution.remote import remote
from marin.utils import fsspec_exists
from zephyr import Dataset, ZephyrContext

from experiments.downstream_scaling.evals.framework.schema import (
    completions_file,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)

VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}


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
    num_workers: int
    worker_resources: ResourceConfig
    chunk_size: int = 512


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
    num_workers: int
    chunk_size: int
    worker_resources: ResourceConfig


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


def _load_vllm(model_path: str, seed: int):
    for key, value in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(key, value)

    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa_kernel

    orig_get_tuned = rpa_kernel.get_tuned_block_sizes

    def patched_get_tuned(*args, **kwargs):
        bkv_p, bq_sz = orig_get_tuned(*args, **kwargs)
        return (max(1, bkv_p // 2), bq_sz)

    rpa_kernel.get_tuned_block_sizes = patched_get_tuned

    from vllm import LLM, SamplingParams

    resolved_model_path = discover_hf_checkpoints(model_path)[-1]
    logger.info("Resolved %s -> %s", model_path, resolved_model_path)
    llm = LLM(
        model=resolved_model_path,
        trust_remote_code=True,
        load_format="runai_streamer",
        seed=seed,
    )
    return llm, SamplingParams


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
            num_workers=config.execution.num_workers,
            chunk_size=versioned(config.execution.chunk_size),  # type: ignore[arg-type]
            worker_resources=config.execution.worker_resources,
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


def _process_iid_shard(
    chunks,
    shard_info,
    *,
    config: IIDCompletionStepConfig,
    prompt_ids: list[str],
    prompts: list[str],
):
    llm, SamplingParams = _load_vllm(config.model_path, config.sampling.seed + shard_info.shard_idx)
    sampling_params = SamplingParams(n=1, **_sampling_kwargs(config.sampling))
    n_samples = config.sampling.n_samples

    for chunk in chunks:
        if fsspec_exists(chunk.success_path):
            yield {"chunk_id": chunk.chunk_id, "output_path": chunk.output_path, "skipped": True}
            continue

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
        with fsspec.open(chunk.success_path, "wt") as f:
            f.write("ok\n")
        yield {"chunk_id": chunk.chunk_id, "output_path": chunk.output_path, "skipped": False}


def run_iid_completion_chunks(config: IIDCompletionStepConfig) -> None:
    prompt_rows = list(read_prompt_rows(config.prompts_path))
    prompt_ids = [row["id"] for row in prompt_rows]
    prompts = [row["prompt"] for row in prompt_rows]
    chunks_dir = os.path.join(config.output_path, "chunks", f"chunk_size={config.chunk_size}")
    chunks = _chunk_specs(chunks_dir, len(prompt_rows), config.sampling.n_samples, config.chunk_size)

    process_shard = partial(
        _process_iid_shard,
        config=config,
        prompt_ids=prompt_ids,
        prompts=prompts,
    )
    pipeline = Dataset.from_list(chunks).reshard(config.num_workers).map_shard(process_shard)
    ZephyrContext(
        name="iid-completion-chunks",
        max_workers=config.num_workers,
        resources=config.worker_resources,
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(pipeline)

    path = completions_file(config.output_path)
    aggregate_pipeline = (
        Dataset.from_files(os.path.join(chunks_dir, "chunk-*.jsonl.gz"))
        .load_jsonl()
        .group_by(
            key=lambda record: record["id"],
            reducer=lambda prompt_id, items: {
                "id": prompt_id,
                "completions": [item["completion"] for item in items],
                "metadata": {
                    "completion_algorithm": "iid",
                },
            },
            sort_by=lambda record: record["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    ZephyrContext(
        name="iid-completions-aggregate",
        max_workers=config.num_workers,
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(aggregate_pipeline)
    logger.info("Wrote IID completion rows to %s", path)
