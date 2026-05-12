# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rerank-decode vLLM TPU completion algorithm for downstream-scaling evals."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import fsspec
from fray import ActorGroup, ResourceConfig, current_client
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.executor import ExecutorStep, InputName, MirroredValue, this_output_path, versioned
from marin.execution.remote import remote
from marin.utils import fsspec_exists
from zephyr import Dataset, ZephyrContext

from experiments.downstream_scaling.evals.algorithms.iid import VLLM_TPU_ENV_VARS
from experiments.downstream_scaling.evals.framework.schema import (
    completions_file,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankSamplingConfig:
    n_rollouts: int
    proposal_samples: int
    temperature: float
    top_p: float
    top_k: int
    chunk_tokens: int
    max_tokens: int
    seed: int
    stop: tuple[str, ...] | None = None
    include_eos_in_score: bool = True


@dataclass(frozen=True)
class RerankScorerConfig:
    max_model_len: int = 8192
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_prefix_caching: bool = True
    enable_prefix_caching_with_prompt_logprobs: bool = True


@dataclass(frozen=True)
class RerankProposalConfig:
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    gpu_memory_utilization: float | None = None


@dataclass(frozen=True)
class RerankExecutionConfig:
    num_workers: int
    requests_per_chunk: int
    worker_resources: ResourceConfig
    scorer_actor_resources: ResourceConfig


@dataclass(frozen=True)
class RerankConfig:
    sampling: RerankSamplingConfig
    scoring_model_path: str | InputName | MirroredValue
    scorer: RerankScorerConfig
    proposal: RerankProposalConfig
    execution: RerankExecutionConfig


@dataclass(frozen=True)
class RerankCompletionStepConfig:
    output_path: str
    proposal_model_path: str
    scoring_model_path: str
    prompts_path: str
    sampling: RerankSamplingConfig
    scorer: RerankScorerConfig
    proposal: RerankProposalConfig
    num_workers: int
    requests_per_chunk: int
    worker_resources: ResourceConfig
    scorer_actor_resources: ResourceConfig


@dataclass(frozen=True)
class RerankChunkSpec:
    chunk_id: int
    chunk_start: int
    chunk_end: int
    output_path: str
    success_path: str


@dataclass(frozen=True)
class RerankStats:
    num_steps: int = 0
    remaining_tokens: int = 0
    finish_reason: str = ""
    total_proposal_time: float = 0.0
    total_scoring_time: float = 0.0


@dataclass(frozen=True)
class RerankCompletionAlgorithm:
    config: RerankConfig

    def make_completions_step(
        self,
        *,
        name: str,
        model_path: str | InputName | MirroredValue,
        prompts_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return make_rerank_completion_step(
            name=name,
            model_path=model_path,
            prompts_path=prompts_path,
            config=self.config,
        )


class VLLMScorerActor:
    def __init__(self, model_path: str, scorer: RerankScorerConfig) -> None:
        _set_vllm_tpu_env_vars()

        from experiments.rerank_decode.scorer import VLLMLogprobScorerTPU

        resolved_model_path = discover_hf_checkpoints(model_path)[-1]
        logger.info("Resolved scoring model %s -> %s", model_path, resolved_model_path)
        self._scorer = VLLMLogprobScorerTPU(
            resolved_model_path,
            max_model_len=scorer.max_model_len,
            tensor_parallel_size=scorer.tensor_parallel_size,
            data_parallel_size=scorer.data_parallel_size,
            enable_prefix_caching=scorer.enable_prefix_caching,
            enable_prefix_caching_with_prompt_logprobs=scorer.enable_prefix_caching_with_prompt_logprobs,
        )
        self._eos_token = self._scorer.tokenizer.eos_token or ""

    def eos_token(self) -> str:
        return self._eos_token

    def reset(self) -> None:
        self._scorer.reset()

    def score(self, prompt: str, completions: list[str]) -> list[float]:
        return self._scorer.score(prompt, completions)

    def accept(self, prompt: str, completion: str) -> None:
        self._scorer.accept(prompt, completion)


def make_rerank_completion_step(
    *,
    name: str,
    model_path: str | InputName | MirroredValue,
    prompts_path: str | InputName | MirroredValue,
    config: RerankConfig,
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(
            run_rerank_completion_chunks,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
            pip_dependency_groups=["vllm", "tpu"],
            env_vars=VLLM_TPU_ENV_VARS,
        ),
        config=RerankCompletionStepConfig(
            output_path=this_output_path(),
            proposal_model_path=version_path(model_path),  # type: ignore[arg-type]
            scoring_model_path=version_path(config.scoring_model_path),  # type: ignore[arg-type]
            prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
            sampling=versioned(config.sampling),  # type: ignore[arg-type]
            scorer=versioned(config.scorer),  # type: ignore[arg-type]
            proposal=versioned(config.proposal),  # type: ignore[arg-type]
            num_workers=config.execution.num_workers,
            requests_per_chunk=versioned(config.execution.requests_per_chunk),  # type: ignore[arg-type]
            worker_resources=config.execution.worker_resources,
            scorer_actor_resources=config.execution.scorer_actor_resources,
        ),
    )


def _set_vllm_tpu_env_vars() -> None:
    for key, value in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(key, value)


def _chunk_specs(chunks_dir: str, num_prompts: int, n_rollouts: int, requests_per_chunk: int) -> list[RerankChunkSpec]:
    total_requests = num_prompts * n_rollouts
    return [
        RerankChunkSpec(
            chunk_id=chunk_id,
            chunk_start=start,
            chunk_end=min(start + requests_per_chunk, total_requests),
            output_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz"),
            success_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.SUCCESS"),
        )
        for chunk_id, start in enumerate(range(0, total_requests, requests_per_chunk))
    ]


def _best_index(scores: list[float]) -> int:
    return max(range(len(scores)), key=scores.__getitem__)


def _load_proposal_vllm(model_path: str, proposal: RerankProposalConfig, seed: int):
    _set_vllm_tpu_env_vars()

    from vllm import LLM, SamplingParams

    resolved_model_path = discover_hf_checkpoints(model_path)[-1]
    logger.info("Resolved proposal model %s -> %s", model_path, resolved_model_path)
    kwargs: dict[str, Any] = {
        "model": resolved_model_path,
        "trust_remote_code": True,
        "load_format": "runai_streamer",
        "seed": seed,
        "tensor_parallel_size": proposal.tensor_parallel_size,
        "data_parallel_size": proposal.data_parallel_size,
    }
    if proposal.max_model_len is not None:
        kwargs["max_model_len"] = proposal.max_model_len
    if proposal.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = proposal.gpu_memory_utilization
    return LLM(**kwargs), SamplingParams


def _generate_proposals(
    proposal_llm,
    SamplingParams,
    *,
    prompt: str,
    sampling: RerankSamplingConfig,
    max_tokens: int,
) -> list[dict[str, Any]]:
    sampling_params = SamplingParams(
        n=sampling.proposal_samples,
        max_tokens=max_tokens,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        stop=list(sampling.stop) if sampling.stop is not None else None,
    )
    outputs = proposal_llm.generate([prompt], sampling_params, use_tqdm=False)
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one proposal output, got {len(outputs)}")
    return [
        {
            "text": output.text,
            "finish_reason": getattr(output, "finish_reason", None),
        }
        for output in outputs[0].outputs
    ]


def _create_scorer_actor(config: RerankCompletionStepConfig, shard_idx: int) -> tuple[Any, list[ActorGroup]]:
    client = current_client()
    actor_suffix = f"{shard_idx}-{uuid.uuid4().hex[:8]}"
    scorer_group = client.create_actor_group(
        VLLMScorerActor,
        config.scoring_model_path,
        config.scorer,
        name=f"rerank-scorer-{actor_suffix}",
        count=1,
        resources=config.scorer_actor_resources,
    )
    try:
        scorer = scorer_group.wait_ready(count=1)[0]
        return scorer, [scorer_group]
    except Exception:
        _shutdown_actor_groups([scorer_group])
        raise


def _shutdown_actor_groups(groups: list[ActorGroup]) -> None:
    for group in groups:
        try:
            group.shutdown()
        except Exception:
            logger.exception("Failed to shut down rerank actor group")


def _rerank_one_prompt(
    *,
    proposal_llm,
    SamplingParams,
    scorer,
    prompt: str,
    sampling: RerankSamplingConfig,
    eos_token: str,
) -> tuple[str, RerankStats]:
    scorer.reset()
    current_prompt = prompt
    generated = ""
    remaining_tokens = sampling.max_tokens
    stats = RerankStats(remaining_tokens=remaining_tokens)

    while remaining_tokens > 0:
        step_tokens = min(sampling.chunk_tokens, remaining_tokens)

        proposal_start = time.perf_counter()
        candidates = _generate_proposals(
            proposal_llm,
            SamplingParams,
            prompt=current_prompt,
            sampling=sampling,
            max_tokens=step_tokens,
        )
        proposal_time = time.perf_counter() - proposal_start

        candidate_texts = [
            (
                candidate["text"] + eos_token
                if sampling.include_eos_in_score and candidate["finish_reason"] == "stop"
                else candidate["text"]
            )
            for candidate in candidates
        ]

        scoring_start = time.perf_counter()
        scores = scorer.score(current_prompt, candidate_texts)
        scoring_time = time.perf_counter() - scoring_start

        if len(scores) != len(candidates):
            raise RuntimeError(f"Expected {len(candidates)} scores, got {len(scores)}")

        best = _best_index(scores)
        best_candidate = candidates[best]
        best_text = best_candidate["text"]
        generated += best_text

        stats = RerankStats(
            num_steps=stats.num_steps + 1,
            remaining_tokens=remaining_tokens,
            finish_reason=stats.finish_reason,
            total_proposal_time=stats.total_proposal_time + proposal_time,
            total_scoring_time=stats.total_scoring_time + scoring_time,
        )

        if best_candidate["finish_reason"] == "stop":
            return generated, RerankStats(
                num_steps=stats.num_steps,
                remaining_tokens=remaining_tokens,
                finish_reason="stop",
                total_proposal_time=stats.total_proposal_time,
                total_scoring_time=stats.total_scoring_time,
            )

        scorer.accept(current_prompt, best_text)
        remaining_tokens -= step_tokens
        current_prompt += best_text

    return generated, RerankStats(
        num_steps=stats.num_steps,
        remaining_tokens=remaining_tokens,
        finish_reason="max_tokens",
        total_proposal_time=stats.total_proposal_time,
        total_scoring_time=stats.total_scoring_time,
    )


def _process_rerank_shard(
    chunks,
    shard_info,
    *,
    config: RerankCompletionStepConfig,
    prompt_ids: list[str],
    prompts: list[str],
):
    n_rollouts = config.sampling.n_rollouts
    proposal_llm, SamplingParams = _load_proposal_vllm(
        config.proposal_model_path,
        config.proposal,
        config.sampling.seed + shard_info.shard_idx,
    )
    scorer, actor_groups = _create_scorer_actor(config, shard_info.shard_idx)

    try:
        eos_token = scorer.eos_token()
        for chunk in chunks:
            if fsspec_exists(chunk.success_path):
                yield {"chunk_id": chunk.chunk_id, "output_path": chunk.output_path, "skipped": True}
                continue

            records = []
            for request_index in range(chunk.chunk_start, chunk.chunk_end):
                prompt_index = request_index // n_rollouts
                completion_index = request_index % n_rollouts
                completion, stats = _rerank_one_prompt(
                    proposal_llm=proposal_llm,
                    SamplingParams=SamplingParams,
                    scorer=scorer,
                    prompt=prompts[prompt_index],
                    sampling=config.sampling,
                    eos_token=eos_token,
                )
                records.append(
                    {
                        "id": prompt_ids[prompt_index],
                        "completion_index": completion_index,
                        "completion": {
                            "text": completion,
                            "metadata": asdict(stats),
                        },
                    }
                )

            with fsspec.open(chunk.output_path, "wt", compression="gzip") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            with fsspec.open(chunk.success_path, "wt") as f:
                f.write("ok\n")
            yield {"chunk_id": chunk.chunk_id, "output_path": chunk.output_path, "skipped": False}
    finally:
        _shutdown_actor_groups(actor_groups)


def run_rerank_completion_chunks(config: RerankCompletionStepConfig) -> None:
    prompt_rows = list(read_prompt_rows(config.prompts_path))
    prompt_ids = [row["id"] for row in prompt_rows]
    prompts = [row["prompt"] for row in prompt_rows]
    chunks_dir = os.path.join(config.output_path, "chunks", f"requests_per_chunk={config.requests_per_chunk}")
    chunks = _chunk_specs(chunks_dir, len(prompt_rows), config.sampling.n_rollouts, config.requests_per_chunk)

    process_shard = partial(
        _process_rerank_shard,
        config=config,
        prompt_ids=prompt_ids,
        prompts=prompts,
    )
    pipeline = Dataset.from_list(chunks).reshard(config.num_workers).map_shard(process_shard)
    ZephyrContext(
        name="rerank-completion-chunks",
        max_workers=config.num_workers,
        resources=config.worker_resources,
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
                    "completion_algorithm": "rerank",
                    "proposal_model_path": config.proposal_model_path,
                    "scoring_model_path": config.scoring_model_path,
                },
            },
            sort_by=lambda record: record["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    ZephyrContext(name="rerank-completions-aggregate", max_workers=config.num_workers).execute(aggregate_pipeline)
    logger.info("Wrote rerank completion rows to %s", path)
