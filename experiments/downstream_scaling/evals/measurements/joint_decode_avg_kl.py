# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KL between a joint-decode-avg rule's weight-zero and recorded-weight distributions.

Replay forces both sides' recorded token ids and measures one KL per decision.
Union rules keep advisor-only candidates in the weight-zero baseline. Models,
prompts, and sampling settings must match the source run to interpret the
result as that run's sampling-rule KL; matching replayed text alone is insufficient.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import load_tokenizer
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo

from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.framework.schema import (
    completions_file,
    read_completion_rows,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.framework.xregion import ledger
from experiments.downstream_scaling.evals.framework.xregion import pool as xregion_pool
from experiments.downstream_scaling.evals.framework.xregion.pool import (
    EnginePlacement,
    WorkerPoolConfig,
    validate_pool_placements,
)
from experiments.downstream_scaling.evals.measurements.schema import read_statistic_rows, statistics_file
from experiments.downstream_scaling.evals.utils import discover_hf_checkpoints, version_path

logger = logging.getLogger(__name__)

# Match the joint_decode_avg_v2 worker environment.
VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_TARGET_DEVICE": "tpu",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
    "RUNAI_STREAMER_MEMORY_LIMIT": "4294967296",
}

DEFAULT_HEARTBEAT_TIMEOUT = 2 * 60
DEFAULT_LEDGER_PREFIX = "gs://marin-us-central2"
DEFAULT_POLL_BACKOFF = 10.0
TOKEN_PATHS_FILENAME = "token_paths.jsonl.gz"

DistributionKey = int | xtok_selection.Key


class XtokSelectionRule(StrEnum):
    BYTES_UNION = "bytes_union"
    ANCHORED_PREFIX_MASS = "anchored_prefix_mass"
    AVG_LOGITS = "avg_logits"
    AVG_PROBS = "avg_probs"
    UNNORMALIZED_ADD = "unnormalized_add"


@dataclass(frozen=True)
class JointDecodeReplaySamplingConfig:
    max_tokens: int
    advisor_max_tokens: int
    top_k_a: int
    top_k_b: int
    seed: int
    selection_rule: XtokSelectionRule
    temperature: float = 1.0
    stop: tuple[str, ...] | None = None
    prefix_credit: float = 1.0

    def __post_init__(self) -> None:
        if self.max_tokens < 1 or self.advisor_max_tokens < 1:
            raise ValueError("max_tokens and advisor_max_tokens must both be >= 1")
        if self.top_k_a < 1 or self.top_k_b < 1:
            raise ValueError("top_k_a and top_k_b must both be >= 1")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError(f"temperature must be finite and > 0 (got {self.temperature})")
        if not 0.0 <= self.prefix_credit <= 1.0:
            raise ValueError(f"prefix_credit must be in [0, 1] (got {self.prefix_credit})")


@dataclass(frozen=True)
class JointDecodeModelConfig:
    max_model_len: int = 8192
    gpu_memory_utilization: float | None = None
    enable_prefix_caching: bool = False
    apply_rpa_block_size_patch: bool = False


@dataclass(frozen=True)
class JointDecodePlacement:
    decoder: EnginePlacement
    advisor: EnginePlacement


@dataclass(frozen=True)
class JointDecodePoolConfig:
    pool: WorkerPoolConfig
    placements: tuple[JointDecodePlacement, ...]

    def __post_init__(self) -> None:
        engines = tuple(engine for placement in self.placements for engine in (placement.decoder, placement.advisor))
        validate_pool_placements(self.pool, engines)


@dataclass(frozen=True)
class JointDecodeExecutionConfig:
    worker_pools: tuple[JointDecodePoolConfig, ...]
    ledger_prefix: str = DEFAULT_LEDGER_PREFIX
    chunk_size: int = 512
    microbatch_size: int | None = None
    heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT
    poll_backoff: float = DEFAULT_POLL_BACKOFF
    barrier_timeout_s: float = 60.0
    aggregate_workers: int = 32
    # As in v2, None derives max_model_len + microbatch_size in the child.
    max_num_batched_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1 (got {self.chunk_size})")
        if self.microbatch_size is not None and self.microbatch_size < 1:
            raise ValueError(f"microbatch_size must be >= 1 or None (got {self.microbatch_size})")
        if self.aggregate_workers < 1:
            raise ValueError(f"aggregate_workers must be >= 1 (got {self.aggregate_workers})")


@dataclass(frozen=True)
class JointDecodeAvgKlConfig:
    sampling: JointDecodeReplaySamplingConfig
    advisor_model_path: str | InputName | MirroredValue
    decoder_model: JointDecodeModelConfig
    advisor_model: JointDecodeModelConfig
    execution: JointDecodeExecutionConfig
    advisor_prompts_path: str | InputName | MirroredValue | None = None


@dataclass(frozen=True)
class JointDecodeAvgKlStepConfig:
    output_path: str
    decoder_model_path: str
    advisor_model_path: str
    prompts_path: str
    advisor_prompts_path: str | None
    alg_output_path: str
    sampling: JointDecodeReplaySamplingConfig
    decoder_model: JointDecodeModelConfig
    advisor_model: JointDecodeModelConfig
    worker_pools: tuple[JointDecodePoolConfig, ...]
    ledger_prefix: str
    chunk_size: int
    microbatch_size: int
    heartbeat_timeout: float
    poll_backoff: float
    barrier_timeout_s: float
    aggregate_workers: int
    max_num_batched_tokens: int | None = None


@dataclass(frozen=True)
class ReplayChunkSpec:
    chunk_id: int
    chunk_start: int
    chunk_end: int
    output_path: str


@dataclass(frozen=True)
class JointDecodeLocalWorkerConfig:
    decoder_model_path: str
    advisor_model_path: str
    prompts_path: str
    advisor_prompts_path: str | None
    alg_output_path: str
    token_paths_path: str
    sampling: JointDecodeReplaySamplingConfig
    decoder_model: JointDecodeModelConfig
    advisor_model: JointDecodeModelConfig
    ledger_path: str
    poll_backoff: float
    microbatch_size: int
    barrier_timeout_s: float
    owner: str
    placement: JointDecodePlacement
    max_num_batched_tokens: int | None = None


@dataclass(frozen=True)
class JointDecodeAvgKl:
    decoder_model_path: str | InputName | MirroredValue
    config: JointDecodeAvgKlConfig

    def make_statistic_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        alg_output_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        """Build a step measuring KL along the completion output's recorded paths."""
        return make_joint_decode_avg_kl_step(
            name=name,
            prompts_path=prompts_path,
            alg_output_path=alg_output_path,
            statistic=self,
        )


def make_joint_decode_avg_kl_step(
    *,
    name: str,
    prompts_path: str | InputName | MirroredValue,
    alg_output_path: str | InputName | MirroredValue,
    statistic: JointDecodeAvgKl,
) -> ExecutorStep:
    """Build the replay coordinator with versioned inputs and local sampling config."""
    config = statistic.config
    microbatch_size = (
        config.execution.chunk_size if config.execution.microbatch_size is None else config.execution.microbatch_size
    )
    return ExecutorStep(
        name=name,
        fn=remote(
            run_joint_decode_avg_kl,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
            pip_dependency_groups=["vllm"],
            env_vars=VLLM_TPU_ENV_VARS,
        ),
        config=JointDecodeAvgKlStepConfig(
            output_path=this_output_path(),
            decoder_model_path=version_path(statistic.decoder_model_path),  # type: ignore[arg-type]
            advisor_model_path=version_path(config.advisor_model_path),  # type: ignore[arg-type]
            prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
            advisor_prompts_path=(
                None if config.advisor_prompts_path is None else version_path(config.advisor_prompts_path)  # type: ignore[arg-type]
            ),
            alg_output_path=version_path(alg_output_path),  # type: ignore[arg-type]
            sampling=versioned(config.sampling),  # type: ignore[arg-type]
            decoder_model=versioned(config.decoder_model),  # type: ignore[arg-type]
            advisor_model=versioned(config.advisor_model),  # type: ignore[arg-type]
            worker_pools=config.execution.worker_pools,
            ledger_prefix=config.execution.ledger_prefix,
            chunk_size=versioned(config.execution.chunk_size),  # type: ignore[arg-type]
            microbatch_size=microbatch_size,
            heartbeat_timeout=config.execution.heartbeat_timeout,
            poll_backoff=config.execution.poll_backoff,
            barrier_timeout_s=config.execution.barrier_timeout_s,
            aggregate_workers=config.execution.aggregate_workers,
            max_num_batched_tokens=config.execution.max_num_batched_tokens,
        ),
    )


def _log_softmax(values: list[float], temperature: float) -> list[float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("selection scores must be nonempty and finite")
    maximum = max(values)
    shifted = [(value - maximum) / temperature for value in values]
    log_total = math.log(sum(math.exp(value) for value in shifted))
    return [value - log_total for value in shifted]


def _avg_anchored_scores(
    a: dict[xtok_selection.Key, xtok_selection.Candidate],
    b: dict[xtok_selection.Key, xtok_selection.Candidate],
    advisor_weight: float,
    *,
    prefix_credit: float,
) -> dict[xtok_selection.Key, float]:
    # Mirror xtok_selection.select_avg_anchored; keep B's floor in log space.
    b_logprobs = _log_softmax([candidate.logit for candidate in b.values()], 1.0)
    b_probs = {key: math.exp(logprob) for key, logprob in zip(b, b_logprobs, strict=True)}
    b_floor = min(b_logprobs)
    scores = {}
    for key, candidate in a.items():
        mass = (
            xtok_selection.prefix_mass(key, b_probs, credit=prefix_credit)
            if isinstance(key, bytes)
            else b_probs.get(xtok_selection.EOS_KEY, 0.0)
        )
        scores[key] = (1.0 - advisor_weight) * candidate.logit + advisor_weight * (
            math.log(mass) if mass > 0.0 else b_floor
        )
    return scores


def _sampling_logprobs(
    rule: str,
    a_topk: list[dict[str, Any]],
    b_topk: list[dict[str, Any]],
    *,
    advisor_weight: float,
    temperature: float,
    vocab_a: xtok_selection.Vocab,
    vocab_b: xtok_selection.Vocab,
    prefix_credit: float,
) -> dict[DistributionKey, float]:
    if rule in (XtokSelectionRule.BYTES_UNION, XtokSelectionRule.ANCHORED_PREFIX_MASS):
        a = xtok_selection.candidates(vocab_a, a_topk)
        b = xtok_selection.candidates(vocab_b, b_topk)
        scores = (
            xtok_selection.avg_bytes_union_scores(a, b, advisor_weight)
            if rule == XtokSelectionRule.BYTES_UNION
            else _avg_anchored_scores(a, b, advisor_weight, prefix_credit=prefix_credit)
        )
        return dict(zip(scores, _log_softmax(list(scores.values()), temperature), strict=True))

    a_logits = {int(item["token_id"]): float(item["logit"]) for item in a_topk}
    b_logits = {int(item["token_id"]): float(item["logit"]) for item in b_topk}
    if not a_logits or not b_logits:
        raise ValueError("both sides must provide at least one top-k logit")
    union = sorted(set(a_logits) | set(b_logits))
    a_floor, b_floor = min(a_logits.values()), min(b_logits.values())
    a_values = [a_logits.get(token_id, a_floor) for token_id in union]
    b_values = [b_logits.get(token_id, b_floor) for token_id in union]

    if rule == XtokSelectionRule.AVG_PROBS:
        log_a = _log_softmax(a_values, temperature)
        log_b = _log_softmax(b_values, temperature)
        if advisor_weight == 0.0:
            return dict(zip(union, log_a, strict=True))
        if advisor_weight == 1.0:
            return dict(zip(union, log_b, strict=True))
        log_weight_a, log_weight_b = math.log1p(-advisor_weight), math.log(advisor_weight)
        mixture = []
        for lp_a, lp_b in zip(log_a, log_b, strict=True):
            left, right = log_weight_a + lp_a, log_weight_b + lp_b
            mixture.append(max(left, right) + math.log1p(math.exp(-abs(left - right))))
        return dict(zip(union, mixture, strict=True))

    if rule == XtokSelectionRule.UNNORMALIZED_ADD:
        scale = 1.0 + advisor_weight
        advisor_weight /= scale
        temperature /= scale
    elif rule != XtokSelectionRule.AVG_LOGITS:
        raise ValueError(f"unknown selection rule: {rule!r}")
    values = [
        (1.0 - advisor_weight) * value_a + advisor_weight * value_b
        for value_a, value_b in zip(a_values, b_values, strict=True)
    ]
    return dict(zip(union, _log_softmax(values, temperature), strict=True))


def _kl(log_p: dict[DistributionKey, float], log_q: dict[DistributionKey, float]) -> float:
    if log_p.keys() != log_q.keys():
        raise ValueError("KL distributions must have the same candidate keys")
    if any(not math.isfinite(value) for value in (*log_p.values(), *log_q.values())):
        raise ValueError("KL log probabilities must be finite")
    value = math.fsum(math.exp(lp) * (lp - log_q[key]) for key, lp in log_p.items())
    if not math.isfinite(value):
        raise ValueError(f"non-finite KL: {value}")
    return max(value, 0.0)


@dataclass(frozen=True)
class XtokPathStep:
    bytes_hex: str
    tokens_a: list[int]
    tokens_b: list[int]


@dataclass(frozen=True)
class TokenPathRow:
    id: str
    completion_index: int
    advisor_weight: float
    steps: list[XtokPathStep]


def _is_token_ids(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in value
    )


def _is_token_path_step(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("bytes_hex"), str)
        and _is_token_ids(value.get("tokens_a"))
        and _is_token_ids(value.get("tokens_b"))
    )


def _token_path_row(raw: Any, path: str, rule: XtokSelectionRule) -> TokenPathRow:
    completion_index = raw.get("completion_index") if isinstance(raw, dict) else None
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("id"), str)
        or not isinstance(completion_index, int)
        or isinstance(completion_index, bool)
        or completion_index < 0
        or not isinstance(raw.get("steps"), list)
        or not all(_is_token_path_step(step) for step in raw["steps"])
    ):
        raise TypeError(f"Invalid token-path row: {path}")
    weight = raw.get("advisor_weight")
    if not isinstance(weight, int | float) or isinstance(weight, bool):
        raise TypeError(f"Token-path advisor_weight must be a number: {path}")
    if not math.isfinite(weight) or weight < 0.0 or (rule is not XtokSelectionRule.UNNORMALIZED_ADD and weight > 1.0):
        raise ValueError(f"Invalid advisor_weight={weight} for {rule.value}: {path}")
    return TokenPathRow(
        id=raw["id"],
        completion_index=completion_index,
        advisor_weight=float(weight),
        steps=[
            XtokPathStep(
                bytes_hex=step["bytes_hex"],
                tokens_a=list(step["tokens_a"]),
                tokens_b=list(step["tokens_b"]),
            )
            for step in raw["steps"]
        ],
    )


def read_token_path_rows(path: str, rule: XtokSelectionRule) -> Iterator[TokenPathRow]:
    """Read validated paths, rejecting duplicate completion keys and invalid weights."""
    seen: set[tuple[str, int]] = set()
    with fsspec.open(path, "rt", compression="gzip") as f:
        for line in f:
            row = _token_path_row(json.loads(line), path, rule)
            key = (row.id, row.completion_index)
            if key in seen:
                raise ValueError(f"Duplicate token-path row {key!r}: {path}")
            seen.add(key)
            yield row


def _validate_token_paths(alg_output_path: str, rule: XtokSelectionRule) -> int:
    expected = {
        (row["id"], index)
        for row in read_completion_rows(completions_file(alg_output_path))
        for index in range(len(row["completions"]))
    }
    actual = {
        (row.id, row.completion_index)
        for row in read_token_path_rows(os.path.join(alg_output_path, TOKEN_PATHS_FILENAME), rule)
    }
    if actual != expected:
        raise ValueError(
            f"Token-path keys do not match completions: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    return len(actual)


def _read_row_range(path: str, start: int, end: int, rule: XtokSelectionRule) -> list[TokenPathRow]:
    rows = []
    with fsspec.open(path, "rt", compression="gzip") as f:
        for index, line in enumerate(f):
            if index >= end:
                break
            if index >= start:
                rows.append(_token_path_row(json.loads(line), path, rule))
    return rows


def _chunk_specs(chunks_dir: str, num_records: int, chunk_size: int) -> list[ReplayChunkSpec]:
    return [
        ReplayChunkSpec(
            chunk_id=chunk_id,
            chunk_start=start,
            chunk_end=min(start + chunk_size, num_records),
            output_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz"),
        )
        for chunk_id, start in enumerate(range(0, num_records, chunk_size))
    ]


@dataclass
class _ReplayRequest:
    steps: list[XtokPathStep]
    advisor_weight: float
    kls: list[float]


def _make_replay_select_token(
    sampling: JointDecodeReplaySamplingConfig,
    vocab_a: xtok_selection.Vocab,
    vocab_b: xtok_selection.Vocab,
    requests: dict[int, _ReplayRequest],
) -> Callable[..., tuple[list[int], list[int]]]:
    params: dict[str, Any] = dict(
        temperature=sampling.temperature,
        vocab_a=vocab_a,
        vocab_b=vocab_b,
        prefix_credit=sampling.prefix_credit,
    )

    def select_token(a_topk, b_topk, *, rng: random.Random, request_index: int):
        request = requests[request_index]
        index = len(request.kls)
        if index >= len(request.steps):
            raise RuntimeError(f"replay ran past the recorded path for request {request_index}")
        log_q0 = _sampling_logprobs(sampling.selection_rule.value, a_topk, b_topk, advisor_weight=0.0, **params)
        log_qw = _sampling_logprobs(
            sampling.selection_rule.value, a_topk, b_topk, advisor_weight=request.advisor_weight, **params
        )
        request.kls.append(_kl(log_q0, log_qw))
        step = request.steps[index]
        return list(step.tokens_a), list(step.tokens_b)

    return select_token


def write_replay_chunk(
    chunk: ReplayChunkSpec,
    *,
    decoder: Any,
    prompts: dict[str, str],
    advisor_prompts: dict[str, str],
    completions: dict[tuple[str, int], str],
    token_paths_path: str,
    selection_rule: XtokSelectionRule,
    requests: dict[int, _ReplayRequest],
) -> None:
    """Replay a sidecar range and write KL values after verifying every output."""
    rows = _read_row_range(token_paths_path, chunk.chunk_start, chunk.chunk_end, selection_rule)
    requests.clear()
    requests.update(
        (index, _ReplayRequest(steps=row.steps, advisor_weight=row.advisor_weight, kls=[]))
        for index, row in enumerate(rows)
    )
    outputs = decoder.generate(
        [prompts[row.id] for row in rows],
        [advisor_prompts[row.id] for row in rows],
    )
    records = []
    for index, (row, output) in enumerate(zip(rows, outputs, strict=True)):
        key = (row.id, row.completion_index)
        request = requests[index]
        if len(request.kls) != len(row.steps):
            raise RuntimeError(f"replay ended before the recorded path for {key!r}: {len(request.kls)}/{len(row.steps)}")
        if output.text != completions[key]:
            raise RuntimeError(f"replayed completion text differs for {key!r}")
        records.append({"id": row.id, "completion_index": row.completion_index, "value": {"kl": request.kls}})
    with fsspec.open(chunk.output_path, "wt", compression="gzip") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _engine_placement_from_dict(data: dict[str, Any]) -> EnginePlacement:
    bounds = data["chips_per_process_bounds"]
    return EnginePlacement(
        visible_chips=tuple(data["visible_chips"]),
        chips_per_process_bounds=(bounds[0], bounds[1], bounds[2]),
        tensor_parallel_size=data["tensor_parallel_size"],
    )


def _child_config_from_file(path: str) -> JointDecodeLocalWorkerConfig:
    with open(path) as f:
        data = json.load(f)
    sampling_data = dict(data["sampling"])
    sampling_data["selection_rule"] = XtokSelectionRule(sampling_data["selection_rule"])
    if sampling_data["stop"] is not None:
        sampling_data["stop"] = tuple(sampling_data["stop"])
    placement_data = data["placement"]
    return JointDecodeLocalWorkerConfig(
        decoder_model_path=data["decoder_model_path"],
        advisor_model_path=data["advisor_model_path"],
        prompts_path=data["prompts_path"],
        advisor_prompts_path=data["advisor_prompts_path"],
        alg_output_path=data["alg_output_path"],
        token_paths_path=data["token_paths_path"],
        sampling=JointDecodeReplaySamplingConfig(**sampling_data),
        decoder_model=JointDecodeModelConfig(**data["decoder_model"]),
        advisor_model=JointDecodeModelConfig(**data["advisor_model"]),
        ledger_path=data["ledger_path"],
        poll_backoff=data["poll_backoff"],
        microbatch_size=data["microbatch_size"],
        barrier_timeout_s=data["barrier_timeout_s"],
        owner=data["owner"],
        placement=JointDecodePlacement(
            decoder=_engine_placement_from_dict(placement_data["decoder"]),
            advisor=_engine_placement_from_dict(placement_data["advisor"]),
        ),
        max_num_batched_tokens=data["max_num_batched_tokens"],
    )


def _load_vocab(model_path: str) -> xtok_selection.Vocab:
    resolved = discover_hf_checkpoints(model_path)[-1]
    logger.info("Resolved tokenizer %s -> %s", model_path, resolved)
    return xtok_selection.load_vocab(load_tokenizer(resolved))


def _load_prompts(prompts_path: str) -> dict[str, str]:
    return {row["id"]: row["prompt"] for row in read_prompt_rows(prompts_path)}


def _run_joint_decode_local_worker(config: JointDecodeLocalWorkerConfig) -> None:
    # The joint-decode package rides the Linux-only vllm extra.
    from experiments.downstream_scaling.evals.algorithms import joint_decode_backend  # noqa: PLC0415

    visible_chips = config.placement.decoder.visible_chips + config.placement.advisor.visible_chips
    expected_visible_chips = ",".join(str(chip) for chip in visible_chips)
    actual_visible_chips = os.environ.get("TPU_VISIBLE_CHIPS")
    if actual_visible_chips != expected_visible_chips:
        raise ValueError(f"TPU_VISIBLE_CHIPS={actual_visible_chips!r}, expected {expected_visible_chips!r}")

    prompts = _load_prompts(config.prompts_path)
    advisor_prompts = prompts if config.advisor_prompts_path is None else _load_prompts(config.advisor_prompts_path)
    if prompts.keys() != advisor_prompts.keys():
        raise ValueError(
            f"Advisor prompt ids do not match decoder prompt ids: "
            f"missing={sorted(prompts.keys() - advisor_prompts.keys())}, "
            f"unexpected={sorted(advisor_prompts.keys() - prompts.keys())}"
        )
    completions = {
        (row["id"], index): completion["text"]
        for row in read_completion_rows(completions_file(config.alg_output_path))
        for index, completion in enumerate(row["completions"])
    }
    vocab_a = _load_vocab(config.decoder_model_path)
    vocab_b = _load_vocab(config.advisor_model_path)
    requests: dict[int, _ReplayRequest] = {}

    def engine_params(model_path: str, model: JointDecodeModelConfig) -> joint_decode_backend.EngineModelParams:
        return joint_decode_backend.EngineModelParams(
            model_path=model_path,
            max_model_len=model.max_model_len,
            gpu_memory_utilization=model.gpu_memory_utilization,
            enable_prefix_caching=model.enable_prefix_caching,
            apply_rpa_block_size_patch=model.apply_rpa_block_size_patch,
        )

    max_num_batched_tokens = (
        config.max_num_batched_tokens
        if config.max_num_batched_tokens is not None
        else max(config.decoder_model.max_model_len, config.advisor_model.max_model_len) + config.microbatch_size
    )
    with joint_decode_backend.open_joint_decoder(
        decoder=engine_params(config.decoder_model_path, config.decoder_model),
        advisor=engine_params(config.advisor_model_path, config.advisor_model),
        max_tokens=config.sampling.max_tokens,
        advisor_max_tokens=config.sampling.advisor_max_tokens,
        top_k_a=config.sampling.top_k_a,
        top_k_b=config.sampling.top_k_b,
        seed=config.sampling.seed,
        stop=tuple(config.sampling.stop or ()),
        select_token=_make_replay_select_token(config.sampling, vocab_a, vocab_b, requests),
        decoder_placement=config.placement.decoder,
        advisor_placement=config.placement.advisor,
        max_microbatch_size=config.microbatch_size,
        max_num_batched_tokens=max_num_batched_tokens,
        barrier_timeout_s=config.barrier_timeout_s,
    ) as decoder:
        while True:
            with ledger.claim_next_chunk(config.ledger_path, config.owner) as claim:
                if claim is None:
                    summary = ledger.summarize(config.ledger_path)
                    if summary.done == summary.total:
                        return
                    time.sleep(config.poll_backoff)
                    continue

                write_replay_chunk(
                    ReplayChunkSpec(**claim.chunk),
                    decoder=decoder,
                    prompts=prompts,
                    advisor_prompts=advisor_prompts,
                    completions=completions,
                    token_paths_path=config.token_paths_path,
                    selection_rule=config.sampling.selection_rule,
                    requests=requests,
                )
                ledger.mark_done(claim)


def joint_decode_tp1_placements(chips_per_vm: int) -> tuple[JointDecodePlacement, ...]:
    if chips_per_vm % 2 != 0:
        raise ValueError(f"joint decode needs an even number of chips per VM, got {chips_per_vm}")
    return tuple(
        JointDecodePlacement(
            decoder=EnginePlacement((chip,), (1, 1, 1), 1),
            advisor=EnginePlacement((chip + 1,), (1, 1, 1), 1),
        )
        for chip in range(0, chips_per_vm, 2)
    )


def joint_decode_pool_configs(
    model_key: str,
    worker_pools: tuple[WorkerPoolConfig, ...],
    overrides: Mapping[tuple[str, str], tuple[JointDecodePlacement, ...]],
) -> tuple[JointDecodePoolConfig, ...]:
    configs = []
    for pool in worker_pools:
        placements = overrides.get((model_key, pool.pool_id))
        if placements is None:
            placements = joint_decode_tp1_placements(pool.chips_per_vm)
        configs.append(JointDecodePoolConfig(pool=pool, placements=placements))
    return tuple(configs)


def _placement_chips(placement: JointDecodePlacement) -> tuple[int, ...]:
    return placement.decoder.visible_chips + placement.advisor.visible_chips


def _child_owner(pool_id: str, shard_idx: int, placement: JointDecodePlacement) -> str:
    chips = ",".join(str(chip) for chip in _placement_chips(placement))
    return f"{pool_id}/shard-{shard_idx}/chips-{chips}"


def _write_child_config(tmpdir: Path, config: JointDecodeLocalWorkerConfig) -> Path:
    chips = "-".join(str(chip) for chip in _placement_chips(config.placement))
    path = tmpdir / f"child_chips_{chips}.json"
    with open(path, "wt") as f:
        json.dump(asdict(config), f, sort_keys=True)
    return path


def _stream_child_output(proc: subprocess.Popen[str], *, label: str) -> list[threading.Thread]:
    threads = []

    def stream(pipe, stream_name: str) -> None:
        assert pipe is not None
        for line in pipe:
            logger.info("joint-decode-avg-kl local worker %s %s: %s", label, stream_name, line.rstrip())

    for pipe, stream_name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
        thread = threading.Thread(target=stream, args=(pipe, stream_name), daemon=True)
        thread.start()
        threads.append(thread)
    return threads


def _spawn_child(
    *,
    tmpdir: Path,
    config: JointDecodeAvgKlStepConfig,
    token_paths_path: str,
    ledger_path: str,
    pool_id: str,
    shard_idx: int,
    placement: JointDecodePlacement,
) -> tuple[subprocess.Popen[str], list[threading.Thread]]:
    child_config = JointDecodeLocalWorkerConfig(
        decoder_model_path=config.decoder_model_path,
        advisor_model_path=config.advisor_model_path,
        prompts_path=config.prompts_path,
        advisor_prompts_path=config.advisor_prompts_path,
        alg_output_path=config.alg_output_path,
        token_paths_path=token_paths_path,
        sampling=config.sampling,
        decoder_model=config.decoder_model,
        advisor_model=config.advisor_model,
        ledger_path=ledger_path,
        poll_backoff=config.poll_backoff,
        microbatch_size=config.microbatch_size,
        barrier_timeout_s=config.barrier_timeout_s,
        owner=_child_owner(pool_id, shard_idx, placement),
        placement=placement,
        max_num_batched_tokens=config.max_num_batched_tokens,
    )
    config_path = _write_child_config(tmpdir, child_config)
    chip_label = ",".join(str(chip) for chip in _placement_chips(placement))

    # The child only partitions chips (a validated invariant of the pair
    # harness); JAX/vLLM process env is owned per engine worker by the
    # joint-decode package.
    env = os.environ.copy()
    env["TPU_VISIBLE_CHIPS"] = chip_label

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.downstream_scaling.evals.measurements.joint_decode_avg_kl",
        "--xregion-worker-child-config",
        str(config_path),
    ]
    logger.info("Launching joint-decode-avg-kl local worker shard=%d chips=%s", shard_idx, chip_label)
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
                    logger.warning(
                        "joint-decode-avg-kl local worker exited after ledger completion with rc=%d",
                        return_code,
                    )
                    continue
                _terminate_children(procs)
                raise RuntimeError(
                    f"joint-decode-avg-kl local worker failed with rc={return_code}; "
                    f"ledger is {summary.done}/{summary.total} done"
                )

        if ledger_complete and all_done:
            break
        if all_done:
            raise RuntimeError(
                f"joint-decode-avg-kl local workers exited before completion: "
                f"{summary.done}/{summary.total} chunks done"
            )

        time.sleep(1.0)

    for thread in threads:
        thread.join(timeout=5)


def _stage_token_paths(token_paths_path: str, tmpdir: Path) -> str:
    local_path = tmpdir / TOKEN_PATHS_FILENAME
    with fsspec.open(token_paths_path, "rb") as source, local_path.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    return str(local_path)


def _supervise_joint_decode_worker(
    _worker_ids: Iterator[int],
    shard_info: ShardInfo,
    *,
    config: JointDecodeAvgKlStepConfig,
    token_paths_path: str,
    ledger_path: str,
    pool: WorkerPoolConfig,
    placements: tuple[JointDecodePlacement, ...],
) -> Iterator[dict[str, object]]:
    if os.environ.get("TPU_VISIBLE_CHIPS") is not None:
        raise ValueError(
            "joint decode avg KL supervisor expects to own the full TPU VM; TPU_VISIBLE_CHIPS is already set"
        )

    logger.info(
        "Starting joint-decode-avg-kl supervisor pool=%s shard=%d chips_per_vm=%d placements=%s",
        pool.pool_id,
        shard_info.shard_idx,
        pool.chips_per_vm,
        placements,
    )

    with tempfile.TemporaryDirectory(prefix="joint_decode_avg_kl_local_workers_") as tmp:
        tmpdir = Path(tmp)
        local_token_paths_path = _stage_token_paths(token_paths_path, tmpdir)
        procs: list[subprocess.Popen[str]] = []
        threads: list[threading.Thread] = []
        try:
            for placement in placements:
                proc, proc_threads = _spawn_child(
                    tmpdir=tmpdir,
                    config=config,
                    token_paths_path=local_token_paths_path,
                    ledger_path=ledger_path,
                    pool_id=pool.pool_id,
                    shard_idx=shard_info.shard_idx,
                    placement=placement,
                )
                procs.append(proc)
                threads.extend(proc_threads)
            _wait_for_children(procs, threads, ledger_path)
        except Exception:
            _terminate_children(procs)
            raise

    yield {"status": "done", "pool_id": pool.pool_id, "shard_idx": shard_info.shard_idx}


def _aggregate_statistic_row(
    prompt_id: str,
    items: Iterator[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    records = list(items)
    indices = [record["completion_index"] for record in records]
    if indices != list(range(len(records))):
        raise ValueError(f"Statistic completion indices for {prompt_id!r} are {indices}, expected 0..{len(records) - 1}")
    return {
        "id": prompt_id,
        "values": [record["value"] for record in records],
        "metadata": metadata,
    }


def run_joint_decode_avg_kl(config: JointDecodeAvgKlStepConfig) -> None:
    """Validate source artifacts, run replay workers, and aggregate per-completion KL."""
    if not config.worker_pools:
        raise ValueError("joint decode avg KL requires at least one worker pool")

    num_records = _validate_token_paths(config.alg_output_path, config.sampling.selection_rule)
    token_paths_path = os.path.join(config.alg_output_path, TOKEN_PATHS_FILENAME)
    chunks_dir = os.path.join(config.output_path, "chunks", f"chunk_size={config.chunk_size}")
    chunks = _chunk_specs(chunks_dir, num_records, config.chunk_size)
    ledger_path = ledger.convert_mirror_path(
        ledger_prefix=config.ledger_prefix,
        output_path=config.output_path,
    )
    ledger.ensure_manifest(ledger_path, chunks)
    pool_configs = {pool_config.pool.pool_id: pool_config for pool_config in config.worker_pools}

    def make_process_shard(pool: WorkerPoolConfig):
        pool_config = pool_configs[pool.pool_id]
        return functools.partial(
            _supervise_joint_decode_worker,
            config=config,
            token_paths_path=token_paths_path,
            ledger_path=ledger_path,
            pool=pool,
            placements=pool_config.placements,
        )

    xregion_pool.run_worker_pools(
        worker_pools=tuple(pool_config.pool for pool_config in config.worker_pools),
        ledger_path=ledger_path,
        make_process_shard=make_process_shard,
        poll_backoff=config.poll_backoff,
        heartbeat_timeout=config.heartbeat_timeout,
    )

    summary = ledger.summarize(ledger_path)
    if summary.done != summary.total:
        raise RuntimeError(f"joint decode avg KL incomplete: {summary.done}/{summary.total} chunks done")

    metadata = {
        "statistic": "joint_decode_avg_kl",
        "decoder_model_path": config.decoder_model_path,
        "advisor_model_path": config.advisor_model_path,
        "selection_rule": config.sampling.selection_rule.value,
        "temperature": config.sampling.temperature,
        "top_k_a": config.sampling.top_k_a,
        "top_k_b": config.sampling.top_k_b,
        "prefix_credit": config.sampling.prefix_credit,
    }
    done_ids = set(ledger.done_chunk_ids(ledger_path))
    path = statistics_file(config.output_path)
    pipeline = (
        Dataset.from_list([chunk.output_path for chunk in chunks if chunk.chunk_id in done_ids])
        .load_jsonl()
        .group_by(
            key=lambda record: record["id"],
            reducer=functools.partial(_aggregate_statistic_row, metadata=metadata),
            sort_by=lambda record: record["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    ZephyrContext(
        name="joint-decode-avg-kl-aggregate",
        max_workers=config.aggregate_workers,
        resources=ResourceConfig(cpu=1, ram="4g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(pipeline)

    written = sum(len(row["values"]) for row in read_statistic_rows(path))
    if written != num_records:
        raise ValueError(f"Wrote {written} KL statistic values for {num_records} token-path records")
    logger.info("Wrote %d joint-decode-avg KL statistic values to %s", written, path)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xregion-worker-child-config", required=True)
    args = parser.parse_args()
    _run_joint_decode_local_worker(_child_config_from_file(args.xregion_worker_child_config))


if __name__ == "__main__":
    _main()
