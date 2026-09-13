# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KL(decoder || advisor) along recorded same-tokenizer token paths.

Two vLLM engines per chip pair teacher-force the decoder and the advisor along
each recorded completion, each under its own side's prompt. At every decision
boundary the KL is taken over the union of the two top-k token-id sets, each
side's minimum top-k logprob standing in for ids it did not rank, both sides
softmaxed over the union at the statistic's configured temperature:
``select_avg_logits``'s union and floor applied to a KL. Each completion value
is ``{"kl": list[float]}``, one entry per recorded decision step.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import fsspec
from fray.cluster import ResourceConfig
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo

from experiments.downstream_scaling.evals.framework.schema import read_prompt_rows
from experiments.downstream_scaling.evals.framework.xregion import ledger
from experiments.downstream_scaling.evals.framework.xregion import pool as xregion_pool
from experiments.downstream_scaling.evals.framework.xregion.pool import (
    EnginePlacement,
    WorkerPoolConfig,
    validate_pool_placements,
)
from experiments.downstream_scaling.evals.measurements.schema import read_statistic_rows, statistics_file
from experiments.downstream_scaling.evals.utils import discover_hf_checkpoints, localize_mirror_path, version_path

logger = logging.getLogger(__name__)

VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
    "RUNAI_STREAMER_MEMORY_LIMIT": "4294967296",
}

DEFAULT_HEARTBEAT_TIMEOUT = 2 * 60
DEFAULT_LEDGER_PREFIX = "gs://marin-us-central2"
DEFAULT_POLL_BACKOFF = 10.0
# How long to wait for an engine's exit code after its stdout hits EOF.
ENGINE_EXIT_TIMEOUT = 60.0
VLLM_CONSTRUCTOR_SEED = 0
# Prompt-logprob scoring allocates a [batched tokens, vocab] logits buffer at
# runtime, above the gpu_memory_utilization cap; at vLLM's default of 8192
# tokens that is ~8 GiB, more than the headroom on a 32 GiB chip.
MAX_NUM_BATCHED_TOKENS = 2048
TOKEN_PATHS_FILENAME = "token_paths.jsonl.gz"
# Engine stdout lines carrying this prefix are protocol; every other line is a
# library log forwarded to the pair child's logger.
IPC_PREFIX = "__TOKEN_PATH_KL__:"


class TokenPathSide(StrEnum):
    A = "a"
    B = "b"


class TokenPathStep(TypedDict):
    bytes_hex: str
    tokens_a: list[int]
    tokens_b: list[int]


class TokenPathRow(TypedDict):
    id: str
    completion_index: int
    steps: list[TokenPathStep]


class _FlatPromptLogprobs(Protocol):
    start_indices: list[int]
    end_indices: list[int]
    token_ids: list[int]
    logprobs: list[float]


@dataclass(frozen=True)
class TokenPathModelConfig:
    model_path: str | InputName | MirroredValue
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    apply_rpa_block_size_patch: bool = False


@dataclass(frozen=True)
class TokenPathPairPlacement:
    """One decoder engine and one advisor engine on disjoint chips of a VM."""

    decoder: EnginePlacement
    advisor: EnginePlacement


@dataclass(frozen=True)
class TokenPathPoolConfig:
    pool: WorkerPoolConfig
    placements: tuple[TokenPathPairPlacement, ...]

    def __post_init__(self) -> None:
        engines = tuple(engine for pair in self.placements for engine in (pair.decoder, pair.advisor))
        validate_pool_placements(self.pool, engines)


@dataclass(frozen=True)
class TokenPathExecutionConfig:
    worker_pools: tuple[TokenPathPoolConfig, ...]
    microbatch_size: int
    ledger_prefix: str = DEFAULT_LEDGER_PREFIX
    chunk_size: int = 512
    heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT
    poll_backoff: float = DEFAULT_POLL_BACKOFF
    aggregate_workers: int = 32

    def __post_init__(self) -> None:
        if self.microbatch_size < 1:
            raise ValueError(f"microbatch_size must be >= 1 (got {self.microbatch_size})")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1 (got {self.chunk_size})")
        if self.aggregate_workers < 1:
            raise ValueError(f"aggregate_workers must be >= 1 (got {self.aggregate_workers})")


@dataclass(frozen=True)
class TokenPathKl:
    decoder_model: TokenPathModelConfig
    advisor_model: TokenPathModelConfig
    execution: TokenPathExecutionConfig
    k: int = 16
    # The advisor's prompts step output when the sweep prompted the advisor
    # differently from the decoder; None means the advisor saw the task prompt
    # and both engines read prompts_path.
    advisor_prompts_path: str | InputName | MirroredValue | None = None
    # Sampling temperature both sides are softmaxed at. Versioned only when it
    # departs from the default, so temperature-1 runs keep their hash.
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1 (got {self.k})")

    def make_statistic_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        alg_output_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return make_token_path_kl_step(
            name=name,
            prompts_path=prompts_path,
            alg_output_path=alg_output_path,
            statistic=self,
        )


@dataclass(frozen=True)
class _EngineConfig:
    max_model_len: int | None
    gpu_memory_utilization: float | None
    apply_rpa_block_size_patch: bool


@dataclass(frozen=True)
class KlStepConfig:
    output_path: str
    prompts_path: str
    advisor_prompts_path: str | None
    alg_output_path: str
    decoder_model_path: str
    advisor_model_path: str
    decoder_model: _EngineConfig
    advisor_model: _EngineConfig
    worker_pools: tuple[TokenPathPoolConfig, ...]
    k: int
    microbatch_size: int
    ledger_prefix: str
    chunk_size: int
    heartbeat_timeout: float
    poll_backoff: float
    aggregate_workers: int
    temperature: float


@dataclass(frozen=True)
class KlChunkSpec:
    chunk_id: int
    chunk_start: int
    chunk_end: int
    output_path: str


@dataclass(frozen=True)
class KlLocalWorkerConfig:
    prompts_path: str
    advisor_prompts_path: str | None
    token_paths_path: str
    decoder_model_path: str
    advisor_model_path: str
    decoder_model: _EngineConfig
    advisor_model: _EngineConfig
    k: int
    microbatch_size: int
    ledger_path: str
    poll_backoff: float
    owner: str
    placement: TokenPathPairPlacement
    temperature: float


@dataclass(frozen=True)
class EngineWorkerConfig:
    side: TokenPathSide
    model_path: str
    prompts_path: str
    token_paths_path: str
    model: _EngineConfig
    k: int
    microbatch_size: int
    placement: EnginePlacement


@dataclass(frozen=True)
class _ScoringRequest:
    id: str
    completion_index: int
    prompt_length: int
    token_ids: list[int]
    step_lengths: tuple[int, ...]


def _engine_config(model: TokenPathModelConfig) -> _EngineConfig:
    return _EngineConfig(
        max_model_len=versioned(model.max_model_len),  # type: ignore[arg-type]
        gpu_memory_utilization=model.gpu_memory_utilization,
        apply_rpa_block_size_patch=model.apply_rpa_block_size_patch,
    )


def make_token_path_kl_step(
    *,
    name: str,
    prompts_path: str | InputName | MirroredValue,
    alg_output_path: str | InputName | MirroredValue,
    statistic: TokenPathKl,
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(
            run_token_path_kl,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
            pip_dependency_groups=["vllm"],
            env_vars=VLLM_TPU_ENV_VARS,
        ),
        config=KlStepConfig(
            output_path=this_output_path(),
            prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
            advisor_prompts_path=(
                None
                if statistic.advisor_prompts_path is None
                else version_path(statistic.advisor_prompts_path)  # type: ignore[arg-type]
            ),
            alg_output_path=version_path(alg_output_path),  # type: ignore[arg-type]
            decoder_model_path=version_path(statistic.decoder_model.model_path),  # type: ignore[arg-type]
            advisor_model_path=version_path(statistic.advisor_model.model_path),  # type: ignore[arg-type]
            decoder_model=_engine_config(statistic.decoder_model),
            advisor_model=_engine_config(statistic.advisor_model),
            worker_pools=statistic.execution.worker_pools,
            k=versioned(statistic.k),  # type: ignore[arg-type]
            temperature=(
                statistic.temperature if statistic.temperature == 1.0 else versioned(statistic.temperature)  # type: ignore[arg-type]
            ),
            microbatch_size=statistic.execution.microbatch_size,
            ledger_prefix=statistic.execution.ledger_prefix,
            chunk_size=versioned(statistic.execution.chunk_size),  # type: ignore[arg-type]
            heartbeat_timeout=statistic.execution.heartbeat_timeout,
            poll_backoff=statistic.execution.poll_backoff,
            aggregate_workers=statistic.execution.aggregate_workers,
        ),
    )


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


def _token_path_row(raw: Any, path: str) -> TokenPathRow:
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

    return {
        "id": raw["id"],
        "completion_index": completion_index,
        "steps": [
            {
                "bytes_hex": step["bytes_hex"],
                "tokens_a": list(step["tokens_a"]),
                "tokens_b": list(step["tokens_b"]),
            }
            for step in raw["steps"]
        ],
    }


def read_token_path_rows(path: str) -> Iterator[TokenPathRow]:
    seen: set[tuple[str, int]] = set()
    with fsspec.open(path, "rt", compression="gzip") as f:
        for line in f:
            row = _token_path_row(json.loads(line), path)
            key = (row["id"], row["completion_index"])
            if key in seen:
                raise ValueError(f"Duplicate token-path row {key!r}: {path}")
            seen.add(key)
            yield row


def _count_same_tokenizer_records(path: str) -> int:
    """Count sidecar rows, failing on any step whose two sides forced different ids.

    The token-id KL is defined only when both models share a tokenizer, in
    which case every same-tokenizer selection rule forces the same id on both
    sides.
    """
    count = 0
    for row in read_token_path_rows(path):
        for step in row["steps"]:
            if step["tokens_a"] != step["tokens_b"]:
                raise ValueError(
                    f"Token-path row {(row['id'], row['completion_index'])!r} has tokens_a != tokens_b; "
                    "token-id KL requires decoder and advisor to share a tokenizer"
                )
        count += 1
    return count


def _read_row_range(path: str, start: int, end: int) -> list[TokenPathRow]:
    rows = []
    with fsspec.open(path, "rt", compression="gzip") as f:
        for index, line in enumerate(f):
            if index >= end:
                break
            if index >= start:
                rows.append(_token_path_row(json.loads(line), path))
    return rows


def _tokens_for_side(step: TokenPathStep, side: TokenPathSide) -> list[int]:
    if side is TokenPathSide.A:
        return step["tokens_a"]
    return step["tokens_b"]


def _prepare_scoring_request(
    row: TokenPathRow,
    prompt: str,
    tokenizer: Any,
    side: TokenPathSide,
) -> _ScoringRequest:
    prompt_ids = list(tokenizer.encode(prompt))
    if not prompt_ids:
        raise ValueError(f"Prompt {row['id']!r} encodes to zero tokens")

    completion_ids: list[int] = []
    step_lengths: list[int] = []
    for step in row["steps"]:
        step_tokens = _tokens_for_side(step, side)
        if not step_tokens:
            raise ValueError(
                f"Token-path row {(row['id'], row['completion_index'])!r} has an empty side-{side.value} step"
            )
        completion_ids.extend(step_tokens)
        step_lengths.append(len(step_tokens))

    return _ScoringRequest(
        id=row["id"],
        completion_index=row["completion_index"],
        prompt_length=len(prompt_ids),
        token_ids=prompt_ids + completion_ids,
        step_lengths=tuple(step_lengths),
    )


def _topk_at_position(
    prompt_logprobs: _FlatPromptLogprobs,
    position: int,
    k: int,
) -> tuple[list[int], list[float]]:
    start = prompt_logprobs.start_indices[position]
    end = prompt_logprobs.end_indices[position]
    if end - start != k + 1:
        raise ValueError(f"Prompt-logprob position {position} has width {end - start}, expected {k + 1}")
    return (
        prompt_logprobs.token_ids[start + 1 : end],
        prompt_logprobs.logprobs[start + 1 : end],
    )


def _boundary_topk_steps(
    request: _ScoringRequest,
    prompt_logprobs: _FlatPromptLogprobs,
    k: int,
) -> list[dict[str, list]]:
    """The top-k at the first token of each recorded step, the position the joint decoder chose at."""
    if len(prompt_logprobs.start_indices) != len(request.token_ids):
        raise ValueError(
            f"Prompt-logprob length {len(prompt_logprobs.start_indices)} does not match "
            f"token length {len(request.token_ids)}"
        )

    position = request.prompt_length
    steps = []
    for step_length in request.step_lengths:
        topk_ids, topk_logprobs = _topk_at_position(prompt_logprobs, position, k)
        steps.append({"topk_ids": topk_ids, "topk_logprobs": topk_logprobs})
        position += step_length
    return steps


def _log_softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    log_total = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    return [value - log_total for value in values]


def _step_kl(step_a: dict[str, list], step_b: dict[str, list], temperature: float) -> float:
    """KL(p || q) over the union of the two top-k id sets, each side's minimum
    as its floor for ids it did not rank (``select_avg_logits``'s convention),
    both sides softmaxed at ``temperature``."""
    a = {i: lp / temperature for i, lp in zip(step_a["topk_ids"], step_a["topk_logprobs"], strict=True)}
    b = {i: lp / temperature for i, lp in zip(step_b["topk_ids"], step_b["topk_logprobs"], strict=True)}
    a_floor, b_floor = min(a.values()), min(b.values())
    union = sorted(set(a) | set(b))
    log_p = _log_softmax([a.get(token_id, a_floor) for token_id in union])
    log_q = _log_softmax([b.get(token_id, b_floor) for token_id in union])
    # Float error can make the sum slightly negative; clamp as kl_bytes_union does.
    return max(sum(math.exp(lp) * (lp - lq) for lp, lq in zip(log_p, log_q, strict=True)), 0.0)


def _load_vllm(model_path: str, tensor_parallel_size: int, model: _EngineConfig):
    for key, value in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(key, value)

    if model.apply_rpa_block_size_patch:
        from joint_decode.tpu.worker import (  # pyrefly: ignore[missing-import]  # noqa: PLC0415
            _patch_rpa_kernel_block_sizes,
        )

        _patch_rpa_kernel_block_sizes()

    from vllm import LLM, SamplingParams, TokensPrompt  # noqa: PLC0415

    resolved_model_path = discover_hf_checkpoints(model_path)[-1]
    resolved_model_path = localize_mirror_path(resolved_model_path)
    logger.info("Resolved %s -> %s", model_path, resolved_model_path)

    model_kwargs: dict[str, Any] = {}
    if model.max_model_len is not None:
        model_kwargs["max_model_len"] = model.max_model_len
    if model.gpu_memory_utilization is not None:
        model_kwargs["gpu_memory_utilization"] = model.gpu_memory_utilization

    llm = LLM(
        model=resolved_model_path,
        trust_remote_code=True,
        load_format="runai_streamer",
        seed=VLLM_CONSTRUCTOR_SEED,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=1,
        enable_prefix_caching=False,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        **model_kwargs,
    )
    return llm, SamplingParams, TokensPrompt, llm.get_tokenizer()


def _load_prompts(prompts_path: str) -> dict[str, str]:
    return {row["id"]: row["prompt"] for row in read_prompt_rows(prompts_path)}


def _emit_ipc(payload: dict[str, Any]) -> None:
    sys.stdout.write(IPC_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _read_ipc(proc: subprocess.Popen[str], *, label: str) -> dict[str, Any]:
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            # EOF arrives while the engine is still tearing down its TPU
            # state, so poll() would race it; wait for the real exit code.
            try:
                return_code: int | None = proc.wait(timeout=ENGINE_EXIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                return_code = None
            raise RuntimeError(f"KL engine {label} closed stdout before responding (rc={return_code})")
        line = line.rstrip("\n")
        if line.startswith(IPC_PREFIX):
            return json.loads(line[len(IPC_PREFIX) :])
        if line:
            logger.info("kl engine %s stdout: %s", label, line)


def _run_engine(config: EngineWorkerConfig) -> None:
    """Score sidecar row ranges read from stdin, one tagged response line per request."""
    expected_visible_chips = ",".join(str(chip) for chip in config.placement.visible_chips)
    actual_visible_chips = os.environ.get("TPU_VISIBLE_CHIPS")
    if actual_visible_chips != expected_visible_chips:
        raise ValueError(f"TPU_VISIBLE_CHIPS={actual_visible_chips!r}, expected {expected_visible_chips!r}")

    llm, SamplingParams, TokensPrompt, tokenizer = _load_vllm(
        config.model_path, config.placement.tensor_parallel_size, config.model
    )
    prompts = _load_prompts(config.prompts_path)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=config.k,
        flat_logprobs=True,
        detokenize=False,
    )

    for line in iter(sys.stdin.readline, ""):
        request = json.loads(line)
        rows = _read_row_range(config.token_paths_path, request["chunk_start"], request["chunk_end"])
        records = []
        for start in range(0, len(rows), config.microbatch_size):
            batch = [
                _prepare_scoring_request(row, prompts[row["id"]], tokenizer, config.side)
                for row in rows[start : start + config.microbatch_size]
            ]
            outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=scoring_request.token_ids) for scoring_request in batch],
                sampling_params,
                use_tqdm=False,
            )
            for scoring_request, output in zip(batch, outputs, strict=True):
                if output.prompt_logprobs is None:
                    raise RuntimeError("vLLM returned no prompt logprobs")
                records.append(
                    {
                        "id": scoring_request.id,
                        "completion_index": scoring_request.completion_index,
                        "steps": _boundary_topk_steps(
                            scoring_request, cast(_FlatPromptLogprobs, output.prompt_logprobs), config.k
                        ),
                    }
                )
        _emit_ipc({"records": records})


class _EngineProcess:
    """One side's scoring engine, driven over its stdin/stdout pipes."""

    def __init__(self, proc: subprocess.Popen[str], *, label: str, stderr_thread: threading.Thread) -> None:
        self.proc = proc
        self.label = label
        self._stderr_thread = stderr_thread

    def submit(self, chunk_start: int, chunk_end: int) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"chunk_start": chunk_start, "chunk_end": chunk_end}) + "\n")
        self.proc.stdin.flush()

    def collect(self) -> list[dict[str, Any]]:
        return _read_ipc(self.proc, label=self.label)["records"]

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        return_code = self.proc.wait()
        self._stderr_thread.join(timeout=5)
        if return_code != 0:
            raise RuntimeError(f"KL engine {self.label} exited with rc={return_code}")


def _run_kl_chunk(chunk: KlChunkSpec, *, decoder: _EngineProcess, advisor: _EngineProcess, temperature: float) -> None:
    decoder.submit(chunk.chunk_start, chunk.chunk_end)
    advisor.submit(chunk.chunk_start, chunk.chunk_end)
    records_a = decoder.collect()
    records_b = advisor.collect()
    if len(records_a) != len(records_b):
        raise ValueError(f"Chunk {chunk.chunk_id}: decoder returned {len(records_a)} records, advisor {len(records_b)}")

    records = []
    for record_a, record_b in zip(records_a, records_b, strict=True):
        key = (record_a["id"], record_a["completion_index"])
        if key != (record_b["id"], record_b["completion_index"]):
            raise ValueError(
                f"Chunk {chunk.chunk_id}: decoder record {key!r} paired with advisor record "
                f"{(record_b['id'], record_b['completion_index'])!r}"
            )
        if len(record_a["steps"]) != len(record_b["steps"]):
            raise ValueError(
                f"Step count mismatch for {key!r}: {len(record_a['steps'])} decoder, {len(record_b['steps'])} advisor"
            )
        kls = [
            _step_kl(step_a, step_b, temperature)
            for step_a, step_b in zip(record_a["steps"], record_b["steps"], strict=True)
        ]
        if not all(math.isfinite(kl) for kl in kls):
            raise ValueError(f"Non-finite KL for {key!r}")
        records.append({"id": key[0], "completion_index": key[1], "value": {"kl": kls}})

    with fsspec.open(chunk.output_path, "wt", compression="gzip") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _chunk_specs(chunks_dir: str, num_records: int, chunk_size: int) -> list[KlChunkSpec]:
    return [
        KlChunkSpec(
            chunk_id=chunk_id,
            chunk_start=start,
            chunk_end=min(start + chunk_size, num_records),
            output_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz"),
        )
        for chunk_id, start in enumerate(range(0, num_records, chunk_size))
    ]


def token_path_pair_tp1_placements(chips_per_vm: int) -> tuple[TokenPathPairPlacement, ...]:
    """Chip pairs (0,1), (2,3), ...: decoder on the even chip, advisor on the odd."""
    if chips_per_vm % 2 != 0:
        raise ValueError(f"token-path KL needs an even number of chips per VM, got {chips_per_vm}")
    return tuple(
        TokenPathPairPlacement(
            decoder=EnginePlacement((chip,), (1, 1, 1), 1),
            advisor=EnginePlacement((chip + 1,), (1, 1, 1), 1),
        )
        for chip in range(0, chips_per_vm, 2)
    )


def _engine_placement_from_dict(data: dict[str, Any]) -> EnginePlacement:
    bounds = data["chips_per_process_bounds"]
    return EnginePlacement(
        visible_chips=tuple(data["visible_chips"]),
        chips_per_process_bounds=(bounds[0], bounds[1], bounds[2]),
        tensor_parallel_size=data["tensor_parallel_size"],
    )


def _child_config_from_file(path: str) -> KlLocalWorkerConfig:
    with open(path) as f:
        data = json.load(f)
    return KlLocalWorkerConfig(
        prompts_path=data["prompts_path"],
        advisor_prompts_path=data["advisor_prompts_path"],
        token_paths_path=data["token_paths_path"],
        decoder_model_path=data["decoder_model_path"],
        advisor_model_path=data["advisor_model_path"],
        decoder_model=_EngineConfig(**data["decoder_model"]),
        advisor_model=_EngineConfig(**data["advisor_model"]),
        k=data["k"],
        microbatch_size=data["microbatch_size"],
        ledger_path=data["ledger_path"],
        poll_backoff=data["poll_backoff"],
        owner=data["owner"],
        placement=TokenPathPairPlacement(
            decoder=_engine_placement_from_dict(data["placement"]["decoder"]),
            advisor=_engine_placement_from_dict(data["placement"]["advisor"]),
        ),
        temperature=data["temperature"],
    )


def _engine_config_from_file(path: str) -> EngineWorkerConfig:
    with open(path) as f:
        data = json.load(f)
    return EngineWorkerConfig(
        side=TokenPathSide(data["side"]),
        model_path=data["model_path"],
        prompts_path=data["prompts_path"],
        token_paths_path=data["token_paths_path"],
        model=_EngineConfig(**data["model"]),
        k=data["k"],
        microbatch_size=data["microbatch_size"],
        placement=_engine_placement_from_dict(data["placement"]),
    )


def _stream_stderr(proc: subprocess.Popen[str], *, label: str) -> threading.Thread:
    def stream() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            logger.info("kl engine %s stderr: %s", label, line.rstrip())

    thread = threading.Thread(target=stream, daemon=True)
    thread.start()
    return thread


def _spawn_engine(tmpdir: Path, config: KlLocalWorkerConfig, side: TokenPathSide) -> _EngineProcess:
    if side is TokenPathSide.A:
        placement, model_path, model, prompts_path = (
            config.placement.decoder,
            config.decoder_model_path,
            config.decoder_model,
            config.prompts_path,
        )
    else:
        placement, model_path, model, prompts_path = (
            config.placement.advisor,
            config.advisor_model_path,
            config.advisor_model,
            config.prompts_path if config.advisor_prompts_path is None else config.advisor_prompts_path,
        )
    engine_config = EngineWorkerConfig(
        side=side,
        model_path=model_path,
        prompts_path=prompts_path,
        token_paths_path=config.token_paths_path,
        model=model,
        k=config.k,
        microbatch_size=config.microbatch_size,
        placement=placement,
    )
    chip_label = ",".join(str(chip) for chip in placement.visible_chips)
    bounds_label = ",".join(str(size) for size in placement.chips_per_process_bounds)
    cache_suffix = chip_label.replace(",", "_")
    config_path = tmpdir / f"engine_{side.value}_chips_{cache_suffix}.json"
    with open(config_path, "wt") as f:
        json.dump(asdict(engine_config), f, sort_keys=True)

    env = os.environ.copy()
    env["TPU_VISIBLE_CHIPS"] = chip_label
    env["TPU_PROCESS_BOUNDS"] = "1,1,1"
    env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = bounds_label
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["JAX_COMPILATION_CACHE_DIR"] = str(tmpdir / f"jax_cache_{cache_suffix}")
    env["VLLM_ASSETS_CACHE"] = str(tmpdir / f"vllm_assets_{cache_suffix}")

    # No new session: the engine stays in the pair child's process group so
    # the supervisor's killpg on a failed child takes it down too.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "experiments.downstream_scaling.evals.measurements.kl",
            "--engine-config",
            str(config_path),
        ],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    label = f"{side.value}/chips-{chip_label}"
    logger.info("Launched KL engine %s", label)
    return _EngineProcess(proc, label=label, stderr_thread=_stream_stderr(proc, label=label))


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


def _terminate_engines(engines: list[_EngineProcess]) -> None:
    for engine in engines:
        if engine.proc.poll() is None:
            engine.proc.terminate()
    for engine in engines:
        if engine.proc.poll() is None:
            try:
                engine.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                engine.proc.kill()
                engine.proc.wait()


def _claim_chunks(config: KlLocalWorkerConfig, *, decoder: _EngineProcess, advisor: _EngineProcess) -> None:
    while True:
        with ledger.claim_next_chunk(config.ledger_path, config.owner) as claim:
            if claim is None:
                summary = ledger.summarize(config.ledger_path)
                if summary.done == summary.total:
                    return
                time.sleep(config.poll_backoff)
                continue

            _run_kl_chunk(KlChunkSpec(**claim.chunk), decoder=decoder, advisor=advisor, temperature=config.temperature)
            ledger.mark_done(claim)


def _run_local_pair_worker(config: KlLocalWorkerConfig) -> None:
    """Pair child: owns no chips; drives one engine per side and reduces their top-k to KL."""
    if os.environ.get("TPU_VISIBLE_CHIPS") is not None:
        raise ValueError("KL pair child expects no TPU visibility of its own; TPU_VISIBLE_CHIPS is already set")

    with tempfile.TemporaryDirectory(prefix="token_path_kl_pair_") as tmp:
        tmpdir = Path(tmp)
        engines: list[_EngineProcess] = []
        try:
            decoder = _spawn_engine(tmpdir, config, TokenPathSide.A)
            engines.append(decoder)
            advisor = _spawn_engine(tmpdir, config, TokenPathSide.B)
            engines.append(advisor)
            _claim_chunks(config, decoder=decoder, advisor=advisor)
        except Exception:
            _terminate_engines(engines)
            raise
        for engine in engines:
            engine.close()


def _child_owner(pool_id: str, shard_idx: int, placement: TokenPathPairPlacement) -> str:
    chips = ",".join(str(chip) for chip in placement.decoder.visible_chips + placement.advisor.visible_chips)
    return f"{pool_id}/shard-{shard_idx}/chips-{chips}"


def _write_child_config(tmpdir: Path, config: KlLocalWorkerConfig) -> Path:
    chips = "-".join(
        str(chip) for chip in config.placement.decoder.visible_chips + config.placement.advisor.visible_chips
    )
    path = tmpdir / f"child_chips_{chips}.json"
    with open(path, "wt") as f:
        json.dump(asdict(config), f, sort_keys=True)
    return path


def _stream_child_output(proc: subprocess.Popen[str], *, label: str) -> list[threading.Thread]:
    threads = []

    def stream(pipe, stream_name: str) -> None:
        assert pipe is not None
        for line in pipe:
            logger.info("kl worker %s %s: %s", label, stream_name, line.rstrip())

    for pipe, stream_name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
        thread = threading.Thread(target=stream, args=(pipe, stream_name), daemon=True)
        thread.start()
        threads.append(thread)
    return threads


def _spawn_child(
    *,
    tmpdir: Path,
    config: KlStepConfig,
    token_paths_path: str,
    ledger_path: str,
    pool_id: str,
    shard_idx: int,
    placement: TokenPathPairPlacement,
) -> tuple[subprocess.Popen[str], list[threading.Thread]]:
    child_config = KlLocalWorkerConfig(
        prompts_path=config.prompts_path,
        advisor_prompts_path=config.advisor_prompts_path,
        token_paths_path=token_paths_path,
        decoder_model_path=config.decoder_model_path,
        advisor_model_path=config.advisor_model_path,
        decoder_model=config.decoder_model,
        advisor_model=config.advisor_model,
        k=config.k,
        microbatch_size=config.microbatch_size,
        ledger_path=ledger_path,
        poll_backoff=config.poll_backoff,
        owner=_child_owner(pool_id, shard_idx, placement),
        placement=placement,
        temperature=config.temperature,
    )
    config_path = _write_child_config(tmpdir, child_config)
    chip_label = ",".join(str(chip) for chip in placement.decoder.visible_chips + placement.advisor.visible_chips)

    # The pair child inherits the supervisor's environment unchanged: it owns
    # no chips, and each engine it spawns sets its own TPU visibility.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "experiments.downstream_scaling.evals.measurements.kl",
            "--xregion-worker-child-config",
            str(config_path),
        ],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    logger.info("Launched KL pair worker shard=%d chips=%s", shard_idx, chip_label)
    return proc, _stream_child_output(proc, label=chip_label)


def _wait_for_children(
    procs: list[subprocess.Popen[str]],
    threads: list[threading.Thread],
    ledger_path: str,
) -> None:
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
                    logger.warning("KL worker exited after ledger completion with rc=%d", return_code)
                    continue
                _terminate_children(procs)
                raise RuntimeError(
                    f"KL worker failed with rc={return_code}; ledger is {summary.done}/{summary.total} done"
                )

        if ledger_complete and all_done:
            break
        if all_done:
            raise RuntimeError(f"KL workers exited before completion: {summary.done}/{summary.total} chunks done")
        time.sleep(1.0)

    for thread in threads:
        thread.join(timeout=5)


def _stage_token_paths(token_paths_path: str, tmpdir: Path) -> str:
    local_path = tmpdir / TOKEN_PATHS_FILENAME
    with fsspec.open(token_paths_path, "rb") as source, local_path.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    return str(local_path)


def _supervise_worker(
    _worker_ids: Iterator[int],
    shard_info: ShardInfo,
    *,
    config: KlStepConfig,
    token_paths_path: str,
    ledger_path: str,
    pool: WorkerPoolConfig,
    placements: tuple[TokenPathPairPlacement, ...],
) -> Iterator[dict[str, object]]:
    if os.environ.get("TPU_VISIBLE_CHIPS") is not None:
        raise ValueError("KL supervisor expects to own the full TPU VM; TPU_VISIBLE_CHIPS is already set")

    with tempfile.TemporaryDirectory(prefix="token_path_kl_") as tmp:
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


def run_token_path_kl(config: KlStepConfig) -> None:
    if not config.worker_pools:
        raise ValueError("Token-path KL requires at least one worker pool")

    token_paths_path = os.path.join(config.alg_output_path, TOKEN_PATHS_FILENAME)
    num_records = _count_same_tokenizer_records(token_paths_path)
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
            _supervise_worker,
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
        raise RuntimeError(f"Token-path KL incomplete: {summary.done}/{summary.total} chunks done")

    metadata = {
        "statistic": "token_path_kl",
        "decoder_model_path": config.decoder_model_path,
        "advisor_model_path": config.advisor_model_path,
        "k": config.k,
    }
    done_ids = set(ledger.done_chunk_ids(ledger_path))
    chunk_paths = [chunk.output_path for chunk in chunks if chunk.chunk_id in done_ids]
    path = statistics_file(config.output_path)
    pipeline = (
        Dataset.from_list(chunk_paths)
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
        name="token-path-kl-aggregate",
        max_workers=config.aggregate_workers,
        resources=ResourceConfig(cpu=1, ram="4g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(pipeline)

    written = sum(len(row["values"]) for row in read_statistic_rows(path))
    if written != num_records:
        raise ValueError(f"Wrote {written} KL statistic values for {num_records} token-path records")
    logger.info("Wrote %d token-path KL statistic values to %s", written, path)


def _main() -> None:
    # Child and engine processes: without this, their INFO output (including
    # every engine stderr line the pair child forwards) is dropped.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s [kl pid=%(process)d] %(message)s",
    )
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--xregion-worker-child-config")
    group.add_argument("--engine-config")
    args = parser.parse_args()
    if args.engine_config is not None:
        _run_engine(_engine_config_from_file(args.engine_config))
    else:
        _run_local_pair_worker(_child_config_from_file(args.xregion_worker_child_config))


if __name__ == "__main__":
    _main()
