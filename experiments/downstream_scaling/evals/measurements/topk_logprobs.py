# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-k model logprobs along recorded cross-tokenizer token paths.

Each completion value is ``{"steps": [{"topk_ids": list[list[int]],
"topk_logprobs": list[list[float]]}, ...]}``. The inner lists follow the
recorded tokens within that decision step.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
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
VLLM_CONSTRUCTOR_SEED = 0
TOKEN_PATHS_FILENAME = "token_paths.jsonl.gz"


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
class TokenPathPoolConfig:
    pool: WorkerPoolConfig
    placements: tuple[EnginePlacement, ...]

    def __post_init__(self) -> None:
        validate_pool_placements(self.pool, self.placements)


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
class TokenPathTopkLogprobs:
    model: TokenPathModelConfig
    execution: TokenPathExecutionConfig
    side: TokenPathSide
    k: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", TokenPathSide(self.side))
        if self.k < 1:
            raise ValueError(f"k must be >= 1 (got {self.k})")

    def make_statistic_step(
        self,
        *,
        name: str,
        prompts_path: str | InputName | MirroredValue,
        alg_output_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return make_token_path_topk_logprobs_step(
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
class TopkLogprobsStepConfig:
    output_path: str
    model_path: str
    prompts_path: str
    alg_output_path: str
    model: _EngineConfig
    worker_pools: tuple[TokenPathPoolConfig, ...]
    side: TokenPathSide
    k: int
    microbatch_size: int
    ledger_prefix: str
    chunk_size: int
    heartbeat_timeout: float
    poll_backoff: float
    aggregate_workers: int


@dataclass(frozen=True)
class TopkLogprobsChunkSpec:
    chunk_id: int
    chunk_start: int
    chunk_end: int
    output_path: str


@dataclass(frozen=True)
class TopkLogprobsLocalWorkerConfig:
    model_path: str
    prompts_path: str
    token_paths_path: str
    model: _EngineConfig
    side: TokenPathSide
    k: int
    microbatch_size: int
    ledger_path: str
    poll_backoff: float
    owner: str
    placement: EnginePlacement


@dataclass(frozen=True)
class _ScoringRequest:
    id: str
    completion_index: int
    prompt_length: int
    token_ids: list[int]
    step_lengths: tuple[int, ...]


def make_token_path_topk_logprobs_step(
    *,
    name: str,
    prompts_path: str | InputName | MirroredValue,
    alg_output_path: str | InputName | MirroredValue,
    statistic: TokenPathTopkLogprobs,
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(
            run_token_path_topk_logprobs,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
            pip_dependency_groups=["vllm"],
            env_vars=VLLM_TPU_ENV_VARS,
        ),
        config=TopkLogprobsStepConfig(
            output_path=this_output_path(),
            model_path=version_path(statistic.model.model_path),  # type: ignore[arg-type]
            prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
            alg_output_path=version_path(alg_output_path),  # type: ignore[arg-type]
            model=_EngineConfig(
                max_model_len=versioned(statistic.model.max_model_len),  # type: ignore[arg-type]
                gpu_memory_utilization=statistic.model.gpu_memory_utilization,
                apply_rpa_block_size_patch=statistic.model.apply_rpa_block_size_patch,
            ),
            worker_pools=statistic.execution.worker_pools,
            side=versioned(statistic.side),  # type: ignore[arg-type]
            k=versioned(statistic.k),  # type: ignore[arg-type]
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


def _topk_value(
    request: _ScoringRequest,
    prompt_logprobs: _FlatPromptLogprobs,
    k: int,
) -> dict[str, Any]:
    if len(prompt_logprobs.start_indices) != len(request.token_ids):
        raise ValueError(
            f"Prompt-logprob length {len(prompt_logprobs.start_indices)} does not match "
            f"token length {len(request.token_ids)}"
        )

    position = request.prompt_length
    steps = []
    for step_length in request.step_lengths:
        topk_ids = []
        topk_logprobs = []
        for _ in range(step_length):
            token_ids, logprobs = _topk_at_position(prompt_logprobs, position, k)
            topk_ids.append(token_ids)
            topk_logprobs.append(logprobs)
            position += 1
        steps.append({"topk_ids": topk_ids, "topk_logprobs": topk_logprobs})
    return {"steps": steps}


@functools.cache
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
        **model_kwargs,
    )
    return llm, SamplingParams, TokensPrompt, llm.get_tokenizer()


@functools.cache
def _load_prompts(prompts_path: str) -> dict[str, str]:
    return {row["id"]: row["prompt"] for row in read_prompt_rows(prompts_path)}


def _run_topk_logprobs_chunk(
    chunk: TopkLogprobsChunkSpec,
    *,
    model_path: str,
    prompts_path: str,
    token_paths_path: str,
    model: _EngineConfig,
    side: TokenPathSide,
    k: int,
    microbatch_size: int,
    tensor_parallel_size: int,
) -> None:
    llm, SamplingParams, TokensPrompt, tokenizer = _load_vllm(model_path, tensor_parallel_size, model)
    prompts = _load_prompts(prompts_path)
    chunk_rows = _read_row_range(token_paths_path, chunk.chunk_start, chunk.chunk_end)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=k,
        flat_logprobs=True,
        detokenize=False,
    )

    records = []
    for start in range(0, len(chunk_rows), microbatch_size):
        batch = [
            _prepare_scoring_request(row, prompts[row["id"]], tokenizer, side)
            for row in chunk_rows[start : start + microbatch_size]
        ]
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=request.token_ids) for request in batch],
            sampling_params,
            use_tqdm=False,
        )
        for request, output in zip(batch, outputs, strict=True):
            if output.prompt_logprobs is None:
                raise RuntimeError("vLLM returned no prompt logprobs")
            value = _topk_value(request, cast(_FlatPromptLogprobs, output.prompt_logprobs), k)
            records.append(
                {
                    "id": request.id,
                    "completion_index": request.completion_index,
                    "value": value,
                }
            )

    with fsspec.open(chunk.output_path, "wt", compression="gzip") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _chunk_specs(chunks_dir: str, num_records: int, chunk_size: int) -> list[TopkLogprobsChunkSpec]:
    return [
        TopkLogprobsChunkSpec(
            chunk_id=chunk_id,
            chunk_start=start,
            chunk_end=min(start + chunk_size, num_records),
            output_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz"),
        )
        for chunk_id, start in enumerate(range(0, num_records, chunk_size))
    ]


def token_path_tp1_placements(chips_per_vm: int) -> tuple[EnginePlacement, ...]:
    return tuple(
        EnginePlacement(
            visible_chips=(chip,),
            chips_per_process_bounds=(1, 1, 1),
            tensor_parallel_size=1,
        )
        for chip in range(chips_per_vm)
    )


def _engine_placement_from_dict(data: dict[str, Any]) -> EnginePlacement:
    bounds = data["chips_per_process_bounds"]
    return EnginePlacement(
        visible_chips=tuple(data["visible_chips"]),
        chips_per_process_bounds=(bounds[0], bounds[1], bounds[2]),
        tensor_parallel_size=data["tensor_parallel_size"],
    )


def _child_config_from_file(path: str) -> TopkLogprobsLocalWorkerConfig:
    with open(path) as f:
        data = json.load(f)
    return TopkLogprobsLocalWorkerConfig(
        model_path=data["model_path"],
        prompts_path=data["prompts_path"],
        token_paths_path=data["token_paths_path"],
        model=_EngineConfig(**data["model"]),
        side=TokenPathSide(data["side"]),
        k=data["k"],
        microbatch_size=data["microbatch_size"],
        ledger_path=data["ledger_path"],
        poll_backoff=data["poll_backoff"],
        owner=data["owner"],
        placement=_engine_placement_from_dict(data["placement"]),
    )


def _run_local_engine_worker(config: TopkLogprobsLocalWorkerConfig) -> None:
    expected_visible_chips = ",".join(str(chip) for chip in config.placement.visible_chips)
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

            _run_topk_logprobs_chunk(
                TopkLogprobsChunkSpec(**claim.chunk),
                model_path=config.model_path,
                prompts_path=config.prompts_path,
                token_paths_path=config.token_paths_path,
                model=config.model,
                side=config.side,
                k=config.k,
                microbatch_size=config.microbatch_size,
                tensor_parallel_size=config.placement.tensor_parallel_size,
            )
            ledger.mark_done(claim)


def _child_owner(pool_id: str, shard_idx: int, placement: EnginePlacement) -> str:
    chips = ",".join(str(chip) for chip in placement.visible_chips)
    return f"{pool_id}/shard-{shard_idx}/chips-{chips}"


def _write_child_config(tmpdir: Path, config: TopkLogprobsLocalWorkerConfig) -> Path:
    chips = "-".join(str(chip) for chip in config.placement.visible_chips)
    path = tmpdir / f"child_chips_{chips}.json"
    with open(path, "wt") as f:
        json.dump(asdict(config), f, sort_keys=True)
    return path


def _stream_child_output(proc: subprocess.Popen[str], *, label: str) -> list[threading.Thread]:
    threads = []

    def stream(pipe, stream_name: str) -> None:
        assert pipe is not None
        for line in pipe:
            logger.info("top-k logprobs worker %s %s: %s", label, stream_name, line.rstrip())

    for pipe, stream_name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
        thread = threading.Thread(target=stream, args=(pipe, stream_name), daemon=True)
        thread.start()
        threads.append(thread)
    return threads


def _spawn_child(
    *,
    tmpdir: Path,
    config: TopkLogprobsStepConfig,
    token_paths_path: str,
    ledger_path: str,
    pool_id: str,
    shard_idx: int,
    placement: EnginePlacement,
) -> tuple[subprocess.Popen[str], list[threading.Thread]]:
    child_config = TopkLogprobsLocalWorkerConfig(
        model_path=config.model_path,
        prompts_path=config.prompts_path,
        token_paths_path=token_paths_path,
        model=config.model,
        side=config.side,
        k=config.k,
        microbatch_size=config.microbatch_size,
        ledger_path=ledger_path,
        poll_backoff=config.poll_backoff,
        owner=_child_owner(pool_id, shard_idx, placement),
        placement=placement,
    )
    config_path = _write_child_config(tmpdir, child_config)
    chip_label = ",".join(str(chip) for chip in placement.visible_chips)
    bounds_label = ",".join(str(size) for size in placement.chips_per_process_bounds)

    env = os.environ.copy()
    env["TPU_VISIBLE_CHIPS"] = chip_label
    env["TPU_PROCESS_BOUNDS"] = "1,1,1"
    env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = bounds_label
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    cache_suffix = chip_label.replace(",", "_")
    env["JAX_COMPILATION_CACHE_DIR"] = str(tmpdir / f"jax_cache_{cache_suffix}")
    env["VLLM_ASSETS_CACHE"] = str(tmpdir / f"vllm_assets_{cache_suffix}")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "experiments.downstream_scaling.evals.measurements.topk_logprobs",
            "--xregion-worker-child-config",
            str(config_path),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    logger.info("Launched top-k logprobs worker shard=%d chips=%s", shard_idx, chip_label)
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
                    logger.warning("Top-k logprobs worker exited after ledger completion with rc=%d", return_code)
                    continue
                _terminate_children(procs)
                raise RuntimeError(
                    f"Top-k logprobs worker failed with rc={return_code}; "
                    f"ledger is {summary.done}/{summary.total} done"
                )

        if ledger_complete and all_done:
            break
        if all_done:
            raise RuntimeError(
                f"Top-k logprobs workers exited before completion: {summary.done}/{summary.total} chunks done"
            )
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
    config: TopkLogprobsStepConfig,
    token_paths_path: str,
    ledger_path: str,
    pool: WorkerPoolConfig,
    placements: tuple[EnginePlacement, ...],
) -> Iterator[dict[str, object]]:
    if os.environ.get("TPU_VISIBLE_CHIPS") is not None:
        raise ValueError("Top-k logprobs supervisor expects to own the full TPU VM; TPU_VISIBLE_CHIPS is already set")

    with tempfile.TemporaryDirectory(prefix="token_path_topk_logprobs_") as tmp:
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


def run_token_path_topk_logprobs(config: TopkLogprobsStepConfig) -> None:
    if not config.worker_pools:
        raise ValueError("Token-path top-k logprobs requires at least one worker pool")

    token_paths_path = os.path.join(config.alg_output_path, TOKEN_PATHS_FILENAME)
    num_records = sum(1 for _ in read_token_path_rows(token_paths_path))
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
        raise RuntimeError(f"Token-path top-k logprobs incomplete: {summary.done}/{summary.total} chunks done")

    metadata = {
        "statistic": "token_path_topk_logprobs",
        "model_path": config.model_path,
        "side": config.side.value,
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
        name="token-path-topk-logprobs-aggregate",
        max_workers=config.aggregate_workers,
        resources=ResourceConfig(cpu=1, ram="4g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(pipeline)

    written = sum(len(row["values"]) for row in read_statistic_rows(path))
    if written != num_records:
        raise ValueError(f"Wrote {written} top-k statistic values for {num_records} token-path records")
    logger.info("Wrote %d token-path top-k statistic values to %s", written, path)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xregion-worker-child-config", required=True)
    args = parser.parse_args()
    _run_local_engine_worker(_child_config_from_file(args.xregion_worker_child_config))


if __name__ == "__main__":
    _main()
