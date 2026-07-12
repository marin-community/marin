# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-tokenizer joint-decode-avg completion algorithm (xregion worker pools).

Decodes from two models with **different tokenizers**. Candidates are
compared as byte strings; each decision round commits one byte chunk, forced
on each side as that side's own token segmentation (the joint-decode
package's variable-length forcing keeps the engines in sync). The selection
rules live in ``xtok_selection``:

- ``bytes_union``: the package's ``select_avg_logits`` re-keyed on bytes.
- ``anchored_prefix_mass``: the decoder proposes its top-k; the advisor
  scores each candidate by byte-prefix probability mass.

Scale-out shape is cloned from ``joint_decode_avg_xregion``: the executor
step is a CPU coordinator that fans out single-VM TPU worker pools via
Zephyr; each pool worker supervises one child process per chip pair, and
children claim chunks from a shared GCS ledger. Each child loads both
tokenizers from the checkpoint roots and builds its selector locally.

One step sweeps ``advisor_weights``: each ledger chunk carries the weight it
decodes with, the child sets that weight on its selector before generating
(engines load once per child, not once per weight), and all weights land in
one completions file — per prompt, ``len(advisor_weights) * n_samples``
completions with ``completion_index = weight_index * n_samples + sample``
and ``advisor_weight`` in each completion's metadata. The unchanged grade
step scores every completion; analysis groups by the metadata weight.

Each chunk also writes a token-path sidecar (aggregated to
``token_paths.jsonl.gz``): per completion, the step-aligned committed byte
chunks and each side's exact forced token ids, recorded from the selector
via the package's ``request_index`` seam. This is the only artifact from
which the decoded path can be rescored — B's forced segmentation is
noncanonical, so no post-hoc retokenization reproduces it. See
.agents/projects/20260709_joint_decode_xtok_token_paths_plan.md.

Plan: .agents/projects/20260708_joint_decode_avg_xtok_plan.md. Once
experiments run, the config dataclasses, their defaults, and the step
construction below are hash-frozen like the sibling modules'.
"""

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
from collections.abc import Callable, Iterator
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
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext

from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.framework.schema import (
    completions_file,
    read_prompt_rows,
)
from experiments.downstream_scaling.evals.framework.xregion import ledger
from experiments.downstream_scaling.evals.framework.xregion import pool as xregion_pool
from experiments.downstream_scaling.evals.framework.xregion.pool import WorkerPoolConfig
from experiments.downstream_scaling.evals.utils import discover_hf_checkpoints, version_path

logger = logging.getLogger(__name__)

VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    # Required at `uv sync` time so vllm's setup.py skips CUDA-version
    # detection (which asserts CUDA_HOME). Propagated to the container build
    # via remote(env_vars=...).
    "VLLM_TARGET_DEVICE": "tpu",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}

DEFAULT_HEARTBEAT_TIMEOUT = 2 * 60
DEFAULT_LEDGER_PREFIX = "gs://marin-us-central2"
DEFAULT_POLL_BACKOFF = 10.0
TOKEN_PATHS_FILENAME = "token_paths.jsonl.gz"


class XtokSelectionRule(StrEnum):
    BYTES_UNION = "bytes_union"
    ANCHORED_PREFIX_MASS = "anchored_prefix_mass"


@dataclass(frozen=True)
class JointDecodeSamplingConfig:
    n_samples: int
    # Side A (decoder) completion cap; the output text is A's.
    max_tokens: int
    # Side B cap: headroom for tokenizer-fertility mismatch on the same text
    # (e.g. 2x max_tokens; Qwen tokenizes digits singly, llama 3 in groups).
    advisor_max_tokens: int
    top_k_a: int
    top_k_b: int
    seed: int
    selection_rule: XtokSelectionRule
    # The sweep axis: one run decodes every prompt at every weight, engines
    # loaded once. Versioned as a whole — extending the grid re-runs the sweep.
    advisor_weights: tuple[float, ...]
    temperature: float = 1.0
    stop: tuple[str, ...] | None = None
    # anchored_prefix_mass only: per-byte credit for advisor tokens that are
    # strict prefixes of a candidate (continuation logits are unobservable).
    prefix_credit: float = 1.0

    def __post_init__(self) -> None:
        if self.max_tokens < 1 or self.advisor_max_tokens < 1:
            raise ValueError("max_tokens and advisor_max_tokens must both be >= 1")
        if self.top_k_a < 1 or self.top_k_b < 1:
            raise ValueError("top_k_a and top_k_b must both be >= 1")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0 (got {self.temperature})")
        if not self.advisor_weights:
            raise ValueError("advisor_weights must be non-empty")
        if any(not 0.0 <= weight <= 1.0 for weight in self.advisor_weights):
            raise ValueError(f"advisor_weights must all be in [0, 1] (got {self.advisor_weights})")
        if len(set(self.advisor_weights)) != len(self.advisor_weights):
            raise ValueError(f"advisor_weights must be distinct (got {self.advisor_weights})")
        if not 0.0 <= self.prefix_credit <= 1.0:
            raise ValueError(f"prefix_credit must be in [0, 1] (got {self.prefix_credit})")


@dataclass(frozen=True)
class JointDecodeModelConfig:
    max_model_len: int = 8192
    gpu_memory_utilization: float | None = None
    enable_prefix_caching: bool = False
    # Halve the RPA-kernel KV-page block size. Required for delphi-shaped
    # models (otherwise vmem error); harms perf on standard models like llama,
    # so default off.
    apply_rpa_block_size_patch: bool = False


@dataclass(frozen=True)
class JointDecodeExecutionConfig:
    worker_pools: tuple[WorkerPoolConfig, ...]
    ledger_prefix: str = DEFAULT_LEDGER_PREFIX
    chunk_size: int = 512
    # Cap on in-flight requests per engine pair. Under the package backend
    # this bounds the sliding admission window (None → whole chunk).
    microbatch_size: int | None = None
    heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT
    poll_backoff: float = DEFAULT_POLL_BACKOFF
    barrier_timeout_s: float = 60.0
    aggregate_workers: int = 32
    # Per-step scheduler token budget (plain field: never enters executor
    # version payloads). None derives max_model_len + microbatch cap in the
    # worker: the package's own derived default (8 x max_model_len) OOMs
    # large advisors at vllm's memory profiling.
    max_num_batched_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.microbatch_size is not None and self.microbatch_size < 1:
            raise ValueError(f"microbatch_size must be >= 1 or None (got {self.microbatch_size})")
        if self.aggregate_workers < 1:
            raise ValueError(f"aggregate_workers must be >= 1 (got {self.aggregate_workers})")


@dataclass(frozen=True)
class JointDecodeConfig:
    sampling: JointDecodeSamplingConfig
    advisor_model_path: str | InputName | MirroredValue
    decoder_model: JointDecodeModelConfig
    advisor_model: JointDecodeModelConfig
    execution: JointDecodeExecutionConfig


@dataclass(frozen=True)
class JointDecodeCompletionStepConfig:
    output_path: str
    decoder_model_path: str
    advisor_model_path: str
    prompts_path: str
    sampling: JointDecodeSamplingConfig
    decoder_model: JointDecodeModelConfig
    advisor_model: JointDecodeModelConfig
    worker_pools: tuple[WorkerPoolConfig, ...]
    ledger_prefix: str
    chunk_size: int
    microbatch_size: int
    heartbeat_timeout: float
    poll_backoff: float
    barrier_timeout_s: float
    aggregate_workers: int
    max_num_batched_tokens: int | None = None


@dataclass(frozen=True)
class XtokChunkSpec:
    """One ledger work unit: a request range decoded at one advisor weight.

    chunk_start/chunk_end index the per-weight request range
    [0, num_prompts * n_samples); the same ranges repeat once per weight.
    """

    chunk_id: int
    advisor_weight: float
    weight_index: int
    chunk_start: int
    chunk_end: int
    output_path: str
    token_paths_path: str


@dataclass(frozen=True)
class JointDecodeLocalWorkerConfig:
    decoder_model_path: str
    advisor_model_path: str
    prompts_path: str
    sampling: JointDecodeSamplingConfig
    decoder_model: JointDecodeModelConfig
    advisor_model: JointDecodeModelConfig
    ledger_path: str
    poll_backoff: float
    microbatch_size: int
    barrier_timeout_s: float
    owner: str
    chip_pair: tuple[int, int]
    max_num_batched_tokens: int | None = None


@dataclass(frozen=True)
class JointDecodeCompletionAlgorithm:
    config: JointDecodeConfig

    def make_completions_step(
        self,
        *,
        name: str,
        model_path: str | InputName | MirroredValue,
        prompts_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return make_joint_decode_completion_step(
            name=name,
            model_path=model_path,
            prompts_path=prompts_path,
            config=self.config,
        )


def make_joint_decode_completion_step(
    *,
    name: str,
    model_path: str | InputName | MirroredValue,
    prompts_path: str | InputName | MirroredValue,
    config: JointDecodeConfig,
) -> ExecutorStep:
    microbatch_size = (
        config.execution.chunk_size if config.execution.microbatch_size is None else config.execution.microbatch_size
    )
    return ExecutorStep(
        name=name,
        fn=remote(
            run_joint_decode_completion_chunks,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4g"),
            pip_dependency_groups=["vllm", "tpu"],
            env_vars=VLLM_TPU_ENV_VARS,
        ),
        config=JointDecodeCompletionStepConfig(
            output_path=this_output_path(),
            decoder_model_path=version_path(model_path),  # type: ignore[arg-type]
            advisor_model_path=version_path(config.advisor_model_path),  # type: ignore[arg-type]
            prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
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


def sweep_chunk_specs(
    chunks_dir: str,
    num_prompts: int,
    n_samples: int,
    advisor_weights: tuple[float, ...],
    chunk_size: int,
) -> list[XtokChunkSpec]:
    total_requests = num_prompts * n_samples
    specs: list[XtokChunkSpec] = []
    for weight_index, advisor_weight in enumerate(advisor_weights):
        for start in range(0, total_requests, chunk_size):
            chunk_id = len(specs)
            specs.append(
                XtokChunkSpec(
                    chunk_id=chunk_id,
                    advisor_weight=advisor_weight,
                    weight_index=weight_index,
                    chunk_start=start,
                    chunk_end=min(start + chunk_size, total_requests),
                    output_path=os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz"),
                    token_paths_path=os.path.join(chunks_dir, f"token-paths-{chunk_id:06d}.jsonl.gz"),
                )
            )
    return specs


@dataclass(frozen=True)
class XtokPathStep:
    """One committed decision: the chunk's bytes (hex; empty for a
    selector-chosen EOS) and each side's exact forced token ids."""

    bytes_hex: str
    tokens_a: list[int]
    tokens_b: list[int]


def _path_step(vocab_a: xtok_selection.Vocab, tokens_a: list[int], tokens_b: list[int]) -> XtokPathStep:
    if tokens_a == [vocab_a.eos_id]:
        chunk = b""  # selector-chosen EOS commits no text bytes
    else:
        pieces = [vocab_a.token_bytes[token_id] for token_id in tokens_a]
        if any(piece is None for piece in pieces):
            raise ValueError(f"non-EOS forced tokens include a special id: {tokens_a}")
        chunk = b"".join(piece for piece in pieces if piece is not None)
    return XtokPathStep(bytes_hex=chunk.hex(), tokens_a=list(tokens_a), tokens_b=list(tokens_b))


def write_sweep_chunk(
    chunk: XtokChunkSpec,
    *,
    decoder: Any,
    prompt_ids: list[str],
    prompts: list[str],
    n_samples: int,
    token_paths: dict[int, list[XtokPathStep]],
) -> None:
    """Generate one chunk at its advisor weight (the caller has already set
    the weight on the selector state and reset token_paths, which the
    selector fills during generate) and write completion + token-path
    records. Trace/output mismatches raise: they mean the request-index
    invariant broke, which must kill the run rather than write silently
    misaligned sidecars.

    completion_index = weight_index * n_samples + sample keeps indices dense
    per prompt across the whole sweep, so the standard sort-by-index
    aggregation and the positional grade alignment work unchanged."""
    request_indices = range(chunk.chunk_start, chunk.chunk_end)
    chunk_prompt_ids = [prompt_ids[i // n_samples] for i in request_indices]
    completion_indices = [chunk.weight_index * n_samples + i % n_samples for i in request_indices]
    chunk_prompts = [prompts[i // n_samples] for i in request_indices]

    outputs = decoder.generate(chunk_prompts, chunk_prompts)

    records = []
    path_records = []
    rows = zip(chunk_prompt_ids, completion_indices, outputs, strict=True)
    for batch_index, (prompt_id, completion_index, output) in enumerate(rows):
        steps = token_paths.pop(batch_index, None)
        if steps is None:
            if output.text:
                raise RuntimeError(
                    f"no token path recorded for batch index {batch_index} with non-empty completion text"
                )
            steps = []
        records.append(
            {
                "id": prompt_id,
                "completion_index": completion_index,
                "completion": {
                    "text": output.text,
                    "metadata": {
                        "finish_reason": output.finish_reason,
                        "advisor_weight": chunk.advisor_weight,
                    },
                },
            }
        )
        path_records.append(
            {
                "id": prompt_id,
                "completion_index": completion_index,
                "advisor_weight": chunk.advisor_weight,
                "weight_index": chunk.weight_index,
                "steps": [asdict(step) for step in steps],
            }
        )
    if token_paths:
        raise RuntimeError(f"token paths recorded for unknown batch indices: {sorted(token_paths)}")

    with fsspec.open(chunk.output_path, "wt", compression="gzip") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    with fsspec.open(chunk.token_paths_path, "wt", compression="gzip") as f:
        for record in path_records:
            f.write(json.dumps(record) + "\n")


# ---- Xregion local workers ----


def _num_prompts(prompts_path: str) -> int:
    return sum(1 for _ in read_prompt_rows(prompts_path))


def _child_config_from_file(path: str) -> JointDecodeLocalWorkerConfig:
    with open(path) as f:
        data = json.load(f)
    sampling_data = dict(data["sampling"])
    sampling_data["selection_rule"] = XtokSelectionRule(sampling_data["selection_rule"])
    sampling_data["advisor_weights"] = tuple(sampling_data["advisor_weights"])
    return JointDecodeLocalWorkerConfig(
        decoder_model_path=data["decoder_model_path"],
        advisor_model_path=data["advisor_model_path"],
        prompts_path=data["prompts_path"],
        sampling=JointDecodeSamplingConfig(**sampling_data),
        decoder_model=JointDecodeModelConfig(**data["decoder_model"]),
        advisor_model=JointDecodeModelConfig(**data["advisor_model"]),
        ledger_path=data["ledger_path"],
        poll_backoff=data["poll_backoff"],
        microbatch_size=data["microbatch_size"],
        barrier_timeout_s=data["barrier_timeout_s"],
        owner=data["owner"],
        chip_pair=tuple(data["chip_pair"]),
        max_num_batched_tokens=data["max_num_batched_tokens"],
    )


def _load_vocab(model_path: str) -> xtok_selection.Vocab:
    resolved = discover_hf_checkpoints(model_path)[-1]
    logger.info("Resolved tokenizer %s -> %s", model_path, resolved)
    return xtok_selection.load_vocab(load_tokenizer(resolved))


@dataclass
class _SweepState:
    """Selector state the child mutates between chunks. The selector is
    baked into the coordinator at engine spawn; chunks run serially, so
    setting these between generate() calls is race-free."""

    advisor_weight: float
    token_paths: dict[int, list[XtokPathStep]]  # request_index -> steps


def _make_select_token(
    sampling: JointDecodeSamplingConfig,
    vocab_a: xtok_selection.Vocab,
    vocab_b: xtok_selection.Vocab,
    state: _SweepState,
) -> Callable[..., tuple[list[int], list[int]]]:
    if sampling.selection_rule is XtokSelectionRule.BYTES_UNION:
        rule = xtok_selection.select_avg_bytes_union
        rule_kwargs: dict[str, float] = {}
    elif sampling.selection_rule is XtokSelectionRule.ANCHORED_PREFIX_MASS:
        rule = xtok_selection.select_avg_anchored
        rule_kwargs = {"prefix_credit": sampling.prefix_credit}
    else:
        raise ValueError(f"unknown selection rule: {sampling.selection_rule!r}")

    def select_token(a_topk, b_topk, *, rng, request_index: int):
        tokens_a, tokens_b = rule(
            a_topk,
            b_topk,
            advisor_weight=state.advisor_weight,
            temperature=sampling.temperature,
            rng=rng,
            vocab_a=vocab_a,
            vocab_b=vocab_b,
            **rule_kwargs,
        )
        state.token_paths.setdefault(request_index, []).append(_path_step(vocab_a, tokens_a, tokens_b))
        return tokens_a, tokens_b

    return select_token


def _run_joint_decode_local_worker(config: JointDecodeLocalWorkerConfig) -> None:
    # Lazy: the joint-decode package rides the linux-only vllm extra; this
    # module must stay importable for step construction everywhere.
    from experiments.downstream_scaling.evals.algorithms import joint_decode_backend  # noqa: PLC0415

    expected_visible_chips = ",".join(str(chip) for chip in config.chip_pair)
    actual_visible_chips = os.environ.get("TPU_VISIBLE_CHIPS")
    if actual_visible_chips != expected_visible_chips:
        raise ValueError(f"TPU_VISIBLE_CHIPS={actual_visible_chips!r}, expected {expected_visible_chips!r}")

    prompt_rows = list(read_prompt_rows(config.prompts_path))
    prompt_ids = [row["id"] for row in prompt_rows]
    prompts = [row["prompt"] for row in prompt_rows]

    vocab_a = _load_vocab(config.decoder_model_path)
    vocab_b = _load_vocab(config.advisor_model_path)
    state = _SweepState(advisor_weight=config.sampling.advisor_weights[0], token_paths={})

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
        select_token=_make_select_token(config.sampling, vocab_a, vocab_b, state),
        chip_a=config.chip_pair[0],
        chip_b=config.chip_pair[1],
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

                chunk = XtokChunkSpec(**claim.chunk)
                state.advisor_weight = chunk.advisor_weight
                state.token_paths = {}
                write_sweep_chunk(
                    chunk,
                    decoder=decoder,
                    prompt_ids=prompt_ids,
                    prompts=prompts,
                    n_samples=config.sampling.n_samples,
                    token_paths=state.token_paths,
                )
                ledger.mark_done(claim)


def _chip_pairs(chips_per_vm: int) -> list[tuple[int, int]]:
    if chips_per_vm % 2 != 0:
        raise ValueError(f"joint decode needs an even number of chips per VM, got {chips_per_vm}")
    return [(start, start + 1) for start in range(0, chips_per_vm, 2)]


def _child_owner(pool_id: str, shard_idx: int, chip_pair: tuple[int, int]) -> str:
    return f"{pool_id}/shard-{shard_idx}/chips-{chip_pair[0]},{chip_pair[1]}"


def _write_child_config(tmpdir: Path, config: JointDecodeLocalWorkerConfig) -> Path:
    chips = "-".join(str(chip) for chip in config.chip_pair)
    path = tmpdir / f"child_chips_{chips}.json"
    with open(path, "wt") as f:
        json.dump(asdict(config), f, sort_keys=True)
    return path


def _stream_child_output(proc: subprocess.Popen[str], *, label: str) -> list[threading.Thread]:
    threads = []

    def stream(pipe, stream_name: str) -> None:
        assert pipe is not None
        for line in pipe:
            logger.info("joint-decode-avg-xtok local worker %s %s: %s", label, stream_name, line.rstrip())

    for pipe, stream_name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
        thread = threading.Thread(target=stream, args=(pipe, stream_name), daemon=True)
        thread.start()
        threads.append(thread)
    return threads


def _spawn_child(
    *,
    tmpdir: Path,
    config: JointDecodeCompletionStepConfig,
    ledger_path: str,
    pool_id: str,
    shard_idx: int,
    chip_pair: tuple[int, int],
) -> tuple[subprocess.Popen[str], list[threading.Thread]]:
    child_config = JointDecodeLocalWorkerConfig(
        decoder_model_path=config.decoder_model_path,
        advisor_model_path=config.advisor_model_path,
        prompts_path=config.prompts_path,
        sampling=config.sampling,
        decoder_model=config.decoder_model,
        advisor_model=config.advisor_model,
        ledger_path=ledger_path,
        poll_backoff=config.poll_backoff,
        microbatch_size=config.microbatch_size,
        barrier_timeout_s=config.barrier_timeout_s,
        owner=_child_owner(pool_id, shard_idx, chip_pair),
        chip_pair=chip_pair,
        max_num_batched_tokens=config.max_num_batched_tokens,
    )
    config_path = _write_child_config(tmpdir, child_config)
    chip_label = ",".join(str(chip) for chip in chip_pair)

    # The child only partitions chips (a validated invariant of the pair
    # harness); JAX/vLLM process env is owned per engine worker by the
    # joint-decode package.
    env = os.environ.copy()
    env["TPU_VISIBLE_CHIPS"] = chip_label

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "experiments.downstream_scaling.evals.algorithms.joint_decode_avg_xtok",
        "--xregion-worker-child-config",
        str(config_path),
    ]
    logger.info("Launching joint-decode-avg-xtok local worker shard=%d chips=%s", shard_idx, chip_label)
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
                        "joint-decode-avg-xtok local worker exited after ledger completion with rc=%d",
                        return_code,
                    )
                    continue
                _terminate_children(procs)
                raise RuntimeError(
                    f"joint-decode-avg-xtok local worker failed with rc={return_code}; "
                    f"ledger is {summary.done}/{summary.total} done"
                )

        if ledger_complete and all_done:
            break
        if all_done:
            raise RuntimeError(
                f"joint-decode-avg-xtok local workers exited before completion: "
                f"{summary.done}/{summary.total} chunks done"
            )

        time.sleep(1.0)

    for thread in threads:
        thread.join(timeout=5)


def _supervise_joint_decode_worker(
    _worker_ids: Iterator[int],
    shard_info: ShardInfo,
    *,
    config: JointDecodeCompletionStepConfig,
    ledger_path: str,
    pool: WorkerPoolConfig,
) -> Iterator[dict[str, object]]:
    if pool.vm_count != 1:
        raise ValueError(f"joint decode avg xtok supports only single-VM TPU pools, got vm_count={pool.vm_count}")
    if os.environ.get("TPU_VISIBLE_CHIPS") is not None:
        raise ValueError(
            "joint decode avg xtok supervisor expects to own the full TPU VM; TPU_VISIBLE_CHIPS is already set"
        )

    chip_pairs = _chip_pairs(pool.chips_per_vm)
    logger.info(
        "Starting joint-decode-avg-xtok supervisor pool=%s shard=%d chips_per_vm=%d chip_pairs=%s",
        pool.pool_id,
        shard_info.shard_idx,
        pool.chips_per_vm,
        chip_pairs,
    )

    with tempfile.TemporaryDirectory(prefix="joint_decode_avg_xtok_local_workers_") as tmp:
        tmpdir = Path(tmp)
        procs: list[subprocess.Popen[str]] = []
        threads: list[threading.Thread] = []
        try:
            for chip_pair in chip_pairs:
                proc, proc_threads = _spawn_child(
                    tmpdir=tmpdir,
                    config=config,
                    ledger_path=ledger_path,
                    pool_id=pool.pool_id,
                    shard_idx=shard_info.shard_idx,
                    chip_pair=chip_pair,
                )
                procs.append(proc)
                threads.extend(proc_threads)
            _wait_for_children(procs, threads, ledger_path)
        except Exception:
            _terminate_children(procs)
            raise

    yield {"status": "done", "pool_id": pool.pool_id, "shard_idx": shard_info.shard_idx}


def run_joint_decode_completion_chunks(config: JointDecodeCompletionStepConfig) -> None:
    if not config.worker_pools:
        raise ValueError("joint decode avg xtok requires at least one worker pool")

    chunks_dir = os.path.join(config.output_path, "chunks", f"chunk_size={config.chunk_size}")
    chunks = sweep_chunk_specs(
        chunks_dir,
        _num_prompts(config.prompts_path),
        config.sampling.n_samples,
        config.sampling.advisor_weights,
        config.chunk_size,
    )
    ledger_path = ledger.convert_mirror_path(
        ledger_prefix=config.ledger_prefix,
        output_path=config.output_path,
    )
    ledger.ensure_manifest(ledger_path, chunks)

    def make_process_shard(pool: WorkerPoolConfig):
        return functools.partial(
            _supervise_joint_decode_worker,
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
        raise RuntimeError(f"joint decode avg xtok incomplete: {summary.done}/{summary.total} chunks done")

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
                    "completion_algorithm": "joint_decode_avg_xtok",
                    "decoder_model_path": config.decoder_model_path,
                    "advisor_model_path": config.advisor_model_path,
                },
            },
            sort_by=lambda record: record["completion_index"],
            num_output_shards=1,
        )
        .write_jsonl(path, skip_existing=True)
    )
    ZephyrContext(
        name="joint-decode-avg-xtok-completions-aggregate",
        max_workers=config.aggregate_workers,
        resources=ResourceConfig(cpu=1, ram="4g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(aggregate_pipeline)
    logger.info("Wrote joint-decode-avg-xtok completion rows to %s", path)

    token_paths_path = os.path.join(config.output_path, TOKEN_PATHS_FILENAME)
    token_paths_pipeline = (
        Dataset.from_list([chunk.token_paths_path for chunk in chunks if chunk.chunk_id in done_ids])
        .load_jsonl()
        .reshard(1)
        .write_jsonl(token_paths_path, skip_existing=True)
    )
    ZephyrContext(
        name="joint-decode-avg-xtok-token-paths-aggregate",
        max_workers=config.aggregate_workers,
        resources=ResourceConfig(cpu=1, ram="4g", preemptible=True),
        coordinator_resources=ResourceConfig(cpu=0.1, ram="1g", preemptible=True),
    ).execute(token_paths_pipeline)
    logger.info("Wrote joint-decode-avg-xtok token paths to %s", token_paths_path)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xregion-worker-child-config", required=True)
    args = parser.parse_args()
    _run_joint_decode_local_worker(_child_config_from_file(args.xregion_worker_child_config))


if __name__ == "__main__":
    _main()
