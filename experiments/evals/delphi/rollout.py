# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rollout primitives for Delphi-style evaluation.

Decouples generation from grading. Generation is split across N parallel
worker steps (each holds one TPU + vLLM engine for one model), and a CPU
aggregator step gathers per-chunk outputs into one final per-problem JSONL.

The dataset's download step is responsible for prompt construction (so prompts
land in the executor hash via the download step's config, and the rollout
worker doesn't need the prompt-construction libraries — e.g. lm-eval-harness
for GSM8K — installed).

Implementation: each ``(problem, sample)`` pair is one flat request to vLLM
(``SamplingParams.n=1``). Flat requests are split into chunks of ``chunk_size``
(``chunk_starts = range(0, total_requests, chunk_size)``); each chunk maps to
one ``llm.generate`` call writing one JSONL.gz file. Worker ``w`` of
``num_workers`` processes ``chunk_starts[w::num_workers]`` (stride-partitioned,
so progress is balanced across workers and chunk ``i`` is always owned by
worker ``i % num_workers``). After all worker steps succeed, the aggregator
step walks chunks in their original order, groups by ``problem_id``, and
writes the final per-problem output. Worst-case pre-emption loss = one chunk
per worker.

Per-problem record schema in the final ``rollouts.jsonl.gz`` (one JSONL line):

    {"problem_id": int, "problem": str, "ground_truth": str,
     "prompt": str, "completions": [str, ...]}

All sampling/model/dataset config is recoverable from the executor's content
hash on ``RolloutWorkerConfig`` and ``AggregateConfig`` — caller should wrap
configurable fields with ``versioned(...)`` (the executor only includes
``VersionedValue`` / ``InputName`` / dataclass-walked / list-walked fields in
the version dict; plain scalars are silently omitted, see
``executor.py:1063-1090``).
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import fsspec
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.executor import InputName
from marin.utils import fsspec_exists, fsspec_glob

logger = logging.getLogger(__name__)


# Mirrors origin/will/delphi-evals:exp1337_eval_suite.py:101-118.
# - MARIN_VLLM_MODE=native: skip docker mode (no docker socket on iris workers).
# - VLLM_ENABLE_V1_MULTIPROCESSING=0: keep APIServer + EngineCore in one process so
#   the TPU stays claimed by a single process; otherwise the spawned EngineCore
#   child can't re-open libtpu and JAX falls back to CPU.
# - VLLM_ALLOW_LONG_MAX_MODEL_LEN, VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION,
#   VLLM_TPU_SKIP_PRECOMPILE: match Harbor's working recipe.
VLLM_TPU_ENV_VARS: dict[str, str] = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
}


@dataclass(frozen=True)
class ChunksRootConfig:
    """Content-addressed root for rollout chunk files shared across workers."""

    model_path: str | InputName
    output_path: str
    dataset_path: str | InputName

    problem_id_field: str = "problem_id"
    problem_field: str = "problem"
    prompt_field: str = "prompt"
    ground_truth_field: str = "ground_truth"

    n_problems: int | None = None
    n_samples: int = 10
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 1000
    max_tokens: int = 2048
    seed: int = 42
    stop: list[str] | None = None
    chunk_size: int = 512


@dataclass(frozen=True)
class RolloutWorkerConfig:
    """One worker's slice of an N-worker rollout for ``model_path`` on ``dataset_path``.

    Worker ``worker_id`` of ``num_workers`` processes the chunks
    ``range(0, n_problems * n_samples, chunk_size)[worker_id::num_workers]``.
    Each worker writes its chunk files to its own ``output_path``; the
    downstream aggregator step gathers across all worker output paths.

    ``dataset_path`` must contain JSONL.gz shards with at least the four fields
    named below; the prompt is taken verbatim from ``prompt_field`` (no further
    formatting is applied — the dataset's download step is responsible for
    constructing the full prompt).

    Output layout under ``output_path``:
      - ``rollouts-chunk-{flat_idx:08d}.jsonl.gz`` — one file per chunk owned
        by this worker. Each line is ``{"problem_id": ..., "completion": ...}``.
      - ``rollouts-chunk-{flat_idx:08d}.SUCCESS`` — written after the chunk
        file is closed successfully; resume skips chunks by this marker.
    """

    model_path: str | InputName
    output_path: str
    dataset_path: str | InputName

    worker_id: int
    num_workers: int
    chunks_path: str | InputName | None = None

    problem_id_field: str = "problem_id"
    problem_field: str = "problem"
    prompt_field: str = "prompt"
    ground_truth_field: str = "ground_truth"

    # If set, restrict to the first N problems (deterministic; debug knob).
    n_problems: int | None = None

    n_samples: int = 10
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 1000
    max_tokens: int = 2048
    seed: int = 42
    # Stop strings passed through to ``vllm.SamplingParams.stop`` — needed e.g. for
    # GSM8K where lm-eval-harness stops at ``["Question:", "</s>", "<|im_end|>"]``
    # so the model doesn't continue into a fake next exemplar.
    stop: list[str] | None = None

    # Number of flat (problem, sample) requests per ``llm.generate`` call and per
    # chunk file. Lower = finer checkpoint granularity / more files; higher = fewer
    # files / more work lost on a mid-chunk pre-emption.
    chunk_size: int = 512


@dataclass(frozen=True)
class AggregateConfig:
    """Aggregate per-chunk outputs across worker steps into the final per-problem JSONL.

    ``worker_paths`` is the list of worker step output paths in worker-id order;
    chunk ``chunk_starts[i]`` is read from ``worker_paths[i % len(worker_paths)]``,
    matching the stride partitioning the workers used.

    Output: ``{output_path}/rollouts.jsonl.gz`` — one record per problem with
    all ``n_samples`` completions merged.
    """

    worker_paths: list[str]
    output_path: str
    dataset_path: str | InputName
    chunks_path: str | InputName | None = None

    problem_id_field: str = "problem_id"
    problem_field: str = "problem"
    prompt_field: str = "prompt"
    ground_truth_field: str = "ground_truth"

    n_problems: int | None = None
    n_samples: int = 10
    chunk_size: int = 512


def init_chunks_root(config: ChunksRootConfig) -> None:
    """Initialize the shared chunk root step."""
    logger.info(f"Using shared rollout chunks root {config.output_path}")


def _load_vllm(model_path: str, seed: int):
    # Set TPU env vars BEFORE any vllm import so vllm-tpu picks them up at engine init.
    for k, v in VLLM_TPU_ENV_VARS.items():
        os.environ.setdefault(k, v)

    # Patch ``ragged_paged_attention.get_tuned_block_sizes`` BEFORE importing
    # vllm so the kernel's bound reference is the patched one. The auto-tuner
    # returns ``bkv_p=128`` paired with the pallas backend's pinned
    # ``page_size=16``, giving ``bkv_sz = 128 * 16 = 2048`` and a
    # ``bkv_double_buf`` VMEM scratch of 88 MB on the 1e22 / 1e23 Delphi models
    # — exceeds the 64 MB VMEM cap. Halving ``bkv_p`` brings scratch to 44 MB.
    # See ``tpu_inference/kernels/ragged_paged_attention/v3/kernel.py:1471, 1500``.
    import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa_kernel

    orig_get_tuned = rpa_kernel.get_tuned_block_sizes

    def patched_get_tuned(*args, **kwargs):
        bkv_p, bq_sz = orig_get_tuned(*args, **kwargs)
        return (max(1, bkv_p // 2), bq_sz)

    rpa_kernel.get_tuned_block_sizes = patched_get_tuned

    from vllm import LLM, SamplingParams

    # Same path resolution lm-eval-harness's evaluator uses with
    # ``discover_latest_checkpoint=True`` (``marin/evaluation/run.py:120-121``):
    # latest by mtime among ``{model_path}/**/config.json``.
    resolved_model_path = discover_hf_checkpoints(model_path)[-1]
    logger.info(f"Resolved {model_path} -> {resolved_model_path}")
    llm = LLM(
        model=resolved_model_path,
        trust_remote_code=True,
        load_format="runai_streamer",
        seed=seed,
    )
    return llm, SamplingParams


def _load_prompt_rows(dataset_path: str, n_problems: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(fsspec_glob(os.path.join(dataset_path, "*.jsonl.gz")))
    if not paths:
        raise FileNotFoundError(f"No prompt files found under {dataset_path}")

    for path in paths:
        with fsspec.open(path, "rt", compression="gzip") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows if n_problems is None else rows[:n_problems]


def _process_chunk(
    chunk_dir: str,
    chunk_start: int,
    chunk_size: int,
    total_requests: int,
    n_samples: int,
    problem_ids: list[Any],
    prompts: list[str],
    sampling: dict[str, Any],
    llm,
    SamplingParams,
) -> int:
    chunk_path = os.path.join(chunk_dir, f"rollouts-chunk-{chunk_start:08d}.jsonl.gz")
    success_path = os.path.join(chunk_dir, f"rollouts-chunk-{chunk_start:08d}.SUCCESS")
    if fsspec_exists(success_path):
        return 0

    chunk_end = min(chunk_start + chunk_size, total_requests)
    chunk_problem_ids = [problem_ids[k // n_samples] for k in range(chunk_start, chunk_end)]
    chunk_prompts = [prompts[k // n_samples] for k in range(chunk_start, chunk_end)]
    sp = SamplingParams(n=1, **sampling)

    outs = llm.generate(chunk_prompts, sp)

    with fsspec.open(chunk_path, "wt", compression="gzip") as f:
        for pid, out in zip(chunk_problem_ids, outs, strict=True):
            f.write(json.dumps({"problem_id": pid, "completion": out.outputs[0].text}) + "\n")

    with fsspec.open(success_path, "wt") as f:
        f.write("ok\n")

    logger.info(f"Wrote chunk {chunk_start} ({len(chunk_prompts)} requests) to {chunk_path}")
    return 1


def run_rollout_worker(config: RolloutWorkerConfig) -> None:
    """Process this worker's stride-partitioned slice of chunks on one TPU."""
    if config.n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {config.n_samples}")
    if config.chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {config.chunk_size}")

    rows = _load_prompt_rows(config.dataset_path, config.n_problems)

    problem_ids = [row[config.problem_id_field] for row in rows]
    prompts = [row[config.prompt_field] for row in rows]

    n_problems = len(rows)
    n_samples = config.n_samples
    total_requests = n_problems * n_samples
    all_chunk_starts = list(range(0, total_requests, config.chunk_size))
    my_chunk_starts = all_chunk_starts[config.worker_id :: config.num_workers]

    logger.info(
        f"Worker {config.worker_id}/{config.num_workers}: "
        f"{len(my_chunk_starts)} of {len(all_chunk_starts)} chunks to process."
    )
    if not my_chunk_starts:
        return

    sampling = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_tokens": config.max_tokens,
        "stop": config.stop,
    }

    chunk_dir = config.chunks_path or config.output_path
    llm, SamplingParams = _load_vllm(config.model_path, config.seed + config.worker_id)
    for chunk_start in my_chunk_starts:
        _process_chunk(
            chunk_dir,
            chunk_start,
            config.chunk_size,
            total_requests,
            n_samples,
            problem_ids,
            prompts,
            sampling,
            llm,
            SamplingParams,
        )


def _chunk_paths(chunk_dir: str, chunk_start: int) -> tuple[str, str]:
    chunk_path = os.path.join(chunk_dir, f"rollouts-chunk-{chunk_start:08d}.jsonl.gz")
    success_path = os.path.join(chunk_dir, f"rollouts-chunk-{chunk_start:08d}.SUCCESS")
    return chunk_path, success_path


def _validate_shared_chunks(chunk_dir: str, chunk_starts: list[int]) -> None:
    expected_files: set[str] = set()
    for chunk_start in chunk_starts:
        chunk_path, success_path = _chunk_paths(chunk_dir, chunk_start)
        expected_files.add(os.path.basename(chunk_path))
        expected_files.add(os.path.basename(success_path))

    chunk_files = fsspec_glob(os.path.join(chunk_dir, "rollouts-chunk-*.jsonl.gz"))
    success_files = fsspec_glob(os.path.join(chunk_dir, "rollouts-chunk-*.SUCCESS"))
    actual_files = {os.path.basename(path) for path in [*chunk_files, *success_files]}

    def summarize_files(files: list[str]) -> str:
        if len(files) <= 10:
            return str(files)
        return f"{files[:10]} ... and {len(files) - 10} more"

    missing_files = sorted(expected_files - actual_files)
    if missing_files:
        raise RuntimeError(f"Cannot aggregate: missing chunk files under {chunk_dir}: {summarize_files(missing_files)}")


def aggregate_rollouts(config: AggregateConfig) -> None:
    """Walk all worker chunk files, group by problem_id, write final rollouts.jsonl.gz."""
    final_path = os.path.join(config.output_path, "rollouts.jsonl.gz")

    rows = _load_prompt_rows(config.dataset_path, config.n_problems)

    problem_ids = [row[config.problem_id_field] for row in rows]
    problems = [row[config.problem_field] for row in rows]
    prompts = [row[config.prompt_field] for row in rows]
    ground_truths = [row[config.ground_truth_field] for row in rows]

    n_problems = len(rows)
    n_samples = config.n_samples
    total_requests = n_problems * n_samples
    chunk_starts = list(range(0, total_requests, config.chunk_size))
    num_workers = len(config.worker_paths)

    shared_chunks_path = config.chunks_path
    if shared_chunks_path is not None:
        _validate_shared_chunks(shared_chunks_path, chunk_starts)

    completions_by_pid: dict = {pid: [] for pid in problem_ids}
    for i, chunk_start in enumerate(chunk_starts):
        chunk_dir = shared_chunks_path if shared_chunks_path is not None else config.worker_paths[i % num_workers]
        chunk_path, success_path = _chunk_paths(chunk_dir, chunk_start)
        if shared_chunks_path is not None and not fsspec_exists(success_path):
            raise RuntimeError(f"Cannot aggregate: missing chunk success marker {success_path}")
        if not fsspec_exists(chunk_path):
            raise RuntimeError(f"Cannot aggregate: missing chunk file {chunk_path}")
        with fsspec.open(chunk_path, "rt", compression="gzip") as f:
            for line in f:
                rec = json.loads(line)
                completions_by_pid[rec["problem_id"]].append(rec["completion"])

    for pid, comps in completions_by_pid.items():
        if len(comps) != n_samples:
            raise RuntimeError(f"Problem {pid} aggregated {len(comps)} completions, expected {n_samples}")

    with fsspec.open(final_path, "wt", compression="gzip") as out:
        for pid, problem, prompt, gt in zip(problem_ids, problems, prompts, ground_truths, strict=True):
            record = {
                "problem_id": pid,
                "problem": problem,
                "ground_truth": gt,
                "prompt": prompt,
                "completions": completions_by_pid[pid],
            }
            out.write(json.dumps(record) + "\n")

    logger.info(f"Aggregated {n_problems} problems' rollouts to {final_path}")
