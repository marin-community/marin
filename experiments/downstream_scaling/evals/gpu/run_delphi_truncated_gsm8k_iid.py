"""Run GPU IID downstream-scaling evals on truncated GSM8K for the Delphi ladder.

Uses normal GSM8K few-shot demonstrations, then gives the target problem a
truncated prefix of its gold solution. Completions are generated with local
vLLM on the GPU box; prompt writing and grading follow the existing GSM8K task
executor behavior.

Install the required packages into the project venv before running::

    uv pip install --python .venv/bin/python 'lm-eval[math,api]@git+https://github.com/stanford-crfm/lm-evaluation-harness@d5e3391f22cde186c827674d5c3ec7c5f4fe0cab'
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import queue
import sys
import time
from dataclasses import dataclass
from typing import Any

import fsspec

from marin.execution.executor import ExecutorStep, InputName, MirroredValue, executor_main
from marin.execution.types import this_output_path, versioned
from marin.utils import fsspec_exists

from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.schema import completions_file, read_prompt_rows
from experiments.downstream_scaling.evals.tasks.gsm8k_truncated import TruncatedGSM8KTask, TruncatedGSM8KTaskConfig
from experiments.downstream_scaling.evals.utils import version_path
from experiments.downstream_scaling.models.delphi import DELPHI_HF_REPOS
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

N_SAMPLES = 32
N_PROBLEMS = 256
NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234
MAX_TOKENS = 512
SEED = 42
STOP_TOKENS: tuple[str, ...] = ("Question:", "</s>", "<|im_end|>")
TEMPERATURE = 0.4
TOP_K = 16
BARRIER_TIMEOUT_S = 600.0
CHUNK_SIZE = 1024
TENSOR_PARALLEL_SIZE = 1
DATA_PARALLEL_SIZE = 2
WORKER_JOIN_TIMEOUT = 60.0

DELPHI_SLUGS = [
    "3e18",
    "9e18",
    "2e19",
    "3e19",
    "9e19",
    "2e20",
    "3e20",
    "1e21",
    "1e22",
    "1e23",
]
TRUNCATE_FRACTIONS = tuple(i / 10 for i in range(11))


@dataclass(frozen=True)
class VllmGpuCompletionStepConfig:
    output_path: str
    prompts_path: str
    model_path: str
    max_tokens: int
    top_k: int
    temperature: float
    n_samples: int
    seed: int
    stop: tuple[str, ...]
    max_model_len: int
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    enforce_eager: bool
    chunk_size: int
    tensor_parallel_size: int
    data_parallel_size: int


@dataclass(frozen=True)
class CompletionRequest:
    request_index: int
    chunk_position: int
    row: dict[str, Any]
    sample_index: int
    prompt: str


CompletionRecord = dict[str, Any]


def _make_llm(config: VllmGpuCompletionStepConfig):
    from vllm import LLM

    return LLM(
        model=config.model_path,
        trust_remote_code=True,
        tensor_parallel_size=config.tensor_parallel_size,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=config.enable_prefix_caching,
        enforce_eager=config.enforce_eager,
        seed=config.seed,
    )


def _make_sampling_params(config: VllmGpuCompletionStepConfig, request_index: int):
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        seed=config.seed + request_index,
        stop=list(config.stop) if config.stop else None,
        ignore_eos=False,
    )


def _request_record(request: CompletionRequest, text: str, finish_reason: str | None) -> CompletionRecord:
    return {
        "id": request.row["id"],
        "sample_index": request.sample_index,
        "text": text,
        "finish_reason": finish_reason or "unknown",
    }


def _write_chunk_file(chunk_file: str, records: list[CompletionRecord]) -> None:
    with fsspec.open(chunk_file, "wt", compression="gzip") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _merge_chunks(config: VllmGpuCompletionStepConfig, rows: list[dict[str, Any]], n_chunks: int) -> None:
    chunks_dir = os.path.join(config.output_path, "chunks")
    by_id: dict[str, list[dict[str, Any]]] = {}
    for chunk_id in range(n_chunks):
        chunk_file = os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz")
        with fsspec.open(chunk_file, "rt", compression="gzip") as f:
            for line in f:
                item = json.loads(line)
                by_id.setdefault(item["id"], []).append(
                    {
                        "text": item["text"],
                        "metadata": {"sample_index": item["sample_index"], "finish_reason": item["finish_reason"]},
                    }
                )

    with fsspec.open(completions_file(config.output_path), "wt", compression="gzip") as f:
        for row in rows:
            f.write(json.dumps({"id": row["id"], "completions": by_id[row["id"]]}) + "\n")


def _gpu_worker_loop(
    config: VllmGpuCompletionStepConfig,
    worker_rank: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_rank)
        llm = _make_llm(config)
        while True:
            item = task_queue.get()
            if item is None:
                return

            chunk_id, requests = item
            prompts = [request.prompt for request in requests]
            sampling_params = [_make_sampling_params(config, request.request_index) for request in requests]
            outputs = llm.generate(prompts, sampling_params)
            records = []
            for request, output in zip(requests, outputs, strict=True):
                completion = output.outputs[0]
                records.append((request.chunk_position, _request_record(request, completion.text, completion.finish_reason)))
            result_queue.put((worker_rank, chunk_id, records))
    except BaseException:
        raise


def _get_worker_result(
    result_queue: multiprocessing.Queue,
    workers: list[multiprocessing.Process],
    chunk_id: int,
) -> tuple[int, list[tuple[int, CompletionRecord]]]:
    while True:
        try:
            worker_rank, result_chunk_id, worker_records = result_queue.get(timeout=30)
        except queue.Empty:
            failed_workers = [
                (worker_rank, worker.exitcode)
                for worker_rank, worker in enumerate(workers)
                if worker.exitcode is not None and worker.exitcode != 0
            ]
            if failed_workers:
                raise RuntimeError(f"GPU worker failed while waiting for chunk {chunk_id}: {failed_workers}")
            continue

        if result_chunk_id != chunk_id:
            raise RuntimeError(f"worker {worker_rank} returned chunk {result_chunk_id}, expected {chunk_id}")
        return worker_rank, worker_records


def _stop_workers(task_queues: list[multiprocessing.Queue], workers: list[multiprocessing.Process]) -> None:
    for task_queue in task_queues:
        task_queue.put(None)
    for worker in workers:
        worker.join(timeout=WORKER_JOIN_TIMEOUT)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=WORKER_JOIN_TIMEOUT)
        if worker.is_alive():
            worker.kill()
            worker.join()


def run_vllm_gpu_completions(config: VllmGpuCompletionStepConfig) -> None:
    rows = list(read_prompt_rows(config.prompts_path))
    flat = [
        CompletionRequest(
            request_index=request_index,
            chunk_position=-1,
            row=row,
            sample_index=sample_index,
            prompt=row["prompt"],
        )
        for request_index, (row, sample_index) in enumerate(
            (row, sample_index) for row in rows for sample_index in range(config.n_samples)
        )
    ]

    chunks_dir = os.path.join(config.output_path, "chunks")
    n_chunks = (len(flat) + config.chunk_size - 1) // config.chunk_size
    worker_count = config.data_parallel_size

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    task_queues = [context.Queue() for _ in range(worker_count)]
    workers = [
        context.Process(
            target=_gpu_worker_loop,
            args=(config, worker_rank, task_queues[worker_rank], result_queue),
        )
        for worker_rank in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    try:
        for chunk_id in range(n_chunks):
            chunk_file = os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz")
            success_file = os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.SUCCESS")
            if fsspec_exists(success_file):
                logger.info("chunk %d/%d already done; skipping", chunk_id + 1, n_chunks)
                continue

            start = chunk_id * config.chunk_size
            end = min(start + config.chunk_size, len(flat))
            chunk_requests = [
                CompletionRequest(
                    request_index=request.request_index,
                    chunk_position=chunk_position,
                    row=request.row,
                    sample_index=request.sample_index,
                    prompt=request.prompt,
                )
                for chunk_position, request in enumerate(flat[start:end])
            ]

            t0 = time.monotonic()
            for worker_rank, task_queue in enumerate(task_queues):
                task_queue.put((chunk_id, chunk_requests[worker_rank::worker_count]))

            records_by_position: dict[int, CompletionRecord] = {}
            for _ in range(worker_count):
                _, worker_records = _get_worker_result(result_queue, workers, chunk_id)
                for chunk_position, record in worker_records:
                    records_by_position[chunk_position] = record

            records = [records_by_position[chunk_position] for chunk_position in range(len(chunk_requests))]
            _write_chunk_file(chunk_file, records)
            with fsspec.open(success_file, "wt"):
                pass
            logger.info("chunk %d/%d done in %.1fs", chunk_id + 1, n_chunks, time.monotonic() - t0)
    finally:
        _stop_workers(task_queues, workers)

    for worker_rank, worker in enumerate(workers):
        if worker.exitcode:
            raise RuntimeError(f"GPU worker {worker_rank} exited with code {worker.exitcode}")

    _merge_chunks(config, rows, n_chunks)

@dataclass(frozen=True)
class VllmGpuCompletionAlgorithm:
    max_tokens: int
    top_k: int
    temperature: float
    n_samples: int
    seed: int
    stop: tuple[str, ...]
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False
    enforce_eager: bool = True
    chunk_size: int = CHUNK_SIZE
    tensor_parallel_size: int = TENSOR_PARALLEL_SIZE
    data_parallel_size: int = DATA_PARALLEL_SIZE

    def make_completions_step(
        self,
        *,
        name: str,
        model_path: str | InputName | MirroredValue,
        prompts_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return ExecutorStep(
            name=name,
            fn=run_vllm_gpu_completions,
            config=VllmGpuCompletionStepConfig(
                output_path=this_output_path(),
                prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
                model_path=version_path(model_path),  # type: ignore[arg-type]
                max_tokens=versioned(self.max_tokens),  # type: ignore[arg-type]
                top_k=versioned(self.top_k),  # type: ignore[arg-type]
                temperature=versioned(self.temperature),  # type: ignore[arg-type]
                n_samples=versioned(self.n_samples),  # type: ignore[arg-type]
                seed=versioned(self.seed),  # type: ignore[arg-type]
                stop=versioned(self.stop),  # type: ignore[arg-type]
                max_model_len=versioned(self.max_model_len),  # type: ignore[arg-type]
                gpu_memory_utilization=versioned(self.gpu_memory_utilization),  # type: ignore[arg-type]
                enable_prefix_caching=versioned(self.enable_prefix_caching),  # type: ignore[arg-type]
                enforce_eager=versioned(self.enforce_eager),  # type: ignore[arg-type]
                chunk_size=versioned(self.chunk_size),  # type: ignore[arg-type]
                tensor_parallel_size=self.tensor_parallel_size,
                data_parallel_size=self.data_parallel_size,
            ),
        )


def make_task(truncate_fraction: float) -> TruncatedGSM8KTask:
    return TruncatedGSM8KTask(
        config=TruncatedGSM8KTaskConfig(
            tokenizer_path=llama3_tokenizer,
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
            truncate_fraction=truncate_fraction,
        )
    )


def build_steps(slugs: list[str]) -> list[ExecutorStep]:
    return [
        make_eval_step(
            name=f"downstream_scaling/evals/gpu/delphi/truncated_gsm8k/iid_v2/truncate_{i:02d}/{slug}",
            model_path=DELPHI_HF_REPOS[slug],
            task=make_task(truncate_fraction),
            alg=VllmGpuCompletionAlgorithm(
                max_tokens=MAX_TOKENS,
                top_k=TOP_K,
                temperature=TEMPERATURE,
                n_samples=N_SAMPLES,
                seed=SEED,
                stop=STOP_TOKENS,
            ),
        )
        for i, truncate_fraction in enumerate(TRUNCATE_FRACTIONS)
        for slug in slugs
    ]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--slugs", nargs="+", default=list(DELPHI_SLUGS))
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]

    unknown_slugs = sorted(set(args.slugs) - set(DELPHI_SLUGS))
    if unknown_slugs:
        parser.error(f"unknown Delphi slugs: {', '.join(unknown_slugs)}")

    executor_main(
        steps=build_steps(args.slugs),
        max_concurrent=1,
        description="Delphi scaling-ladder IID evals on truncated GSM8K with local GPU vLLM.",
    )


if __name__ == "__main__":
    main()
