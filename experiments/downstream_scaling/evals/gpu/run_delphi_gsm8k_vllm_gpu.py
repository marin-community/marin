"""Regular-sampling GSM8K baseline against the Delphi ladder (no joint decode).

Mirrors ``run_delphi_gsm8k_joint_decode_gpu.py`` exactly except that the
completions step uses a single vanilla ``vllm.LLM`` instead of two workers and
the joint-decode coordinator. Same prompts (``GSM8KTask``), same constants
(``MAX_TOKENS``, ``TEMPERATURE``, ``TOP_K``, ``CHUNK_SIZE``, ...), same chunked
write protocol with per-chunk ``SUCCESS`` markers, same output schema, same
grade step. Lets us measure the regular-sampling wall clock on the same model,
same hardware, same workload — the baseline that the joint-decode runner is
supposed to be ``almost as fast as``.

Outputs land under the prefix from ``--prefix`` if given, otherwise from the
``MARIN_PREFIX`` env var, otherwise the GCE region bucket
(``gs://marin-{region}``) when running on GCP, otherwise ``/tmp/marin``.

Install the required packages into the project venv before running::

    uv pip install --python .venv/bin/python 'lm-eval[math,api]@git+https://github.com/stanford-crfm/lm-evaluation-harness@d5e3391f22cde186c827674d5c3ec7c5f4fe0cab'
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import fsspec

from marin.execution.executor import ExecutorStep, InputName, MirroredValue, executor_main
from marin.execution.types import this_output_path, versioned
from marin.utils import fsspec_exists

from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.schema import completions_file, read_prompt_rows
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.evals.utils import version_path
from experiments.downstream_scaling.models.delphi import DELPHI_HF_REPOS

from vllm import LLM, SamplingParams

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
CHUNK_SIZE = 64

DELPHI_SLUGS = ["1e22"]


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


def run_vllm_gpu_completions(config: VllmGpuCompletionStepConfig) -> None:
    rows = list(read_prompt_rows(config.prompts_path))
    flat = [(row, sample_index) for row in rows for sample_index in range(config.n_samples)]
    prompts = [row["prompt"] for row, _ in flat]

    chunks_dir = os.path.join(config.output_path, "chunks")

    llm = LLM(
        model=config.model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=config.enable_prefix_caching,
        enforce_eager=config.enforce_eager,
        seed=config.seed,
    )
    sampling_params = SamplingParams(
        n=1,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_k=config.top_k,
        seed=config.seed,
        stop=list(config.stop) if config.stop else None,
        ignore_eos=False,
    )

    n_chunks = (len(flat) + config.chunk_size - 1) // config.chunk_size

    for chunk_id in range(n_chunks):
        chunk_file = os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz")
        success_file = os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.SUCCESS")
        if fsspec_exists(success_file):
            logger.info("chunk %d/%d already done; skipping", chunk_id + 1, n_chunks)
            continue
        start = chunk_id * config.chunk_size
        end = min(start + config.chunk_size, len(flat))
        chunk_flat = flat[start:end]
        chunk_prompts = prompts[start:end]
        t0 = time.monotonic()
        outputs = llm.generate(chunk_prompts, sampling_params)
        with fsspec.open(chunk_file, "wt", compression="gzip") as f:
            for (row, sample_index), output in zip(chunk_flat, outputs, strict=True):
                completion = output.outputs[0]
                f.write(json.dumps({
                    "id": row["id"],
                    "sample_index": sample_index,
                    "text": completion.text,
                    "finish_reason": completion.finish_reason or "unknown",
                }) + "\n")
        with fsspec.open(success_file, "wt"):
            pass
        logger.info("chunk %d/%d done in %.1fs", chunk_id + 1, n_chunks, time.monotonic() - t0)

    by_id: dict[str, list[dict[str, Any]]] = {}
    for chunk_id in range(n_chunks):
        chunk_file = os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz")
        with fsspec.open(chunk_file, "rt", compression="gzip") as f:
            for line in f:
                item = json.loads(line)
                by_id.setdefault(item["id"], []).append({
                    "text": item["text"],
                    "metadata": {"sample_index": item["sample_index"], "finish_reason": item["finish_reason"]},
                })

    with fsspec.open(completions_file(config.output_path), "wt", compression="gzip") as f:
        for row in rows:
            f.write(json.dumps({"id": row["id"], "completions": by_id[row["id"]]}) + "\n")


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
            ),
        )


def main() -> None:
    task = GSM8KTask(
        config=GSM8KTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
        )
    )

    steps = [
        make_eval_step(
            name=f"downstream_scaling/evals/gpu/delphi/gsm8k/vllm/{slug}",
            model_path=DELPHI_HF_REPOS[slug],
            task=task,
            alg=VllmGpuCompletionAlgorithm(
                max_tokens=MAX_TOKENS,
                top_k=TOP_K,
                temperature=TEMPERATURE,
                n_samples=N_SAMPLES,
                seed=SEED,
                stop=STOP_TOKENS,
            ),
        )
        for slug in DELPHI_SLUGS
    ]

    executor_main(
        steps=steps,
        max_concurrent=1,
        description="Delphi GSM8K vLLM baseline (no joint decode) on GPU.",
    )


if __name__ == "__main__":
    main()
