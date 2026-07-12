# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run unnormalized-add joint-decode-gpu GSM8K evals on Delphi 1e22.

Decoder (A) = Delphi 1e22.
Advisor (B) = Llama 3.1 8B.

At each token, sample from the top-k union with score:

    score(token) = logit_a(token) + alpha * logit_b(token)

Tokens missing from one side use that side's top-k floor logit.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any

import fsspec
from joint_decode_gpu.config import JointDecodeConfig, JointDecodeModelConfig, JointDecodeSamplingConfig
from joint_decode_gpu.coordinator import JointDecoder
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue, executor_main
from thalas.execution.types import this_output_path, versioned

from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.schema import completions_file, read_prompt_rows
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.evals.utils import fsspec_exists, version_path
from experiments.downstream_scaling.models.delphi import DELPHI_HF_REPOS

logger = logging.getLogger(__name__)

N_SAMPLES = 32
N_PROBLEMS = 256
NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234
MAX_TOKENS = 512
SEED = 42
STOP_TOKENS: tuple[str, ...] = ("Question:", "</s>", "<|im_end|>")
TEMPERATURE = 0.4
TOP_K_A = 16
TOP_K_B = 16
MICROBATCH_SIZE = 8
BARRIER_TIMEOUT_S = 600.0
CHUNK_SIZE = 64

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
ADVISOR_MODEL = "meta-llama/Llama-3.1-8B"
ALPHAS = [i / 10.0 for i in range(21)]


def select_unnormalized_add(
    a_topk: list[dict[str, Any]],
    b_topk: list[dict[str, Any]],
    *,
    alpha: float,
    temperature: float,
    rng: random.Random,
    request_index: int,
) -> int:
    del request_index
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    a_logits = {int(t["token_id"]): float(t["logit"]) for t in a_topk}
    b_logits = {int(t["token_id"]): float(t["logit"]) for t in b_topk}
    if not a_logits or not b_logits:
        raise ValueError("both sides must provide at least one top-k logit")

    a_floor = min(a_logits.values())
    b_floor = min(b_logits.values())
    union = list(set(a_logits) | set(b_logits))
    scores = [a_logits.get(token_id, a_floor) + alpha * b_logits.get(token_id, b_floor) for token_id in union]

    if temperature == 0.0:
        return union[scores.index(max(scores))]

    max_score = max(scores)
    weights = [math.exp((score - max_score) / temperature) for score in scores]
    return rng.choices(union, weights=weights, k=1)[0]


@dataclass(frozen=True)
class JointDecodeGpuCompletionStepConfig:
    output_path: str
    prompts_path: str
    decoder_model_path: str
    advisor_model_path: str
    sampling: JointDecodeSamplingConfig
    n_samples: int
    alpha: float
    temperature: float
    max_model_len: int
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    enforce_eager: bool
    chunk_size: int


def run_joint_decode_gpu_completions(config: JointDecodeGpuCompletionStepConfig) -> None:
    rows = list(read_prompt_rows(config.prompts_path))
    flat = [(row, sample_index) for row in rows for sample_index in range(config.n_samples)]
    prompts = [row["prompt"] for row, _ in flat]

    chunks_dir = os.path.join(config.output_path, "chunks")

    jd_config = JointDecodeConfig(
        model_a=JointDecodeModelConfig(
            model_path=config.decoder_model_path,
            gpu_index=0,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=config.enable_prefix_caching,
            enforce_eager=config.enforce_eager,
        ),
        model_b=JointDecodeModelConfig(
            model_path=config.advisor_model_path,
            gpu_index=1,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=config.enable_prefix_caching,
            enforce_eager=config.enforce_eager,
        ),
        sampling=config.sampling,
    )
    select_token = functools.partial(
        select_unnormalized_add,
        alpha=config.alpha,
        temperature=config.temperature,
    )

    n_chunks = (len(flat) + config.chunk_size - 1) // config.chunk_size

    with JointDecoder(jd_config, select_token=select_token) as decoder:
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
            outputs = decoder.generate(chunk_prompts, chunk_prompts)
            with fsspec.open(chunk_file, "wt", compression="gzip") as f:
                for (row, sample_index), output in zip(chunk_flat, outputs, strict=True):
                    f.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "sample_index": sample_index,
                                "text": output.text,
                                "finish_reason": output.finish_reason or "unknown",
                            }
                        )
                        + "\n"
                    )
            with fsspec.open(success_file, "wt"):
                pass
            logger.info("chunk %d/%d done in %.1fs", chunk_id + 1, n_chunks, time.monotonic() - t0)

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


@dataclass(frozen=True)
class JointDecodeGpuCompletionAlgorithm:
    advisor_model_path: str | InputName | MirroredValue
    sampling: JointDecodeSamplingConfig
    n_samples: int
    alpha: float
    temperature: float
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
            fn=run_joint_decode_gpu_completions,
            config=JointDecodeGpuCompletionStepConfig(
                output_path=this_output_path(),
                prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
                decoder_model_path=version_path(model_path),  # type: ignore[arg-type]
                advisor_model_path=version_path(self.advisor_model_path),  # type: ignore[arg-type]
                sampling=versioned(self.sampling),  # type: ignore[arg-type]
                n_samples=versioned(self.n_samples),  # type: ignore[arg-type]
                alpha=versioned(self.alpha),  # type: ignore[arg-type]
                temperature=versioned(self.temperature),  # type: ignore[arg-type]
                max_model_len=versioned(self.max_model_len),  # type: ignore[arg-type]
                gpu_memory_utilization=versioned(self.gpu_memory_utilization),  # type: ignore[arg-type]
                enable_prefix_caching=versioned(self.enable_prefix_caching),  # type: ignore[arg-type]
                enforce_eager=versioned(self.enforce_eager),  # type: ignore[arg-type]
                chunk_size=versioned(self.chunk_size),  # type: ignore[arg-type]
            ),
        )


def make_task() -> GSM8KTask:
    return GSM8KTask(
        config=GSM8KTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
        )
    )


def build_steps(slugs: list[str], skip_grades: bool) -> list[ExecutorStep]:
    task = make_task()
    sampling = JointDecodeSamplingConfig(
        max_tokens=MAX_TOKENS,
        top_k_a=TOP_K_A,
        top_k_b=TOP_K_B,
        microbatch_size=MICROBATCH_SIZE,
        barrier_timeout_s=BARRIER_TIMEOUT_S,
        seed=SEED,
        stop=STOP_TOKENS,
    )

    return [
        make_eval_step(
            name=(
                f"downstream_scaling/evals/gpu/delphi/gsm8k/joint_decode/unnormalized_add/"
                f"alpha{round(alpha * 100):03d}/{slug}"
            ),
            model_path=DELPHI_HF_REPOS[slug],
            task=task,
            alg=JointDecodeGpuCompletionAlgorithm(
                advisor_model_path=ADVISOR_MODEL,
                sampling=sampling,
                n_samples=N_SAMPLES,
                alpha=alpha,
                temperature=TEMPERATURE,
            ),
            skip_grades=skip_grades,
        )
        for slug in slugs
        for alpha in ALPHAS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--slugs", nargs="+", default=list(DELPHI_SLUGS))
    parser.add_argument("--skip-grades", action="store_true")
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]

    unknown_slugs = sorted(set(args.slugs) - set(DELPHI_SLUGS))
    if unknown_slugs:
        parser.error(f"unknown Delphi slugs: {', '.join(unknown_slugs)}")

    executor_main(
        steps=build_steps(args.slugs, skip_grades=args.skip_grades),
        max_concurrent=1,
        description="Delphi 1e22 joint-decode-gpu GSM8K evals with unnormalized-add logits.",
    )


if __name__ == "__main__":
    main()
