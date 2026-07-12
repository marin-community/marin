# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run cross-tokenizer joint-decode-avg GSM8K evals on two local GPUs.

Decoder (A) is each Delphi checkpoint and advisor (B) is Qwen3-4B-Base.
The completion step loads both engines once per checkpoint, then runs the full
anchored-prefix-mass advisor-weight sweep. GSM8K prompts and grading use the
existing task implementation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any

import fsspec
from joint_decode.config import JointDecodeSamplingConfig
from joint_decode.gpu.config import JointDecodeConfig, JointDecodeModelConfig
from joint_decode.gpu.decoder import joint_decoder
from thalas.execution.executor import ExecutorStep, InputName, MirroredValue, executor_main
from thalas.execution.types import this_output_path, versioned
from transformers import AutoTokenizer

from experiments.downstream_scaling.evals.algorithms import xtok_selection
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
ADVISOR_MAX_TOKENS = 2 * MAX_TOKENS
MAX_MODEL_LEN = 4096
SEED = 42
STOP_TOKENS: tuple[str, ...] = ("Question:", "</s>", "<|im_end|>")

TEMPERATURE = 0.4
TOP_K_A = 16
TOP_K_B = 64
PREFIX_CREDIT = 0.1
ADVISOR_WEIGHTS: tuple[float, ...] = tuple(i / 10.0 for i in range(11))

CHUNK_SIZE = 512
MAX_MICROBATCH_SIZE = 512
BARRIER_TIMEOUT_S = 1200.0
GPU_MEMORY_UTILIZATION = 0.9

DELPHI_SLUGS: tuple[str, ...] = tuple(DELPHI_HF_REPOS)
ADVISOR_MODEL = "Qwen/Qwen3-4B-Base"


@dataclass(frozen=True)
class JointDecodeGpuCompletionStepConfig:
    output_path: str
    prompts_path: str
    decoder_model_path: str
    advisor_model_path: str
    sampling: JointDecodeSamplingConfig
    n_samples: int
    advisor_weights: tuple[float, ...]
    temperature: float
    prefix_credit: float
    max_model_len: int
    gpu_memory_utilization: float
    enable_prefix_caching: bool
    enforce_eager: bool
    chunk_size: int


@dataclass
class SelectorState:
    advisor_weight: float


def _load_vocab(model_path: str) -> xtok_selection.Vocab:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return xtok_selection.load_vocab(tokenizer)


def run_joint_decode_gpu_completions(config: JointDecodeGpuCompletionStepConfig) -> None:
    rows = list(read_prompt_rows(config.prompts_path))
    flat = [(row, sample_index) for row in rows for sample_index in range(config.n_samples)]
    prompts = [row["prompt"] for row, _ in flat]
    chunks_dir = os.path.join(config.output_path, "chunks")
    chunks_per_weight = (len(flat) + config.chunk_size - 1) // config.chunk_size

    vocab_a = _load_vocab(config.decoder_model_path)
    vocab_b = _load_vocab(config.advisor_model_path)
    selector_state = SelectorState(advisor_weight=config.advisor_weights[0])

    def select_token(
        a_topk: list[dict[str, Any]],
        b_topk: list[dict[str, Any]],
        *,
        rng: random.Random,
        request_index: int,
    ) -> tuple[list[int], list[int]]:
        del request_index
        return xtok_selection.select_avg_anchored(
            a_topk,
            b_topk,
            advisor_weight=selector_state.advisor_weight,
            temperature=config.temperature,
            prefix_credit=config.prefix_credit,
            rng=rng,
            vocab_a=vocab_a,
            vocab_b=vocab_b,
        )

    decode_config = JointDecodeConfig(
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

    with joint_decoder(decode_config, select_token=select_token) as decoder:
        for weight_index, advisor_weight in enumerate(config.advisor_weights):
            selector_state.advisor_weight = advisor_weight
            for chunk_index in range(chunks_per_weight):
                chunk_id = weight_index * chunks_per_weight + chunk_index
                chunk_file = os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz")
                success_file = os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.SUCCESS")
                if fsspec_exists(success_file):
                    logger.info("chunk %d already done; skipping", chunk_id)
                    continue

                start = chunk_index * config.chunk_size
                end = min(start + config.chunk_size, len(flat))
                started = time.monotonic()
                outputs = decoder.generate(prompts[start:end], prompts[start:end])
                with fsspec.open(chunk_file, "wt", compression="gzip") as f:
                    for (row, sample_index), output in zip(flat[start:end], outputs, strict=True):
                        f.write(
                            json.dumps(
                                {
                                    "id": row["id"],
                                    "completion_index": weight_index * config.n_samples + sample_index,
                                    "completion": {
                                        "text": output.text,
                                        "metadata": {
                                            "finish_reason": output.finish_reason,
                                            "advisor_weight": advisor_weight,
                                        },
                                    },
                                }
                            )
                            + "\n"
                        )
                with fsspec.open(success_file, "wt"):
                    pass
                logger.info("chunk %d done in %.1fs", chunk_id, time.monotonic() - started)

    by_id: dict[str, list[dict[str, Any]]] = {row["id"]: [] for row in rows}
    for chunk_id in range(chunks_per_weight * len(config.advisor_weights)):
        chunk_file = os.path.join(chunks_dir, f"chunk-{chunk_id:06d}.jsonl.gz")
        with fsspec.open(chunk_file, "rt", compression="gzip") as f:
            for line in f:
                record = json.loads(line)
                by_id[record["id"]].append(record)

    with fsspec.open(completions_file(config.output_path), "wt", compression="gzip") as f:
        for row in rows:
            records = sorted(by_id[row["id"]], key=lambda record: record["completion_index"])
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "completions": [record["completion"] for record in records],
                        "metadata": {
                            "completion_algorithm": "joint_decode_avg_xtok",
                            "decoder_model_path": config.decoder_model_path,
                            "advisor_model_path": config.advisor_model_path,
                        },
                    }
                )
                + "\n"
            )


@dataclass(frozen=True)
class JointDecodeGpuCompletionAlgorithm:
    advisor_model_path: str | InputName | MirroredValue
    sampling: JointDecodeSamplingConfig
    n_samples: int
    advisor_weights: tuple[float, ...]
    temperature: float
    prefix_credit: float
    max_model_len: int = MAX_MODEL_LEN
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION
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
                advisor_weights=versioned(self.advisor_weights),  # type: ignore[arg-type]
                temperature=versioned(self.temperature),  # type: ignore[arg-type]
                prefix_credit=versioned(self.prefix_credit),  # type: ignore[arg-type]
                max_model_len=versioned(self.max_model_len),  # type: ignore[arg-type]
                gpu_memory_utilization=versioned(self.gpu_memory_utilization),  # type: ignore[arg-type]
                enable_prefix_caching=versioned(self.enable_prefix_caching),  # type: ignore[arg-type]
                enforce_eager=versioned(self.enforce_eager),  # type: ignore[arg-type]
                chunk_size=versioned(self.chunk_size),  # type: ignore[arg-type]
            ),
        )


def build_steps(slugs: list[str], *, skip_grades: bool) -> list[ExecutorStep]:
    task = GSM8KTask(
        config=GSM8KTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
        )
    )
    sampling = JointDecodeSamplingConfig(
        max_tokens_a=MAX_TOKENS,
        max_tokens_b=ADVISOR_MAX_TOKENS,
        top_k_a=TOP_K_A,
        top_k_b=TOP_K_B,
        barrier_timeout_s=BARRIER_TIMEOUT_S,
        seed=SEED,
        stop=STOP_TOKENS,
        max_microbatch_size=MAX_MICROBATCH_SIZE,
        max_num_batched_tokens=MAX_MICROBATCH_SIZE + 8 * MAX_MODEL_LEN,
    )
    return [
        make_eval_step(
            name=(
                "downstream_scaling/evals/gpu/delphi/gsm8k/joint_decode_avg_xtok/"
                f"anchored_prefix_mass/qwen3_4b_base/{slug}"
            ),
            model_path=DELPHI_HF_REPOS[slug],
            task=task,
            alg=JointDecodeGpuCompletionAlgorithm(
                advisor_model_path=ADVISOR_MODEL,
                sampling=sampling,
                n_samples=N_SAMPLES,
                advisor_weights=ADVISOR_WEIGHTS,
                temperature=TEMPERATURE,
                prefix_credit=PREFIX_CREDIT,
            ),
            skip_grades=skip_grades,
        )
        for slug in slugs
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
        description="Delphi cross-tokenizer joint-decode-avg GSM8K evals on two local GPUs.",
    )


if __name__ == "__main__":
    main()
