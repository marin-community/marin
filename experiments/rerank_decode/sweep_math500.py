# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a single rerank-decode eval on MATH-500 for one (chunk_size, num_samples) config.

Designed to be launched as an independent SLURM job by launch_sweep_math500.py.
Each invocation evaluates one (chunk_size, num_samples) pair. The executor
framework gives each config a unique output_path via this_output_path().
"""

import atexit
import json
import logging
import os
from dataclasses import dataclass

import fsspec
import openai
from datasets import load_dataset
from transformers import AutoTokenizer

from experiments.rerank_decode.scorer import KVCacheScorer, VLLMLogprobScorer
from experiments.rerank_decode.serve import launch_vllm_server, shutdown_servers, wait_for_server
from experiments.rerank_decode.utils import rerank
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.rl.environments.tinker_environments.math_grading import extract_boxed, grade_answer
from marin.utils import fsspec_exists

logger = logging.getLogger(__name__)

PROPOSAL_MODEL = "meta-llama/Llama-3.2-1B"
SCORING_MODEL = "Qwen/Qwen3-4B"
CHECKPOINT_BATCH_SIZE = 10

QUESTION_SUFFIX = " Write your answer in \\boxed{} format."


@dataclass(frozen=True)
class RerankDecodeMath500Config:
    output_path: str

    chunk_size: int = 1
    num_samples: int = 1
    scorer_type: str = "kv_cache"  # "kv_cache" or "vllm"
    proposal_model: str = PROPOSAL_MODEL
    scoring_model: str = SCORING_MODEL
    max_tokens: int = 2048
    temperature: float = 1.0

    proposal_gpus: tuple[int, ...] = (1,)
    scoring_gpus: tuple[int, ...] = (0,)
    proposal_port: int = 8000
    scoring_port: int = 8001


def run_single_eval(config: RerankDecodeMath500Config):
    """Run rerank decode eval on MATH-500 for a single (chunk_size, num_samples) pair."""
    logging.basicConfig(level=logging.INFO)

    eos_token = AutoTokenizer.from_pretrained(config.scoring_model).eos_token

    # Load MATH-500 directly from HuggingFace (small dataset, no need for zephyr)
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [row["problem"] for row in dataset]
    answers = [row["answer"] for row in dataset]
    prompts = [problem + QUESTION_SUFFIX for problem in problems]

    logger.info("Loaded %d MATH-500 problems", len(problems))

    # Launch proposal server
    procs = []
    proposal_proc = launch_vllm_server(
        model=config.proposal_model,
        port=config.proposal_port,
        gpu_ids=list(config.proposal_gpus),
    )
    procs.append(proposal_proc)

    # Launch scoring server only for vllm scorer
    if config.scorer_type == "vllm":
        scoring_proc = launch_vllm_server(
            model=config.scoring_model,
            port=config.scoring_port,
            gpu_ids=list(config.scoring_gpus),
        )
        procs.append(scoring_proc)

    atexit.register(shutdown_servers, *procs)

    try:
        wait_for_server(config.proposal_port)
        proposal_client = openai.Client(
            base_url=f"http://localhost:{config.proposal_port}/v1", api_key="none"
        )

        if config.scorer_type == "vllm":
            wait_for_server(config.scoring_port)
            scoring_client = openai.Client(
                base_url=f"http://localhost:{config.scoring_port}/v1", api_key="none"
            )
            scorer = VLLMLogprobScorer(client=scoring_client, model=config.scoring_model)
        elif config.scorer_type == "kv_cache":
            scorer = KVCacheScorer(model_name=config.scoring_model, score_batch_size=4)
        else:
            raise ValueError(f"Unknown scorer_type: {config.scorer_type}")

        result_path = os.path.join(config.output_path, "results.json.gz")

        def process_batch(batch_rows: list[dict]) -> list[dict]:
            batch_results = []
            for row in batch_rows:
                i = row["problem_id"]
                scorer.reset()
                if (i + 1) % 50 == 0:
                    logger.info("  %d/%d", i + 1, len(prompts))

                generated = rerank(
                    proposal_client=proposal_client,
                    proposal_model=config.proposal_model,
                    scorer=scorer,
                    prompt=row["prompt"],
                    num_samples=config.num_samples,
                    max_tokens=config.max_tokens,
                    chunk_size=config.chunk_size,
                    temperature=config.temperature,
                    eos_token=eos_token,
                )

                try:
                    extracted = extract_boxed(generated)
                except ValueError:
                    extracted = None

                is_correct = False
                if extracted is not None:
                    is_correct = grade_answer(extracted, row["ground_truth"])

                batch_results.append({
                    "problem": row["problem"],
                    "ground_truth": row["ground_truth"],
                    "samples": [{"generated": generated, "extracted_answer": extracted, "correct": is_correct}],
                })

            return batch_results

        for batch_start in range(0, len(prompts), CHECKPOINT_BATCH_SIZE):
            batch_result_path = os.path.join(config.output_path, f"results-batch-{batch_start}.json.gz")
            if fsspec_exists(batch_result_path):
                continue

            batch_end = min(batch_start + CHECKPOINT_BATCH_SIZE, len(prompts))
            logger.info("Processing problems %d-%d/%d", batch_start, batch_end, len(prompts))
            batch_rows = [
                {
                    "problem_id": i,
                    "problem": problems[i],
                    "ground_truth": answers[i],
                    "prompt": prompts[i],
                }
                for i in range(batch_start, batch_end)
            ]
            batch_results = process_batch(batch_rows)
            with fsspec.open(batch_result_path, "wt", compression="gzip") as f:
                json.dump(batch_results, f, indent=2)

        results = []
        pass_at_1 = 0
        for batch_start in range(0, len(prompts), CHECKPOINT_BATCH_SIZE):
            batch_result_path = os.path.join(config.output_path, f"results-batch-{batch_start}.json.gz")
            with fsspec.open(batch_result_path, "rt", compression="gzip") as f:
                batch_results = json.load(f)
                results.extend(batch_results)

        for result in results:
            is_correct = result["samples"][0]["correct"]
            if is_correct:
                pass_at_1 += 1

        accuracy = pass_at_1 / len(results)
        print(f"chunk_size={config.chunk_size} num_samples={config.num_samples}: {accuracy:.4f} ({pass_at_1}/{len(results)})")

        output = {
            "pass_at_1": accuracy,
            "chunk_size": config.chunk_size,
            "num_samples": config.num_samples,
            "results": results,
        }

        with fsspec.open(result_path, "wt", compression="gzip") as f:
            json.dump(output, f, indent=2)

    finally:
        shutdown_servers(*procs)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--scorer", choices=["vllm", "kv_cache"], default="kv_cache")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    sweep_step = ExecutorStep(
        name="rerank_decode/math500_sweep",
        fn=run_single_eval,
        config=RerankDecodeMath500Config(
            output_path=this_output_path(),
            chunk_size=versioned(args.chunk_size),
            num_samples=versioned(args.num_samples),
            scorer_type=versioned(args.scorer),
        ),
    )

    executor_main(steps=[sweep_step], description="Rerank decode eval on MATH-500.")
