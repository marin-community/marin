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

"""Sweep rerank decode on MATH-500 over chunk_size and num_samples.

Proposal model: Llama-3.2-1B
Scoring model: Qwen3-4B

Evaluates accuracy (pass@1 via reranking) for:
  chunk_size in [1, 5, 10, 20, 50]
  num_samples in [1, 2, 4, 8]
"""

import atexit
import itertools
import json
import logging
import os
from dataclasses import dataclass

import fsspec
import openai
from transformers import AutoTokenizer

from marin.utils import fsspec_exists

from experiments.evals.vllm_math500_eval import (
    PROMPT_FORMAT_REGISTRY,
    download_math500_step,
)
from experiments.rerank_decode.scorer import VLLMLogprobScorer
from experiments.rerank_decode.serve import launch_vllm_servers, shutdown_servers
from experiments.rerank_decode.utils import rerank
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)
from marin.rl.environments.tinker_environments.math_grading import extract_boxed, grade_answer
from zephyr import Backend, Dataset, load_jsonl

logger = logging.getLogger(__name__)

PROPOSAL_MODEL = "meta-llama/Llama-3.2-1B"
SCORING_MODEL = "Qwen/Qwen3-4B"

CHUNK_SIZES = [1, 5, 10, 20, 50]
NUM_SAMPLES = [1, 2, 4, 8]


@dataclass(frozen=True)
class SweepConfig:
    output_path: str
    math500_path: str | InputName = output_path_of(download_math500_step)

    proposal_model: str = PROPOSAL_MODEL
    scoring_model: str = SCORING_MODEL # TODO: add eos_token
    prompt_format: str = "question_with_boxed_suffix"
    max_tokens: int = 2048
    temperature: float = 1.0

    proposal_gpus: tuple[int, ...] = (0,)
    scoring_gpus: tuple[int, ...] = (1,)
    proposal_port: int = 8000
    scoring_port: int = 8001


def run_single_eval(
    proposal_client: openai.Client,
    scorer: VLLMLogprobScorer,
    config: SweepConfig,
    prompts: list[str],
    problems: list[str],
    answers: list[str],
    chunk_size: int,
    num_samples: int,
    result_path: str,
):
    """Run rerank decode on all prompts and grade results.

    Copied from vllm_math500_eval.run_math500_eval with the generation step
    replaced by rerank() and the inner sample loop removed (rerank returns one
    best completion per prompt).
    """

    eos_token = AutoTokenizer.from_pretrained(config.scoring_model).eos_token

    # -- generation (replaces llm.generate) --
    generations = []
    for i, prompt in enumerate(prompts):
        scorer.reset()
        if (i + 1) % 50 == 0:
            logger.info("  %d/%d", i + 1, len(prompts))
        generations.append(
            rerank(
                proposal_client=proposal_client,
                proposal_model=config.proposal_model,
                scorer=scorer,
                prompt=prompt,
                num_samples=num_samples,
                max_tokens=config.max_tokens,
                chunk_size=chunk_size,
                temperature=config.temperature,
                eos_token=eos_token,
            )
        )

    # -- grading (from vllm_math500_eval.py:149-197, adapted for single generation per prompt) --
    results = []
    pass_at_1 = 0
    for i, generated in enumerate(generations):
        ground_truth = answers[i]

        try:
            extracted = extract_boxed(generated)
        except ValueError:
            extracted = None

        is_correct = False
        if extracted is not None:
            is_correct = grade_answer(extracted, ground_truth)

        if is_correct:
            pass_at_1 += 1

        results.append({
            "problem": problems[i],
            "ground_truth": ground_truth,
            "samples": [{"generated": generated, "extracted_answer": extracted, "correct": is_correct}],
        })

    accuracy = pass_at_1 / len(results)
    print(f"chunk_size={chunk_size} num_samples={num_samples}: {accuracy:.4f} ({pass_at_1}/{len(results)})")

    output = {
        "pass_at_1": accuracy,
        "chunk_size": chunk_size,
        "num_samples": num_samples,
        "results": results,
    }

    with fsspec.open(result_path, "wt", compression="gzip") as f:
        json.dump(output, f, indent=2)

    return accuracy


def run_sweep(config: SweepConfig):
    logging.basicConfig(level=logging.INFO)

    # Load MATH-500 (same as vllm_math500_eval.py:130-137)
    dataset = Dataset.from_files(os.path.join(config.math500_path, "*.jsonl.gz")).flat_map(load_jsonl)
    rows = Backend.execute(dataset)

    problems = [row["problem"] for row in rows]
    answers = [row["answer"] for row in rows]

    prompt_formatter = PROMPT_FORMAT_REGISTRY[config.prompt_format]
    prompts = [prompt_formatter(problem) for problem in problems]

    logger.info("Loaded %d MATH-500 problems", len(problems))

    # Launch servers once for all sweep configs
    proc_a, proc_b = launch_vllm_servers(
        proposal_model=config.proposal_model,
        scoring_model=config.scoring_model,
        proposal_gpus=list(config.proposal_gpus),
        scoring_gpus=list(config.scoring_gpus),
        proposal_port=config.proposal_port,
        scoring_port=config.scoring_port,
    )
    atexit.register(shutdown_servers, proc_a, proc_b)

    proposal_client = openai.Client(
        base_url=f"http://localhost:{config.proposal_port}/v1", api_key="none"
    )
    scoring_client = openai.Client(
        base_url=f"http://localhost:{config.scoring_port}/v1", api_key="none"
    )
    scorer = VLLMLogprobScorer(client=scoring_client, model=config.scoring_model)

    all_results = {}

    try:
        for chunk_size, num_samples in itertools.product(CHUNK_SIZES, NUM_SAMPLES):
            key = f"chunk{chunk_size}_n{num_samples}"
            result_path = os.path.join(config.output_path, f"{key}.json.gz")

            # Skip if already computed
            if fsspec_exists(result_path):
                logger.info("Skipping %s (already exists)", key)
                with fsspec.open(result_path, "rt", compression="gzip") as f:
                    saved = json.load(f)
                all_results[key] = saved["pass_at_1"]
                continue

            logger.info("Running chunk_size=%d num_samples=%d", chunk_size, num_samples)
            accuracy = run_single_eval(
                proposal_client=proposal_client,
                scorer=scorer,
                config=config,
                prompts=prompts,
                problems=problems,
                answers=answers,
                chunk_size=chunk_size,
                num_samples=num_samples,
                result_path=result_path,
            )
            all_results[key] = accuracy

        # Print summary table
        print("\n=== MATH-500 Rerank Decode Sweep ===")
        print(f"Proposal: {config.proposal_model}")
        print(f"Scoring:  {config.scoring_model}")
        print()
        header = "chunk_size\\num_samples | " + " | ".join(f"n={n:>2}" for n in NUM_SAMPLES)
        print(header)
        print("-" * len(header))
        for chunk_size in CHUNK_SIZES:
            row = f"chunk={chunk_size:>3}              | "
            row += " | ".join(
                f"{all_results.get(f'chunk{chunk_size}_n{n}', float('nan')):.3f}" for n in NUM_SAMPLES
            )
            print(row)

        # Save summary
        summary = {
            "proposal_model": config.proposal_model,
            "scoring_model": config.scoring_model,
            "results": {k: v for k, v in all_results.items()},
        }
        with fsspec.open(os.path.join(config.output_path, "summary.json"), "wt") as f:
            json.dump(summary, f, indent=2)

    finally:
        shutdown_servers(proc_a, proc_b)


sweep_step = ExecutorStep(
    name="rerank_decode/math500_sweep",
    fn=run_sweep,
    config=SweepConfig(
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[download_math500_step, sweep_step], description="Rerank decode sweep on MATH-500.")
