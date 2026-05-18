# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Roll out the 10 Delphi scaling-ladder checkpoints on MATH-500.

Generation only — no grading at rollout time. Each model produces ``N_SAMPLES``
completions per problem. Grading runs as a separate downstream step that uses
marin's existing ``extract_boxed`` + ``safe_grade`` (sympy) — same code path as
``MathEnv.check_answer`` in ``experiments.evals.math500_eval``.

Prompts are precomputed in the download step using the same ``standard_fewshot``
formatter the rest of the math500 eval pipeline uses (see
``experiments/evals/vllm_math500_eval.py:92-98``). The strings are inlined here
so the executor's hash captures the exact prompt content and we don't pull in
``MathEnv``'s deps.

Edit the constants in the ``Configuration`` block to change sampling / problem
subset / prompt — anything wrapped in ``versioned(...)`` participates in the
executor's content hash, so a change there triggers re-runs of the affected steps.
"""

import logging
from dataclasses import dataclass

from datasets import load_dataset
from fray.cluster import ResourceConfig

from experiments.evals.delphi.delphi_checkpoints import DELPHI_CHECKPOINTS
from experiments.evals.delphi.grade import MATH_BOXED, GradingConfig, grade_rollouts
from experiments.evals.delphi.rollout import (
    VLLM_TPU_ENV_VARS,
    AggregateConfig,
    ChunksRootConfig,
    RolloutWorkerConfig,
    aggregate_rollouts,
    init_chunks_root,
    run_rollout_worker,
)
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.execution.remote import remote
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)


# === Configuration ===
N_SAMPLES = 4096
N_PROBLEMS: int | None = 256  # None = all 500 problems; int to subset
TEMPERATURE = 0.6
TOP_P = 1.0
TOP_K = 1000
MAX_TOKENS = 2048
SEED: int = 42
CHUNK_SIZE = 512
NUM_WORKERS = 4  # parallel TPU workers per model (each owns one v5p-8)
MODEL_ROLLOUT_OVERRIDES: dict[str, dict[str, int]] = {
    "1e22": {"n_samples": 512},
    "1e23": {"n_samples": 256},
}

# Resolved form of the ``standard_fewshot`` prompt format used in the rest of the
# math500 eval pipeline (``experiments/evals/vllm_math500_eval.py:92-98``,
# ``experiments/evals/math500_eval.py:97-103``). Matches
# ``MathEnv.standard_fewshot_prefix() + question_suffix()`` byte-for-byte.
MATH500_PROMPT_PREFIX = (
    "How many r's are in strawberry? Write your answer in \\boxed{} format.\n\n"
    "Let's spell the word out and number all the letters: "
    "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
    "We have r's at positions 3, 8, and 9. \\boxed{3}\n\n"
)
MATH500_PROMPT_SUFFIX = " Write your answer in \\boxed{} format."


# === Dataset download ===


@dataclass(frozen=True)
class DownloadMath500Config:
    output_path: str
    prompt_prefix: str = ""
    prompt_suffix: str = ""


def download_math500(config: DownloadMath500Config) -> None:
    """Materialize ``HuggingFaceH4/MATH-500`` (test split) as JSONL with precomputed prompts.

    Each row written: ``{"problem_id": int, "problem": str, "ground_truth": str,
    "prompt": str, "solution": str, "subject": str, "level": int, "unique_id": str}``.
    ``prompt`` is ``prompt_prefix + problem + prompt_suffix`` precomputed once; the
    rollout reads it verbatim. The dataset's ``answer`` field is already the clean
    boxed answer; we rename it to ``ground_truth`` to match the rollout primitive's
    expected schema.
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for i, raw in enumerate(dataset):
        row = dict(raw)
        problem = row["problem"]
        rows.append(
            {
                "problem_id": i,
                "problem": problem,
                "ground_truth": row["answer"],
                "prompt": config.prompt_prefix + problem + config.prompt_suffix,
                "solution": row["solution"],
                "subject": row.get("subject", ""),
                "level": row.get("level", 0),
                "unique_id": row.get("unique_id", ""),
            }
        )

    pipeline = Dataset.from_list(rows).reshard(1).write_jsonl(f"{config.output_path}/math500-{{shard:05d}}.jsonl.gz")
    ZephyrContext(name="download-math500").execute(pipeline)


download_math500_step = ExecutorStep(
    name="raw/math500",
    fn=download_math500,
    config=DownloadMath500Config(
        output_path=this_output_path(),
        prompt_prefix=versioned(MATH500_PROMPT_PREFIX),
        prompt_suffix=versioned(MATH500_PROMPT_SUFFIX),
    ),
)


# === Per-model rollout + grade steps ===


def _rollout_setting(slug: str, key: str, default: int) -> int:
    return MODEL_ROLLOUT_OVERRIDES.get(slug, {}).get(key, default)


def make_rollout_steps(
    slug: str,
    ckpt_rel_path: str,
    *,
    n_samples: int,
    num_workers: int,
) -> tuple[list[ExecutorStep], ExecutorStep]:
    chunks_root_step = ExecutorStep(
        name=f"rollouts/delphi/math500/{slug}/chunks",
        fn=remote(init_chunks_root, resources=ResourceConfig.with_cpu(ram="1g")),
        config=ChunksRootConfig(
            model_path=InputName.hardcoded(ckpt_rel_path),
            output_path=this_output_path(),
            dataset_path=output_path_of(download_math500_step),
            problem_id_field=versioned("problem_id"),
            problem_field=versioned("problem"),
            prompt_field=versioned("prompt"),
            ground_truth_field=versioned("ground_truth"),
            n_problems=versioned(N_PROBLEMS),
            n_samples=versioned(n_samples),
            temperature=versioned(TEMPERATURE),
            top_p=versioned(TOP_P),
            top_k=versioned(TOP_K),
            max_tokens=versioned(MAX_TOKENS),
            seed=versioned(SEED),
            chunk_size=versioned(CHUNK_SIZE),
        ),
    )
    chunks_path = output_path_of(chunks_root_step)
    workers = [
        ExecutorStep(
            name=f"rollouts/delphi/math500/{slug}/worker-{i:02d}",
            fn=remote(
                run_rollout_worker,
                resources=ResourceConfig.with_tpu("v5p-8"),
                pip_dependency_groups=["vllm", "tpu"],
                env_vars=VLLM_TPU_ENV_VARS,
            ),
            config=RolloutWorkerConfig(
                model_path=InputName.hardcoded(ckpt_rel_path),
                output_path=this_output_path(),
                dataset_path=output_path_of(download_math500_step),
                worker_id=versioned(i),
                num_workers=versioned(num_workers),
                problem_id_field=versioned("problem_id"),
                problem_field=versioned("problem"),
                prompt_field=versioned("prompt"),
                ground_truth_field=versioned("ground_truth"),
                n_problems=versioned(N_PROBLEMS),
                n_samples=versioned(n_samples),
                temperature=versioned(TEMPERATURE),
                top_p=versioned(TOP_P),
                top_k=versioned(TOP_K),
                max_tokens=versioned(MAX_TOKENS),
                seed=versioned(SEED),
                chunk_size=versioned(CHUNK_SIZE),
            ),
        )
        for i in range(num_workers)
    ]
    aggregator = ExecutorStep(
        name=f"rollouts/delphi/math500/{slug}",
        fn=remote(aggregate_rollouts, resources=ResourceConfig.with_cpu(ram="32g")),
        config=AggregateConfig(
            worker_paths=[output_path_of(w) for w in workers],
            output_path=this_output_path(),
            dataset_path=output_path_of(download_math500_step),
            problem_id_field=versioned("problem_id"),
            problem_field=versioned("problem"),
            prompt_field=versioned("prompt"),
            ground_truth_field=versioned("ground_truth"),
            n_problems=versioned(N_PROBLEMS),
            n_samples=versioned(n_samples),
            chunk_size=versioned(CHUNK_SIZE),
        ),
    )
    return [chunks_root_step, *workers], aggregator


def make_grade_step(slug: str, rollout_step: ExecutorStep) -> ExecutorStep:
    return ExecutorStep(
        name=f"grades/delphi/math500/{slug}",
        fn=remote(grade_rollouts, pip_dependency_groups=["eval", "math"]),
        config=GradingConfig(
            rollout_path=output_path_of(rollout_step),
            output_path=this_output_path(),
            grader=versioned(MATH_BOXED),
        ),
    )


def build_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = [download_math500_step]
    for slug, ckpt in DELPHI_CHECKPOINTS.items():
        workers, aggregator = make_rollout_steps(
            slug,
            ckpt,
            n_samples=_rollout_setting(slug, "n_samples", N_SAMPLES),
            num_workers=_rollout_setting(slug, "num_workers", NUM_WORKERS),
        )
        steps.extend(workers)
        steps.append(aggregator)
        steps.append(make_grade_step(slug, aggregator))
    return steps


if __name__ == "__main__":
    executor_main(
        steps=build_steps(),
        description=(
            "Delphi scaling-ladder rollouts + boxed-answer grading on MATH-500 "
            "(10 checkpoints, generation + per-problem accuracy)."
        ),
    )
