# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Roll out the 10 Delphi scaling-ladder checkpoints on GSM8K (test split).

Generation only — no grading at rollout time. Each model produces ``N_SAMPLES``
completions per problem. Grading runs as a separate downstream step that uses
lm-eval-harness's ``gsm8k`` task filters and exact_match metric — i.e. exactly
what Will's ``EvalTaskConfig("gsm8k", 5)`` invokes.

Prompts are precomputed in the download step via lm-eval-harness's own
``Task.fewshot_context`` (with ``fewshot_random_seed=1234``, the default Will
inherits from ``lm_eval.evaluator.simple_evaluate``). So the rollout reads
prompts verbatim from the dataset and grading uses the same ``Task`` object —
no reimplementation of either the prompt or the metric.

Edit the constants in the ``Configuration`` block to change sampling / problem
subset — anything wrapped in ``versioned(...)`` participates in the executor's
content hash, so a change there triggers re-runs of the affected steps.
"""

import logging
from dataclasses import dataclass

from fray.cluster import ResourceConfig

from experiments.evals.delphi.delphi_checkpoints import DELPHI_CHECKPOINTS
from experiments.evals.delphi.grade import GSM8K_LM_EVAL, GradingConfig, grade_rollouts
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
N_PROBLEMS: int | None = 256  # None = all 1319 test problems; int to subset
TEMPERATURE = 0.1
TOP_P = 1.0
TOP_K = 1000
MAX_TOKENS = 512
SEED: int = 42
CHUNK_SIZE = 512
NUM_WORKERS = 16  # parallel TPU workers per model (each owns one v5p-8)
MODEL_ROLLOUT_OVERRIDES: dict[str, dict[str, int]] = {
    "1e22": {"n_samples": 512},
    "1e23": {"n_samples": 256},
}

# lm-eval-harness's default for the gsm8k task is num_fewshot=5; default
# fewshot_random_seed=1234 (lm_eval/evaluator.py:84). Will doesn't override
# either, so these values reproduce his prompts exactly.
NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234

# Match lm-eval-harness ``gsm8k.yaml`` ``generation_kwargs.until``: stop generation
# before the model rolls into a fake next ``Question:`` exemplar.
STOP_TOKENS = ["Question:", "</s>", "<|im_end|>"]

TARGET_SLUGS = ["1e23"]


# === Dataset download ===


@dataclass(frozen=True)
class DownloadGsm8kConfig:
    output_path: str
    num_fewshot: int = 5
    fewshot_seed: int = 1234


def download_gsm8k(config: DownloadGsm8kConfig) -> None:
    """Materialize ``openai/gsm8k`` (test split) as JSONL with precomputed prompts.

    Each row written: ``{"problem_id": int, "problem": str, "ground_truth": str,
    "prompt": str, "solution": str}``. ``prompt`` is the full lm-eval-harness
    fewshot context (``Task.fewshot_context``) for this problem; ``ground_truth``
    is the integer answer extracted from the trailing ``#### N`` line; ``solution``
    is the original full chain-of-thought from the dataset.
    """
    import re

    import lm_eval.tasks

    # lm-eval-harness's gsm8k task — same one Will runs.
    task = lm_eval.tasks.get_task_dict(["gsm8k"])["gsm8k"]
    task.set_fewshot_seed(config.fewshot_seed)
    test_docs = list(task.test_docs())

    answer_pattern = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
    rows = []
    for i, doc in enumerate(test_docs):
        prompt = task.fewshot_context(doc, num_fewshot=config.num_fewshot)
        match = answer_pattern.search(doc["answer"])
        if not match:
            raise ValueError(f"GSM8K problem_id={i} has no '####' answer marker: {doc!r}")
        rows.append(
            {
                "problem_id": i,
                "problem": doc["question"],
                "ground_truth": match.group(1).replace(",", ""),
                "prompt": prompt,
                "solution": doc["answer"],
            }
        )

    pipeline = Dataset.from_list(rows).reshard(1).write_jsonl(f"{config.output_path}/gsm8k-{{shard:05d}}.jsonl.gz")
    ZephyrContext(name="download-gsm8k").execute(pipeline)


download_gsm8k_step = ExecutorStep(
    name="raw/gsm8k",
    fn=remote(download_gsm8k, pip_dependency_groups=["eval"]),
    config=DownloadGsm8kConfig(
        output_path=this_output_path(),
        num_fewshot=versioned(NUM_FEWSHOT),
        fewshot_seed=versioned(FEWSHOT_SEED),
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
        name=f"rollouts/delphi/gsm8k/{slug}/chunks",
        fn=remote(init_chunks_root, resources=ResourceConfig.with_cpu(ram="1g")),
        config=ChunksRootConfig(
            model_path=InputName.hardcoded(ckpt_rel_path),
            output_path=this_output_path(),
            dataset_path=output_path_of(download_gsm8k_step),
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
            stop=versioned(STOP_TOKENS),
            chunk_size=versioned(CHUNK_SIZE),
        ),
    )
    chunks_path = output_path_of(chunks_root_step)
    workers = [
        ExecutorStep(
            name=f"rollouts/delphi/gsm8k/{slug}/worker-{i:02d}",
            fn=remote(
                run_rollout_worker,
                resources=ResourceConfig.with_tpu("v5p-8"),
                pip_dependency_groups=["vllm", "tpu"],
                env_vars=VLLM_TPU_ENV_VARS,
            ),
            config=RolloutWorkerConfig(
                model_path=InputName.hardcoded(ckpt_rel_path),
                output_path=this_output_path(),
                dataset_path=output_path_of(download_gsm8k_step),
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
                stop=versioned(STOP_TOKENS),
                chunk_size=versioned(CHUNK_SIZE),
                chunks_path=chunks_path,
            ),
        )
        for i in range(num_workers)
    ]
    aggregator = ExecutorStep(
        name=f"rollouts/delphi/gsm8k/{slug}",
        fn=remote(aggregate_rollouts, resources=ResourceConfig.with_cpu(ram="32g")),
        config=AggregateConfig(
            worker_paths=[output_path_of(w) for w in workers],
            output_path=this_output_path(),
            dataset_path=output_path_of(download_gsm8k_step),
            problem_id_field=versioned("problem_id"),
            problem_field=versioned("problem"),
            prompt_field=versioned("prompt"),
            ground_truth_field=versioned("ground_truth"),
            n_problems=versioned(N_PROBLEMS),
            n_samples=versioned(n_samples),
            chunk_size=versioned(CHUNK_SIZE),
            chunks_path=chunks_path
        ),
    )
    return [chunks_root_step, *workers], aggregator


def make_grade_step(slug: str, rollout_step: ExecutorStep) -> ExecutorStep:
    return ExecutorStep(
        name=f"grades/delphi/gsm8k/{slug}",
        fn=remote(grade_rollouts, pip_dependency_groups=["eval", "math"]),
        config=GradingConfig(
            rollout_path=output_path_of(rollout_step),
            output_path=this_output_path(),
            grader=versioned(GSM8K_LM_EVAL),
        ),
    )


def build_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = [download_gsm8k_step]
    for slug, ckpt in DELPHI_CHECKPOINTS.items():
        if slug in TARGET_SLUGS:
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
            "Delphi scaling-ladder rollouts + lm-eval-harness grading on GSM8K "
            "(10 checkpoints, generation + per-problem accuracy)."
        ),
    )
