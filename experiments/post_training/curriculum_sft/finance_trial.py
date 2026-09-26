# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finance curriculum SFT trial on the September Snowball HF checkpoint, evaluated on FinanceBench.

Problems come from a seeded simulation of company financials, so reference answers are computed
rather than authored and no teacher model is involved. The base checkpoint solves each problem
several times in thinking mode; its own numerically correct solutions become the SFT rows.

All stages run on ``cw-rno2a`` with ``MARIN_PREFIX`` set to the ``ttl=7d`` trial prefix. Problem
generation is a small CPU step, so the distill stage builds it in the same graph:

    uv run iris --cluster=cw-rno2a job run --no-wait --job-name curriculum-finance-distill-<date> \\
      -e MARIN_PREFIX s3://marin-us-east-02a/tmp/ttl=7d/curriculum-math-20260924 \\
      -- uv run python experiments/post_training/curriculum_sft/finance_trial.py \\
      --stage distill --version <version> --run
"""

import hashlib
import json

import click
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.post_training.curriculum_sft.finance import (
    REPORTING_ANALYSIS,
    REPORTING_DISCLOSURES,
    WORKING_CAPITAL,
    generate_finance_problems,
)
from experiments.post_training.curriculum_sft.self_distill import SELF_CHAT_FILENAME, AnswerCheck, self_distill_step
from experiments.post_training.curriculum_sft.trial import (
    CONTEXT_LENGTH,
    HF_MODEL,
    HF_REVISION,
    build_trial,
    snowball_model,
)
from experiments.sft.launcher import ArtifactDatasetSpec

PROBLEMS_VERSION = "2026.09.26.1"
SELF_DISTILL_VERSION = "2026.09.26.1"
CURRICULUM_IDS = (REPORTING_ANALYSIS, REPORTING_DISCLOSURES, WORKING_CAPITAL)
EVALS = "financebench"
EVAL_NAME = "curriculum-finance-sep20"
PROBLEMS_PER_CAPABILITY = 300
SAMPLES_PER_PROBLEM = 4
SOLUTIONS_PER_PROBLEM = 1
SEED = 17
SELF_DISTILL_TEMPERATURE = 0.7
# Rendered problems reach about 1,400 tokens, so this leaves room for the prompt inside CONTEXT_LENGTH;
# longer rows are rejected anyway.
SELF_DISTILL_MAX_COMPLETION_TOKENS = 2560


def build_generation() -> dict[str, ArtifactStep[Artifact]]:
    """Build one programmatic problem step per capability."""
    return {
        capability_id: generate_finance_problems(
            capability_id, version=PROBLEMS_VERSION, count=PROBLEMS_PER_CAPABILITY, seed=SEED
        )
        for capability_id in CURRICULUM_IDS
    }


def build_self_distill() -> ArtifactStep[Artifact]:
    """Sample the base checkpoint on the generated problems and keep its numerically correct solutions."""
    return self_distill_step(
        build_generation(),
        name="documents/curriculum-sft/finance-self-distill",
        version=SELF_DISTILL_VERSION,
        answer_check=AnswerCheck.NUMERIC,
        model=snowball_model(f"{EVAL_NAME}-self-distill", HF_MODEL, HF_REVISION),
        accelerator=AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
        samples_per_problem=SAMPLES_PER_PROBLEM,
        solutions_per_problem=SOLUTIONS_PER_PROBLEM,
        temperature=SELF_DISTILL_TEMPERATURE,
        max_completion_tokens=SELF_DISTILL_MAX_COMPLETION_TOKENS,
        max_sequence_tokens=CONTEXT_LENGTH,
        seed=SEED,
    )


def build_finance_trial(version: str, learning_rate: float, warmup: int) -> dict[str, ArtifactStep]:
    """Build baseline and trained FinanceBench evaluations around one SFT run on self-distilled rows."""
    distilled = build_self_distill()
    datasets = [
        ArtifactDatasetSpec(
            slug=capability_id,
            artifact=distilled,
            train_glob=SELF_CHAT_FILENAME.format(capability_id=capability_id),
            weight=1.0,
        )
        for capability_id in CURRICULUM_IDS
    ]
    curriculum_key = hashlib.sha256(json.dumps(sorted(CURRICULUM_IDS)).encode()).hexdigest()[:12]
    return build_trial(
        name=f"{curriculum_key}/snowball-self",
        eval_name=EVAL_NAME,
        datasets=datasets,
        evals=EVALS,
        version=version,
        learning_rate=learning_rate,
        warmup=warmup,
    )


@click.command()
@click.option(
    "--stage", type=click.Choice(["generate", "distill", "baseline", "train", "after", "full"]), default="baseline"
)
@click.option("--learning-rate", type=float, help="Peak Adam learning rate; required for baseline through full.")
@click.option("--warmup", type=int, help="Linear warmup steps; required for baseline through full.")
@build_options
def main(stage: str, learning_rate: float | None, warmup: int | None) -> dict[str, ArtifactStep]:
    if stage == "generate":
        return build_generation()
    if stage == "distill":
        return {"distill": build_self_distill()}
    version = resolve_version(EVAL_NAME, None)
    if learning_rate is None or warmup is None:
        raise click.UsageError(f"--stage {stage} requires --learning-rate and --warmup")
    trial = build_finance_trial(version, learning_rate, warmup)
    if stage == "full":
        return {"baseline": trial["baseline"], "after": trial["after"]}
    return {stage: trial[stage]}


if __name__ == "__main__":
    main()
