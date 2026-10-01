# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPQA-style science curriculum SFT trial on the September Snowball HF checkpoint.

GLM writes and blind-verifies four-option questions for NMR assignment, polar organic mechanisms,
and special relativity. The base checkpoint answers each kept question several times in thinking
mode; its own solutions that box the keyed letter become the SFT rows.

The generate stage calls the GLM relay, so submit it through the hub to ``cw-us-east-08a`` with
``MARIN_PREFIX`` set to the ``ttl=30d`` source prefix:

    uv run iris --cluster=marin job run --no-wait --target-cluster cw-us-east-08a \\
      --enable-extra-resources --cpu 2 --memory 16GB --disk 32GB \\
      --job-name curriculum-science-generate-<date> \\
      -e MARIN_PREFIX s3://marin-us-east-02a/tmp/ttl=30d/curriculum-math-20260924 \\
      -e GLM_BULK_TOKEN "$GLM_BULK_TOKEN" \\
      -- uv run python experiments/post_training/baby_rsi/science_trial.py \\
      --stage generate --version <version> --run

The distill, baseline, train, and after stages run on ``cw-rno2a`` with ``MARIN_PREFIX`` set to the
``ttl=7d`` trial prefix and adopt the questions from the source prefix at ``QUESTIONS_VERSION``.
"""

import hashlib
import json

import click
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.baby_rsi.science import (
    NMR_CAPABILITY,
    ORGANIC_MECHANISMS_CAPABILITY,
    SPECIAL_RELATIVITY_CAPABILITY,
    generate_science_questions,
)
from experiments.post_training.baby_rsi.self_distill import SELF_CHAT_FILENAME, AnswerCheck, self_distill_step
from experiments.post_training.baby_rsi.trial import (
    CONTEXT_LENGTH,
    HF_MODEL,
    HF_REVISION,
    SOURCE_PREFIX,
    build_trial,
    snowball_model,
)
from experiments.sft.launcher import ArtifactDatasetSpec

QUESTIONS_VERSION = "2026.09.26.1"
SELF_DISTILL_VERSION = "2026.09.26.1"
CURRICULUM_IDS = (NMR_CAPABILITY, ORGANIC_MECHANISMS_CAPABILITY, SPECIAL_RELATIVITY_CAPABILITY)
EVALS = "gpqa-diamond,mmlu-pro"
EVAL_NAME = "curriculum-science-sep20"
REQUESTED_QUESTIONS_PER_CAPABILITY = 320
VERIFY_SAMPLES = 3
GLM_MAX_COMPLETION_TOKENS = 32768
SAMPLES_PER_PROBLEM = 4
SOLUTIONS_PER_PROBLEM = 1
SEED = 17
SELF_DISTILL_TEMPERATURE = 0.7
# Leaves room for the prompt and template inside CONTEXT_LENGTH; longer samples are rejected anyway.
SELF_DISTILL_MAX_COMPLETION_TOKENS = 3584


def build_generation() -> dict[str, ArtifactStep[Artifact]]:
    """Build one GLM question step per capability; run it on `cw-us-east-08a`, where the GLM relay is reachable."""
    return {
        capability_id: generate_science_questions(
            capability_id,
            version=QUESTIONS_VERSION,
            requested=REQUESTED_QUESTIONS_PER_CAPABILITY,
            verify_samples=VERIFY_SAMPLES,
            seed=SEED,
            max_completion_tokens=GLM_MAX_COMPLETION_TOKENS,
        )
        for capability_id in CURRICULUM_IDS
    }


def _adopted_questions() -> dict[str, ArtifactStep[Artifact]]:
    sources: dict[str, ArtifactStep[Artifact]] = {}
    for capability_id in CURRICULUM_IDS:
        questions_name = user_owned_name(f"documents/curriculum-sft/{capability_id}/science-questions")
        sources[capability_id] = ArtifactStep.adopt(
            name=user_owned_name(f"documents/curriculum-sft/{capability_id}/staged-science-questions"),
            version=QUESTIONS_VERSION,
            source=prefix_join(prefix_join(SOURCE_PREFIX, questions_name), QUESTIONS_VERSION),
            kind=Artifact,
        )
    return sources


def build_self_distill() -> ArtifactStep[Artifact]:
    """Sample the base checkpoint on the verified questions and keep its solutions that box the keyed letter."""
    return self_distill_step(
        _adopted_questions(),
        name="documents/curriculum-sft/science-self-distill",
        version=SELF_DISTILL_VERSION,
        answer_check=AnswerCheck.CHOICE,
        model=snowball_model(f"{EVAL_NAME}-self-distill", HF_MODEL, HF_REVISION),
        accelerator=AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
        samples_per_problem=SAMPLES_PER_PROBLEM,
        solutions_per_problem=SOLUTIONS_PER_PROBLEM,
        temperature=SELF_DISTILL_TEMPERATURE,
        max_completion_tokens=SELF_DISTILL_MAX_COMPLETION_TOKENS,
        max_sequence_tokens=CONTEXT_LENGTH,
        seed=SEED,
    )


def build_science_trial(version: str, learning_rate: float, warmup: int) -> dict[str, ArtifactStep]:
    """Build baseline and trained GPQA Diamond and MMLU-Pro evaluations around one SFT run on self-distilled rows."""
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
    trial = build_science_trial(version, learning_rate, warmup)
    if stage == "full":
        return {"baseline": trial["baseline"], "after": trial["after"]}
    return {stage: trial[stage]}


if __name__ == "__main__":
    main()
