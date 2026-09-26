# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge-free math curriculum SFT trial on the September Snowball HF checkpoint."""

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

from experiments.post_training.curriculum_sft.generation import (
    CHAT_FILENAME,
    generate_curriculum_problems,
    solve_curriculum_problems,
)
from experiments.post_training.curriculum_sft.self_distill import SELF_CHAT_FILENAME, AnswerCheck, self_distill_step
from experiments.post_training.curriculum_sft.trial import (
    CONTEXT_LENGTH,
    HF_MODEL,
    HF_REVISION,
    SOURCE_PREFIX,
    build_trial,
    snowball_model,
)
from experiments.sft.launcher import ArtifactDatasetSpec

# Problems and solutions carry separate versions so a solve-recipe change reuses the accepted problems.
PROBLEMS_VERSION = "2026.09.26.1"
SOLUTIONS_VERSION = "2026.09.26.3"
CURRICULUM_IDS = (
    "d01.algebra.exact-symbolic-evaluation",
    "d01.algebra.scalar-equations",
    "d01.algebra.number-theoretic-reasoning",
)
EVALS = "olympiadbench-deterministic,math500"
REQUESTED_PROBLEMS_PER_CAPABILITY = 320
SAMPLES_PER_PROBLEM = 4
SOLUTIONS_PER_PROBLEM = 1
SEED = 17
PROBLEM_MAX_COMPLETION_TOKENS = 16384
SOLUTION_MAX_COMPLETION_TOKENS = 32768
SELF_DISTILL_VERSION = "2026.09.26.1"
SELF_DISTILL_TEMPERATURE = 0.7
# Leaves room for the prompt and template inside CONTEXT_LENGTH; longer samples are rejected anyway.
SELF_DISTILL_MAX_COMPLETION_TOKENS = 3584
TASK_SPECIFICATION = (
    "Target the difficulty of MATH levels 3-5 and AMC 12. Vary the setting, the quantities, and the "
    "structure across problems; do not default to solving one radical or logarithmic equation."
)


def build_generation() -> dict[str, ArtifactStep[Artifact]]:
    """Build GLM problem and blind-solve steps; run them on `cw-us-east-08a`, where the GLM relay is reachable."""
    steps: dict[str, ArtifactStep[Artifact]] = {}
    for capability_id in CURRICULUM_IDS:
        problems = generate_curriculum_problems(
            capability_id,
            version=PROBLEMS_VERSION,
            requested_problems=REQUESTED_PROBLEMS_PER_CAPABILITY,
            seed=SEED,
            max_completion_tokens=PROBLEM_MAX_COMPLETION_TOKENS,
            task_specification=TASK_SPECIFICATION,
        )
        steps[capability_id] = solve_curriculum_problems(
            problems,
            capability_id=capability_id,
            version=SOLUTIONS_VERSION,
            samples_per_problem=SAMPLES_PER_PROBLEM,
            solutions_per_problem=SOLUTIONS_PER_PROBLEM,
            tokenizer=HF_MODEL,
            tokenizer_revision=HF_REVISION,
            max_sequence_tokens=CONTEXT_LENGTH,
            seed=SEED,
            max_completion_tokens=SOLUTION_MAX_COMPLETION_TOKENS,
        )
    return steps


def _adopted_problems() -> dict[str, ArtifactStep[Artifact]]:
    sources: dict[str, ArtifactStep[Artifact]] = {}
    for capability_id in CURRICULUM_IDS:
        problems_name = user_owned_name(f"documents/curriculum-sft/{capability_id}/problems")
        sources[capability_id] = ArtifactStep.adopt(
            name=user_owned_name(f"documents/curriculum-sft/{capability_id}/staged-problems"),
            version=PROBLEMS_VERSION,
            source=prefix_join(prefix_join(SOURCE_PREFIX, problems_name), PROBLEMS_VERSION),
            kind=Artifact,
        )
    return sources


def build_self_distill() -> ArtifactStep[Artifact]:
    """Sample the base checkpoint on the accepted problems and keep its own verified solutions."""
    return self_distill_step(
        _adopted_problems(),
        name="documents/curriculum-sft/math-self-distill",
        version=SELF_DISTILL_VERSION,
        answer_check=AnswerCheck.MATH,
        model=snowball_model("curriculum-math-sep20-self-distill", HF_MODEL, HF_REVISION),
        accelerator=AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
        samples_per_problem=SAMPLES_PER_PROBLEM,
        solutions_per_problem=SOLUTIONS_PER_PROBLEM,
        temperature=SELF_DISTILL_TEMPERATURE,
        max_completion_tokens=SELF_DISTILL_MAX_COMPLETION_TOKENS,
        max_sequence_tokens=CONTEXT_LENGTH,
        seed=SEED,
    )


def _datasets(data: str) -> list[ArtifactDatasetSpec]:
    if data == "self":
        distilled = build_self_distill()
        return [
            ArtifactDatasetSpec(
                slug=capability_id,
                artifact=distilled,
                train_glob=SELF_CHAT_FILENAME.format(capability_id=capability_id),
                weight=1.0,
            )
            for capability_id in CURRICULUM_IDS
        ]
    generated = _staged_generation()
    return [
        ArtifactDatasetSpec(slug=capability_id, artifact=generated[capability_id], train_glob=CHAT_FILENAME, weight=1.0)
        for capability_id in CURRICULUM_IDS
    ]


def _staged_generation() -> dict[str, ArtifactStep[Artifact]]:
    sources: dict[str, ArtifactStep[Artifact]] = {}
    for capability_id in CURRICULUM_IDS:
        name = user_owned_name(f"documents/curriculum-sft/{capability_id}/staged-solved-chat")
        solved_name = user_owned_name(f"documents/curriculum-sft/{capability_id}/solved-chat")
        sources[capability_id] = ArtifactStep.adopt(
            name=name,
            version=SOLUTIONS_VERSION,
            source=prefix_join(prefix_join(SOURCE_PREFIX, solved_name), SOLUTIONS_VERSION),
            kind=Artifact,
        )
    return sources


def build_math_trial(version: str, learning_rate: float, warmup: int, data: str) -> dict[str, ArtifactStep]:
    """Build the math trial; ``data`` is ``glm`` for GLM-solved rows or ``self`` for self-distilled rows."""
    curriculum_key = hashlib.sha256(json.dumps(sorted(CURRICULUM_IDS)).encode()).hexdigest()[:12]
    return build_trial(
        name=f"{curriculum_key}/snowball{'-self' if data == 'self' else ''}",
        eval_name="curriculum-math-sep20",
        datasets=_datasets(data),
        evals=EVALS,
        version=version,
        learning_rate=learning_rate,
        warmup=warmup,
    )


@click.command()
@click.option(
    "--stage", type=click.Choice(["generate", "distill", "baseline", "train", "after", "full"]), default="baseline"
)
@click.option("--data", type=click.Choice(["glm", "self"]), default="glm", help="Training rows for train/after.")
@click.option("--learning-rate", type=float, help="Peak Adam learning rate; required except for generation.")
@click.option("--warmup", type=int, help="Linear warmup steps; required except for generation.")
@build_options
def main(stage: str, data: str, learning_rate: float | None, warmup: int | None) -> dict[str, ArtifactStep]:
    if stage == "generate":
        return build_generation()
    if stage == "distill":
        return {"distill": build_self_distill()}
    version = resolve_version("curriculum-math-sep20", None)
    if learning_rate is None or warmup is None:
        raise click.UsageError(f"--stage {stage} requires --learning-rate and --warmup")
    trial = build_math_trial(version, learning_rate, warmup, data)
    if stage == "full":
        return {"baseline": trial["baseline"], "after": trial["after"]}
    return {stage: trial[stage]}


if __name__ == "__main__":
    main()
