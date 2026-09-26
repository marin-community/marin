# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execution-verified code curriculum SFT trial on the September Snowball HF checkpoint.

Training rows are the base model's own solutions that pass the task's check (self-distillation), so
each task kind trains only once ``self_distill.AnswerCheck`` can grade it; see ``SELF_DISTILL_CHECKS``.
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

from experiments.post_training.curriculum_sft.code_tasks import (
    DYNAMIC_SEMANTICS_CAPABILITY,
    IMPLEMENTATION_CAPABILITY,
    TaskKind,
    generate_code_tasks,
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

TASKS_VERSION = "2026.09.26.1"
SELF_DISTILL_VERSION = "2026.09.26.1"
CODE_CAPABILITIES = {
    IMPLEMENTATION_CAPABILITY: TaskKind.IMPLEMENT,
    DYNAMIC_SEMANTICS_CAPABILITY: TaskKind.TRACE,
}
# Task kinds that self-distillation can grade. Trace tasks need a literal-equality check
# (``code_tasks.literals_equal``) and implement tasks a test-running check (``code_tasks.run_tests``);
# add a kind here once ``AnswerCheck`` supports it.
SELF_DISTILL_CHECKS: dict[TaskKind, AnswerCheck] = {}
EVALS = "humanevalplus,mbppplus,cruxeval"
EVAL_NAME = "curriculum-code-sep20"
REQUESTED_TASKS_PER_CAPABILITY = 320
SEED = 17
TASK_MAX_COMPLETION_TOKENS = 16384
SAMPLES_PER_PROBLEM = 4
SOLUTIONS_PER_PROBLEM = 1
SELF_DISTILL_TEMPERATURE = 0.7
# Leaves room for the prompt and template inside CONTEXT_LENGTH; longer samples are rejected anyway.
SELF_DISTILL_MAX_COMPLETION_TOKENS = 3584


def build_generation() -> dict[str, ArtifactStep[Artifact]]:
    """Build GLM task-generation steps; run them on `cw-us-east-08a`, where the GLM relay is reachable."""
    return {
        capability_id: generate_code_tasks(
            capability_id,
            kind=kind,
            version=TASKS_VERSION,
            requested=REQUESTED_TASKS_PER_CAPABILITY,
            seed=SEED,
            max_completion_tokens=TASK_MAX_COMPLETION_TOKENS,
        )
        for capability_id, kind in CODE_CAPABILITIES.items()
    }


def _adopted_tasks(capability_id: str, kind: TaskKind) -> ArtifactStep[Artifact]:
    tasks_name = user_owned_name(f"documents/curriculum-sft/{capability_id}/{kind}-tasks")
    return ArtifactStep.adopt(
        name=user_owned_name(f"documents/curriculum-sft/{capability_id}/staged-{kind}-tasks"),
        version=TASKS_VERSION,
        source=prefix_join(prefix_join(SOURCE_PREFIX, tasks_name), TASKS_VERSION),
        kind=Artifact,
    )


def build_self_distill() -> dict[str, ArtifactStep[Artifact]]:
    """Self-distill every capability whose task kind has an answer check, keyed by capability ID."""
    steps: dict[str, ArtifactStep[Artifact]] = {}
    for capability_id, kind in CODE_CAPABILITIES.items():
        if kind not in SELF_DISTILL_CHECKS:
            continue
        steps[capability_id] = self_distill_step(
            {capability_id: _adopted_tasks(capability_id, kind)},
            name=f"documents/curriculum-sft/code-self-distill-{kind}",
            version=SELF_DISTILL_VERSION,
            answer_check=SELF_DISTILL_CHECKS[kind],
            model=snowball_model(f"{EVAL_NAME}-self-distill-{kind}", HF_MODEL, HF_REVISION),
            accelerator=AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
            samples_per_problem=SAMPLES_PER_PROBLEM,
            solutions_per_problem=SOLUTIONS_PER_PROBLEM,
            temperature=SELF_DISTILL_TEMPERATURE,
            max_completion_tokens=SELF_DISTILL_MAX_COMPLETION_TOKENS,
            max_sequence_tokens=CONTEXT_LENGTH,
            seed=SEED,
        )
    if not steps:
        raise click.UsageError(
            "no code task kind has a self-distillation answer check yet; add AnswerCheck support for trace or "
            "implement tasks in self_distill.py and register it in code_trial.SELF_DISTILL_CHECKS"
        )
    return steps


def _datasets() -> list[ArtifactDatasetSpec]:
    return [
        ArtifactDatasetSpec(
            slug=capability_id,
            artifact=distilled,
            train_glob=SELF_CHAT_FILENAME.format(capability_id=capability_id),
            weight=1.0,
        )
        for capability_id, distilled in build_self_distill().items()
    ]


def build_code_trial(
    version: str, learning_rate: float, warmup: int, datasets: list[ArtifactDatasetSpec]
) -> dict[str, ArtifactStep]:
    curriculum_key = hashlib.sha256(json.dumps(sorted(CODE_CAPABILITIES)).encode()).hexdigest()[:12]
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
@click.option("--learning-rate", type=float, help="Peak Adam learning rate; required except for generate/distill.")
@click.option("--warmup", type=int, help="Linear warmup steps; required except for generate/distill.")
@build_options
def main(stage: str, learning_rate: float | None, warmup: int | None) -> dict[str, ArtifactStep]:
    if stage == "generate":
        return build_generation()
    if stage == "distill":
        return {f"distill-{capability_id}": step for capability_id, step in build_self_distill().items()}
    version = resolve_version(EVAL_NAME, None)
    if learning_rate is None or warmup is None:
        raise click.UsageError(f"--stage {stage} requires --learning-rate and --warmup")
    # The baseline evaluation does not depend on training data, so it runs before self-distillation exists.
    datasets = [] if stage == "baseline" else _datasets()
    trial = build_code_trial(version, learning_rate, warmup, datasets)
    if stage == "full":
        return {"baseline": trial["baseline"], "after": trial["after"]}
    return {stage: trial[stage]}


if __name__ == "__main__":
    main()
