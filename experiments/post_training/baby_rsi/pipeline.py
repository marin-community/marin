# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a finite artifact graph for the Baby RSI interface sketch."""

from dataclasses import dataclass
from typing import TypeVar

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec
from taskcompendium.parquet import read_tasks, write_tasks

from experiments.post_training.baby_rsi.dummy import (
    DummyAnalysisEngine,
    DummyProblemGenerator,
    DummyTrainer,
    accept_all_tasks,
    fixed_evaluation_tasks,
    initial_policy,
    run_policy,
    teacher_policy,
)
from experiments.post_training.baby_rsi.interfaces import (
    BabyRsiReport,
    EvaluationSummary,
    PolicyState,
    RolloutBuffer,
    RoundSummary,
    TaskPromptSet,
)

DATA_FILENAME = "data.json"
TASKS_FILENAME = "tasks.parquet"
REPORT_FILENAME = "report.md"
CPU_RESOURCES = ResourceConfig.with_cpu(cpu=1, ram="1g")

ModelT = TypeVar("ModelT", bound=BaseModel)


def _write_model(output_path: str, value: BaseModel) -> Artifact:
    output = StoragePath(output_path)
    output.mkdirs()
    (output / DATA_FILENAME).write_text(value.model_dump_json(indent=2) + "\n")
    return Artifact(path=output_path)


def _read_model(path: str, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate_json((StoragePath(path) / DATA_FILENAME).read_text())


def _write_tasks(output_path: str, tasks: tuple[TaskSpec, ...]) -> Artifact:
    output = StoragePath(output_path)
    output.mkdirs()
    write_tasks(str(output / TASKS_FILENAME), iter(tasks))
    return Artifact(path=output_path)


def _read_tasks(path: str) -> tuple[TaskSpec, ...]:
    return tuple(read_tasks(str(StoragePath(path) / TASKS_FILENAME)))


def write_initial_policy(*, output_path: str) -> Artifact:
    return _write_model(output_path, initial_policy())


def write_teacher_policy(*, output_path: str) -> Artifact:
    return _write_model(output_path, teacher_policy())


def write_evaluation_tasks(*, output_path: str) -> Artifact:
    return _write_tasks(output_path, fixed_evaluation_tasks())


def evaluate_policy(*, policy_path: str, tasks_path: str, output_path: str) -> Artifact:
    policy = _read_model(policy_path, PolicyState)
    tasks = _read_tasks(tasks_path)
    return _write_model(output_path, run_policy(policy, tasks))


def analyze_rollouts(*, rollouts_path: str, output_path: str) -> Artifact:
    rollouts = _read_model(rollouts_path, RolloutBuffer)
    return _write_model(output_path, DummyAnalysisEngine().analyze_failed_runs(rollouts))


def generate_training_tasks(*, prompts_path: str, inference_policy_path: str, output_path: str) -> Artifact:
    prompts = _read_model(prompts_path, TaskPromptSet)
    inference_policy = _read_model(inference_policy_path, PolicyState)
    tasks = DummyProblemGenerator().generate_tasks(prompts.prompts, inference_policy, accept_all_tasks)
    return _write_tasks(output_path, tasks)


def train_policy(*, policy_path: str, rollouts_path: str, output_path: str) -> Artifact:
    policy = _read_model(policy_path, PolicyState)
    rollouts = _read_model(rollouts_path, RolloutBuffer)
    return _write_model(output_path, DummyTrainer(policy).train(rollouts))


def _evaluation_summary(buffer: RolloutBuffer) -> EvaluationSummary:
    return EvaluationSummary(
        passed=sum(rollout.reward == 1.0 for rollout in buffer.rollouts),
        total=len(buffer.rollouts),
    )


def write_report(
    *,
    initial_evaluation_path: str,
    final_evaluation_path: str,
    analysis_paths: tuple[str, ...],
    output_path: str,
) -> Artifact:
    initial_evaluation = _read_model(initial_evaluation_path, RolloutBuffer)
    final_evaluation = _read_model(final_evaluation_path, RolloutBuffer)
    summaries = []
    for round_index, analysis_path in enumerate(analysis_paths):
        prompts = _read_model(analysis_path, TaskPromptSet)
        summaries.append(
            RoundSummary(
                round_index=round_index,
                targeted_capabilities=tuple(prompt.capability_id for prompt in prompts.prompts),
            )
        )
    report = BabyRsiReport(
        initial=_evaluation_summary(initial_evaluation),
        final=_evaluation_summary(final_evaluation),
        rounds=tuple(summaries),
    )
    output = StoragePath(output_path)
    output.mkdirs()
    (output / DATA_FILENAME).write_text(report.model_dump_json(indent=2) + "\n")
    lines = [
        "# Baby RSI dummy report",
        "",
        f"Initial evaluation: {report.initial.passed}/{report.initial.total} passed.",
        f"Final evaluation: {report.final.passed}/{report.final.total} passed.",
        "",
        "This report comes from deterministic dummy components. It does not report a model-training result.",
        "",
    ]
    for summary in report.rounds:
        targets = ", ".join(summary.targeted_capabilities) or "none"
        lines.append(f"Round {summary.round_index} targets: {targets}.")
    (output / REPORT_FILENAME).write_text("\n".join(lines) + "\n")
    return Artifact(path=output_path)


@dataclass(frozen=True)
class BabyRsiRoundSteps:
    evaluation_before: ArtifactStep[Artifact]
    analysis: ArtifactStep[Artifact]
    training_tasks: ArtifactStep[Artifact]
    training_rollouts: ArtifactStep[Artifact]
    policy_after: ArtifactStep[Artifact]
    evaluation_after: ArtifactStep[Artifact]


@dataclass(frozen=True)
class BabyRsiWorkflow:
    evaluation_tasks: ArtifactStep[Artifact]
    initial_policy: ArtifactStep[Artifact]
    teacher_policy: ArtifactStep[Artifact]
    initial_evaluation: ArtifactStep[Artifact]
    rounds: tuple[BabyRsiRoundSteps, ...]
    report: ArtifactStep[Artifact]


def build_workflow(*, rounds: int, version: str | None = None) -> BabyRsiWorkflow:
    """Build a finite feedback graph with one artifact per document interface."""
    if rounds <= 0:
        raise ValueError("Baby RSI requires at least one round")

    evaluation_tasks = apply(
        user_owned_name("documents/baby-rsi/evaluation-tasks"),
        remote(write_evaluation_tasks, resources=CPU_RESOURCES),
        version=version,
        output_path=OUT,
    )
    base_policy = apply(
        user_owned_name("models/baby-rsi/base-policy"),
        remote(write_initial_policy, resources=CPU_RESOURCES),
        version=version,
        output_path=OUT,
    )
    inference_policy = apply(
        user_owned_name("models/baby-rsi/inference-policy"),
        remote(write_teacher_policy, resources=CPU_RESOURCES),
        version=version,
        output_path=OUT,
    )
    initial_evaluation = apply(
        user_owned_name("evaluations/baby-rsi/baseline"),
        remote(evaluate_policy, resources=CPU_RESOURCES),
        version=version,
        policy_path=base_policy,
        tasks_path=evaluation_tasks,
        output_path=OUT,
    )

    policy = base_policy
    evaluation = initial_evaluation
    round_steps = []
    for round_index in range(rounds):
        prefix = f"baby-rsi/round-{round_index:02d}"
        analysis = apply(
            user_owned_name(f"documents/{prefix}/analysis"),
            remote(analyze_rollouts, resources=CPU_RESOURCES),
            version=version,
            rollouts_path=evaluation,
            output_path=OUT,
        )
        training_tasks = apply(
            user_owned_name(f"documents/{prefix}/tasks"),
            remote(generate_training_tasks, resources=CPU_RESOURCES),
            version=version,
            prompts_path=analysis,
            inference_policy_path=inference_policy,
            output_path=OUT,
        )
        training_rollouts = apply(
            user_owned_name(f"rollouts/{prefix}/training"),
            remote(evaluate_policy, resources=CPU_RESOURCES),
            version=version,
            policy_path=inference_policy,
            tasks_path=training_tasks,
            output_path=OUT,
        )
        trained_policy = apply(
            user_owned_name(f"models/{prefix}/trained"),
            remote(train_policy, resources=CPU_RESOURCES),
            version=version,
            policy_path=policy,
            rollouts_path=training_rollouts,
            output_path=OUT,
        )
        evaluation_after = apply(
            user_owned_name(f"evaluations/{prefix}/after"),
            remote(evaluate_policy, resources=CPU_RESOURCES),
            version=version,
            policy_path=trained_policy,
            tasks_path=evaluation_tasks,
            output_path=OUT,
        )
        round_steps.append(
            BabyRsiRoundSteps(
                evaluation_before=evaluation,
                analysis=analysis,
                training_tasks=training_tasks,
                training_rollouts=training_rollouts,
                policy_after=trained_policy,
                evaluation_after=evaluation_after,
            )
        )
        policy = trained_policy
        evaluation = evaluation_after

    report = apply(
        user_owned_name("reports/baby-rsi"),
        remote(write_report, resources=CPU_RESOURCES),
        version=version,
        initial_evaluation_path=initial_evaluation,
        final_evaluation_path=evaluation,
        analysis_paths=tuple(round_step.analysis for round_step in round_steps),
        output_path=OUT,
    )
    return BabyRsiWorkflow(
        evaluation_tasks=evaluation_tasks,
        initial_policy=base_policy,
        teacher_policy=inference_policy,
        initial_evaluation=initial_evaluation,
        rounds=tuple(round_steps),
        report=report,
    )


@click.command(help=__doc__)
@click.option("--rounds", type=int, default=1, show_default=True)
@build_options
def main(rounds: int) -> dict[str, ArtifactStep]:
    return {"report": build_workflow(rounds=rounds).report}


if __name__ == "__main__":
    main()
