# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Phase 1a's M1 and M4 runs and the Phase 0c learning-rate sweep on Snowball.

    uv run python -m experiments.post_training.phase1a.launch --version YYYY.MM.DD \\
        --stage phase1a --learning-rate 1e-6 --b-uri S3_URI --b-version V \\
        --run-data-uri S3_URI --run-data-version V \\
        --expert-x PATH@REVISION --expert-r PATH@REVISION --b-teacher PATH@REVISION

Without ``--run`` the graph is built and validated only. ``--run-data-uri`` is the
root written by ``python -m experiments.post_training.phase1a.data``: one
``<condition>/seed<seed>/`` directory per cell, holding both arms' files.
"""

from __future__ import annotations

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.post_training.chat_template import FAMILY_TOKENIZER_REVISIONS
from experiments.post_training.phase1a.data import M1_FILE, M4_FILE
from experiments.post_training.phase1a.matrix import (
    CHECKPOINT_STEPS,
    Condition,
    Method,
    RunPlan,
    Task,
    TeacherModel,
    phase0c_lr_sweep,
    phase1a_runs,
    skyrl_recipe,
    topology,
)

CLUSTER = "cw-us-east-02a"
# Every scheduled checkpoint stays until the weight analysis has read it.
RETENTION = SkyRLRetentionPolicy(resume_checkpoint_count=len(CHECKPOINT_STEPS), temporary_storage_ttl_days=90)


def teacher_model(value: str) -> TeacherModel:
    path, separator, revision = value.rpartition("@")
    if not separator or not path or len(revision) != 40:
        raise click.BadParameter(f"expected PATH@<40-character revision>, got {value!r}")
    return TeacherModel(path, revision)


def run_data_relative_path(run: RunPlan) -> str:
    """Learning-rate sweep runs read the same cell data as theta_X at their seed."""
    file = M1_FILE if run.method is Method.M1 else M4_FILE
    return f"{run.condition}/seed{run.seed}/{file}"


def run_spec(
    run: RunPlan,
    *,
    model: ArtifactStep[LevanterCheckpoint],
    run_data: ArtifactStep,
    teachers: dict[Task, TeacherModel],
) -> SkyRLSpec:
    repo, revision = FAMILY_TOKENIZER_REVISIONS["grug"]
    recipe = skyrl_recipe(run, teachers)
    name = user_owned_name(f"checkpoints/phase1a/{run.name}")
    return SkyRLSpec(
        name=name,
        version=resolve_version(name, None),
        config_yaml=yaml.safe_dump(recipe, sort_keys=False),
        model=ArtifactHfModel(step=model, tokenizer_uri=repo, tokenizer_revision=revision),
        train_data=(ArtifactDataSource(run_data, relative_path=run_data_relative_path(run)),),
        validation_data=(),
        topology=topology(run),
        retention=RETENTION,
        seed=run.seed,
    )


def execution() -> IrisSkyRLExecution:
    return IrisSkyRLExecution(
        cluster=CLUSTER,
        cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
        cpu=16,
        memory="768GB",
        disk="2TB",
        priority="interactive",
        max_retries=1,
        target_cluster=CLUSTER,
        parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
        coordinator_timeout_hours=72,
        wandb_entity=None,
    )


def selected_runs(
    stage: str,
    learning_rate: float | None,
    methods: tuple[str, ...],
    conditions: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[RunPlan]:
    if stage == "lr-sweep":
        runs = phase0c_lr_sweep()
    elif learning_rate is None:
        raise click.UsageError("--learning-rate is required for phase1a; take it from the Phase 0c sweep")
    else:
        runs = phase1a_runs(learning_rate)
    return [
        run
        for run in runs
        if (not methods or run.method in methods)
        and (not conditions or run.condition in conditions)
        and (not seeds or run.seed in seeds)
    ]


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(["phase1a", "lr-sweep"]), required=True)
@click.option("--learning-rate", type=float)
@click.option("--method", "methods", multiple=True, type=click.Choice([str(method) for method in Method]))
@click.option("--condition", "conditions", multiple=True, type=click.Choice([str(c) for c in Condition]))
@click.option("--seed", "seeds", multiple=True, type=int)
@click.option("--b-uri", required=True, help="Stage 0 Snowball B: the artifact root holding its HF export.")
@click.option("--b-version", required=True)
@click.option("--run-data-uri", required=True)
@click.option("--run-data-version", required=True)
@click.option("--expert-x", required=True, type=teacher_model, help="E_X as PATH@REVISION.")
@click.option("--expert-r", required=True, type=teacher_model, help="E_R as PATH@REVISION.")
@click.option("--b-teacher", required=True, type=teacher_model, help="B as PATH@REVISION, the procedure nulls' teacher.")
@rl_build_options
def main(
    stage: str,
    learning_rate: float | None,
    methods: tuple[str, ...],
    conditions: tuple[str, ...],
    seeds: tuple[int, ...],
    b_uri: str,
    b_version: str,
    run_data_uri: str,
    run_data_version: str,
    expert_x: TeacherModel,
    expert_r: TeacherModel,
    b_teacher: TeacherModel,
) -> list[ArtifactStep[SkyRLRun]]:
    model = ArtifactStep.adopt(user_owned_name("models/phase1a-snowball-b"), b_version, b_uri, kind=LevanterCheckpoint)
    run_data = ArtifactStep.adopt(user_owned_name("documents/phase1a-run-data"), run_data_version, run_data_uri)
    pool = {Task.X: expert_x, Task.R: expert_r, Task.GENERAL: b_teacher}
    steps = []
    for run in selected_runs(stage, learning_rate, methods, conditions, seeds):
        teachers = {task: pool[task] for task in run.tasks}
        spec = run_spec(run, model=model, run_data=run_data, teachers=teachers)
        steps.append(skyrl_step(spec, execution(), export_hf=True))
    return steps


if __name__ == "__main__":
    main()
