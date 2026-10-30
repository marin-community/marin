# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Phase 1a's M1 and M4 runs and the Phase 0c learning-rate sweep on Snowball.

    uv run python -m experiments.post_training.phase1a.launch --version YYYY.MM.DD \\
        --stage phase1a --learning-rate 1e-6 --b-uri S3_URI --b-version V \\
        --run-data-uri S3_URI --run-data-version V \\
        --expert-x s3://E_X_EXPORT@sha256:ID --expert-r s3://E_R_EXPORT@sha256:ID --b-teacher REPO@REVISION

A teacher is a Hub ``REPO@<40-character revision>`` or an object-store HF export pinned by
its manifest identity, ``s3://...@sha256:<64 hex>``, which is checked before any run is
built and again when the runtime stages it.

Without ``--run`` the graph is built and validated only. ``--run-data-uri`` is the
root written by ``python -m experiments.post_training.phase1a.data``: one
``<condition>/seed<seed>/`` directory per cell, holding both arms' files.
"""

from __future__ import annotations

import json
import re

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
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

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

CLUSTER = "cw-rno2a"  # where B, the experts and the run data live
# Rollouts render prompts through the shared template B was trained on, not the base
# tokenizer's own; SkyRL forces this file onto the policy tokenizer on every node. SkyRL
# resolves it against its own root, vendor/skyrl.
POLICY_CHAT_TEMPLATE = "../../experiments/post_training/chat_templates/grug_shared.jinja"
# Eight CPU-offloaded ranks peak near 120 GiB each during restore. Reserve room for
# checkpoint copies and Ray above that peak; cw-rno2a H100 nodes have 2 TiB RAM.
LEARNER_MEMORY = "1500GB"
# Every scheduled checkpoint stays until the weight analysis has read it.
RETENTION = SkyRLRetentionPolicy(resume_checkpoint_count=len(CHECKPOINT_STEPS), temporary_storage_ttl_days=90)


# The manifest SkyRL writes beside every HF export; its identity hashes every file.
MODEL_MANIFEST_FILENAME = ".marinskyrl-model-manifest.json"
_MANIFEST_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}")


def teacher_model(value: str) -> TeacherModel:
    """A teacher pin: a Hub ``REPO@<40-character revision>`` or an ``s3://EXPORT@sha256:<64 hex>``.

    An object-store export is pinned by its manifest identity, which the runtime checks
    again when it stages the export.
    """
    path, separator, revision = value.rpartition("@")
    if not separator or not path:
        raise click.BadParameter(f"expected PATH@REVISION, got {value!r}")
    if path.startswith("s3://"):
        if not _MANIFEST_IDENTITY.fullmatch(revision):
            raise click.BadParameter(f"an s3:// teacher is pinned as @sha256:<64 hex>, got {value!r}")
    elif len(revision) != 40:
        raise click.BadParameter(f"a Hub teacher is pinned as REPO@<40-character revision>, got {value!r}")
    return TeacherModel(path, revision)


def require_teacher_pin(model: TeacherModel) -> None:
    """Check an object-store teacher's export against its pinned manifest identity; Hub pins pass."""
    if "://" not in model.path:
        return
    identity = json.loads(StoragePath(f"{model.path.rstrip('/')}/{MODEL_MANIFEST_FILENAME}").read_text())["identity"]
    if identity != model.revision:
        raise ValueError(f"teacher {model.path} has manifest identity {identity}, not the pinned {model.revision}")


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
    resume_from: str | None = None,
) -> SkyRLSpec:
    repo, revision = FAMILY_TOKENIZER_REVISIONS["grug"]
    recipe = skyrl_recipe(run, teachers)
    if resume_from is not None:
        recipe["trainer"].update(resume_mode="from_path", resume_path=resume_from)
    name = user_owned_name(f"checkpoints/phase1a/{run.name}")
    return SkyRLSpec(
        name=name,
        version=resolve_version(name, None),
        config_yaml=yaml.safe_dump(recipe, sort_keys=False),
        model=ArtifactHfModel(
            step=model, tokenizer_uri=repo, tokenizer_revision=revision, chat_template=POLICY_CHAT_TEMPLATE
        ),
        train_data=(ArtifactDataSource(run_data, relative_path=run_data_relative_path(run)),),
        validation_data=(),
        topology=topology(run),
        retention=RETENTION,
        seed=run.seed,
    )


def execution(cluster: str) -> IrisSkyRLExecution:
    return IrisSkyRLExecution(
        cluster=cluster,
        cluster_config=f"lib/iris/config/{cluster}.yaml",
        cpu=16,
        memory=LEARNER_MEMORY,
        disk="2TB",
        priority="interactive",
        max_retries=1,
        target_cluster=cluster,
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
        if learning_rate is not None:
            # The sweep runs serially, one learning rate at a time, smallest first.
            runs = [run for run in runs if run.learning_rate == learning_rate]
            if not runs:
                grid = sorted({run.learning_rate for run in phase0c_lr_sweep()})
                raise click.UsageError(f"--learning-rate {learning_rate} is not in the lr-sweep grid {grid}")
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
@click.option(
    "--learning-rate",
    type=float,
    help="phase1a: the sweep's chosen LR, required. lr-sweep: optionally restrict the sweep to one LR.",
)
@click.option("--method", "methods", multiple=True, type=click.Choice([str(method) for method in Method]))
@click.option("--condition", "conditions", multiple=True, type=click.Choice([str(c) for c in Condition]))
@click.option("--seed", "seeds", multiple=True, type=int)
@click.option("--b-uri", required=True, help="Stage 0 Snowball B: the artifact root holding its HF export.")
@click.option("--b-version", required=True)
@click.option("--run-data-uri", required=True)
@click.option("--run-data-version", required=True)
@click.option("--expert-x", type=teacher_model, help="E_X as REPO@REVISION or s3://EXPORT@sha256:ID.")
@click.option("--expert-r", type=teacher_model, help="E_R as REPO@REVISION or s3://EXPORT@sha256:ID.")
@click.option("--b-teacher", type=teacher_model, help="B as PATH@REVISION, the procedure nulls' teacher.")
@click.option(
    "--resume-from", help="Full optimizer checkpoint global_step directory; requires exactly one selected run."
)
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
    expert_x: TeacherModel | None,
    expert_r: TeacherModel | None,
    b_teacher: TeacherModel | None,
    resume_from: str | None,
) -> list[ArtifactStep[SkyRLRun]]:
    model = ArtifactStep.adopt(user_owned_name("models/phase1a-snowball-b"), b_version, b_uri, kind=LevanterCheckpoint)
    run_data = ArtifactStep.adopt(user_owned_name("documents/phase1a-run-data"), run_data_version, run_data_uri)
    configure_coreweave_s3()  # S3 teacher pins are read at launch; a no-op inside a CoreWeave pod
    pool = {Task.X: expert_x, Task.R: expert_r, Task.GENERAL: b_teacher}
    runs = selected_runs(stage, learning_rate, methods, conditions, seeds)
    if resume_from is not None and len(runs) != 1:
        raise click.UsageError("--resume-from requires exactly one selected run")
    for task in {task for run in runs if run.method is Method.M4 for task in run.tasks}:
        if pool[task] is None:
            raise click.UsageError(f"the selected M4 runs score {task} and need its teacher")
        require_teacher_pin(pool[task])
    steps = []
    for run in runs:
        teachers = {task: pool[task] for task in run.tasks} if run.method is Method.M4 else {}
        spec = run_spec(run, model=model, run_data=run_data, teachers=teachers, resume_from=resume_from)
        steps.append(skyrl_step(spec, execution(CLUSTER), export_hf=True))
    return steps


if __name__ == "__main__":
    main()
