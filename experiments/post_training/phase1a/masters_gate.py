# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Snowball-scale gate for FP32 master checkpoints (master_records), in two short runs.

``save`` trains Phase 1a's M1 recipe on Snowball's B for 3 steps on E_X's training rows,
saving a masters-only checkpoint at step 2 and a full one (dp_reshardable optimizer plus
master records) at step 3: the saves that crashed E_X's 67B checkpoint. ``reload``
resumes from that step-3 checkpoint at learning rate 0 and saves a full checkpoint at
step 4; with AdamW, weight decay 0 and a zero learning rate the masters cannot move, so
``compare`` must find step 4's masters bit-identical to step 3's, and step 2's complete.

    uv run python -m experiments.post_training.phase1a.masters_gate save --version V \\
        --b-uri S3_URI --b-version V --rows-uri S3_URI --rows-version V --run
    uv run python -m experiments.post_training.phase1a.masters_gate reload --version V ... \\
        --resume-path s3://.../checkpoints/global_step_3 --run
    python -m experiments.post_training.phase1a.masters_gate compare --step2 ... --step3 ... --step4 ...
"""

from __future__ import annotations

import dataclasses

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactDataSource, ArtifactHfModel, SkyRLRun, SkyRLSpec, SkyRLTopology, skyrl_step
from marin.training.training import LevanterCheckpoint

from experiments.post_training.chat_template import FAMILY_TOKENIZER_REVISIONS
from experiments.post_training.phase1a.experts import EXPERT_CLUSTER, EXPERT_SEQUENCES_PER_STEP, ROWS_FILES
from experiments.post_training.phase1a.launch import POLICY_CHAT_TEMPLATE, RETENTION, execution
from experiments.post_training.phase1a.matrix import (
    GPUS_PER_NODE,
    POLICY_NODES,
    Condition,
    Method,
    RunPlan,
    Task,
    role_plan,
    skyrl_recipe,
)

GATE_LEARNING_RATE = 1e-5
GATE_WARMUP_STEPS = 1


def gate_recipe(*, learning_rate: float, max_steps: int, weight_steps: list[int], optimizer_steps: list[int]) -> dict:
    recipe = skyrl_recipe(RunPlan(Method.M1, Condition.X, 0, learning_rate), {})
    recipe["trainer"].update(
        epochs=1,
        max_steps=max_steps,
        checkpoint_steps=weight_steps,
        optimizer_checkpoint_steps=optimizer_steps,
        ckpt_interval=weight_steps[0],
    )
    # Megatron's scheduler needs fewer warmup steps than the run's few total steps.
    recipe["trainer"]["policy"]["optimizer_config"]["num_warmup_steps"] = GATE_WARMUP_STEPS
    return recipe


def save_recipe() -> dict:
    return gate_recipe(learning_rate=GATE_LEARNING_RATE, max_steps=3, weight_steps=[2, 3], optimizer_steps=[3])


def reload_recipe(resume_path: str, to_step: int = 4) -> dict:
    """Resume ``resume_path`` at learning rate 0 and save a full checkpoint at ``to_step``."""
    recipe = gate_recipe(learning_rate=0.0, max_steps=to_step, weight_steps=[to_step], optimizer_steps=[to_step])
    recipe["trainer"].update(resume_mode="from_path", resume_path=resume_path)
    return recipe


def gate_topology() -> SkyRLTopology:
    base = role_plan(RunPlan(Method.M1, Condition.X, 0, GATE_LEARNING_RATE))
    plan = dataclasses.replace(
        base, train_batch_size=EXPERT_SEQUENCES_PER_STEP, policy_mini_batch_size=EXPERT_SEQUENCES_PER_STEP
    )
    return SkyRLTopology(num_nodes=POLICY_NODES, gpus_per_node=GPUS_PER_NODE, gpu_variant="H100", role_plan=plan)


def gate_step(name: str, recipe: dict, b_uri: str, b_version: str, rows_uri: str, rows_version: str):
    model = ArtifactStep.adopt(user_owned_name("models/phase1a-snowball-b"), b_version, b_uri, kind=LevanterCheckpoint)
    rows = ArtifactStep.adopt(user_owned_name("documents/phase0c-expert-rows"), rows_version, rows_uri)
    repo, revision = FAMILY_TOKENIZER_REVISIONS["grug"]
    spec_name = user_owned_name(f"checkpoints/phase0c/{name}")
    spec = SkyRLSpec(
        name=spec_name,
        version=resolve_version(spec_name, None),
        config_yaml=yaml.safe_dump(recipe, sort_keys=False),
        model=ArtifactHfModel(
            step=model, tokenizer_uri=repo, tokenizer_revision=revision, chat_template=POLICY_CHAT_TEMPLATE
        ),
        train_data=(ArtifactDataSource(rows, relative_path=ROWS_FILES[Task.X]),),
        validation_data=(),
        topology=gate_topology(),
        retention=RETENTION,
        seed=0,
    )
    return skyrl_step(spec, execution(EXPERT_CLUSTER), export_hf=False)


def compare_masters(step2: str, step3: str, step4: str) -> dict[str, int]:
    """Read every master record a tensor at a time: step 2 complete, step 4 bit-identical to step 3."""
    # torch and skyrl_train exist only in a SkyRL runtime, not the root environment.
    import torch  # noqa: PLC0415
    from skyrl_train.distributed.megatron.master_export import (  # noqa: PLC0415
        fp32_master_names,
        load_fp32_master,
    )

    names = fp32_master_names(step3)
    if fp32_master_names(step2) != names or fp32_master_names(step4) != names:
        raise ValueError("master record sets differ between steps 2, 3 and 4")
    mismatched = sum(not torch.equal(load_fp32_master(step3, name), load_fp32_master(step4, name)) for name in names)
    return {"records": len(names), "mismatched": mismatched}


_common = [
    click.option("--b-uri", required=True),
    click.option("--b-version", required=True),
    click.option("--rows-uri", required=True),
    click.option("--rows-version", required=True),
]


def _with_common(fn):
    for option in reversed(_common):
        fn = option(fn)
    return fn


@click.group()
def cli() -> None:
    pass


@cli.command()
@_with_common
@rl_build_options
def save(b_uri: str, b_version: str, rows_uri: str, rows_version: str) -> list[ArtifactStep[SkyRLRun]]:
    return [gate_step("masters-gate-save", save_recipe(), b_uri, b_version, rows_uri, rows_version)]


@cli.command()
@_with_common
@click.option("--resume-path", required=True, help="a global_step_N checkpoint of a gate run")
@click.option("--to-step", type=int, default=4, show_default=True, help="the step to train to and save")
@click.option("--name", default="masters-gate-reload", show_default=True)
@rl_build_options
def reload(
    b_uri: str, b_version: str, rows_uri: str, rows_version: str, resume_path: str, to_step: int, name: str
) -> list[ArtifactStep[SkyRLRun]]:
    return [gate_step(name, reload_recipe(resume_path, to_step), b_uri, b_version, rows_uri, rows_version)]


@cli.command()
@click.option("--step2", required=True)
@click.option("--step3", required=True)
@click.option("--step4", required=True)
def compare(step2: str, step3: str, step4: str) -> None:
    result = compare_masters(step2, step3, step4)
    click.echo(result)
    if result["mismatched"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
