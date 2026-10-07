# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect native Snowball counterparts for seeded Qwen BFCL preferences."""

from dataclasses import replace

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import skyrl_step

from experiments.post_training.bfcl_rl.collect import (
    COLLECTION_EXECUTION,
    NATIVE_AGENT_PROFILES,
    SMOKE_TASKS,
    collection_spec,
    native_agent_profiles,
)
from experiments.post_training.bfcl_rl.launch import recovered_model


def native_student_collection_step(
    recovery_version: str,
    policy_export_version: str,
    policy_checkpoint_step: int,
    seed: int,
    task: str | None,
    generation_batch_size: int,
    harnesses: tuple[str, ...],
) -> ArtifactStep:
    """Bind the recovered student to the teacher's task and native harness schedule."""
    spec = collection_spec("student", task)
    recipe = yaml.safe_load(spec.config_yaml)
    harbor = recipe["terminal_bench"]["harbor"]
    harbor.update(name="opencode", version="1.18.2", agent_profiles=native_agent_profiles(harnesses))
    harbor.pop("thinking_format")
    generator = recipe["generator"]
    generator["model_loading"] = "stage_local"
    generator["engine_init_kwargs"]["model_loader_extra_config"] = {"concurrency": 4}
    recipe["trainer"]["seed"] = seed
    recipe["trainer"]["eval_batch_size"] = generation_batch_size
    policy = replace(
        recovered_model(recovery_version, policy_export_version), relative_path=f"hf/step-{policy_checkpoint_step}"
    )
    name = user_owned_name(f"rollouts/bfcl-rl-native-student-seed-{seed}-{task or 'full'}")
    return skyrl_step(
        replace(
            spec,
            name=name,
            version=resolve_version(name, None),
            config_yaml=yaml.safe_dump(recipe, sort_keys=False),
            model=policy,
            seed=seed,
        ),
        replace(COLLECTION_EXECUTION, coordinator_timeout_hours=48, job_timeout_seconds=48 * 60 * 60),
    )


@click.command(help=__doc__)
@click.option("--recovery-version", required=True)
@click.option("--policy-export-version", required=True)
@click.option("--policy-checkpoint-step", type=click.IntRange(min=0), required=True)
@click.option("--seed", type=click.IntRange(min=0, max=2**31 - 1), required=True)
@click.option("--generation-batch-size", type=click.IntRange(min=1), required=True)
@click.option("--task", type=click.Choice(SMOKE_TASKS), default=None)
@click.option(
    "--harness",
    "harnesses",
    type=click.Choice([p["name"] for p in NATIVE_AGENT_PROFILES]),
    multiple=True,
    required=True,
    help="Ordered schedule; repeat a harness to increase its collection share.",
)
@rl_build_options
def main(
    recovery_version: str,
    policy_export_version: str,
    policy_checkpoint_step: int,
    seed: int,
    generation_batch_size: int,
    harnesses: tuple[str, ...],
    task: str | None,
) -> ArtifactStep:
    return native_student_collection_step(
        recovery_version,
        policy_export_version,
        policy_checkpoint_step,
        seed,
        task,
        generation_batch_size,
        harnesses,
    )


if __name__ == "__main__":
    main()
