# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect seeded Qwen demonstrations on the BFCL complement with native harnesses."""

from dataclasses import replace

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactHfModel, SkyRLRolePlan, SkyRLTopology, skyrl_step
from marin.training.training import LevanterCheckpoint

from experiments.post_training.bfcl_rl.collect import (
    COLLECTION_EXECUTION,
    NATIVE_AGENT_PROFILES,
    SMOKE_TASKS,
    collection_spec,
    native_agent_profiles,
)

TEACHER_MODEL = "Qwen/Qwen3.6-35B-A3B"
TEACHER_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=True,
    policy_num_nodes=1,
    policy_num_gpus_per_node=4,
    colocate_policy_ref=True,
    reference_num_nodes=1,
    reference_num_gpus_per_node=4,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=2,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=2,
    inference_engine_expert_parallel_size=1,
    train_batch_size=8,
    policy_mini_batch_size=8,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=1,
)


def offline_collection_step(
    teacher_source: str,
    seed: int,
    task: str | None,
    generation_batch_size: int,
    harnesses: tuple[str, ...],
) -> ArtifactStep:
    """Bind the pinned Qwen teacher and unchanged complement to a generation-only run."""
    spec = collection_spec("teacher", task)
    recipe = yaml.safe_load(spec.config_yaml)
    harbor = recipe["terminal_bench"]["harbor"]
    harbor.update(name="opencode", version="1.18.2", agent_profiles=native_agent_profiles(harnesses))
    harbor.pop("thinking_format")
    generator = recipe["generator"]
    generator["model_loading"] = "stage_local"
    generator["engine_init_kwargs"] = {
        "enable_auto_tool_choice": True,
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "gdn_prefill_backend": "triton",
        "limit_mm_per_prompt": {"image": 0, "video": 0},
    }
    # Seed the teacher engines independently for each collection pass.
    recipe["trainer"]["seed"] = seed
    recipe["trainer"]["eval_batch_size"] = generation_batch_size
    model_step = ArtifactStep.adopt(
        user_owned_name("inputs/bfcl-rl-qwen36-teacher"),
        "2026.10.04",
        teacher_source,
        kind=LevanterCheckpoint,
        config={"model": TEACHER_MODEL, "revision": TEACHER_REVISION},
    )
    name = user_owned_name(f"rollouts/bfcl-rl-qwen36-native-seed-{seed}-{task or 'full'}")
    spec = replace(
        spec,
        name=name,
        version=resolve_version(name, None),
        config_yaml=yaml.safe_dump(recipe, sort_keys=False),
        model=ArtifactHfModel(model_step, TEACHER_MODEL, TEACHER_REVISION, relative_path=""),
        topology=SkyRLTopology(num_nodes=1, gpus_per_node=4, gpu_variant="H100", role_plan=ROLE_PLAN),
        seed=seed,
    )
    execution = replace(COLLECTION_EXECUTION, memory="320Gi", disk="1024Gi")
    return skyrl_step(spec, execution)


@click.command(help=__doc__)
@click.option("--teacher-source", required=True, help="Completed region-local snapshot of the pinned teacher.")
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
    teacher_source: str,
    seed: int,
    generation_batch_size: int,
    harnesses: tuple[str, ...],
    task: str | None,
) -> ArtifactStep:
    return offline_collection_step(teacher_source, seed, task, generation_batch_size, harnesses)


if __name__ == "__main__":
    main()
