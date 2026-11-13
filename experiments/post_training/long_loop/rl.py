# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment-owned asynchronous GS8 RL recipe and artifact binding."""

from dataclasses import replace

import yaml
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRunConfig,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.long_loop.stage_config import validate_stage_config


def configured_rl_step(
    *,
    version: str,
    model: ArtifactHfModel,
    train_data: tuple[ArtifactDataSource, ...],
    validation_data: tuple[ArtifactDataSource, ...],
    config: dict,
    plan: SkyRLRolePlan,
    execution: IrisSkyRLExecution,
    name: str,
    retained: int,
    seed: int,
    ray_timeouts: dict[str, int],
) -> ArtifactStep[SkyRLRun]:
    """Bind supplied training settings to admitted model/data artifacts and runtime paths."""
    validate_stage_config(config, "rl")
    buffer = config.get("trainer", {}).get("rollout_buffer", {})
    quotas = buffer.get("consumed_route_quotas", {})
    if quotas and set(buffer.get("source_routes", {}).values()) != set(quotas):
        raise ValueError("RL route admission is incomplete: supply verified source manifests and classifications")
    step = skyrl_step(
        SkyRLSpec(
            name=user_owned_name(f"checkpoints/{name}"),
            version=version,
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            model=model,
            train_data=train_data,
            validation_data=validation_data,
            topology=SkyRLTopology(
                num_nodes=plan.num_nodes,
                gpus_per_node=plan.policy_num_gpus_per_node,
                gpu_variant="H100",
                role_plan=plan,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=retained),
            seed=seed,
        ),
        execution,
        export_hf=True,
    )

    original_builder = step.build_config

    def bind_runtime(context: StepContext) -> SkyRLRunConfig:
        launch = original_builder(context)
        document = yaml.safe_load(launch.launch_config_yaml)
        document["run"]["finalization"] = "any_terminal"
        document["ray"].update(ray_timeouts)
        recipe = document["skyrl"]
        for callback in recipe["trainer"].get("callbacks", []):
            if callback["type"] == "best_checkpoint":
                callback["initial_model_uri"] = document["inputs"]["model"]["uri"]
        traces = prefix_join(document["artifacts"]["attempts_root"], "traces")
        retention = recipe["generator"].get("trajectory_retention")
        if retention is not None:
            retention["run_id"] = document["run"]["id"]
        installed = recipe.get("terminal_bench", {}).get("installed_agent")
        if installed is not None:
            installed["output_path"] = prefix_join(traces, "installed-agent")
        return replace(launch, launch_config_yaml=yaml.safe_dump(document, sort_keys=False))

    return replace(step, build_config=bind_runtime)
