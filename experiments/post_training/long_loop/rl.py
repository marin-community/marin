# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plumbing-smoke RL binding using the maintained task-compendium recipe."""

from dataclasses import replace

import yaml
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)

from experiments.post_training.async_rl import SNOWBALL_RECIPE
from experiments.post_training.inventory.train_quick_loop import training_recipe
from experiments.post_training.task_curation.inventory import TRAIN_FILENAME, VALIDATION_FILENAME

SMOKE_PLAN = replace(SNOWBALL_RECIPE.role_plan, train_batch_size=64, policy_mini_batch_size=64, n_samples_per_prompt=8)


def smoke_recipe() -> dict:
    """Retain the quick-loop task/runtime policy; change only GS8 and staleness."""
    config = training_recipe(3)
    config["trainer"]["rollout_buffer"]["max_staleness_steps"] = 2
    config["trainer"]["project_name"] = "marin-long-loop"
    _materialize_role_plan_config(config, SMOKE_PLAN)
    return config


def smoke_rl_step(
    *,
    version: str,
    model: ArtifactHfModel,
    selection: ArtifactStep[Artifact],
) -> ArtifactStep[SkyRLRun]:
    """Consume the preceding verified BF16 handoff, never a fixed released SFT model."""
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name("checkpoints/long-loop-plumbing-rl"),
            version=version,
            config_yaml=yaml.safe_dump(smoke_recipe(), sort_keys=False),
            model=model,
            train_data=(ArtifactDataSource(selection, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(selection, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=SMOKE_PLAN.num_nodes,
                gpus_per_node=SMOKE_PLAN.policy_num_gpus_per_node,
                gpu_variant="H100",
                role_plan=SMOKE_PLAN,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=20261020,
        ),
        IrisSkyRLExecution(
            cluster="cw-rno2a",
            cluster_config="lib/iris/config/cw-rno2a.yaml",
            cpu=32,
            memory=SNOWBALL_RECIPE.host_memory,
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster="cw-rno2a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=36,
            job_timeout_seconds=36 * 60 * 60,
            wandb_entity="marin-community",
            required_env_vars=("DAYTONA_EVAL_API_KEY", "TOGETHER_API_KEY"),
        ),
        export_hf=True,
    )
