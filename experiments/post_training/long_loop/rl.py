# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment-owned asynchronous GS8 RL recipe and artifact binding."""

from pathlib import Path

import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_step,
)

from experiments.post_training.long_loop.archive_selection import trace_retention
from experiments.post_training.task_curation.inventory import TRAIN_FILENAME, VALIDATION_FILENAME

SELECTION_PREFIX = "rl-data/task-selection"
DEFAULT_SELECTION_URI = "s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.11.04"
SMOKE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=64,
    policy_mini_batch_size=64,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=8,
)


def smoke_recipe() -> dict:
    """Load the local recipe; final evaluation belongs to the separate Eval-org stage."""
    config = yaml.safe_load(Path(__file__).with_name("rl-smoke.yaml").read_text())
    _materialize_role_plan_config(config, SMOKE_PLAN)
    return config


def smoke_rl_step(
    *,
    version: str,
    model: ArtifactHfModel,
    selection: ArtifactStep[Artifact],
) -> ArtifactStep[SkyRLRun]:
    """Consume the preceding verified BF16 handoff, never a fixed released SFT model."""
    config = smoke_recipe()
    config["trainer"]["dump_training_order"] = True
    config["generator"]["trajectory_retention"] = trace_retention(
        f"s3://marin-us-east-02a/marin/long-loop/traces/{version}/rl", version, fraction=0.1, phase="train"
    )
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name("checkpoints/long-loop-plumbing-rl"),
            version=version,
            config_yaml=yaml.safe_dump(config, sort_keys=False),
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
            memory="1800GB",
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster="cw-rno2a",
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=36,
            coordinator_name=f"long-loop-plumbing-{version}",
            job_timeout_seconds=36 * 60 * 60,
            wandb_entity="marin-community",
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
            required_env_vars=(
                "DAYTONA_RL_API_KEY",
                "DAYTONA_EVAL_API_KEY",
                "DAYTONA_TARGET",
                "TOGETHER_API_KEY",
                "MARIN_PREFIX",
            ),
        ),
        export_hf=True,
    )
