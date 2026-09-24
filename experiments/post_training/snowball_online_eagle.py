# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and hot-refresh a Snowball EAGLE-3 draft during ordinary SkyRL rollouts.

This replaces the offline hidden-state pipeline with MarinSkyRL's managed online
trainer. One rollout node captures bounded proposal-aware windows, a dedicated
one-GPU DraftTrainer updates the draft asynchronously, and accepted checkpoints
refresh vLLM at the next safe boundary. Captures and draft checkpoints live
under the SkyRL checkpoint root; no separate hidden-state artifact is created.

Plan or run the experiment::

    python -m experiments.post_training.snowball_online_eagle
    python -m experiments.post_training.snowball_online_eagle --run
"""

from __future__ import annotations

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.drafting_sft import MegatronDraftPolicy, OnlineEagleTraining, sft_draft
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.curriculum_rl.pool import (
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    pool_step,
)

TARGET_MODEL_URI = StoragePath(
    "s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b-sft-s3-agentic/step-1903/hf-bf16-vllm/"
)
TARGET_TOKENIZER = "marin-community/marin-tokenizer"
TARGET_TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
INITIAL_DRAFT_URI = StoragePath("hf://laion/snowball-64k-eagle3-draft-r2egym")
INITIAL_DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"

ARTIFACT_NAME = "checkpoints/snowball-online-eagle"
POOL_ARTIFACT_NAME = "documents/curriculum-rl-pool"
CLUSTER = "cw-us-east-02a"
GPU_VARIANT = "H100"
GPUS_PER_NODE = 8
SEED = 17

DRAFT_SFT_PLAN = sft_draft(
    target_model=TARGET_MODEL_URI,
    initial_draft=INITIAL_DRAFT_URI,
    initial_draft_identity=INITIAL_DRAFT_REVISION,
    policy=MegatronDraftPolicy(
        policy_num_nodes=4,
        gpus_per_node=GPUS_PER_NODE,
        inference_engine_expert_parallel_size=GPUS_PER_NODE,
        train_batch_size=32,
        policy_mini_batch_size=32,
        micro_train_batch_size_per_gpu=1,
        n_samples_per_prompt=4,
        eval_batch_size=64,
        micro_forward_batch_size_per_gpu=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        learning_rate=1.0e-6,
    ),
    training=OnlineEagleTraining(
        num_speculative_tokens=3,
        interval_steps=4,
        max_tokens_per_update=131072,
        max_window_tokens=16384,
        max_tokens_per_micro_batch=8192,
        max_sequences_per_prompt_group=2,
        min_train_sequences=6,
        holdout_fraction=0.25,
        min_holdout_sequences=3,
        epochs_per_update=1,
        learning_rate=5.0e-5,
        max_validation_loss_increase=0.05,
        max_validation_agreement_decrease=0.01,
        reserved_gpu_memory_gib=12,
    ),
    environment="aime",
    project_name="marin-snowball-online-eagle",
    request_window_tokens=9856,
    max_new_tokens_per_turn=8192,
    max_steps=25,
    checkpoint_interval=5,
)

TARGET_MODEL = ArtifactStep.adopt(
    "models/snowball-67b-a2b-sft-s3-agentic-step1903",
    "2026.09.21",
    str(DRAFT_SFT_PLAN.target_model),
    kind=LevanterCheckpoint,
)


def online_eagle_step() -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(ARTIFACT_NAME)
    version = resolve_version(name, None)
    pool = pool_step(POOL_ARTIFACT_NAME, resolve_version(POOL_ARTIFACT_NAME, None))
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=DRAFT_SFT_PLAN.config_yaml,
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(
                step=TARGET_MODEL,
                tokenizer_uri=TARGET_TOKENIZER,
                tokenizer_revision=TARGET_TOKENIZER_REVISION,
                relative_path="",
            ),
            train_data=(ArtifactDataSource(pool, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(pool, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=DRAFT_SFT_PLAN.num_nodes,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=DRAFT_SFT_PLAN.role_plan,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=32,
            memory="512GB",
            disk="2TB",
            priority="interactive",
            max_retries=1,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=24,
            wandb_entity="marin-community",
        ),
    )


@click.command(help=__doc__)
@rl_build_options
def main() -> ArtifactStep[SkyRLRun]:
    return online_eagle_step()


if __name__ == "__main__":
    main()
