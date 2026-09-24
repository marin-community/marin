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
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.post_training.curriculum_rl.pool import (
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    pool_step,
)

TARGET_MODEL_URI = "s3://marin-us-east-02a/marin/exports/grug/" "june-67b-a2b-sft-s3-agentic/step-1903/hf-bf16-vllm/"
TARGET_MODEL = ArtifactStep.adopt(
    "models/snowball-67b-a2b-sft-s3-agentic-step1903",
    "2026.09.21",
    TARGET_MODEL_URI,
    kind=LevanterCheckpoint,
)
TARGET_TOKENIZER = "marin-community/marin-tokenizer"
TARGET_TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
INITIAL_DRAFT_URI = "hf://laion/snowball-64k-eagle3-draft-r2egym"
INITIAL_DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"

ARTIFACT_NAME = "checkpoints/snowball-online-eagle"
POOL_ARTIFACT_NAME = "documents/curriculum-rl-pool"
CLUSTER = "cw-us-east-02a"
GPU_VARIANT = "H100"
GPUS_PER_NODE = 8
NUM_NODES = 6
SEED = 17

ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=GPUS_PER_NODE,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=GPUS_PER_NODE,
    inference_engine_expert_parallel_size=GPUS_PER_NODE,
    train_batch_size=32,
    policy_mini_batch_size=32,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=4,
)


def online_eagle_config_yaml() -> str:
    """Return the qualified Megatron and online-EAGLE recipe."""
    return f"""\
entrypoint: standard

context_budget:
  request_window_tokens: 9856
  max_new_tokens_per_turn: 8192
  max_turns: 1

environment:
  env_class: aime

trainer:
  strategy: megatron
  flash_attn: false
  use_sample_packing: false
  offload_optimizer_during_rollouts: true
  gradient_checkpointing: true
  algorithm:
    advantage_estimator: grpo
    use_kl_loss: false
  epochs: 1
  max_steps: 25
  update_epochs_per_batch: 1
  eval_batch_size: 64
  micro_forward_batch_size_per_gpu: 1
  eval_before_train: false
  eval_interval: -1
  ckpt_interval: 5
  resume_mode: latest
  logger: wandb
  project_name: marin-snowball-online-eagle
  policy:
    optimizer_config:
      lr: 1.0e-6
      max_grad_norm: 1.0
    megatron_config:
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 2
      context_parallel_size: 1
      expert_model_parallel_size: 8
      expert_tensor_parallel_size: 1

generator:
  backend: vllm
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  gpu_memory_utilization: 0.75
  enforce_eager: false
  run_engines_locally: true
  weight_sync_backend: nccl
  async_engine: true
  batched: false
  engine_init_kwargs:
    async_scheduling: false
  speculative_decoding:
    method: eagle3
    model:
      source_uri: {INITIAL_DRAFT_URI}
      source_identity: {INITIAL_DRAFT_REVISION}
    num_speculative_tokens: 3
    training:
      interval_steps: 4
      max_tokens_per_update: 131072
      max_window_tokens: 16384
      max_tokens_per_micro_batch: 8192
      max_sequences_per_prompt_group: 2
      min_train_sequences: 6
      holdout_fraction: 0.25
      min_holdout_sequences: 3
      epochs_per_update: 1
      learning_rate: 5.0e-5
      max_validation_loss_increase: 0.05
      max_validation_agreement_decrease: 0.01
      reserved_gpu_memory_gib: 12
  sampling_params:
    temperature: 1.0
    top_p: 1.0

data:
  kind: parquet
  train_data: []
  val_data: []

extra_env:
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
"""


def online_eagle_step() -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(ARTIFACT_NAME)
    version = resolve_version(name, None)
    pool = pool_step(POOL_ARTIFACT_NAME, resolve_version(POOL_ARTIFACT_NAME, None))
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=online_eagle_config_yaml(),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(
                step=TARGET_MODEL,
                tokenizer_uri=TARGET_TOKENIZER,
                tokenizer_revision=TARGET_TOKENIZER_REVISION,
                relative_path="",
            ),
            train_data=(ArtifactDataSource(pool, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(pool, relative_path=VALIDATION_FILENAME),),
            # Four policy nodes, one rollout node, and one independently derived
            # DraftTrainer node.
            topology=SkyRLTopology(
                num_nodes=NUM_NODES,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant=GPU_VARIANT,
                role_plan=ROLE_PLAN,
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
