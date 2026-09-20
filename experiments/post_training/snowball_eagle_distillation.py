# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distill a reusable EAGLE-3 draft for the Snowball SFT checkpoint.

The target model captures verifier hidden states while generating long-form
math responses on 64 H100s. The rollout engines are then torn down and one of
those GPUs performs a bounded four-epoch draft update.
"""

from __future__ import annotations

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    EagleDraftDistillationSpec,
    EagleDraftModel,
    EagleDraftSource,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLTopology,
    eagle_draft_distillation_step,
)
from marin.training.training import LevanterCheckpoint

from experiments.evaluation.models import SNOWBALL_SFT_EXPORT_URI
from experiments.post_training.curriculum_rl.pool import VALIDATION_FILENAME, pool_step

ARTIFACT_NAME = "models/snowball-67b-a2b-eagle3-distilled"
POOL_ARTIFACT_NAME = "documents/curriculum-rl-pool"
POOL_ARTIFACT_VERSION = "2026.09.18"
INITIAL_DRAFT_URI = "hf://laion/snowball-64k-eagle3-draft-r2egym"
INITIAL_DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
MARIN_TOKENIZER = "marin-community/marin-tokenizer"
MARIN_TOKENIZER_REVISION = "a5ca45f"
GPUS_PER_NODE = 8
MARINSKYRL_COMMIT = "98b48499118f427c0efc54e80c4e99639adca616"

SNOWBALL_MODEL = ArtifactStep.adopt(
    "models/snowball-67b-a2b-sft-s2-thinking",
    "2026.08.30",
    SNOWBALL_SFT_EXPORT_URI,
    kind=LevanterCheckpoint,
)

DISTILLATION_CONFIG = """
entrypoint: generate

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
  algorithm:
    advantage_estimator: grpo
    use_kl_loss: false
  train_batch_size: 64
  policy_mini_batch_size: 64
  eval_batch_size: 64
  micro_forward_batch_size_per_gpu: 1
  micro_train_batch_size_per_gpu: 1
  logger: console
  project_name: snowball-eagle3-offline-distillation
  placement:
    colocate_all: false
    policy_num_nodes: 1
    policy_num_gpus_per_node: 8

generator:
  backend: vllm
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  inference_engine_tensor_parallel_size: 1
  inference_engine_pipeline_parallel_size: 1
  inference_engine_data_parallel_size: 64
  inference_engine_expert_parallel_size: 64
  num_inference_engines: 1
  n_samples_per_prompt: 16
  eval_n_samples_per_prompt: 4
  gpu_memory_utilization: 0.75
  max_num_seqs: 16
  max_num_batched_tokens: 16384
  enforce_eager: false
  run_engines_locally: true
  weight_sync_backend: nccl
  async_engine: true
  batched: false
  engine_init_kwargs:
    async_scheduling: false
  speculative_decoding:
    method: eagle3
    model: {}
    num_speculative_tokens: 3
    training:
      interval_steps: 1
      max_tokens_per_update: 1000000
      max_window_tokens: 16384
      max_tokens_per_micro_batch: 8192
      max_sequences_per_prompt_group: 2
      min_train_sequences: 32
      holdout_fraction: 0.1
      min_holdout_sequences: 16
      epochs_per_update: 4
      optimizer: hybrid_muon
      learning_rate: 1.0e-5
      muon_learning_rate: 0.02
      max_validation_loss_increase: 0.05
      max_validation_agreement_decrease: 0.01
      reserved_gpu_memory_gib: 12
  sampling_params:
    temperature: 1.0
    top_p: 1.0
  eval_sampling_params:
    temperature: 1.0
    top_p: 1.0

data:
  kind: parquet
  train_data: []
  val_data: []
  shuffle: false

extra_env:
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
"""


def build_distillation(version: str | None = None) -> ArtifactStep[EagleDraftModel]:
    """Build the Snowball EAGLE distillation artifact."""
    resolved_version = version or resolve_version(ARTIFACT_NAME, None)
    pool = pool_step(POOL_ARTIFACT_NAME, POOL_ARTIFACT_VERSION)
    return eagle_draft_distillation_step(
        EagleDraftDistillationSpec(
            name=user_owned_name(ARTIFACT_NAME),
            version=resolved_version,
            config_yaml=DISTILLATION_CONFIG,
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON, commit=MARINSKYRL_COMMIT),
            target_model=ArtifactHfModel(
                step=SNOWBALL_MODEL,
                tokenizer_uri=MARIN_TOKENIZER,
                tokenizer_revision=MARIN_TOKENIZER_REVISION,
            ),
            initial_draft=EagleDraftSource(uri=INITIAL_DRAFT_URI, identity=INITIAL_DRAFT_REVISION),
            data=(ArtifactDataSource(pool, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                num_nodes=8,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant="H100",
                role_plan=SkyRLRolePlan(
                    colocate_all=False,
                    policy_num_nodes=1,
                    policy_num_gpus_per_node=GPUS_PER_NODE,
                    num_inference_engines=1,
                    inference_engine_tensor_parallel_size=1,
                    inference_engine_pipeline_parallel_size=1,
                    inference_engine_data_parallel_size=64,
                    inference_engine_expert_parallel_size=64,
                    train_batch_size=64,
                    policy_mini_batch_size=64,
                    micro_train_batch_size_per_gpu=1,
                    n_samples_per_prompt=16,
                    evaluation_only=True,
                ),
            ),
            retention=SkyRLRetentionPolicy(),
            seed=17,
        ),
        IrisSkyRLExecution(
            cluster="cw-us-east-02a",
            cluster_config="lib/iris/config/cw-us-east-02a.yaml",
            cpu=16,
            memory="512GB",
            disk="2TB",
            priority="interactive",
            max_retries=1,
            wandb_entity="marin-community",
        ),
    )


@click.command(help=__doc__)
@build_options
def main() -> dict[str, ArtifactStep]:
    return {"draft": build_distillation()}


if __name__ == "__main__":
    main()
