# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distill a reusable EAGLE-3 draft for the Snowball SFT checkpoint.

The source evaluation is a durable FineStore artifact. Draft versions replay its
cached responses through the target as bulk prefills, then train and validate the
EAGLE head from the ephemeral verifier features.
"""

from __future__ import annotations

import click
from marin.evaluation.evalchemy.result import FineStoreEvalchemyResult
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.rl.eagle import EagleRolloutCorpus, eagle_rollout_corpus_step
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactEagleDraft,
    ArtifactHfModel,
    EagleDraftDistillationSpec,
    EagleDraftModel,
    EagleDraftSource,
    IrisSkyRLExecution,
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    eagle_draft_distillation_step,
    skyrl_step,
)

from experiments.post_training.curriculum_rl.launch import (
    MARIN_TOKENIZER,
    MARIN_TOKENIZER_REVISION,
    POOL_ARTIFACT_NAME,
    SNOWBALL_MODEL,
)
from experiments.post_training.curriculum_rl.pool import VALIDATION_FILENAME, pool_step

ARTIFACT_NAME = "models/snowball-67b-a2b-eagle3-distilled"
RL_ARTIFACT_NAME = "checkpoints/snowball-67b-a2b-eagle3-frozen-draft-smoke"
ROLLOUT_VERSION = "2026.09.20"
ROLLOUT_ARCHIVE_URI = (
    "s3://marin-us-east-02a/marin/evaluation/evalchemy/"
    "snowball-67b-a2b-sft-s2-thinking/snowball-eagle-math/2026.09.20"
)
ROLLOUT_SOURCE_ARTIFACT_NAME = "data/snowball-67b-a2b-eagle3-math-rollouts"
CORPUS_ARTIFACT_NAME = "data/snowball-67b-a2b-eagle3-math-corpus"
CORPUS_VERSION = "2026.09.20"
INITIAL_DRAFT_URI = "hf://laion/snowball-64k-eagle3-draft-r2egym"
INITIAL_DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
CLUSTER = "cw-rno2a"
GPUS_PER_NODE = 8
MARINSKYRL_COMMIT = "18c335dba3ecf2fd241dc3851042c9bd8983374d"
RL_DATA_VERSION = "2026.09.18"

DISTILLATION_CONFIG = """
entrypoint: generate

context_budget:
  request_window_tokens: 32768
  max_new_tokens_per_turn: 1
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
  eval_batch_size: 4096
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
  # Keep expert collectives within one eight-GPU node while retaining eight
  # node-local request schedulers across the eight pools.
  inference_engine_data_parallel_size: 8
  inference_engine_expert_parallel_size: 8
  inference_engine_node_local: true
  num_inference_engines: 8
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
  # The launcher's record-file route stages both Parquet and JSONL inputs.
  kind: parquet
  train_data: []
  val_data: []
  shuffle: false
  eagle_replay: true
  eagle_replay_batch_size: 4096

extra_env:
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
"""

RL_SMOKE_CONFIG = """
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
    advantage_estimator: rloo_n
    group_advantage_min_size: 4
    use_kl_loss: false
  epochs: 1
  max_steps: 1
  update_epochs_per_batch: 1
  train_batch_size: 512
  policy_mini_batch_size: 64
  eval_batch_size: 512
  micro_forward_batch_size_per_gpu: 1
  micro_train_batch_size_per_gpu: 1
  eval_before_train: false
  eval_interval: -1
  ckpt_interval: 100
  resume_mode: none
  logger: console
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
      optimizer_checkpoint_sharding_type: dp_reshardable
      ddp_config:
        overlap_grad_reduce: true
        overlap_param_gather: true
        grad_reduce_in_fp32: false
  placement:
    colocate_all: false
    policy_strict_spread_pg: true
    policy_num_nodes: 4
    policy_num_gpus_per_node: 8

generator:
  backend: vllm
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  inference_engine_tensor_parallel_size: 1
  inference_engine_pipeline_parallel_size: 1
  inference_engine_data_parallel_size: 8
  inference_engine_expert_parallel_size: 8
  inference_engine_node_local: true
  # Eight engines are eight node-local pools, not eight GPU workers.
  num_inference_engines: 8
  n_samples_per_prompt: 16
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
    enable_mfu_metrics: true
  speculative_decoding:
    method: eagle3
    model: {}
    num_speculative_tokens: 3
  sampling_params:
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


def build_rollout_corpus() -> ArtifactStep[EagleRolloutCorpus]:
    """Extract replay JSONL from the normalized, immutable evaluation archive."""
    evaluation = ArtifactStep.adopt(
        name=user_owned_name(ROLLOUT_SOURCE_ARTIFACT_NAME),
        version=ROLLOUT_VERSION,
        source=ROLLOUT_ARCHIVE_URI,
        kind=FineStoreEvalchemyResult,
    )
    return eagle_rollout_corpus_step(
        name=user_owned_name(CORPUS_ARTIFACT_NAME),
        version=CORPUS_VERSION,
        evaluations=(evaluation,),
    )


def build_distillation(
    version: str | None = None,
    *,
    corpus: ArtifactStep[EagleRolloutCorpus] | None = None,
) -> ArtifactStep[EagleDraftModel]:
    """Build the Snowball EAGLE distillation artifact."""
    resolved_version = version or resolve_version(ARTIFACT_NAME, None)
    corpus = corpus or build_rollout_corpus()
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
            data=(ArtifactDataSource(corpus, relative_path="corpus.jsonl"),),
            topology=SkyRLTopology(
                num_nodes=8,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant="H100",
                role_plan=SkyRLRolePlan(
                    colocate_all=False,
                    policy_num_nodes=1,
                    policy_num_gpus_per_node=GPUS_PER_NODE,
                    num_inference_engines=8,
                    inference_engine_tensor_parallel_size=1,
                    inference_engine_pipeline_parallel_size=1,
                    inference_engine_data_parallel_size=8,
                    inference_engine_expert_parallel_size=8,
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
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=16,
            memory="512GB",
            disk="2TB",
            priority="interactive",
            max_retries=1,
            wandb_entity="marin-community",
        ),
    )


def build_rl_smoke(
    draft: ArtifactStep[EagleDraftModel],
    version: str | None = None,
) -> ArtifactStep[SkyRLModel]:
    """Run one production-shaped RLOO-N step with the frozen distilled draft."""
    resolved_version = version or resolve_version(RL_ARTIFACT_NAME, None)
    pool = pool_step(POOL_ARTIFACT_NAME, RL_DATA_VERSION)
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(RL_ARTIFACT_NAME),
            version=resolved_version,
            config_yaml=RL_SMOKE_CONFIG,
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON, commit=MARINSKYRL_COMMIT),
            model=ArtifactHfModel(
                step=SNOWBALL_MODEL,
                tokenizer_uri=MARIN_TOKENIZER,
                tokenizer_revision=MARIN_TOKENIZER_REVISION,
            ),
            train_data=(ArtifactDataSource(pool, relative_path=VALIDATION_FILENAME),),
            validation_data=(),
            topology=SkyRLTopology(
                num_nodes=12,
                gpus_per_node=GPUS_PER_NODE,
                gpu_variant="H100",
                role_plan=SkyRLRolePlan(
                    colocate_all=False,
                    policy_num_nodes=4,
                    policy_num_gpus_per_node=GPUS_PER_NODE,
                    num_inference_engines=8,
                    inference_engine_tensor_parallel_size=1,
                    inference_engine_pipeline_parallel_size=1,
                    inference_engine_data_parallel_size=8,
                    inference_engine_expert_parallel_size=8,
                    train_batch_size=512,
                    policy_mini_batch_size=64,
                    micro_train_batch_size_per_gpu=1,
                    n_samples_per_prompt=16,
                ),
            ),
            retention=SkyRLRetentionPolicy(),
            seed=17,
            draft_model=ArtifactEagleDraft(draft),
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
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
    corpus = build_rollout_corpus()
    draft = build_distillation(corpus=corpus)
    smoke = build_rl_smoke(draft)
    return {"corpus": corpus, "draft": draft, "smoke": smoke}


if __name__ == "__main__":
    main()
