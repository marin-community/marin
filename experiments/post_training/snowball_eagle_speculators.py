# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a reusable Snowball EAGLE-3 draft with offline Speculators artifacts.

Run on the RNO2A controller so the target checkpoint and generated hidden states
stay in the same object-store region::

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
      --enable-extra-resources --target-cluster cw-rno2a \
      -- python experiments/post_training/snowball_eagle_speculators.py \
      --version 2026.09.20 --stage draft --run
"""

from __future__ import annotations

from dataclasses import dataclass

import click
from fray.types import ResourceConfig
from marin.evaluation.evalchemy.result import FineStoreEvalchemyResult
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.external_dependencies import SPECULATORS
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    EagleDraftArtifact,
    IrisSkyRLExecution,
    SkyRLModel,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.speculators import (
    SPECULATORS_DATA_FILENAME,
    DraftTrainingConfig,
    HfSnapshotConfig,
    HiddenStateCaptureConfig,
    RolloutConversationConfig,
    VerifierViewConfig,
    build_verifier_view,
    capture_hidden_states,
    mirror_hf_snapshot,
    train_draft,
    write_rollout_conversations,
)
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.curriculum_rl.launch import (
    GPUS_PER_NODE,
    MARIN_TOKENIZER,
    MARIN_TOKENIZER_REVISION,
    POOL_ARTIFACT_NAME,
    SNOWBALL_MODEL,
)
from experiments.post_training.curriculum_rl.pool import VALIDATION_FILENAME, pool_step

VERSION = "2026.09.20"
ROLLOUT_ARCHIVE_URI = (
    "s3://marin-us-east-02a/marin/evaluation/evalchemy/snowball-67b-a2b-sft-s2-thinking/snowball-eagle-math/2026.09.20"
)
INITIAL_DRAFT_REPO = "laion/snowball-64k-eagle3-draft-r2egym"
INITIAL_DRAFT_REVISION = "4bdb47c08e5b5190bea3c7a93c3e14470230e469"
TARGET_LAYER_IDS = (2, 13, 23)
VERIFIER_NUM_HIDDEN_LAYERS = 26
SEQUENCE_LENGTH = 32768
RL_DATA_VERSION = "2026.09.18"
RL_ARTIFACT_NAME = "checkpoints/snowball-67b-a2b-eagle3-speculators-smoke"
CLUSTER = "cw-rno2a"
_DRAFT_GPU_COUNT = 8
# Speculators installs torchaudio through its multimodal dependencies. Pin the
# CUDA 12.8 wheel used by the Iris H100 PyTorch runtime so Transformers imports.
_TORCHAUDIO_CU128_REQUIREMENT = (
    "torchaudio @ https://download.pytorch.org/whl/cu128/"
    "torchaudio-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
    "#sha256=78b86a17f164bdaabdcee93fdfde2587fc43b9ebf15cd61dcf730b4f8615176b"
)

RL_SMOKE_ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=GPUS_PER_NODE,
    num_inference_engines=GPUS_PER_NODE,
    inference_engine_tensor_parallel_size=1,
    inference_engine_data_parallel_size=GPUS_PER_NODE,
    inference_engine_expert_parallel_size=GPUS_PER_NODE,
    train_batch_size=512,
    policy_mini_batch_size=64,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=16,
)

RL_SMOKE_CONFIG = f"""
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
  train_batch_size: {RL_SMOKE_ROLE_PLAN.train_batch_size}
  policy_mini_batch_size: {RL_SMOKE_ROLE_PLAN.policy_mini_batch_size}
  eval_batch_size: 512
  micro_forward_batch_size_per_gpu: 1
  micro_train_batch_size_per_gpu: {RL_SMOKE_ROLE_PLAN.micro_train_batch_size_per_gpu}
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
    policy_num_nodes: {RL_SMOKE_ROLE_PLAN.policy_num_nodes}
    policy_num_gpus_per_node: {RL_SMOKE_ROLE_PLAN.policy_num_gpus_per_node}

generator:
  backend: vllm
  model_dtype: bfloat16
  vllm_attention_backend: FLASH_ATTN
  inference_engine_tensor_parallel_size: {RL_SMOKE_ROLE_PLAN.inference_engine_tensor_parallel_size}
  inference_engine_pipeline_parallel_size: 1
  inference_engine_data_parallel_size: {RL_SMOKE_ROLE_PLAN.inference_engine_data_parallel_size}
  inference_engine_expert_parallel_size: {RL_SMOKE_ROLE_PLAN.inference_engine_expert_parallel_size}
  num_inference_engines: {RL_SMOKE_ROLE_PLAN.num_inference_engines}
  n_samples_per_prompt: {RL_SMOKE_ROLE_PLAN.n_samples_per_prompt}
  gpu_memory_utilization: 0.75
  max_num_seqs: 16
  max_num_batched_tokens: 16384
  enforce_eager: false
  vllm_v1_disable_multiproc: false
  run_engines_locally: true
  weight_sync_backend: nccl
  async_engine: true
  batched: false
  engine_init_kwargs:
    async_scheduling: false
    enable_mfu_metrics: true
  speculative_decoding:
    method: eagle3
    model: {{}}
    num_speculative_tokens: 3
    training: null
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


def _conversation_step(evaluation: ArtifactStep) -> ArtifactStep[Artifact]:
    name = "data/snowball-eagle-math-conversations"

    def build_config(ctx: StepContext) -> RolloutConversationConfig:
        return RolloutConversationConfig(
            source_archives=(ctx.artifact_path(evaluation),),
            output_path=ctx.output_path,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            write_rollout_conversations,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="16g"),
        ),
        build_config=build_config,
        deps=(evaluation,),
    )


def _initial_draft_step() -> ArtifactStep[Artifact]:
    name = "models/snowball-eagle3-initial-draft"

    def build_config(ctx: StepContext) -> HfSnapshotConfig:
        return HfSnapshotConfig(
            repo_id=INITIAL_DRAFT_REPO,
            revision=INITIAL_DRAFT_REVISION,
            output_path=ctx.output_path,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            mirror_hf_snapshot,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="64g"),
        ),
        build_config=build_config,
    )


def _verifier_step() -> ArtifactStep[Artifact]:
    name = "models/snowball-eagle3-verifier-view"

    def build_config(ctx: StepContext) -> VerifierViewConfig:
        return VerifierViewConfig(
            source_model=ctx.artifact_path(SNOWBALL_MODEL),
            transformers_model_type="llama",
            output_path=ctx.output_path,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            build_verifier_view,
            resources=ResourceConfig.with_cpu(cpu=8, ram="32g", disk="32g"),
        ),
        build_config=build_config,
        deps=(SNOWBALL_MODEL,),
    )


def _capture_step(dataset: ArtifactStep) -> ArtifactStep[Artifact]:
    name = "data/snowball-eagle3-hidden-states"

    def build_config(ctx: StepContext) -> HiddenStateCaptureConfig:
        return HiddenStateCaptureConfig(
            dataset_path=prefix_join(ctx.artifact_path(dataset), SPECULATORS_DATA_FILENAME),
            target_model=ctx.artifact_path(SNOWBALL_MODEL),
            processor_model=MARIN_TOKENIZER,
            output_path=ctx.output_path,
            target_layer_ids=TARGET_LAYER_IDS,
            verifier_num_hidden_layers=VERIFIER_NUM_HIDDEN_LAYERS,
            sequence_length=SEQUENCE_LENGTH,
            data_parallel_size=_DRAFT_GPU_COUNT,
            concurrency=64,
            max_samples=None,
            gpu_memory_utilization=0.9,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(
            capture_hidden_states,
            resources=ResourceConfig.with_gpu("H100", count=_DRAFT_GPU_COUNT, cpu=96, ram="512g", disk="1t"),
            pip_packages=[SPECULATORS.requirement(), _TORCHAUDIO_CU128_REQUIREMENT],
            max_retries_failure=2,
        ),
        build_config=build_config,
        deps=(dataset, SNOWBALL_MODEL),
    )


def _draft_step(
    captured_data: ArtifactStep,
    verifier: ArtifactStep,
    initial_draft: ArtifactStep,
) -> ArtifactStep[EagleDraftArtifact]:
    name = "models/snowball-eagle3-speculators"

    def build_config(ctx: StepContext) -> DraftTrainingConfig:
        return DraftTrainingConfig(
            captured_data_path=ctx.artifact_path(captured_data),
            verifier_path=ctx.artifact_path(verifier),
            initial_draft_path=ctx.artifact_path(initial_draft),
            output_path=ctx.output_path,
            target_layer_ids=TARGET_LAYER_IDS,
            sequence_length=SEQUENCE_LENGTH,
            epochs=4,
            learning_rate=1e-5,
            muon_learning_rate=0.02,
            num_processes=_DRAFT_GPU_COUNT,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=EagleDraftArtifact,
        run=remote(
            train_draft,
            resources=ResourceConfig.with_gpu("H100", count=_DRAFT_GPU_COUNT, cpu=96, ram="512g", disk="1t"),
            pip_packages=[SPECULATORS.requirement(), _TORCHAUDIO_CU128_REQUIREMENT],
        ),
        build_config=build_config,
        deps=(captured_data, verifier, initial_draft),
    )


@dataclass(frozen=True)
class SnowballDraftPipeline:
    conversations: ArtifactStep[Artifact]
    initial_draft: ArtifactStep[Artifact]
    verifier: ArtifactStep[Artifact]
    captured_data: ArtifactStep[Artifact]
    draft: ArtifactStep[EagleDraftArtifact]
    smoke: ArtifactStep[SkyRLModel]


def build_rl_smoke(draft: ArtifactStep[EagleDraftArtifact]) -> ArtifactStep[SkyRLModel]:
    """Run one production-shaped RLOO-N step with eight node-local rollout pools."""
    pool = pool_step(POOL_ARTIFACT_NAME, RL_DATA_VERSION)
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(RL_ARTIFACT_NAME),
            version=resolve_version(RL_ARTIFACT_NAME, None),
            config_yaml=RL_SMOKE_CONFIG,
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
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
                role_plan=RL_SMOKE_ROLE_PLAN,
            ),
            retention=SkyRLRetentionPolicy(),
            seed=17,
            draft_model=draft,
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


def build_pipeline() -> SnowballDraftPipeline:
    """Build the offline capture and draft-training artifact graph."""
    evaluation = ArtifactStep.adopt(
        name="data/snowball-eagle-math-rollouts",
        version=VERSION,
        source=ROLLOUT_ARCHIVE_URI,
        kind=FineStoreEvalchemyResult,
    )
    conversations = _conversation_step(evaluation)
    initial_draft = _initial_draft_step()
    verifier = _verifier_step()
    captured_data = _capture_step(conversations)
    draft = _draft_step(captured_data, verifier, initial_draft)
    smoke = build_rl_smoke(draft)
    return SnowballDraftPipeline(
        conversations=conversations,
        initial_draft=initial_draft,
        verifier=verifier,
        captured_data=captured_data,
        draft=draft,
        smoke=smoke,
    )


@click.command(help=__doc__)
@click.option(
    "--stage",
    type=click.Choice(("conversations", "initial_draft", "verifier", "captured_data", "draft", "smoke")),
    default="draft",
    show_default=True,
)
@build_options
def main(stage: str) -> ArtifactStep:
    return getattr(build_pipeline(), stage)


if __name__ == "__main__":
    main()
