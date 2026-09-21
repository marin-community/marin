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
import yaml
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


def _rl_smoke_role_plan() -> SkyRLRolePlan:
    return SkyRLRolePlan(
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


def _rl_smoke_config(role_plan: SkyRLRolePlan) -> str:
    config = {
        "entrypoint": "standard",
        "context_budget": {
            "request_window_tokens": 9856,
            "max_new_tokens_per_turn": 8192,
            "max_turns": 1,
        },
        "environment": {"env_class": "aime"},
        "trainer": {
            "strategy": "megatron",
            "flash_attn": False,
            "use_sample_packing": False,
            "offload_optimizer_during_rollouts": True,
            "gradient_checkpointing": True,
            "algorithm": {
                "advantage_estimator": "rloo_n",
                "group_advantage_min_size": 4,
                "use_kl_loss": False,
            },
            "epochs": 1,
            "max_steps": 1,
            "update_epochs_per_batch": 1,
            "train_batch_size": role_plan.train_batch_size,
            "policy_mini_batch_size": role_plan.policy_mini_batch_size,
            "eval_batch_size": 512,
            "micro_forward_batch_size_per_gpu": 1,
            "micro_train_batch_size_per_gpu": role_plan.micro_train_batch_size_per_gpu,
            "eval_before_train": False,
            "eval_interval": -1,
            "ckpt_interval": 100,
            "resume_mode": "none",
            "logger": "console",
            "policy": {
                "optimizer_config": {"lr": 1.0e-6, "max_grad_norm": 1.0},
                "megatron_config": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 2,
                    "context_parallel_size": 1,
                    "expert_model_parallel_size": 8,
                    "expert_tensor_parallel_size": 1,
                    "optimizer_checkpoint_sharding_type": "dp_reshardable",
                    "ddp_config": {
                        "overlap_grad_reduce": True,
                        "overlap_param_gather": True,
                        "grad_reduce_in_fp32": False,
                    },
                },
            },
            "placement": {
                "colocate_all": False,
                "policy_strict_spread_pg": True,
                "policy_num_nodes": role_plan.policy_num_nodes,
                "policy_num_gpus_per_node": role_plan.policy_num_gpus_per_node,
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "inference_engine_tensor_parallel_size": role_plan.inference_engine_tensor_parallel_size,
            "inference_engine_pipeline_parallel_size": 1,
            "inference_engine_data_parallel_size": role_plan.inference_engine_data_parallel_size,
            "inference_engine_expert_parallel_size": role_plan.inference_engine_expert_parallel_size,
            "num_inference_engines": role_plan.num_inference_engines,
            "n_samples_per_prompt": role_plan.n_samples_per_prompt,
            "gpu_memory_utilization": 0.75,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 16384,
            "enforce_eager": False,
            "vllm_v1_disable_multiproc": False,
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "async_engine": True,
            "batched": False,
            "engine_init_kwargs": {"async_scheduling": False, "enable_mfu_metrics": True},
            "speculative_decoding": {
                "method": "eagle3",
                "model": {},
                "num_speculative_tokens": 3,
                "training": None,
            },
            "sampling_params": {"temperature": 1.0, "top_p": 1.0},
        },
        "data": {"kind": "parquet", "train_data": [], "val_data": [], "shuffle": False},
        "extra_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    }
    return yaml.safe_dump(config, sort_keys=False)


@dataclass(frozen=True)
class DraftSftArtifactNames:
    """Artifact names for each reusable draft-SFT stage."""

    conversations: str
    initial_draft: str
    verifier: str
    captured_data: str
    draft: str


@dataclass(frozen=True)
class DraftSftExecutionConfig:
    """Model-specific hidden-state capture and training settings."""

    processor_model: str
    transformers_model_type: str
    target_layer_ids: tuple[int, ...]
    verifier_num_hidden_layers: int
    sequence_length: int
    gpu_count: int


def _conversation_step(*, name: str, rollouts: ArtifactStep) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> RolloutConversationConfig:
        return RolloutConversationConfig(
            source_archives=(ctx.artifact_path(rollouts),),
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
        deps=(rollouts,),
    )


def _initial_draft_step(*, name: str, repo_id: str, revision: str) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> HfSnapshotConfig:
        return HfSnapshotConfig(
            repo_id=repo_id,
            revision=revision,
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


def _verifier_step(
    *,
    name: str,
    target_model: ArtifactStep,
    transformers_model_type: str,
) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> VerifierViewConfig:
        return VerifierViewConfig(
            source_model=ctx.artifact_path(target_model),
            transformers_model_type=transformers_model_type,
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
        deps=(target_model,),
    )


def _capture_step(
    *,
    name: str,
    dataset: ArtifactStep,
    target_model: ArtifactStep,
    execution: DraftSftExecutionConfig,
) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> HiddenStateCaptureConfig:
        return HiddenStateCaptureConfig(
            dataset_path=prefix_join(ctx.artifact_path(dataset), SPECULATORS_DATA_FILENAME),
            target_model=ctx.artifact_path(target_model),
            processor_model=execution.processor_model,
            output_path=ctx.output_path,
            target_layer_ids=execution.target_layer_ids,
            verifier_num_hidden_layers=execution.verifier_num_hidden_layers,
            sequence_length=execution.sequence_length,
            data_parallel_size=execution.gpu_count,
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
            resources=ResourceConfig.with_gpu("H100", count=execution.gpu_count, cpu=96, ram="512g", disk="1t"),
            pip_packages=[SPECULATORS.requirement(), _TORCHAUDIO_CU128_REQUIREMENT],
            max_retries_failure=2,
        ),
        build_config=build_config,
        deps=(dataset, target_model),
    )


def _draft_step(
    *,
    name: str,
    captured_data: ArtifactStep,
    verifier: ArtifactStep,
    initial_draft: ArtifactStep,
    execution: DraftSftExecutionConfig,
) -> ArtifactStep[EagleDraftArtifact]:
    def build_config(ctx: StepContext) -> DraftTrainingConfig:
        return DraftTrainingConfig(
            captured_data_path=ctx.artifact_path(captured_data),
            verifier_path=ctx.artifact_path(verifier),
            initial_draft_path=ctx.artifact_path(initial_draft),
            output_path=ctx.output_path,
            target_layer_ids=execution.target_layer_ids,
            sequence_length=execution.sequence_length,
            epochs=4,
            learning_rate=1e-5,
            muon_learning_rate=0.02,
            num_processes=execution.gpu_count,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=EagleDraftArtifact,
        run=remote(
            train_draft,
            resources=ResourceConfig.with_gpu("H100", count=execution.gpu_count, cpu=96, ram="512g", disk="1t"),
            pip_packages=[SPECULATORS.requirement(), _TORCHAUDIO_CU128_REQUIREMENT],
        ),
        build_config=build_config,
        deps=(captured_data, verifier, initial_draft),
    )


@dataclass(frozen=True)
class DraftSftPipeline:
    """Artifact handles produced by a draft-SFT pipeline."""

    conversations: ArtifactStep[Artifact]
    initial_draft: ArtifactStep[Artifact]
    verifier: ArtifactStep[Artifact]
    captured_data: ArtifactStep[Artifact]
    draft: ArtifactStep[EagleDraftArtifact]


def sft_draft_model(
    *,
    names: DraftSftArtifactNames,
    rollouts: ArtifactStep,
    target_model: ArtifactStep,
    initial_draft_repo: str,
    initial_draft_revision: str,
    execution: DraftSftExecutionConfig,
) -> DraftSftPipeline:
    """Build reusable Speculators SFT artifacts for a target model."""
    conversations = _conversation_step(name=names.conversations, rollouts=rollouts)
    initial_draft = _initial_draft_step(
        name=names.initial_draft,
        repo_id=initial_draft_repo,
        revision=initial_draft_revision,
    )
    verifier = _verifier_step(
        name=names.verifier,
        target_model=target_model,
        transformers_model_type=execution.transformers_model_type,
    )
    captured_data = _capture_step(
        name=names.captured_data,
        dataset=conversations,
        target_model=target_model,
        execution=execution,
    )
    draft = _draft_step(
        name=names.draft,
        captured_data=captured_data,
        verifier=verifier,
        initial_draft=initial_draft,
        execution=execution,
    )
    return DraftSftPipeline(
        conversations=conversations,
        initial_draft=initial_draft,
        verifier=verifier,
        captured_data=captured_data,
        draft=draft,
    )


def snowball_eagle_sft() -> DraftSftPipeline:
    """Build the Snowball-specific EAGLE-3 SFT pipeline."""
    rollouts = ArtifactStep.adopt(
        name="data/snowball-eagle-math-rollouts",
        version=VERSION,
        source=ROLLOUT_ARCHIVE_URI,
        kind=FineStoreEvalchemyResult,
    )
    return sft_draft_model(
        names=DraftSftArtifactNames(
            conversations="data/snowball-eagle-math-conversations",
            initial_draft="models/snowball-eagle3-initial-draft",
            verifier="models/snowball-eagle3-verifier-view",
            captured_data="data/snowball-eagle3-hidden-states",
            draft="models/snowball-eagle3-speculators",
        ),
        rollouts=rollouts,
        target_model=SNOWBALL_MODEL,
        initial_draft_repo=INITIAL_DRAFT_REPO,
        initial_draft_revision=INITIAL_DRAFT_REVISION,
        execution=DraftSftExecutionConfig(
            processor_model=MARIN_TOKENIZER,
            transformers_model_type="llama",
            target_layer_ids=TARGET_LAYER_IDS,
            verifier_num_hidden_layers=VERIFIER_NUM_HIDDEN_LAYERS,
            sequence_length=SEQUENCE_LENGTH,
            gpu_count=_DRAFT_GPU_COUNT,
        ),
    )


@dataclass(frozen=True)
class SnowballDraftPipeline:
    """Snowball draft SFT stages and its RL smoke run."""

    conversations: ArtifactStep[Artifact]
    initial_draft: ArtifactStep[Artifact]
    verifier: ArtifactStep[Artifact]
    captured_data: ArtifactStep[Artifact]
    draft: ArtifactStep[EagleDraftArtifact]
    smoke: ArtifactStep[SkyRLModel]


def build_rl_smoke(draft: ArtifactStep[EagleDraftArtifact]) -> ArtifactStep[SkyRLModel]:
    """Run one production-shaped RLOO-N step with eight node-local rollout pools."""
    pool = pool_step(POOL_ARTIFACT_NAME, RL_DATA_VERSION)
    role_plan = _rl_smoke_role_plan()
    return skyrl_step(
        SkyRLSpec(
            name=user_owned_name(RL_ARTIFACT_NAME),
            version=resolve_version(RL_ARTIFACT_NAME, None),
            config_yaml=_rl_smoke_config(role_plan),
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
                role_plan=role_plan,
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
    sft = snowball_eagle_sft()
    smoke = build_rl_smoke(sft.draft)
    return SnowballDraftPipeline(
        conversations=sft.conversations,
        initial_draft=sft.initial_draft,
        verifier=sft.verifier,
        captured_data=sft.captured_data,
        draft=sft.draft,
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
