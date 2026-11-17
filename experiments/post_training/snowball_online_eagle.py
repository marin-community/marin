# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and hot-refresh a Snowball EAGLE-3 draft during ordinary SkyRL rollouts.

This replaces the offline hidden-state pipeline with MarinSkyRL's managed online
trainer. One rollout node captures bounded proposal-aware windows, a dedicated
one-GPU DraftTrainer updates the draft asynchronously, and accepted checkpoints
refresh vLLM at the next safe boundary. Captures and draft checkpoints live
under the SkyRL checkpoint root; no separate hidden-state artifact is created.

Plan or run the experiment::

    python -m experiments.post_training.snowball_online_eagle --scale smoke
    python -m experiments.post_training.snowball_online_eagle --scale full --run
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    MegatronGeometry,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from skyrlrecipe import ContextBudget, RLEntrypoint, SkyRLRecipe, sections

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


@dataclass(frozen=True)
class _OnlineEaglePreset:
    label: str
    artifact_suffix: str
    max_steps: int
    checkpoint_interval: int


# Capture at step 4, then leave three rollout steps for training and refresh without starting step 8.
SMOKE = _OnlineEaglePreset(label="smoke", artifact_suffix="-smoke", max_steps=7, checkpoint_interval=0)
FULL = _OnlineEaglePreset(label="full", artifact_suffix="", max_steps=25, checkpoint_interval=5)
PRESETS = {preset.label: preset for preset in (SMOKE, FULL)}

GEOMETRY = MegatronGeometry(1, 2, 1, 8, 1)
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
TRAINING = sections.SpeculatorTraining(
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
)


def training_recipe(preset: _OnlineEaglePreset) -> SkyRLRecipe:
    return SkyRLRecipe(
        entrypoint=RLEntrypoint.STANDARD,
        context_budget=ContextBudget(request_window_tokens=9856, max_new_tokens_per_turn=8192, max_turns=1),
        environment=sections.Environment(env_class="aime"),
        trainer=sections.Trainer(
            strategy="megatron",
            flash_attn=False,
            use_sample_packing=False,
            offload_optimizer_during_rollouts=True,
            gradient_checkpointing=True,
            algorithm=sections.Algorithm(advantage_estimator="grpo", use_kl_loss=False),
            epochs=1,
            max_steps=preset.max_steps,
            update_epochs_per_batch=1,
            eval_batch_size=64,
            micro_forward_batch_size_per_gpu=1,
            eval_before_train=False,
            eval_interval=-1,
            ckpt_interval=preset.checkpoint_interval,
            resume_mode="latest",
            logger="wandb",
            project_name="marin-snowball-online-eagle",
            policy=sections.Policy(
                optimizer_config=sections.PolicyOptimizerConfig(lr=1.0e-6, max_grad_norm=1.0),
                megatron_config=sections.PolicyMegatronConfig(**asdict(GEOMETRY)),
            ),
        ),
        generator=sections.Generator(
            backend="vllm",
            model_dtype="bfloat16",
            vllm_attention_backend="FLASH_ATTN",
            gpu_memory_utilization=0.75,
            enforce_eager=False,
            run_engines_locally=True,
            weight_sync_backend="nccl",
            engine_init_kwargs={"async_scheduling": False},
            speculative_decoding=sections.SpeculativeDecoding(
                method="eagle3",
                model=sections.SpeculatorModel(
                    source_uri=str(INITIAL_DRAFT_URI), source_identity=INITIAL_DRAFT_REVISION
                ),
                num_speculative_tokens=3,
                training=TRAINING,
            ),
            sampling_params=sections.SamplingParams(temperature=1.0, top_p=1.0),
        ),
        data=sections.Data(kind="parquet", train_data=(), val_data=()),
        extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


TARGET_MODEL = ArtifactStep.adopt(
    "models/snowball-67b-a2b-sft-s3-agentic-step1903",
    "2026.09.21",
    str(TARGET_MODEL_URI),
    kind=LevanterCheckpoint,
)


def online_eagle_step(preset: _OnlineEaglePreset) -> ArtifactStep[SkyRLRun]:
    name = user_owned_name(f"{ARTIFACT_NAME}{preset.artifact_suffix}")
    version = resolve_version(name, None)
    pool = pool_step(POOL_ARTIFACT_NAME, resolve_version(POOL_ARTIFACT_NAME, None))
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(training_recipe(preset).to_skyrl(), sort_keys=False),
            model=ArtifactHfModel(
                step=TARGET_MODEL,
                tokenizer_uri=TARGET_TOKENIZER,
                tokenizer_revision=TARGET_TOKENIZER_REVISION,
                relative_path="",
            ),
            train_data=(ArtifactDataSource(pool, relative_path=TRAIN_FILENAME),),
            validation_data=(ArtifactDataSource(pool, relative_path=VALIDATION_FILENAME),),
            topology=SkyRLTopology(
                # One rollout node and one DraftTrainer node accompany the policy nodes.
                num_nodes=ROLE_PLAN.policy_num_nodes + 2,
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
@click.option("--scale", type=click.Choice(sorted(PRESETS)), required=True)
@rl_build_options
def main(scale: str) -> ArtifactStep[SkyRLRun]:
    return online_eagle_step(PRESETS[scale])


if __name__ == "__main__":
    main()
