# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a native Grug checkpoint on conversations generated from one capability."""

import dataclasses
from dataclasses import dataclass

from fray.types import ResourceConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.optim.config import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from marin.datakit.chat_render import render_chat_to_parquet
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugMoeSFTConfig, run_grug_moe_sft_trial
from experiments.june_tpu_67b_a2b.moe.train import GrugTrainerConfig
from experiments.post_training.curriculum_sft.generation import generate_curriculum_sft

SEPTEMBER_GRUG_CHECKPOINT = ArtifactStep.adopt(
    name="models/grug-67b-sft-20260920-native",
    version="2026.09.20",
    source=(
        "gs://marin-us-central2/users/held/grug_sft/"
        "grug-67b-sft-20260920-special-token-lr-frozen-bias-1000-cpfix/checkpoints/step-158000"
    ),
    kind=LevanterCheckpoint,
)
SEPTEMBER_GRUG_TOKENIZER = "gs://marin-us-central2/grug_sft/tokenizer/2026.09.18"


@dataclass(frozen=True)
class RenderConfig:
    input_path: str
    output_path: str


def render_generated_chat(config: RenderConfig) -> Artifact:
    """Render Datakit Harmony rows with the Marin chat template."""
    render_chat_to_parquet(input_path=prefix_join(config.input_path, "chat"), output_path=config.output_path)
    return Artifact(path=config.output_path)


def packed_grug_data(*, rendered_path: str, cache_path: str, tokenizer: str, context_length: int) -> LmDataConfig:
    """Pack rendered conversations without attention across conversation boundaries."""
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        components={
            "curriculum": DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[prefix_join(rendered_path, "*.parquet")]),
                cache_dir=cache_path,
                format=TextLmDatasetFormat(),
                pack=context_length,
                split="train",
            )
        },
        train_weights={"curriculum": 1.0},
        auto_build_caches=True,
        shuffle=True,
        block_cross_document_attention=True,
    )


def curriculum_grug_sft(
    capability_id: str,
    *,
    version: str,
    requested_examples: int,
    accepted_examples: int,
    seed: int,
    max_completion_tokens: int,
    checkpoint: ArtifactStep[LevanterCheckpoint],
    tokenizer: str,
    optimizer: OptimizerConfig,
    resources: ResourceConfig,
    context_length: int,
    batch_size: int,
    steps: int,
    expert_parallel: int,
) -> ArtifactStep[LevanterCheckpoint]:
    """Generate, render, and pack curriculum conversations for native Grug SFT.

    The checkpoint and tokenizer must belong to the same Grug architecture. A
    native checkpoint is required; an HF export cannot initialize this trainer.
    """
    generated = generate_curriculum_sft(
        capability_id,
        version=version,
        requested_examples=requested_examples,
        accepted_examples=accepted_examples,
        seed=seed,
        max_completion_tokens=max_completion_tokens,
    )
    rendered = ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/{capability_id}/rendered-chat"),
        version=version,
        artifact_type=Artifact,
        run=render_generated_chat,
        build_config=lambda ctx: RenderConfig(ctx.artifact_path(generated), ctx.output_path),
        deps=(generated,),
    )

    model = dataclasses.replace(
        MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=context_length),
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=2048,
        use_array_stacked_blocks=True,
        qk_mult=1.75,
        max_seq_len=context_length,
    )

    def build_train_config(ctx: StepContext) -> GrugMoeSFTConfig:
        output_path = ctx.output_path
        return GrugMoeSFTConfig(
            model=model,
            data=packed_grug_data(
                rendered_path=ctx.artifact_path(rendered),
                cache_path=prefix_join(output_path, "token-cache"),
                tokenizer=tokenizer,
                context_length=context_length,
            ),
            output_path=output_path,
            run_id=f"curriculum-sft-{capability_id}-{version}",
            resources=resources,
            steps=steps,
            batch_size=batch_size,
            seed=seed,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(project="marin_moe_sft"),
            optimizer=optimizer,
            init_from_path=ctx.artifact_path(checkpoint),
            expert_parallel=expert_parallel,
            per_device_parallelism=1,
            grug_trainer=GrugTrainerConfig(replica_axis_size=1, z_loss_weight=1e-4),
        )

    return ArtifactStep(
        name=user_owned_name(f"checkpoints/curriculum-sft/{capability_id}/grug"),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_sft_trial,
        build_config=build_train_config,
        deps=(rendered, checkpoint),
    )
