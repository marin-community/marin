# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a native Grug checkpoint on conversations from selected capabilities."""

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pyarrow as pa
from fray.types import ResourceConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.optim.config import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from marin.datakit.chat_normalize import message_text
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.training.training import LevanterCheckpoint
from openai_harmony import Message, Role
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugMoeSFTConfig, run_grug_moe_sft_trial
from experiments.june_tpu_67b_a2b.moe.train import GrugTrainerConfig
from experiments.post_training.curriculum_sft.generation import generate_curriculum_sft
from experiments.post_training.task_curriculum.catalog_artifact import TASK_CURRICULUM, TaskCurriculumCatalogArtifact

GRUG_CHECKPOINTS_DIR = "checkpoints"
SFT_CHAT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field(
            "messages",
            pa.list_(pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])),
            nullable=False,
        ),
    ]
)


@dataclass(frozen=True)
class CurriculumGenerationSpec:
    requested_examples: int
    accepted_examples: int
    seed: int
    max_completion_tokens: int
    task_specification: str


@dataclass(frozen=True)
class PrepareConfig:
    input_path: str
    output_path: str


def prepare_chat_record(record: dict) -> dict:
    """Project generated text-only Harmony into Levanter's OpenAI chat schema."""
    messages = [Message.from_dict(message) for message in record["messages"]]
    if any(message.author.role not in (Role.USER, Role.ASSISTANT) or message.recipient for message in messages):
        raise ValueError("curriculum SFT requires text-only user/assistant conversations")
    return {
        "id": record["id"],
        "messages": [{"role": message.author.role.value, "content": message_text(message)} for message in messages],
    }


def prepare_generated_chat(config: PrepareConfig) -> Artifact:
    """Write OpenAI-style chat Parquet for assistant-only packed SFT."""
    pipeline = (
        Dataset.from_files(prefix_join(prefix_join(config.input_path, "chat"), "*.parquet"))
        .flat_map(load_parquet)
        .map(prepare_chat_record)
        .write_parquet(
            prefix_join(config.output_path, "part-{shard:05d}-of-{total:05d}.parquet"), schema=SFT_CHAT_SCHEMA
        )
    )
    ZephyrContext(name="prepare-curriculum-chat", resources=ResourceConfig(cpu=2, ram="4g"), max_workers=4).execute(
        pipeline
    )
    return Artifact(path=config.output_path)


def packed_grug_data(*, prepared_paths: Mapping[str, str], cache_path: str, tokenizer: str) -> LmDataConfig:
    """Pack conversations with assistant-only loss and no cross-conversation attention."""
    if not prepared_paths:
        raise ValueError("at least one prepared curriculum source is required")
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        components={
            capability_id: DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[prefix_join(path, "*.parquet")]),
                cache_dir=prefix_join(cache_path, capability_id),
                format=ChatLmDatasetFormat(
                    chat_template=MARIN_CHAT_TEMPLATE,
                    chat_template_kwargs=None,
                    mask_user_turns=True,
                    slice_strategy="drop",
                ),
                pack=True,
                split="train",
            )
            for capability_id, path in prepared_paths.items()
        },
        train_weights={capability_id: 1 / len(prepared_paths) for capability_id in prepared_paths},
        auto_build_caches=True,
        shuffle=True,
        block_cross_document_attention=True,
    )


def september_grug_model(context_length: int) -> GrugModelConfig:
    """Return the September SFT architecture at a chosen training sequence length."""
    return dataclasses.replace(
        MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=context_length),
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=2048,
        use_array_stacked_blocks=True,
        head_dim=128,
        qk_mult=1.75,
        max_seq_len=context_length,
        attention_implementation="gpu_fa4_cute",
        ce_implementation="batched_xla",
    )


def curriculum_generation_steps(
    curriculum_ids: Sequence[str],
    *,
    version: str,
    generation: CurriculumGenerationSpec,
    catalog: ArtifactStep[TaskCurriculumCatalogArtifact] = TASK_CURRICULUM,
) -> dict[str, ArtifactStep[Artifact]]:
    """Bind one GLM generation artifact per capability."""
    return {
        capability_id: generate_curriculum_sft(
            capability_id,
            catalog=catalog,
            version=version,
            requested_examples=generation.requested_examples,
            accepted_examples=generation.accepted_examples,
            seed=generation.seed,
            max_completion_tokens=generation.max_completion_tokens,
            task_specification=generation.task_specification,
        )
        for capability_id in curriculum_ids
    }


def curriculum_grug_sft(
    curriculum_ids: Sequence[str],
    *,
    version: str,
    generation: CurriculumGenerationSpec,
    generated: Mapping[str, ArtifactStep[Artifact]] | None = None,
    checkpoint: ArtifactStep[LevanterCheckpoint],
    checkpoint_subpath: str,
    tokenizer: str,
    optimizer: OptimizerConfig,
    resources: ResourceConfig,
    context_length: int,
    batch_size: int,
    steps: int,
    expert_parallel: int,
) -> ArtifactStep[LevanterCheckpoint]:
    """Generate, prepare, and mix curriculum conversations for native Grug SFT.

    The checkpoint and tokenizer must belong to the same Grug architecture. Pass
    a Hugging Face tokenizer ID, which Levanter can stage on each trainer rank.
    A native checkpoint is required; an HF export cannot initialize this trainer.
    """
    ids = tuple(sorted(curriculum_ids))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("curriculum_ids must be nonempty and distinct")
    curriculum_key = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12]
    if generated is None:
        generated = curriculum_generation_steps(ids, version=version, generation=generation)
    if set(generated) != set(ids):
        raise ValueError("generated sources must match curriculum_ids")
    prepared = tuple(
        _prepare_step(
            generated[capability_id],
            capability_id=capability_id,
            version=version,
        )
        for capability_id in ids
    )

    model = september_grug_model(context_length)

    def build_train_config(ctx: StepContext) -> GrugMoeSFTConfig:
        output_path = ctx.output_path
        return GrugMoeSFTConfig(
            model=model,
            data=packed_grug_data(
                prepared_paths={
                    capability_id: ctx.artifact_path(step) for capability_id, step in zip(ids, prepared, strict=True)
                },
                cache_path=prefix_join(output_path, "token-cache"),
                tokenizer=tokenizer,
            ),
            output_path=output_path,
            run_id=f"curriculum-sft-{curriculum_key}-{version}",
            resources=resources,
            steps=steps,
            batch_size=batch_size,
            seed=generation.seed,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(project="marin_moe_sft"),
            optimizer=optimizer,
            init_from_path=(
                prefix_join(ctx.artifact_path(checkpoint), checkpoint_subpath)
                if checkpoint_subpath
                else ctx.artifact_path(checkpoint)
            ),
            expert_parallel=expert_parallel,
            per_device_parallelism=1,
            grug_trainer=GrugTrainerConfig(replica_axis_size=1, z_loss_weight=1e-4),
        )

    return ArtifactStep(
        name=user_owned_name(f"checkpoints/curriculum-sft/{curriculum_key}/grug"),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_sft_trial,
        build_config=build_train_config,
        deps=(*prepared, checkpoint),
    )


def _prepare_step(generated: ArtifactStep[Artifact], *, capability_id: str, version: str) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> PrepareConfig:
        return PrepareConfig(ctx.artifact_path(generated), ctx.output_path)

    return ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/{capability_id}/prepared-chat"),
        version=version,
        artifact_type=Artifact,
        run=prepare_generated_chat,
        build_config=build_config,
        deps=(generated,),
    )
