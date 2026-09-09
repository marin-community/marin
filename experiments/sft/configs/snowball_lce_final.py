# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Five-way Snowball LCE SFT campaign reconstructed from marin #8225 and #8977.

Each base is a pinned HF export of a native step-157000 checkpoint. The first stage loads HF
weights into the first-class Snowball model with the architecture resolved from that exact Hub
revision; later stages use weights-only native checkpoint initialization. Every stage therefore
starts with a fresh optimizer and step counter while retaining the base-specific ``qk_mult``.

Launch a one-update full-shape smoke on RNO2A before the campaign fan-out::

    source ../secrets.env
    uv run iris --config lib/iris/config/marin.yaml job run \
      --target-cluster cw-rno2a --job-name snowball-final-qk157-smoke3-coord \
      --cpu 2 --memory 2G --extra cpu --priority interactive --max-retries 10 --no-wait \
      -e MARIN_PREFIX s3://marin-us-east-02a/marin \
      -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \
      -e IRIS_PORT_JAX 19403 -- \
      python -m experiments.sft.configs.snowball_lce_final \
      --base qk157 --stage smoke --version 2026.09.08.3 --run
"""

import dataclasses
import json
from typing import Literal

import click
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat, LmDatasetFormatBase
from levanter.models.snowball import SnowballConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.checkpoints import resolve_lm_config
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.datasets.grug_a2b_agentic_sft_eot import (
    GRUG_A2B_AGENTIC_SFT_FORMAT,
    grug_a2b_agentic_sft_eot_dataset,
)
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
from experiments.marin_tokenizer import MARIN_CHAT_TEMPLATE
from experiments.sft.delphi_chat_template import DELPHI_V0_CHAT_TEMPLATE
from experiments.sft.launcher import DatasetSpec, HFModel, LevanterCheckpointModel, SFTSpec, sft_step

_SEQ = 32_768
_BATCH = 64
_NODES = 8
_EXPERT_PARALLEL = 8
_WANDB_PROJECT = "marin_moe_sft_snowball_final"
_AGENTIC_STEPS = 1_888
_PREBUILT_TRAIN_RESOURCES = "prebuilt_train_resources"
_OPENCODE_DATASET_REVISION = "a9805934c9c98908c611236bbfc87799f1ff6fe5"
_NEMOTRON_DATASET_REVISION = "a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9"

# These are immutable tags created only after the uploader validated all 44 source files. Add the
# three skew revisions after their export/upload jobs finish; never train from a moving ``main``.
_BASE_REVISIONS: dict[str, tuple[str, str | None]] = {
    "qk157": ("open-athena/snowball-67b-a2b-base-262k-qk157", "2b1f526273b8968b307a0098c08fb4321bb91e35"),
    "qk175": ("open-athena/snowball-67b-a2b-base-262k-qk175", "1934e71f2bb0fbeb19e5ce82372136e5297bf0a4"),
    "qk175-skew2": (
        "open-athena/snowball-67b-a2b-base-262k-qk175-skew2",
        "ce41c24df0afc10079210521ea7e231115ad5a92",
    ),
    "qk175-skew4": (
        "open-athena/snowball-67b-a2b-base-262k-qk175-skew4",
        "5052e68c4d88c9e0de87f7595a25ee4005aef1cf",
    ),
    "qk175-skew8": (
        "open-athena/snowball-67b-a2b-base-262k-qk175-skew8",
        "058ecaf27b9e4f37219df221a51e7d490d58ec3d",
    ),
}

_CHAT_DATASET = DatasetSpec(
    slug="wildchat_386k",
    hf_dataset_id="nyu-dice-lab/wildchat50m-rewild-sft-385700",
    revision="46a5bb5",
    adapter_kwargs={"conversation_column": "conversation"},
    weight=1.0,
)
_THINKING_DATASET = DatasetSpec(
    slug="nemotron_science_think",
    hf_dataset_id="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
    revision="bae881d",
    adapter_kwargs={},
    weight=1.0,
)
_OPENCODE_DATASET = DatasetSpec(
    slug="grug_a2b_agentic_sft_eot",
    hf_dataset_id="open-athena/grug-67b-a2b-agentic-sft-training-data",
    revision=_OPENCODE_DATASET_REVISION,
    adapter_kwargs={},
    weight=1.0,
)
_NEMOTRON_DATASET = DatasetSpec(
    slug="nemotron_terminal_full",
    hf_dataset_id="nvidia/Nemotron-Terminal-Corpus",
    revision=_NEMOTRON_DATASET_REVISION,
    adapter_kwargs={"conversation_column": "conversations"},
    weight=1.0,
)

# The existing immutable cache below was built from precisely these 29 files. Retain the selection
# beside the adopted cache so an audit or future rebuild cannot silently expand to the full corpus.
_NEMOTRON_PARQUET_FILES: tuple[str, ...] = (
    "dataset_adapters/code.parquet",
    "dataset_adapters/math.parquet",
    "dataset_adapters/swe.parquet",
    "synthetic_tasks/skill_based/easy/data_processing/data_filtered.parquet",
    "synthetic_tasks/skill_based/easy/data_querying/data_filtered.parquet",
    "synthetic_tasks/skill_based/easy/data_science/data_filtered.parquet",
    "synthetic_tasks/skill_based/easy/debugging/data_filtered.parquet",
    "synthetic_tasks/skill_based/easy/dependency_management/data_filtered.parquet",
    "synthetic_tasks/skill_based/easy/file_operations/data_filtered.parquet",
    "synthetic_tasks/skill_based/easy/scientific_computing/data_filtered.parquet",
    "synthetic_tasks/skill_based/easy/security/data_filtered.parquet",
    "synthetic_tasks/skill_based/easy/software_engineering/data_filtered.parquet",
    "synthetic_tasks/skill_based/medium/data_processing/data_filtered.parquet",
    "synthetic_tasks/skill_based/medium/data_querying/data_filtered.parquet",
    "synthetic_tasks/skill_based/medium/data_science/data_filtered.parquet",
    "synthetic_tasks/skill_based/medium/debugging/data_filtered.parquet",
    "synthetic_tasks/skill_based/medium/dependency_management/data_filtered.parquet",
    "synthetic_tasks/skill_based/medium/file_operations/data_filtered.parquet",
    "synthetic_tasks/skill_based/medium/model_training/data_filtered.parquet",
    "synthetic_tasks/skill_based/medium/scientific_computing/data_filtered.parquet",
    "synthetic_tasks/skill_based/medium/security/data_filtered.parquet",
    "synthetic_tasks/skill_based/medium/software_engineering/data_filtered.parquet",
    "synthetic_tasks/skill_based/medium/system_administration/data_filtered.parquet",
    "synthetic_tasks/skill_based/mixed/data_processing/data_filtered.parquet",
    "synthetic_tasks/skill_based/mixed/data_science/data_filtered.parquet",
    "synthetic_tasks/skill_based/mixed/debugging/data_filtered.parquet",
    "synthetic_tasks/skill_based/mixed/file_operations/data_filtered.parquet",
    "synthetic_tasks/skill_based/mixed/scientific_computing/data_filtered.parquet",
    "synthetic_tasks/skill_based/mixed/security/data_filtered.parquet",
)
_NEMOTRON_CACHE_SOURCE = "s3://marin-us-east-02a/marin/tokenized/nemotron_terminal_full-chat-7adc64/2026.07.17"

_TRAIN_MESH = MeshConfig(
    axes={"expert": _EXPERT_PARALLEL},
    dcn_axes={"data": -1},
    compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
)


def _resources() -> ResourceConfig:
    return ResourceConfig.with_gpu(
        "H100",
        count=8,
        cpu=64,
        ram="768g",
        disk="512g",
        replicas=_NODES,
        preemptible=True,
    )


def _optimizer(learning_rate: float) -> GrugMoeAdamHConfig:
    return GrugMoeAdamHConfig(
        learning_rate=learning_rate,
        adam_lr=learning_rate,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
        weight_decay=0.0,
        min_lr_ratio=0.1,
        warmup=0.03,
        lr_schedule="cosine",
    )


def _base_model(base: str) -> tuple[HFModel, SnowballConfig, str]:
    repo, revision = _BASE_REVISIONS[base]
    if revision is None:
        raise ValueError(f"Base {base} has not completed its validated HF upload.")
    model = resolve_lm_config("snowball", repo, revision)
    if not isinstance(model, SnowballConfig):
        raise TypeError(f"Expected SnowballConfig for {repo}@{revision}, got {type(model).__name__}.")
    model = dataclasses.replace(
        model,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="ring",
    )
    ref = f"{repo}@{revision}"
    return (
        HFModel(
            model_ref=ref,
            tokenizer_path=ref,
            model_type="snowball",
            model_config=model,
            eos_token_ids=(128001, 128009),
            trainer_mesh=_TRAIN_MESH,
            use_explicit_mesh_axes=True,
        ),
        model,
        ref,
    )


def _native_model(parent: ArtifactStep[LevanterCheckpoint], model: SnowballConfig, tokenizer: str):
    return LevanterCheckpointModel(
        init_from=parent,
        model=model,
        tokenizer_path=tokenizer,
        eos_token_ids=(128001, 128009),
        trainer_mesh=_TRAIN_MESH,
        use_explicit_mesh_axes=True,
    )


def _spec(
    *,
    base: str,
    stage: str,
    version: str | None,
    model,
    dataset: DatasetSpec,
    steps: int | None = None,
    epochs: int | None = None,
    expected_epoch_steps: int | None = None,
    learning_rate: float = 5e-5,
    chat_template: str = DELPHI_V0_CHAT_TEMPLATE,
) -> SFTSpec:
    step_name = f"snowball-final/{base}/{stage}"
    resolved_version = resolve_version(step_name, version)
    return SFTSpec(
        name=user_namespaced_name(step_name, resolved_version),
        version=resolved_version,
        model=model,
        chat_template=chat_template,
        datasets=[dataset],
        optimizer=_optimizer(learning_rate),
        seq_len=_SEQ,
        batch_size=_BATCH,
        num_train_steps=steps,
        num_train_epochs=epochs,
        expected_epoch_steps=expected_epoch_steps,
        wandb_project=_WANDB_PROJECT,
    )


def build_smoke(base: str, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    model, _, _ = _base_model(base)
    return sft_step(
        _spec(base=base, stage="hf-smoke", version=version, model=model, dataset=_CHAT_DATASET, steps=1),
        _resources(),
    )


def build_chat(base: str, version: str | None = None) -> tuple[ArtifactStep[LevanterCheckpoint], SnowballConfig, str]:
    model, config, tokenizer = _base_model(base)
    chat = sft_step(
        _spec(
            base=base,
            stage="chat",
            version=version,
            model=model,
            dataset=_CHAT_DATASET,
            epochs=1,
            expected_epoch_steps=257,
        ),
        _resources(),
    )
    return chat, config, tokenizer


def build_thinking(base: str, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    chat, config, tokenizer = build_chat(base, version)
    thinking_model = _native_model(chat, config, tokenizer)
    return sft_step(
        _spec(
            base=base,
            stage="thinking",
            version=version,
            model=thinking_model,
            dataset=_THINKING_DATASET,
            epochs=1,
            expected_epoch_steps=630,
        ),
        _resources(),
    )


def _adopt_nemotron_cache() -> ArtifactStep[TokenizedCache]:
    return ArtifactStep.adopt(
        name="tokenized/nemotron_terminal_full-chat-7adc64",
        version="2026.07.17",
        source=_NEMOTRON_CACHE_SOURCE,
        kind=TokenizedCache,
    )


def _prebuilt_data_config(
    *,
    tokenizer: str,
    cache_path: str,
    slug: str,
    data_format: LmDatasetFormatBase,
    packed_slice_strategy: Literal["left", "right", "raise"] = "left",
) -> LmDataConfig:
    source = UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=cache_path,
        format=data_format,
    )
    return LmDataConfig(
        tokenizer=tokenizer,
        auto_build_caches=False,
        components={
            slug: DatasetComponent(
                source=source,
                cache_dir=cache_path,
                format=data_format,
                pack=True,
                packed_slice_strategy=packed_slice_strategy,
                split="train",
            )
        },
        train_weights={slug: 1.0},
    )


def _build_prebuilt_stage(
    *,
    base: str,
    stage: str,
    version: str | None,
    parent: ArtifactStep[LevanterCheckpoint],
    config: SnowballConfig,
    tokenizer: str,
    cache: ArtifactStep[TokenizedCache],
    dataset: DatasetSpec,
    data_format: LmDatasetFormatBase,
    chat_template: str,
    packed_slice_strategy: Literal["left", "right", "raise"] = "left",
) -> ArtifactStep[LevanterCheckpoint]:
    model = _native_model(parent, config, tokenizer)
    spec = _spec(
        base=base,
        stage=stage,
        version=version,
        model=model,
        dataset=dataset,
        steps=_AGENTIC_STEPS,
        learning_rate=5e-6,
        chat_template=chat_template,
    )

    def build_config(ctx: StepContext):
        data = _prebuilt_data_config(
            tokenizer=model.resolve_tokenizer(ctx),
            cache_path=ctx.artifact_path(cache),
            slug=dataset.slug,
            data_format=data_format,
            packed_slice_strategy=packed_slice_strategy,
        )
        return model.build_train_config(
            ctx,
            spec,
            data,
            ctx.runtime_arg(_PREBUILT_TRAIN_RESOURCES),
            _AGENTIC_STEPS,
        )

    return ArtifactStep(
        name=spec.name,
        version=spec.version,
        artifact_type=LevanterCheckpoint,
        run=model.run,
        build_config=build_config,
        deps=(cache, *model.init_deps()),
        runtime_args={_PREBUILT_TRAIN_RESOURCES: _resources()},
    )


def _build_prefix(base: str, version: str | None = None) -> tuple[ArtifactStep[LevanterCheckpoint], SnowballConfig, str]:
    chat, config, tokenizer = build_chat(base, version)
    thinking = sft_step(
        _spec(
            base=base,
            stage="thinking",
            version=version,
            model=_native_model(chat, config, tokenizer),
            dataset=_THINKING_DATASET,
            epochs=1,
            expected_epoch_steps=630,
        ),
        _resources(),
    )
    return thinking, config, tokenizer


def _build_opencode_from(
    *,
    base: str,
    version: str | None,
    thinking: ArtifactStep[LevanterCheckpoint],
    config: SnowballConfig,
    tokenizer: str,
) -> ArtifactStep[LevanterCheckpoint]:
    return _build_prebuilt_stage(
        base=base,
        stage="opencode",
        version=version,
        parent=thinking,
        config=config,
        tokenizer=tokenizer,
        cache=grug_a2b_agentic_sft_eot_dataset(),
        dataset=_OPENCODE_DATASET,
        data_format=GRUG_A2B_AGENTIC_SFT_FORMAT,
        chat_template=DELPHI_V0_CHAT_TEMPLATE,
        packed_slice_strategy="right",
    )


def _build_nemotron_terminal_from(
    *,
    base: str,
    version: str | None,
    thinking: ArtifactStep[LevanterCheckpoint],
    config: SnowballConfig,
    tokenizer: str,
) -> ArtifactStep[LevanterCheckpoint]:
    return _build_prebuilt_stage(
        base=base,
        stage="nemotron-terminal",
        version=version,
        parent=thinking,
        config=config,
        tokenizer=tokenizer,
        cache=_adopt_nemotron_cache(),
        dataset=_NEMOTRON_DATASET,
        data_format=ChatLmDatasetFormat(
            messages_field="messages",
            chat_template=MARIN_CHAT_TEMPLATE,
            mask_user_turns=True,
            pack=None,
        ),
        chat_template=MARIN_CHAT_TEMPLATE,
    )


def build_opencode(base: str, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    thinking, config, tokenizer = _build_prefix(base, version)
    return _build_opencode_from(
        base=base,
        version=version,
        thinking=thinking,
        config=config,
        tokenizer=tokenizer,
    )


def build_nemotron_terminal(base: str, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    thinking, config, tokenizer = _build_prefix(base, version)
    return _build_nemotron_terminal_from(
        base=base,
        version=version,
        thinking=thinking,
        config=config,
        tokenizer=tokenizer,
    )


@dataclasses.dataclass(frozen=True)
class CampaignCompleteConfig:
    output_path: str
    base: str
    opencode_path: str
    nemotron_terminal_path: str


def _write_campaign_complete(config: CampaignCompleteConfig) -> None:
    manifest = {
        "base": config.base,
        "opencode_path": config.opencode_path,
        "nemotron_terminal_path": config.nemotron_terminal_path,
    }
    StoragePath(prefix_join(config.output_path, "manifest.json")).write_text(json.dumps(manifest, sort_keys=True))


def build_all(base: str, version: str | None = None) -> ArtifactStep[Artifact]:
    """Build one shared Chat/Thinking prefix followed by both independent agentic branches."""
    thinking, config, tokenizer = _build_prefix(base, version)
    opencode = _build_opencode_from(
        base=base,
        version=version,
        thinking=thinking,
        config=config,
        tokenizer=tokenizer,
    )
    nemotron_terminal = _build_nemotron_terminal_from(
        base=base,
        version=version,
        thinking=thinking,
        config=config,
        tokenizer=tokenizer,
    )
    step_name = f"snowball-final/{base}/complete"
    resolved_version = resolve_version(step_name, version)

    def build_config(ctx: StepContext) -> CampaignCompleteConfig:
        return CampaignCompleteConfig(
            output_path=ctx.output_path,
            base=base,
            opencode_path=ctx.artifact_path(opencode),
            nemotron_terminal_path=ctx.artifact_path(nemotron_terminal),
        )

    return ArtifactStep(
        name=user_namespaced_name(step_name, resolved_version),
        version=resolved_version,
        artifact_type=Artifact,
        run=_write_campaign_complete,
        build_config=build_config,
        deps=(opencode, nemotron_terminal),
    )


@click.command()
@click.option("--base", type=click.Choice(tuple(_BASE_REVISIONS)), required=True)
@click.option(
    "--stage",
    type=click.Choice(("smoke", "chat", "thinking", "opencode", "nemotron-terminal", "all")),
    required=True,
)
@build_options
def main(base: str, stage: str) -> ArtifactStep[LevanterCheckpoint]:
    if stage == "smoke":
        return build_smoke(base)
    if stage == "chat":
        return build_chat(base)[0]
    if stage == "thinking":
        return build_thinking(base)
    if stage == "opencode":
        return build_opencode(base)
    if stage == "nemotron-terminal":
        return build_nemotron_terminal(base)
    return build_all(base)


if __name__ == "__main__":
    main()
