# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball Stage 0 (CL12): the three-stage #8225 curriculum on the 262k base.

Chat, then thinking, then the terminal protocol — each stage initializes from the
previous stage's checkpoint artifact with a fresh optimizer, per the #8225 stage rule.
The base is the current 262k Snowball, ``open-athena/snowball-67b-a2b-base-262k-qk175``,
pinned. PR #8172 (the original launcher) was closed unmerged; this config follows
``experiments/sft/configs/snowball_h100.py`` against the current ``experiments/sft/launcher.py``.

Every stage trains on the converted CL12 rows (:func:`stage0_convert`), never the raw HF
datasets: the conversion strips both source marker styles into ``reasoning_content``, keeps
tool calls structured, and applies the one shared row-id length filter, so this run and the
Axolotl stage configs consume byte-identical files. The chat template is the Grug instance
of the shared template — the base repo's embedded Marin template predates
``reasoning_content`` and would train reasoning as answer text; the spec records the shared
template (the chat caches fork on its hash), and the run contract is that the exported
tokenizer carries it too.
"""

from dataclasses import dataclass, replace

from levanter.optim.config import AdamConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.checkpoints import hf_to_levanter
from marin.training.training import LevanterCheckpoint

from experiments.post_training.chat_template import GRUG_TOKENS, shared_template
from experiments.post_training.stage0.budget import (
    ADAM_BETA1,
    ADAM_BETA2,
    EPOCHS_PER_STAGE,
    GLOBAL_BATCH,
    GPUS_PER_NODE,
    LEARNING_RATE,
    LR_SCHEDULE,
    MAX_GRAD_NORM,
    SEQUENCE_LEN,
    TRAIN_NODES,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from experiments.post_training.stage0.convert import STAGE_ORDER, build_manifest, source_filename
from experiments.sft.launcher import (
    ArtifactDatasetSpec,
    ConvertedCheckpointModel,
    LevanterCheckpointModel,
    SFTSpec,
    resources_from_accelerator,
    sft_step,
)

STAGE0_VERSION = "2026.10.22"
# The design decision log (2026-10-22): the current 262k base, not the 5.7T cooldown.
BASE_HF_ID = "open-athena/snowball-67b-a2b-base-262k-qk175"
BASE_REVISION = "11c9430da3075ef4596f92ee0af142d9d08215c1"

# The #8225 stage slugs, in curriculum order (parallel to convert.STAGE_ORDER).
STAGE_NAMES = ("s1-chat", "s2-thinking", "s3-terminal")


@dataclass(frozen=True)
class Stage0ConvertConfig:
    """Inputs for the CL12 conversion; ``output_path`` is pulled from the step context."""

    output_path: str


def run_stage0_convert(config: Stage0ConvertConfig) -> None:
    """Convert all three #8225 sources to the shared protocol under the step's output path."""
    build_manifest(config.output_path)


def stage0_convert() -> ArtifactStep[Artifact]:
    """The CL12 conversion as a lazy artifact step: one ``{source}.jsonl.gz`` per stage.

    Every Stage 0 consumer addresses this one artifact — the Snowball stages below via
    :class:`ArtifactDatasetSpec`, the Axolotl stage configs via its resolved path as their
    ``data_root`` — so the trainers cannot read different files.
    """
    return ArtifactStep(
        name="documents/stage0-cl12",
        version=STAGE0_VERSION,
        artifact_type=Artifact,
        run=run_stage0_convert,
        build_config=lambda ctx: Stage0ConvertConfig(output_path=ctx.output_path),
    )


def _stage_optimizer() -> AdamConfig:
    """The budget's optimizer, fresh at every stage; identical to the Axolotl stages'."""
    return AdamConfig(
        learning_rate=LEARNING_RATE,
        beta1=ADAM_BETA1,
        beta2=ADAM_BETA2,
        epsilon=1e-8,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        lr_schedule=LR_SCHEDULE,
        # Levanter takes a warmup fraction; the budget's zero steps is the zero fraction.
        warmup=0.0 if WARMUP_STEPS == 0 else WARMUP_STEPS,
    )


def _mesh() -> MeshConfig:
    """Snowball shards parameters over expert x (data, context); context spans all 8 nodes."""
    return MeshConfig(
        axes={"data": 1, "replica": 1, "model": 1, "expert": -1},
        dcn_axes={"context": TRAIN_NODES},
        compute_mapping={
            "batch": ["replica_dcn", "data", "expert"],
            "position": "context",
            "vocab": "model",
        },
    )


def train_resources():
    """#8225's 64 GPUs: 8 nodes x 8 H100s — the mesh's context axis needs all 8 nodes."""
    return replace(resources_from_accelerator(f"{GPUS_PER_NODE}xH100"), replicas=TRAIN_NODES)


def _stage_dataset(source: str, convert: ArtifactStep[Artifact]) -> ArtifactDatasetSpec:
    """One stage's whole mixture: the converted rows of exactly its curriculum source."""
    return ArtifactDatasetSpec(
        slug=source,
        artifact=convert,
        train_glob=source_filename(source),
        weight=1.0,
    )


@dataclass(frozen=True)
class SnowballStage0:
    """The Stage 0 graph: the conversion, the three specs, and their launchable steps."""

    convert: ArtifactStep[Artifact]
    specs: dict[str, SFTSpec]
    steps: dict[str, ArtifactStep[LevanterCheckpoint]]


def snowball_stage0() -> SnowballStage0:
    """The three Stage 0 steps, initialized in sequence; stage 3 is the run's ``B``.

    Each stage trains on exactly one converted source for one epoch of that source
    (#8225's per-stage schedule), so the steps resolve their counts from the token totals
    and the launcher threads the chat caches as dependencies. Stage 2 and 3 hold the
    previous stage's ``ArtifactStep`` as ``init_from``, so the checkpoint path they load is
    resolved by the launcher from the real output — never a hand-built string.
    """
    convert = stage0_convert()
    base_conversion = hf_to_levanter(
        BASE_HF_ID,
        model_type="snowball",
        hf_revision=BASE_REVISION,
        version=STAGE0_VERSION,
        tokenizer=BASE_HF_ID,
        resources=resources_from_accelerator("8xH100"),
    )

    def stage_spec(stage_name: str, source: str, model) -> SFTSpec:
        return SFTSpec(
            name=f"checkpoints/stage0-snowball/{stage_name}",
            version=STAGE0_VERSION,
            model=model,
            chat_template=shared_template(GRUG_TOKENS),
            datasets=[_stage_dataset(source, convert)],
            optimizer=_stage_optimizer(),
            mesh=_mesh(),
            seq_len=SEQUENCE_LEN,
            pack=False,
            batch_size=GLOBAL_BATCH,
            num_train_epochs=EPOCHS_PER_STAGE,
            hf_save_dtype="bfloat16",
            wandb_project="marin_moe_sft",
        )

    spec1 = stage_spec(
        "s1-chat", STAGE_ORDER[0], ConvertedCheckpointModel(conversion=base_conversion)
    )
    step1 = sft_step(spec1, train_resources())

    spec2 = stage_spec(
        "s2-thinking",
        STAGE_ORDER[1],
        LevanterCheckpointModel(init_from=step1, model=base_conversion.model, tokenizer_path=BASE_HF_ID),
    )
    step2 = sft_step(spec2, train_resources())

    spec3 = stage_spec(
        "s3-terminal",
        STAGE_ORDER[2],
        LevanterCheckpointModel(init_from=step2, model=base_conversion.model, tokenizer_path=BASE_HF_ID),
    )
    step3 = sft_step(spec3, train_resources())

    return SnowballStage0(
        convert=convert,
        specs=dict(zip(STAGE_NAMES, (spec1, spec2, spec3), strict=True)),
        steps=dict(zip(STAGE_NAMES, (step1, step2, step3), strict=True)),
    )


def snowball_stage0_steps() -> dict[str, ArtifactStep[LevanterCheckpoint]]:
    """The launchable Stage 0 steps, in curriculum order."""
    return snowball_stage0().steps
