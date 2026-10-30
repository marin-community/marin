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
tokenizer carries it too (the launcher passes the spec's template to the HF export).

Each stage is one packed epoch, as in #8225: conversations pack into 32,768-token
sequences and Snowball's attention masks at the packed segment boundaries. The step
count resolves from the chat cache's token total, so it counts packed sequences.

Launch (a CPU coordinator; the conversion runs in it, the GPU work dispatches as
child jobs, strictly one after another)::

    python -m experiments.post_training.stage0.snowball_stage0 convert --num-workers 30
    python -m experiments.post_training.stage0.snowball_stage0 train --run smoke --through s2-thinking
    python -m experiments.post_training.stage0.snowball_stage0 train --run full --through s3-terminal

``--dry-run`` lowers the graph and builds every config without submitting anything.
"""

import argparse
import os
from dataclasses import dataclass, replace

from levanter.optim.config import AdamConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.checkpoints import hf_to_levanter
from marin.training.training import LevanterCheckpoint

from experiments.post_training.chat_template import GRUG_TOKENS, shared_template
from experiments.post_training.stage0.budget import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPSILON,
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
    chat_tokenize,
    resources_from_accelerator,
    sft_step,
)

STAGE0_VERSION = "2026.10.22"
# The design decision log (2026-10-22): the current 262k base, not the 5.7T cooldown.
BASE_HF_ID = "open-athena/snowball-67b-a2b-base-262k-qk175"
BASE_REVISION = "11c9430da3075ef4596f92ee0af142d9d08215c1"
# The design's pipeline-debugging model: Snowball's router at a fraction of the cost.
LADDER_HF_ID = "open-athena/Snowball-Ladder-d1280-Datakit-SFT-262K-2026.10.17"
LADDER_REVISION = "952acbd4381a6ffc641ea605cb9bbfdfce6f274c"

# The #8225 stage slugs, in curriculum order (parallel to convert.STAGE_ORDER).
STAGE_NAMES = ("s1-chat", "s2-thinking", "s3-terminal")

# The smoke conversion keeps this many rows per source: about 15 packed steps per stage
# at the smoke batch.
SMOKE_ROWS_PER_SOURCE = 2000
SMOKE_BATCH = 8

_CONVERT_WORKERS = "num_workers"


@dataclass(frozen=True)
class Stage0ConvertConfig:
    """Inputs for the CL12 conversion; ``output_path`` is pulled from the step context."""

    output_path: str
    max_rows: int | None
    num_workers: int


def run_stage0_convert(config: Stage0ConvertConfig) -> None:
    """Convert all three #8225 sources to the shared protocol under the step's output path."""
    build_manifest(config.output_path, max_rows=config.max_rows, num_workers=config.num_workers)


def stage0_convert(*, max_rows: int | None = None, num_workers: int = 1) -> ArtifactStep[Artifact]:
    """The CL12 conversion as a lazy artifact step: one ``{source}.jsonl.gz`` per stage.

    Every Stage 0 consumer addresses this one artifact — the Snowball stages below via
    :class:`ArtifactDatasetSpec`, the Axolotl stage configs via its resolved path as their
    ``data_root`` — so the trainers cannot read different files. ``max_rows`` truncates
    every source (the smoke artifact, under its own name); ``num_workers`` is an execution
    choice and never enters the artifact's identity.
    """
    name = "documents/stage0-cl12" if max_rows is None else f"documents/stage0-cl12-first{max_rows}"

    def build_config(ctx: StepContext) -> Stage0ConvertConfig:
        return Stage0ConvertConfig(
            output_path=ctx.output_path, max_rows=max_rows, num_workers=ctx.runtime_arg(_CONVERT_WORKERS)
        )

    return ArtifactStep(
        name=name,
        version=STAGE0_VERSION,
        artifact_type=Artifact,
        run=run_stage0_convert,
        build_config=build_config,
        runtime_args={_CONVERT_WORKERS: num_workers},
    )


def _stage_optimizer() -> AdamConfig:
    """The budget's optimizer, fresh at every stage; identical to the Axolotl stages'."""
    return AdamConfig(
        learning_rate=LEARNING_RATE,
        beta1=ADAM_BETA1,
        beta2=ADAM_BETA2,
        epsilon=ADAM_EPSILON,
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


def _smoke_mesh() -> MeshConfig:
    """One node, with a context axis of 2 so the smoke exercises packed segments under context parallelism."""
    return MeshConfig(
        axes={"data": 1, "replica": 1, "model": 1, "context": 2, "expert": -1},
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
class SnowballRun:
    """Which model, data and hardware a Stage 0 chain runs on; the recipe is fixed."""

    name: str
    hf_id: str
    hf_revision: str
    convert: ArtifactStep[Artifact]
    batch_size: int
    mesh: MeshConfig
    nodes: int


def full_run(*, num_workers: int = 1) -> SnowballRun:
    """The Stage 0 run that produces Snowball's ``B``: the 67B base on 8 nodes."""
    return SnowballRun(
        name="stage0-snowball",
        hf_id=BASE_HF_ID,
        hf_revision=BASE_REVISION,
        convert=stage0_convert(num_workers=num_workers),
        batch_size=GLOBAL_BATCH,
        mesh=_mesh(),
        nodes=TRAIN_NODES,
    )


def smoke_run(*, num_workers: int = 1) -> SnowballRun:
    """The pipeline smoke: the d1280 ladder model on one node, over truncated sources."""
    return SnowballRun(
        name="stage0-snowball-smoke",
        hf_id=LADDER_HF_ID,
        hf_revision=LADDER_REVISION,
        convert=stage0_convert(max_rows=SMOKE_ROWS_PER_SOURCE, num_workers=num_workers),
        batch_size=SMOKE_BATCH,
        mesh=_smoke_mesh(),
        nodes=1,
    )


@dataclass(frozen=True)
class SnowballStage0:
    """The Stage 0 graph: the conversion, the three specs, and their launchable steps."""

    convert: ArtifactStep[Artifact]
    specs: dict[str, SFTSpec]
    steps: dict[str, ArtifactStep[LevanterCheckpoint]]


def snowball_stage0(run: SnowballRun | None = None) -> SnowballStage0:
    """The three Stage 0 steps, initialized in sequence; stage 3 is the run's ``B``.

    Each stage trains on exactly one converted source for one packed epoch of that source
    (#8225's per-stage schedule), so the steps resolve their counts from the token totals
    and the launcher threads the chat caches as dependencies. Stage 2 and 3 hold the
    previous stage's ``ArtifactStep`` as ``init_from``, so the checkpoint path they load is
    resolved by the launcher from the real output — never a hand-built string.
    """
    run = full_run() if run is None else run
    base_conversion = hf_to_levanter(
        run.hf_id,
        model_type="snowball",
        hf_revision=run.hf_revision,
        version=STAGE0_VERSION,
        tokenizer=run.hf_id,
        resources=resources_from_accelerator(f"{GPUS_PER_NODE}xH100"),
    )
    resources = replace(resources_from_accelerator(f"{GPUS_PER_NODE}xH100"), replicas=run.nodes)

    def stage_spec(stage_name: str, source: str, model) -> SFTSpec:
        return SFTSpec(
            name=f"checkpoints/{run.name}/{stage_name}",
            version=STAGE0_VERSION,
            model=model,
            chat_template=shared_template(GRUG_TOKENS),
            datasets=[_stage_dataset(source, run.convert)],
            optimizer=_stage_optimizer(),
            mesh=run.mesh,
            seq_len=SEQUENCE_LEN,
            pack=True,
            batch_size=run.batch_size,
            num_train_epochs=EPOCHS_PER_STAGE,
            hf_save_dtype="bfloat16",
            wandb_project="marin_moe_sft",
        )

    specs: list[SFTSpec] = []
    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    for stage_name, source in zip(STAGE_NAMES, STAGE_ORDER, strict=True):
        if not steps:
            model = ConvertedCheckpointModel(conversion=base_conversion)
        else:
            # The pinned base's tokenizer: the hub id's main is the pinned revision.
            model = LevanterCheckpointModel(init_from=steps[-1], model=base_conversion.model, tokenizer_path=run.hf_id)
        specs.append(stage_spec(stage_name, source, model))
        steps.append(sft_step(specs[-1], resources))

    return SnowballStage0(
        convert=run.convert,
        specs=dict(zip(STAGE_NAMES, specs, strict=True)),
        steps=dict(zip(STAGE_NAMES, steps, strict=True)),
    )


def snowball_stage0_steps() -> dict[str, ArtifactStep[LevanterCheckpoint]]:
    """The launchable Stage 0 steps, in curriculum order."""
    return snowball_stage0().steps


_RUNS = {"full": full_run, "smoke": smoke_run}


def stage_chat_cache(graph: SnowballStage0, stage_name: str) -> ArtifactStep:
    """A later stage's chat cache with only the data as its dependency, to build ahead of training.

    The launcher's cache step also depends on the stage's init step, because a converted
    checkpoint supplies its tokenizer. Stages 2 and 3 read the pinned base tokenizer, so their
    caches need only the converted rows; this handle has the same name, version and output
    path as the launcher's, which then finds it built.
    """
    spec = graph.specs[stage_name]
    if not isinstance(spec.model, LevanterCheckpointModel):
        raise ValueError(f"{stage_name}'s tokenizer comes from its init step; build its cache in the chain")
    cache = chat_tokenize(spec, spec.datasets[0], graph.convert)
    return replace(cache, deps=(graph.convert,))


def main() -> None:
    parser = argparse.ArgumentParser(description="Snowball Stage 0 (CL12): conversion and the three SFT stages.")
    parser.add_argument("--dry-run", action="store_true", help="lower and build every config; submit nothing")
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert", help="build the CL12 artifact in this process")
    convert.add_argument("--run", choices=sorted(_RUNS), default="full")
    convert.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    train = subparsers.add_parser("train", help="run the stage chain through --through, one stage at a time")
    train.add_argument("--run", choices=sorted(_RUNS), required=True)
    train.add_argument("--through", choices=STAGE_NAMES, required=True)
    train.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    cache = subparsers.add_parser("cache", help="build a later stage's chat cache ahead of its training")
    cache.add_argument("--run", choices=sorted(_RUNS), required=True)
    cache.add_argument("--stage", choices=STAGE_NAMES[1:], required=True)
    args = parser.parse_args()

    if args.command == "cache":
        target = stage_chat_cache(snowball_stage0(_RUNS[args.run]()), args.stage)
        StepRunner().run([lower(target)], dry_run=args.dry_run)
        return
    run = _RUNS[args.run](num_workers=args.num_workers)
    target = run.convert if args.command == "convert" else snowball_stage0(run).steps[args.through]
    # The GPU steps (the HF conversion, then each stage) form one dependency chain, so at most
    # one GPU job runs at a time; CPU steps (conversion, chat caches) may overlap with them.
    StepRunner().run([lower(target)], dry_run=args.dry_run)


if __name__ == "__main__":
    main()
