# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a completed Marin SFT export and a frozen chat mix to offline DSpark."""

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import click
import draccus
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, lower
from marin.execution.step_runner import StepRunner
from marin.training.dspark import SPECULATORS_REVISION, VLLM_SOURCE_COMMIT, LocalGpuRuntime, train_and_export
from marin.training.dspark_data import DraftRecipe, GrugShape, inspect_mix, inspect_teacher, prepare_data
from rigging.filesystem.cluster_config import marin_prefix


class TeacherFamily(StrEnum):
    SNOWBALL = "snowball"
    MOCHI = "mochi"


_SHAPES = {
    TeacherFamily.SNOWBALL: GrugShape(2560, 26, 20, 5, 128, 128256),
    TeacherFamily.MOCHI: GrugShape(6144, 48, 48, 12, 128, 128256, sconv=True),
}
_FEATURE_TAPS = {
    TeacherFamily.SNOWBALL: (2, 12, 22),
    TeacherFamily.MOCHI: (2, 24, 45),
}
_RECIPE_VERSION = "2026.11.22"


@dataclass(frozen=True)
class CompletedSFT:
    """Identity of a completed HF SFT export staged on the execution host."""

    family: TeacherFamily
    identity: str
    directory: str
    manifest_sha256: str


@dataclass(frozen=True)
class TrainingMix:
    """Local copy of a versioned set of normalized Marin chat parquet files."""

    directory: str
    manifest_sha256: str


@dataclass(frozen=True)
class OfflineDSpark:
    teacher: CompletedSFT
    data: TrainingMix
    max_steps: int
    teacher_replicas: int
    trainer_ranks: int
    max_sequence_length: int = 32768
    max_anchors: int = 12288
    seed: int = 42

    def training_recipe(self) -> DraftRecipe:
        return DraftRecipe(
            shape=_SHAPES[self.teacher.family],
            feature_taps=_FEATURE_TAPS[self.teacher.family],
            mask_token_id=128004,
            max_sequence_length=self.max_sequence_length,
            max_anchors=self.max_anchors,
            max_steps=self.max_steps,
            trainer_ranks=self.trainer_ranks,
            teacher_replicas=self.teacher_replicas,
            seed=self.seed,
        )


def artifacts(config: OfflineDSpark, runtime: LocalGpuRuntime | None = None) -> tuple[ArtifactStep, ArtifactStep]:
    """Bind the CPU dataset and the terminal vLLM checkpoint artifact."""

    if not config.teacher.identity.strip():
        raise ValueError("Provide the completed SFT export's provenance identity")
    for name, path, digest in (
        ("teacher", config.teacher.directory, config.teacher.manifest_sha256),
        ("mix", config.data.directory, config.data.manifest_sha256),
    ):
        if not Path(path).is_absolute():
            raise ValueError(f"The staged {name} directory must be an absolute POSIX path")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"The {name} manifest must be pinned by SHA-256")

    family = config.teacher.family.value
    recipe = config.training_recipe()
    teacher = ArtifactStep.adopt(
        f"dspark/teachers/{family}/{config.teacher.manifest_sha256}",
        _RECIPE_VERSION,
        config.teacher.directory,
        config={"export_identity": config.teacher.identity},
    )
    mix = ArtifactStep.adopt(f"dspark/mixes/{config.data.manifest_sha256}", _RECIPE_VERSION, config.data.directory)
    inputs = (config.teacher.identity, config.teacher.manifest_sha256, config.data.manifest_sha256, recipe)
    data_key = hashlib.sha256(canonical_json(inputs).encode()).hexdigest()
    draft_key = hashlib.sha256(canonical_json((data_key, SPECULATORS_REVISION, VLLM_SOURCE_COMMIT)).encode()).hexdigest()
    prepared = apply(
        f"dspark/data/{family}/{data_key}",
        prepare_data,
        version=_RECIPE_VERSION,
        output=OUT,
        teacher=teacher,
        teacher_sha256=config.teacher.manifest_sha256,
        mix=mix,
        mix_sha256=config.data.manifest_sha256,
        recipe=recipe,
    )

    def build_config(ctx: StepContext) -> dict:
        return {
            "output": ctx.output_path,
            "teacher": ctx.artifact_path(teacher),
            "teacher_identity": config.teacher.identity,
            "teacher_sha256": config.teacher.manifest_sha256,
            "prepared": ctx.artifact_path(prepared),
            "mix_sha256": config.data.manifest_sha256,
            "recipe": recipe,
            "runtime": ctx.runtime_arg("runtime"),
        }

    draft = ArtifactStep(
        name=f"checkpoints/dspark/{family}/{draft_key}",
        version=_RECIPE_VERSION,
        artifact_type=Artifact,
        run=lambda kwargs: train_and_export(**kwargs),
        build_config=build_config,
        deps=(teacher, prepared),
        runtime_args={"runtime": runtime},
    )
    return prepared, draft


@click.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--stage", type=click.Choice(("plan", "preflight", "prepare", "train")), default="plan")
@click.option("--runtime", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(config: Path, stage: str, runtime: Path | None) -> None:
    """Build the offline DSpark artifact from a typed JSON recipe."""

    if not Path(marin_prefix()).is_absolute():
        raise click.UsageError("Set MARIN_PREFIX to an absolute artifact directory on the execution host")
    recipe = draccus.decode(OfflineDSpark, json.loads(config.read_text()))
    allocation = draccus.decode(LocalGpuRuntime, json.loads(runtime.read_text())) if runtime is not None else None
    if stage == "train" and allocation is None:
        raise click.UsageError("--stage train requires --runtime for an already allocated GPU host")
    prepared, draft = artifacts(recipe, allocation)
    if stage == "preflight":
        settings = recipe.training_recipe()
        decoder = inspect_teacher(Path(recipe.teacher.directory), recipe.teacher.manifest_sha256, settings)
        train, validation = inspect_mix(Path(recipe.data.directory), recipe.data.manifest_sha256)
        click.echo(
            json.dumps(
                {
                    "decoder": decoder,
                    "extraction_taps": settings.extraction_taps,
                    "train_shards": len(train),
                    "validation_shards": len(validation),
                },
                indent=2,
            )
        )
        return
    terminal = prepared if stage == "prepare" else draft
    StepRunner().run([lower(terminal)], dry_run=stage == "plan")
    if stage == "train":
        click.echo(str(Path(draft.path()) / "hf"))


if __name__ == "__main__":
    main()
