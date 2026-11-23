# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a completed Marin SFT export and a frozen chat mix to offline DSpark."""

import hashlib
import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import click
import draccus
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import OUT, ArtifactStep, apply, lower
from marin.execution.step_runner import StepRunner
from marin.training.drafting.dspark_data import (
    DraftDataConfig,
    DraftRecipe,
    GrugShape,
    inspect_mix,
    inspect_teacher,
    prepare_data,
    require_sha256,
)
from marin.training.drafting.dspark_distillation import (
    SPECULATORS_REVISION,
    VLLM_SOURCE_COMMIT,
    TrainingProvenance,
    require_qualified_teacher,
    require_supported_vllm_release,
    train_and_export,
)
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.cross_region import is_cross_region_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath


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
_RECIPE_VERSION = "2026.11.22.2"


@dataclass(frozen=True)
class CompletedSFT:
    """Identity and storage directory of a completed HF SFT export."""

    family: TeacherFamily
    identity: str
    directory: str
    manifest_uri: str
    manifest_sha256: str


@dataclass(frozen=True)
class TrainingMix:
    """Versioned normalized Marin chat Parquet files in storage."""

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
            data=DraftDataConfig(
                shape=_SHAPES[self.teacher.family],
                mask_token_id=128004,
                max_sequence_length=self.max_sequence_length,
            ),
            feature_taps=_FEATURE_TAPS[self.teacher.family],
            max_anchors=self.max_anchors,
            max_steps=self.max_steps,
            trainer_ranks=self.trainer_ranks,
            teacher_replicas=self.teacher_replicas,
            cross_entropy_weight=0.1,
            total_variation_weight=0.9,
            optimizer="muon",
            lr_schedule="linear",
            seed=self.seed,
        )


def artifacts(config: OfflineDSpark) -> tuple[ArtifactStep, ArtifactStep]:
    """Bind the prepared dataset and published DSpark draft artifact."""

    require_supported_vllm_release()
    if not config.teacher.identity.strip():
        raise ValueError("Provide the completed SFT export's provenance identity")
    for name, path, digest in (
        ("teacher", config.teacher.directory, config.teacher.manifest_sha256),
        ("mix", config.data.directory, config.data.manifest_sha256),
    ):
        if StoragePath(path).scheme not in ("gs", "s3") and not Path(path).is_absolute():
            raise ValueError(f"The {name} directory must be a gs:// or s3:// URI or an absolute local path")
        require_sha256(digest)

    family = config.teacher.family.value
    recipe = config.training_recipe()
    teacher = ArtifactStep.adopt(
        f"dspark/teachers/{family}/{config.teacher.manifest_sha256}",
        _RECIPE_VERSION,
        config.teacher.directory,
        config={"export_identity": config.teacher.identity},
    )
    mix = ArtifactStep.adopt(f"dspark/mixes/{config.data.manifest_sha256}", _RECIPE_VERSION, config.data.directory)
    provenance = TrainingProvenance(config.teacher.identity, config.teacher.manifest_sha256, config.data.manifest_sha256)
    data_key = hashlib.sha256(canonical_json((provenance, recipe.data)).encode()).hexdigest()
    draft_key = hashlib.sha256(
        canonical_json((data_key, recipe, SPECULATORS_REVISION, VLLM_SOURCE_COMMIT)).encode()
    ).hexdigest()
    prepared = apply(
        f"dspark/data/{family}/{data_key}",
        prepare_data,
        version=_RECIPE_VERSION,
        output=OUT,
        teacher=teacher,
        teacher_manifest=config.teacher.manifest_uri,
        teacher_sha256=config.teacher.manifest_sha256,
        mix=mix,
        mix_sha256=config.data.manifest_sha256,
        config=recipe.data,
    )

    draft = apply(
        f"checkpoints/dspark/{family}/{draft_key}",
        train_and_export,
        version=_RECIPE_VERSION,
        output=OUT,
        teacher=teacher,
        teacher_manifest=config.teacher.manifest_uri,
        provenance=provenance,
        prepared=prepared,
        recipe=recipe,
    )
    return prepared, draft


@click.command()
@click.option("--config", required=True, help="Path or object storage URI of the DSpark JSON recipe.")
@click.option("--stage", type=click.Choice(("plan", "preflight", "prepare", "train")), default="plan")
def main(config: str, stage: str) -> None:
    """Plan and preflight offline DSpark or execute it in one Iris GPU job."""

    if stage == "train" and "IRIS_TASK_ID" not in os.environ:
        raise click.UsageError("Run the train stage with iris job run on one GPU worker")
    configure_coreweave_s3()
    recipe = draccus.decode(OfflineDSpark, json.loads(StoragePath(config).read_text()))
    settings = recipe.training_recipe()
    prepared, draft = artifacts(recipe)
    prefix = marin_prefix()
    teacher_root = StoragePath(recipe.teacher.directory)
    manifest = StoragePath(recipe.teacher.manifest_uri)
    mix_root = StoragePath(recipe.data.directory)
    if stage == "train":
        require_qualified_teacher(settings)
        for location in (teacher_root, manifest, mix_root, StoragePath(prefix)):
            if location.scheme not in ("gs", "s3"):
                raise click.UsageError(f"Iris training requires a shared gs:// or s3:// location, got {location}")
            if is_cross_region_url(str(location)):
                raise click.UsageError(f"Iris training requires regional inputs and output, got {location}")
    if stage == "preflight":
        decoder = inspect_teacher(teacher_root, manifest, recipe.teacher.manifest_sha256, settings.data)
        train, validation = inspect_mix(mix_root, recipe.data.manifest_sha256)
        click.echo(
            json.dumps(
                {
                    "decoder": decoder,
                    "extraction_taps": settings.extraction_taps,
                    "train_shards": len(train),
                    "validation_shards": len(validation),
                    "gpus": settings.teacher_replicas + settings.trainer_ranks,
                    "training_supported": not settings.data.shape.sconv,
                },
                indent=2,
            )
        )
        return
    terminal = prepared if stage == "prepare" else draft
    StepRunner().run([lower(terminal)], dry_run=stage == "plan")
    if stage == "train":
        click.echo(str(StoragePath(draft.path(prefix)) / "hf"))


if __name__ == "__main__":
    main()
