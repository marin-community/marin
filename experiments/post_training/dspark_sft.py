# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a completed Marin SFT export and a frozen chat mix to offline DSpark."""

import json
import os
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import click
import draccus
from marin.execution.artifact import Artifact, is_mutable_version, read_record
from marin.execution.lazy import OUT, ArtifactStep, apply, artifact_identity, lower
from marin.execution.step_runner import StepRunner, check_cache
from marin.training.drafting.dspark_data import (
    DraftDataConfig,
    GrugShape,
    inspect_mix,
    inspect_teacher,
    prepare_data,
)
from marin.training.drafting.dspark_distillation import (
    HF_DRAFT_SUBDIRECTORY,
    DraftRecipe,
    DSparkExport,
    require_qualified_teacher,
    require_supported_vllm_release,
    train_and_export,
)
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.cross_region import is_cross_region_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.models import ModelConfig, download_model


class TeacherFamily(StrEnum):
    SNOWBALL = "snowball"
    MOCHI = "mochi"


_GRUG_HEAD_DIM = 128
_GRUG_VOCAB_SIZE = 128256
_SHAPES = {
    TeacherFamily.SNOWBALL: GrugShape(2560, 26, 20, 5, _GRUG_HEAD_DIM, _GRUG_VOCAB_SIZE),
    TeacherFamily.MOCHI: GrugShape(6144, 48, 48, 12, _GRUG_HEAD_DIM, _GRUG_VOCAB_SIZE, sconv=True),
}
_FEATURE_TAPS = {
    TeacherFamily.SNOWBALL: (2, 12, 22),
    TeacherFamily.MOCHI: (2, 24, 45),
}
_RECIPE_VERSION = "2026.11.22.3"


@dataclass(frozen=True)
class CompletedSFT:
    """A Hub commit or a completed Marin producer and its HF subdirectory."""

    family: TeacherFamily
    source: str
    fingerprint: str | None = None
    relative_path: str = ""


@dataclass(frozen=True)
class TrainingMix:
    """Versioned normalized Marin chat Parquet files in storage."""

    directory: str
    fingerprint: str


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


def _completed_artifact(path: str, fingerprint: str) -> ArtifactStep:
    if StoragePath(path).scheme not in ("gs", "s3") and not Path(path).is_absolute():
        raise ValueError(f"Expected a gs:// or s3:// artifact root or an absolute local path, got {path}")
    record = read_record(path)
    if record is None or not check_cache(path):
        raise ValueError(f"No completed Marin artifact at {path}")
    if not record.name or not record.version or is_mutable_version(record.version):
        raise ValueError(f"Expected a fixed-version producer at {path}")
    if record.source is not None or record.fingerprint_payload is None:
        raise ValueError(f"Use the completed producer's artifact directory at {path}")
    if not fingerprint or record.fingerprint != fingerprint:
        raise ValueError(f"Artifact fingerprint at {path} is {record.fingerprint}, expected {fingerprint}")
    payload = json.loads(record.fingerprint_payload)

    def missing_producer(_: object) -> None:
        raise ValueError(f"The completed producer at {path} is no longer available")

    return ArtifactStep(
        name=record.name,
        version=record.version,
        artifact_type=Artifact,
        build_config=lambda _: payload,
        run=missing_producer,
        override_path=path,
        expected_fingerprint=fingerprint,
    )


def _teacher_artifact(teacher: CompletedSFT) -> ArtifactStep:
    if ".." in Path(teacher.relative_path).parts or Path(teacher.relative_path).is_absolute():
        raise ValueError("The HF export must be below its artifact root")
    if teacher.source.startswith("hf://"):
        repo, _, revision = teacher.source.removeprefix("hf://").partition("@")
        if not re.fullmatch(r"[0-9a-f]{40}", revision) or repo.count("/") != 1:
            raise ValueError("Specify the Hub teacher as hf://org/model@<full commit>")
        return download_model(ModelConfig(hf_repo_id=repo, hf_revision=revision))
    if teacher.fingerprint is None:
        raise ValueError("Provide the producer fingerprint from the Marin teacher's .artifact.json")
    return _completed_artifact(teacher.source, teacher.fingerprint)


def artifacts(config: OfflineDSpark) -> tuple[ArtifactStep, ArtifactStep[DSparkExport]]:
    """Bind the prepared dataset and published DSpark draft artifact."""

    require_supported_vllm_release()
    family = config.teacher.family.value
    recipe = config.training_recipe()
    teacher = _teacher_artifact(config.teacher)
    mix = _completed_artifact(config.data.directory, config.data.fingerprint)
    prepared = apply(
        f"dspark/data/{family}",
        prepare_data,
        version=_RECIPE_VERSION,
        output=OUT,
        teacher=teacher,
        teacher_subdirectory=config.teacher.relative_path,
        mix=mix,
        config=recipe.data,
    )
    fingerprint = prepared.fingerprint()
    prepared = replace(prepared, name=f"{prepared.name}/{fingerprint}", expected_fingerprint=fingerprint)

    draft = apply(
        f"checkpoints/dspark/{family}",
        train_and_export,
        version=_RECIPE_VERSION,
        artifact_type=DSparkExport,
        output=OUT,
        teacher=teacher,
        teacher_subdirectory=config.teacher.relative_path,
        teacher_identity=artifact_identity(teacher),
        prepared=prepared,
        recipe=recipe,
    )
    fingerprint = draft.fingerprint()
    return prepared, replace(draft, name=f"{draft.name}/{fingerprint}", expected_fingerprint=fingerprint)


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
    prefix = marin_prefix()
    mix_root = StoragePath(recipe.data.directory)
    if stage == "train":
        require_qualified_teacher(settings)
        locations = [mix_root, StoragePath(prefix)]
        if not recipe.teacher.source.startswith("hf://"):
            locations.append(StoragePath(recipe.teacher.source))
        for location in locations:
            if location.scheme not in ("gs", "s3"):
                raise click.UsageError(f"Iris training requires a shared gs:// or s3:// location, got {location}")
            if is_cross_region_url(str(location)):
                raise click.UsageError(f"Iris training requires regional inputs and output, got {location}")
    prepared, draft = artifacts(recipe)
    if stage == "preflight":
        teacher = prepared.deps[0]
        if not check_cache(teacher.path(prefix)):
            raise click.UsageError("Run the prepare stage to stage the pinned Hub teacher in Marin storage")
        teacher_root = StoragePath(teacher.path(prefix)) / recipe.teacher.relative_path
        decoder = inspect_teacher(teacher_root, settings.data)
        train, validation = inspect_mix(mix_root)
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
        click.echo(str(StoragePath(draft.path(prefix)) / HF_DRAFT_SUBDIRECTORY))


if __name__ == "__main__":
    main()
