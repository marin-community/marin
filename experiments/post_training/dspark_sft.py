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
from levanter.models.snowball import SnowballConfig
from marin.execution.artifact import is_mutable_version
from marin.execution.lazy import OUT, ArtifactStep, apply, artifact_identity, lower
from marin.execution.step_runner import StepRunner
from marin.training.drafting.dspark_data import (
    DraftDataConfig,
    GrugShape,
    inspect_mix,
    inspect_teacher,
    prepare_data,
)
from marin.training.drafting.dspark_distillation import (
    DEFAULT_DRAFT_SEED,
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

from experiments.grug.moe_hero_ep.heuristic import HERO_MODEL
from experiments.models import ModelConfig, download_model
from experiments.post_training.drafting.mochi_mix import smoke_mix


class TeacherFamily(StrEnum):
    SNOWBALL = "snowball"
    MOCHI = "mochi"


_SNOWBALL = SnowballConfig()
_TARGETS = {
    TeacherFamily.SNOWBALL: (
        GrugShape.from_hf_config(_SNOWBALL.to_hf_config(_SNOWBALL.vocab_size).to_dict()),
        (2, 12, 22),
    ),
    TeacherFamily.MOCHI: (
        GrugShape.from_hf_config(HERO_MODEL.to_hf_config(HERO_MODEL.vocab_size).to_dict()),
        (2, 24, 45),
    ),
}
_RECIPE_VERSION = "2026.11.22.4"


@dataclass(frozen=True)
class CompletedSFT:
    """A Hub commit or a completed Marin producer and its HF subdirectory."""

    family: TeacherFamily
    source: str
    relative_path: str = ""
    tokenizer_source: str | None = None
    artifact_version: str | None = None


MOCHI_STEP_400407 = CompletedSFT(
    family=TeacherFamily.MOCHI,
    source=(
        "s3://marin-us-east-02a/marin/exports/mochi-sft/" "hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1"
    ),
    tokenizer_source="hf://marin-community/marin-tokenizer-tools@3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09",
    artifact_version="2026.11.13",
)


@dataclass(frozen=True)
class OfflineDSpark:
    teacher: CompletedSFT
    data: str
    max_steps: int
    teacher_replicas: int
    trainer_ranks: int
    max_sequence_length: int = 32768
    max_anchors: int = 12288
    seed: int = DEFAULT_DRAFT_SEED

    def training_recipe(self) -> DraftRecipe:
        shape, taps = _TARGETS[self.teacher.family]
        mochi = self.teacher.family is TeacherFamily.MOCHI
        return DraftRecipe(
            data=DraftDataConfig(
                shape=shape,
                mask_token_id=128004,
                max_sequence_length=self.max_sequence_length,
                draft_layers=2 if mochi else 3,
                draft_intermediate_size=shape.hidden_size if mochi else None,
            ),
            feature_taps=taps,
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


def _adopt_input(source: str, *, version: str | None = None) -> ArtifactStep:
    path = StoragePath(source)
    if path.scheme not in ("gs", "s3") and not Path(source).is_absolute():
        raise ValueError(f"Expected a gs:// or s3:// artifact root or an absolute local path, got {source}")
    if is_mutable_version(path.name) or (version is not None and is_mutable_version(version)):
        raise ValueError(f"Use a completed artifact from a fixed-version directory, got {source}")
    key = path.key if version is not None else path.parent.key
    name = "/".join(filter(None, ("dspark/inputs", path.scheme or "local", path.bucket, key)))
    return ArtifactStep.adopt(name, version or path.name, str(path))


def _teacher_artifact(teacher: CompletedSFT) -> ArtifactStep:
    if ".." in Path(teacher.relative_path).parts or Path(teacher.relative_path).is_absolute():
        raise ValueError("The HF export must be below its artifact root")
    if teacher.source.startswith("hf://"):
        return download_model(_hub_model(teacher.source))
    return _adopt_input(teacher.source, version=teacher.artifact_version)


def _hub_model(source: str) -> ModelConfig:
    repo, _, revision = source.removeprefix("hf://").partition("@")
    if not source.startswith("hf://") or not re.fullmatch(r"[0-9a-f]{40}", revision) or repo.count("/") != 1:
        raise ValueError("Specify the Hub model or tokenizer as hf://org/repo@<full commit>")
    return ModelConfig(hf_repo_id=repo, hf_revision=revision)


def artifacts(
    config: OfflineDSpark, *, teacher: ArtifactStep, mix: ArtifactStep
) -> tuple[ArtifactStep, ArtifactStep[DSparkExport]]:
    """Bind the prepared dataset and published DSpark draft artifact."""

    require_supported_vllm_release()
    family = config.teacher.family.value
    recipe = config.training_recipe()
    tokenizer = download_model(_hub_model(config.teacher.tokenizer_source)) if config.teacher.tokenizer_source else None
    prepared = apply(
        f"dspark/data/{family}",
        prepare_data,
        version=_RECIPE_VERSION,
        output=OUT,
        teacher=teacher,
        teacher_subdirectory=config.teacher.relative_path,
        tokenizer_source=tokenizer,
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
        tokenizer_source=tokenizer,
        teacher_identity=artifact_identity(teacher),
        prepared=prepared,
        recipe=recipe,
    )
    fingerprint = draft.fingerprint()
    return prepared, replace(draft, name=f"{draft.name}/{fingerprint}", expected_fingerprint=fingerprint)


@click.command()
@click.option("--config", help="Path or object storage URI of the DSpark JSON recipe.")
@click.option("--preset", type=click.Choice(("mochi-agentic",)))
@click.option("--data", help="Replace the named preset's smoke mix with a completed train/validation chat mix.")
@click.option("--stage", type=click.Choice(("plan", "preflight", "prepare", "train")), default="plan")
def main(config: str | None, preset: str | None, data: str | None, stage: str) -> None:
    """Plan and preflight offline DSpark or execute it in one Iris GPU job."""

    if stage == "train" and "IRIS_TASK_ID" not in os.environ:
        raise click.UsageError("Run the train stage with iris job run on one GPU worker")
    configure_coreweave_s3()
    prefix = marin_prefix()
    if config is not None:
        if preset is not None or data is not None:
            raise click.UsageError("Use --config or --preset")
        recipe = draccus.decode(OfflineDSpark, json.loads(StoragePath(config).read_text()))
        mix = _adopt_input(recipe.data)
    elif preset == "mochi-agentic":
        mix = _adopt_input(data) if data is not None else smoke_mix()
        recipe = OfflineDSpark(
            teacher=MOCHI_STEP_400407,
            data=mix.path(prefix),
            max_steps=3 if data is None else 4_000,
            teacher_replicas=8,
            trainer_ranks=2,
            max_sequence_length=8192,
            max_anchors=1024,
        )
    else:
        raise click.UsageError("Specify --config or --preset mochi-agentic")
    settings = recipe.training_recipe()
    mix_root = StoragePath(recipe.data)
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
    teacher = _teacher_artifact(recipe.teacher)
    prepared, draft = artifacts(recipe, teacher=teacher, mix=mix)
    if stage == "preflight":
        teacher_root = StoragePath(teacher.path(prefix)) / recipe.teacher.relative_path
        if recipe.teacher.source.startswith("hf://") and not teacher_root.isdir():
            raise click.UsageError("Run the prepare stage to stage the pinned Hub teacher in Marin storage")
        tokenizer = _hub_model(recipe.teacher.tokenizer_source) if recipe.teacher.tokenizer_source else None
        tokenizer_ref = f"{tokenizer.hf_repo_id}@{tokenizer.hf_revision}" if tokenizer else None
        decoder = inspect_teacher(teacher_root, settings.data, tokenizer_source=tokenizer_ref)
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
