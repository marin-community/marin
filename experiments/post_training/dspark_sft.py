# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a completed Marin SFT export and a frozen chat mix to offline DSpark."""

import json
import os
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

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
    require_supported_vllm_release,
    train_and_export,
)
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.cross_region import is_cross_region_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.heuristic import HERO_MODEL
from experiments.models import ModelConfig, download_model
from experiments.post_training.drafting import mochi
from experiments.post_training.drafting.mochi_mix import scaled_mix, smoke_mix


class TeacherFamily(StrEnum):
    SNOWBALL = "snowball"
    MOCHI = "mochi"


_SNOWBALL = SnowballConfig()
_TARGETS = MappingProxyType(
    {
        TeacherFamily.SNOWBALL: (
            GrugShape.from_hf_config(_SNOWBALL.to_hf_config(_SNOWBALL.vocab_size).to_dict()),
            (2, 12, 22),
        ),
        TeacherFamily.MOCHI: (
            GrugShape.from_hf_config(HERO_MODEL.to_hf_config(HERO_MODEL.vocab_size).to_dict()),
            (2, 24, 45),
        ),
    }
)
_RECIPE_VERSION = "2026.11.22.4"
_MOCHI_AGENTIC_PRESET = "mochi-agentic"
_MOCHI_SCALE_PRESET = "mochi-agentic-50m"


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
    mochi_sample: mochi.Sample = mochi.Sample.ALL
    mochi_draft_size: mochi.DraftSize = mochi.DraftSize.ONE_LAYER

    def training_recipe(self) -> DraftRecipe:
        shape, taps = _TARGETS[self.teacher.family]
        layers, intermediate_size = (
            self.mochi_draft_size.architecture if self.teacher.family is TeacherFamily.MOCHI else (3, None)
        )
        return DraftRecipe(
            data=DraftDataConfig(
                shape=shape,
                mask_token_id=128004,
                max_sequence_length=self.max_sequence_length,
                draft_layers=layers,
                draft_intermediate_size=intermediate_size,
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

    if config.teacher.family is TeacherFamily.MOCHI:
        draft = mochi.checkpoint_step(
            teacher,
            config.teacher.relative_path,
            prepared,
            mochi.cached_recipe(recipe, config.mochi_sample),
            _RECIPE_VERSION,
            config.mochi_sample,
        )
    else:
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


def _resolve_recipe(
    config: str | None,
    preset: str | None,
    data: str | None,
    prefix: str,
    *,
    steps: int | None = None,
    draft_size: mochi.DraftSize | None = None,
) -> tuple[OfflineDSpark, ArtifactStep]:
    if config is not None:
        if preset is not None or data is not None or steps is not None or draft_size is not None:
            raise click.UsageError("Use --config or --preset")
        recipe = draccus.decode(OfflineDSpark, json.loads(StoragePath(config).read_text()))
        return recipe, _adopt_input(recipe.data)
    if preset in (_MOCHI_AGENTIC_PRESET, _MOCHI_SCALE_PRESET):
        scale = preset == _MOCHI_SCALE_PRESET
        if data is not None:
            mix = _adopt_input(data)
        else:
            mix = scaled_mix() if scale else smoke_mix()
        sample = mochi.Sample.SMOKE if not scale and data is None and steps is None else mochi.Sample.ALL
        step_budget = 15_000 if scale else 4_000
        if sample is mochi.Sample.SMOKE:
            step_budget = 3
        recipe = OfflineDSpark(
            teacher=MOCHI_STEP_400407,
            data=mix.path(prefix),
            max_steps=steps if steps is not None else step_budget,
            teacher_replicas=8,
            trainer_ranks=1,
            max_sequence_length=8192,
            max_anchors=1024,
            mochi_sample=sample,
        )
        if draft_size is not None:
            recipe = replace(recipe, mochi_draft_size=draft_size)
        return recipe, mix
    raise click.UsageError("Specify --config or choose a --preset")


def _preflight(recipe: OfflineDSpark, teacher: ArtifactStep, prefix: str) -> dict:
    settings = recipe.training_recipe()
    teacher_root = StoragePath(teacher.path(prefix)) / recipe.teacher.relative_path
    if recipe.teacher.source.startswith("hf://") and not teacher_root.isdir():
        raise click.UsageError("Run the prepare stage to stage the pinned Hub teacher in Marin storage")
    tokenizer = _hub_model(recipe.teacher.tokenizer_source) if recipe.teacher.tokenizer_source else None
    tokenizer_ref = f"{tokenizer.hf_repo_id}@{tokenizer.hf_revision}" if tokenizer else None
    decoder = inspect_teacher(teacher_root, settings.data, tokenizer_source=tokenizer_ref)
    train, validation = inspect_mix(StoragePath(recipe.data))
    total_gpus = settings.teacher_replicas
    if recipe.teacher.family is TeacherFamily.SNOWBALL:
        total_gpus += settings.trainer_ranks
    return {
        "decoder": decoder,
        "extraction_taps": settings.extraction_taps,
        "train_shards": len(train),
        "validation_shards": len(validation),
        "gpus": total_gpus,
        "trainer_ranks": settings.trainer_ranks,
    }


def _validate_training_locations(recipe: OfflineDSpark, prefix: str) -> None:
    locations = [recipe.data, prefix]
    if not recipe.teacher.source.startswith("hf://"):
        locations.append(recipe.teacher.source)
    for location in locations:
        if StoragePath(location).scheme not in ("gs", "s3"):
            raise click.UsageError(f"Iris training requires a shared gs:// or s3:// location, got {location}")
        if is_cross_region_url(location):
            raise click.UsageError(f"Iris training requires regional inputs and output, got {location}")


@click.command()
@click.option("--config", help="Path or object storage URI of the DSpark JSON recipe.")
@click.option("--preset", type=click.Choice((_MOCHI_AGENTIC_PRESET, _MOCHI_SCALE_PRESET)))
@click.option("--data", help="Replace the named preset's smoke mix with a completed train/validation chat mix.")
@click.option("--steps", type=click.IntRange(min=1), help="Optimizer step cap for the named preset.")
@click.option(
    "--draft-size",
    type=click.Choice([size.value for size in mochi.DraftSize]),
    help="Mochi draft architecture for the named preset.",
)
@click.option("--stage", type=click.Choice(("plan", "preflight", "prepare", "train")), default="plan")
def main(
    config: str | None,
    preset: str | None,
    data: str | None,
    steps: int | None,
    draft_size: str | None,
    stage: str,
) -> None:
    """Plan and preflight offline DSpark or execute it in one Iris GPU job."""

    if stage == "train" and "IRIS_TASK_ID" not in os.environ:
        raise click.UsageError("Submit the train stage through Iris")
    configure_coreweave_s3()
    prefix = marin_prefix()
    size = mochi.DraftSize(draft_size) if draft_size is not None else None
    recipe, mix = _resolve_recipe(config, preset, data, prefix, steps=steps, draft_size=size)
    if stage == "train":
        _validate_training_locations(recipe, prefix)
    teacher = _teacher_artifact(recipe.teacher)
    prepared, draft = artifacts(recipe, teacher=teacher, mix=mix)
    if stage == "preflight":
        click.echo(json.dumps(_preflight(recipe, teacher, prefix), indent=2))
        return
    if stage == "train" and recipe.teacher.family is TeacherFamily.MOCHI:
        if mochi.train_mochi(
            teacher=str(StoragePath(teacher.path(prefix)) / recipe.teacher.relative_path),
            prepared=prepared,
            draft=draft,
            recipe=mochi.cached_recipe(recipe.training_recipe(), recipe.mochi_sample),
            prefix=prefix,
            sample=recipe.mochi_sample,
        ):
            click.echo(str(StoragePath(draft.path(prefix)) / HF_DRAFT_SUBDIRECTORY))
        return
    terminal = prepared if stage == "prepare" else draft
    StepRunner().run([lower(terminal)], dry_run=stage == "plan")
    if stage == "train":
        click.echo(str(StoragePath(draft.path(prefix)) / HF_DRAFT_SUBDIRECTORY))


if __name__ == "__main__":
    main()
