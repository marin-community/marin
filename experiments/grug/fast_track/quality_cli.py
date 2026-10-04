# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a raw pool, select a fixed token budget, and start a quality run."""

import hashlib
import importlib
import json
import math
from collections.abc import Callable
from enum import StrEnum

import click
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name

from experiments.grug.fast_track.corpus_sample import RawCorpusPool
from experiments.grug.fast_track.label_exclusion import read_label_exclusion
from experiments.grug.fast_track.launch import (
    H100_LADDER_SIZES,
    MatchMode,
    ThroughputResult,
    build_h100_ladder_run,
    resolve_h100_ladder_budget,
)
from experiments.grug.fast_track.quality import DocumentQualityScorer
from experiments.grug.fast_track.quality_features import PreparedQualityPool
from experiments.grug.fast_track.quality_pipeline import (
    QUALITY_FRACTION,
    QualityData,
    QualitySpec,
    QualityTrainingSource,
    SelectionMethod,
    build_quality_data,
    build_quality_features,
    build_ridge_scored_pool,
    build_scored_pool,
)


class QualityStage(StrEnum):
    FEATURES = "features"
    SELECT = "select"
    TRAIN = "train"


def _factory(path: str) -> Callable[[], DocumentQualityScorer]:
    module_name, separator, name = path.partition(":")
    if not separator or not module_name or not name:
        raise click.BadParameter("use module:factory syntax")
    value = getattr(importlib.import_module(module_name), name)
    if not callable(value):
        raise click.BadParameter("scorer factory must be callable")
    return value


def _identity(value: str) -> dict[str, str | int | float | bool]:
    try:
        identity = json.loads(value)
    except json.JSONDecodeError as exc:
        raise click.BadParameter("classifier identity must be JSON") from exc
    if not isinstance(identity, dict):
        raise click.BadParameter("classifier identity must be a JSON object")
    return identity


def _adopt_path(path: str, *, name: str, kind: type[Artifact]) -> ArtifactStep:
    digest = hashlib.sha256(path.encode()).hexdigest()[:20]
    artifact_name = f"fast-track/{name}/{digest}"
    version = resolve_version(artifact_name, None)
    return ArtifactStep.adopt(user_namespaced_name(artifact_name, version), version, path, kind=kind)


@click.command()
@click.option("--raw-pool", required=True, help="RawCorpusPool artifact path.")
@click.option("--scorer-factory", help="Worker-loaded document scorer factory as module:callable.")
@click.option("--classifier-identity", help="Stable classifier identity as JSON.")
@click.option("--ridge-head-artifact", help="Fitted ridge artifact. Its labels and Harrier pins stay paired.")
@click.option("--incumbent-scorer-factory", help="Optional worker-loaded incumbent scorer factory.")
@click.option("--incumbent-identity", help="Stable incumbent identity as JSON.")
@click.option("--label-exclusion-manifest", help="Frozen label exclusion manifest path.")
@click.option("--prepared-features", help="Prepared Harrier feature pool artifact path for a generic scorer.")
@click.option("--run-id", help="Run identifier for training.")
@click.option("--size", type=click.Choice(H100_LADDER_SIZES), default="d512", show_default=True)
@click.option(
    "--training-tokens", type=click.IntRange(min=1), help="Override the rung token budget for preparation only."
)
@click.option(
    "--selection-method",
    type=click.Choice([item.value for item in SelectionMethod]),
    default=SelectionMethod.CANDIDATE.value,
    show_default=True,
)
@click.option("--tie-seed", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--seed", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--data-seed", type=click.IntRange(min=0), default=0, show_default=True)
@click.option(
    "--stage",
    type=click.Choice([stage.value for stage in QualityStage]),
    default=QualityStage.TRAIN.value,
    show_default=True,
    help="Artifact graph boundary: features, select, or train.",
)
@build_options
def main(
    raw_pool: str,
    scorer_factory: str | None,
    classifier_identity: str | None,
    ridge_head_artifact: str | None,
    incumbent_scorer_factory: str | None,
    incumbent_identity: str | None,
    label_exclusion_manifest: str | None,
    prepared_features: str | None,
    run_id: str | None,
    size: str,
    training_tokens: int | None,
    selection_method: str,
    tie_seed: int,
    seed: int,
    data_seed: int,
    stage: str,
) -> ArtifactStep[QualityData] | ArtifactStep[ThroughputResult] | ArtifactStep[PreparedQualityPool]:
    selected_stage = QualityStage(stage)
    rung_tokens = resolve_h100_ladder_budget(
        size=size,
        dense=True,
        match=MatchMode.DATA,
        num_steps=None,
        batch_size=None,
    ).token_count
    if selected_stage is QualityStage.TRAIN and training_tokens is not None and training_tokens != rung_tokens:
        raise click.UsageError("--training-tokens must equal the resolved rung budget when training")
    selected_tokens = (
        training_tokens if selected_stage is not QualityStage.TRAIN and training_tokens is not None else rung_tokens
    )
    raw_step = _adopt_path(raw_pool, name="raw-pool", kind=RawCorpusPool)
    token_cap = math.ceil(selected_tokens / QUALITY_FRACTION)
    if selected_stage is QualityStage.FEATURES:
        if any(
            (
                scorer_factory,
                classifier_identity,
                ridge_head_artifact,
                incumbent_scorer_factory,
                incumbent_identity,
                label_exclusion_manifest,
                prepared_features,
                run_id,
            )
        ):
            raise click.UsageError("--stage features cannot combine with scoring or training options")
        return build_quality_features(raw_step, token_budget=token_cap)
    if ridge_head_artifact is not None:
        if any((scorer_factory, classifier_identity, label_exclusion_manifest, prepared_features)):
            raise click.UsageError("--ridge-head-artifact supplies its scorer, labels, and features")
        if incumbent_scorer_factory is not None or incumbent_identity is not None:
            raise click.UsageError("--ridge-head-artifact cannot combine with an incumbent scorer")
        scored = build_ridge_scored_pool(
            raw_step,
            head_artifact_path=ridge_head_artifact,
            token_budget=token_cap,
        )
    else:
        if scorer_factory is None or classifier_identity is None or label_exclusion_manifest is None:
            raise click.UsageError(
                "generic scoring requires --scorer-factory, --classifier-identity, and --label-exclusion-manifest"
            )
        identity = _identity(classifier_identity)
        incumbent_factory = _factory(incumbent_scorer_factory) if incumbent_scorer_factory else None
        incumbent_id = _identity(incumbent_identity) if incumbent_identity else None
        if (incumbent_factory is None) != (incumbent_id is None):
            raise click.UsageError("--incumbent-scorer-factory and --incumbent-identity must appear together")
        label_exclusion = read_label_exclusion(label_exclusion_manifest)
        feature_step = None
        if prepared_features is not None:
            feature_step = _adopt_path(prepared_features, name="quality-features", kind=PreparedQualityPool)
        scored = build_scored_pool(
            raw_step,
            scorer_factory=_factory(scorer_factory),
            classifier_identity=identity,
            label_exclusion=label_exclusion,
            token_budget=token_cap,
            prepared_features=feature_step,
            incumbent_scorer_factory=incumbent_factory,
            incumbent_identity=incumbent_id,
        )
    selection = build_quality_data(QualitySpec(scored, selected_tokens, SelectionMethod(selection_method), tie_seed))
    if selected_stage is QualityStage.SELECT:
        return selection
    if run_id is None or not run_id.strip():
        raise click.UsageError("--run-id is required when training")
    return build_h100_ladder_run(
        run_id=run_id,
        size=size,
        dense=True,
        training_source=QualityTrainingSource(selection, selected_tokens),
        seed=seed,
        data_seed=data_seed,
    )


if __name__ == "__main__":
    main()
