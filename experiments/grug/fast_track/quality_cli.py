# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a raw pool, select a fixed token budget, and start a quality run."""

import importlib
import json
from collections.abc import Callable
from enum import StrEnum
from typing import cast

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.fast_track.contracts import adopt_artifact_path
from experiments.grug.fast_track.corpus_sample import QUALITY_POOL_MULTIPLIER, RawCorpusPool
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
    QualityData,
    QualitySpec,
    QualityTrainingSource,
    ScoredPool,
    ScorerSpec,
    SelectionMethod,
    build_quality_data,
    build_quality_features,
    build_ridge_scored_pool,
    build_scored_pool,
)


class QualityStage(StrEnum):
    FEATURES = "features"
    SCORE = "score"
    SELECT = "select"
    TRAIN = "train"


def _factory(path: str) -> Callable[[], DocumentQualityScorer]:
    module_name, separator, name = path.partition(":")
    if not separator or not module_name or not name:
        raise click.BadParameter("use module:factory syntax")
    value = getattr(importlib.import_module(module_name), name)
    if not callable(value):
        raise click.BadParameter("scorer factory must be callable")
    return cast(Callable[[], DocumentQualityScorer], value)


def _identity(value: str) -> dict[str, str | int | float | bool]:
    try:
        identity = json.loads(value)
    except json.JSONDecodeError as exc:
        raise click.BadParameter("classifier identity must be JSON") from exc
    if not isinstance(identity, dict):
        raise click.BadParameter("classifier identity must be a JSON object")
    return identity


def _reject_options(options: dict[str, object], reason: str) -> None:
    given = [flag for flag, value in options.items() if value is not None]
    if given:
        raise click.UsageError(f"{', '.join(given)}: {reason}")


@click.command()
@click.option("--raw-pool", help="RawCorpusPool artifact path.")
@click.option(
    "--scored-pool",
    help=f"Reusable ScoredPool artifact path with at least {QUALITY_POOL_MULTIPLIER}x the requested training tokens.",
)
@click.option("--scorer-factory", help="Worker-loaded document scorer factory as module:callable.")
@click.option("--classifier-identity", help="Stable classifier identity as JSON.")
@click.option("--ridge-head-artifact", help="Fitted ridge artifact. Its labels and Harrier pins stay paired.")
@click.option("--incumbent-scorer-factory", help="Optional worker-loaded incumbent scorer factory.")
@click.option("--incumbent-identity", help="Stable incumbent identity as JSON.")
@click.option("--label-exclusion-manifest", help="Frozen label exclusion manifest path.")
@click.option("--prepared-features", help="Prepared Harrier feature pool artifact path for a generic scorer.")
@click.option("--run-id", help="Run identifier for training.")
@click.option("--size", type=click.Choice(H100_LADDER_SIZES), default="d512", show_default=True)
@click.option("--dense/--moe", default=False, show_default=True, help="Select the dense or MoE model.")
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
@click.option("--seed", type=click.IntRange(min=0), default=0, show_default=True, help="Model initialization seed.")
@click.option(
    "--data-seed",
    type=click.IntRange(min=0),
    default=None,
    help="Training data order seed. Omit to derive it from --seed.",
)
@click.option(
    "--stage",
    type=click.Choice([stage.value for stage in QualityStage]),
    default=QualityStage.TRAIN.value,
    show_default=True,
    help="Artifact graph boundary: features, score, select, or train.",
)
@build_options
def main(
    raw_pool: str | None,
    scored_pool: str | None,
    scorer_factory: str | None,
    classifier_identity: str | None,
    ridge_head_artifact: str | None,
    incumbent_scorer_factory: str | None,
    incumbent_identity: str | None,
    label_exclusion_manifest: str | None,
    prepared_features: str | None,
    run_id: str | None,
    size: str,
    dense: bool,
    training_tokens: int | None,
    selection_method: str,
    tie_seed: int,
    seed: int,
    data_seed: int | None,
    stage: str,
) -> (
    ArtifactStep[QualityData]
    | ArtifactStep[ThroughputResult]
    | ArtifactStep[PreparedQualityPool]
    | ArtifactStep[ScoredPool]
):
    selected_stage = QualityStage(stage)
    method = SelectionMethod(selection_method)
    rung_tokens = resolve_h100_ladder_budget(
        size=size,
        dense=dense,
        match=MatchMode.DATA,
        num_steps=None,
        batch_size=None,
    ).token_count
    if selected_stage is QualityStage.TRAIN and training_tokens is not None:
        raise click.UsageError("--training-tokens is only valid for preparation stages")
    selected_tokens = training_tokens if training_tokens is not None else rung_tokens
    if selected_stage is not QualityStage.TRAIN and run_id is not None:
        raise click.UsageError(f"--stage {selected_stage.value} cannot combine with --run-id")
    if (raw_pool is None) == (scored_pool is None):
        raise click.UsageError("provide exactly one of --raw-pool or --scored-pool")
    scoring_options = {
        "--scorer-factory": scorer_factory,
        "--classifier-identity": classifier_identity,
        "--ridge-head-artifact": ridge_head_artifact,
        "--incumbent-scorer-factory": incumbent_scorer_factory,
        "--incumbent-identity": incumbent_identity,
        "--label-exclusion-manifest": label_exclusion_manifest,
        "--prepared-features": prepared_features,
    }
    if scored_pool is not None:
        if selected_stage in (QualityStage.FEATURES, QualityStage.SCORE):
            raise click.UsageError("--scored-pool requires --stage select or --stage train")
        _reject_options(scoring_options, "--scored-pool already fixes the scorer, labels, and features")
        scored = adopt_artifact_path(scored_pool, kind=ScoredPool)
    else:
        assert raw_pool is not None
        raw_step = adopt_artifact_path(raw_pool, kind=RawCorpusPool)
        token_cap = selected_tokens * QUALITY_POOL_MULTIPLIER
        if selected_stage is QualityStage.FEATURES:
            _reject_options(scoring_options, "--stage features does not score")
            return build_quality_features(raw_step, token_budget=token_cap)
        if (
            selected_stage in (QualityStage.SELECT, QualityStage.TRAIN)
            and method is SelectionMethod.INCUMBENT
            and incumbent_scorer_factory is None
        ):
            raise click.UsageError("--selection-method incumbent requires --incumbent-scorer-factory")
        if ridge_head_artifact is not None:
            _reject_options(
                {flag: value for flag, value in scoring_options.items() if flag != "--ridge-head-artifact"},
                "--ridge-head-artifact supplies its scorer, labels, and features, and has no incumbent",
            )
            scored = build_ridge_scored_pool(raw_step, head_artifact_path=ridge_head_artifact, token_budget=token_cap)
        else:
            if scorer_factory is None or classifier_identity is None or label_exclusion_manifest is None:
                raise click.UsageError(
                    "generic scoring requires --scorer-factory, --classifier-identity, and --label-exclusion-manifest"
                )
            if (incumbent_scorer_factory is None) != (incumbent_identity is None):
                raise click.UsageError("--incumbent-scorer-factory and --incumbent-identity must appear together")
            incumbent = (
                ScorerSpec(_factory(incumbent_scorer_factory), _identity(incumbent_identity))
                if incumbent_scorer_factory is not None and incumbent_identity is not None
                else None
            )
            scored = build_scored_pool(
                raw_step,
                candidate=ScorerSpec(_factory(scorer_factory), _identity(classifier_identity)),
                incumbent=incumbent,
                label_exclusion=read_label_exclusion(label_exclusion_manifest),
                token_budget=token_cap,
                prepared_features=(
                    None
                    if prepared_features is None
                    else adopt_artifact_path(prepared_features, kind=PreparedQualityPool)
                ),
            )
    if selected_stage is QualityStage.SCORE:
        return scored
    selection = build_quality_data(QualitySpec(scored, selected_tokens, method, tie_seed))
    if selected_stage is QualityStage.SELECT:
        return selection
    if run_id is None or not run_id.strip():
        raise click.UsageError("--run-id is required when training")
    return build_h100_ladder_run(
        run_id=run_id,
        size=size,
        dense=dense,
        training_source=QualityTrainingSource(selection, selected_tokens),
        seed=seed,
        data_seed=data_seed,
    )


if __name__ == "__main__":
    main()
