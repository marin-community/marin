# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a bounded Hugging Face prefix or add it to a fast-track run."""

import hashlib

import click
from levanter.tokenizers import tokenizer_content_hash
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from pydantic import ValidationError

from experiments.grug.fast_track.add_dataset import (
    AddDatasetPreparationConfig,
    AddDatasetTrainingSource,
    add_dataset_cache_step,
)
from experiments.grug.fast_track.contracts import (
    AddDatasetConfig,
    AddDatasetSamplingPolicy,
    DatasetPrefix,
    PreparedAddDatasetCache,
    unique_token_sample_cap,
)
from experiments.grug.fast_track.hero_sample import HeroTrainingSource, PreparedHeroSample
from experiments.grug.fast_track.launch import (
    H100_LADDER_SIZES,
    V16384_TOKENIZER,
    MatchMode,
    TrainingSource,
    build_h100_ladder_run,
    maximum_h100_ladder_tokens,
    resolve_h100_ladder_budget,
)
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import TOTAL_TOKENS


@click.command()
@click.option("--run-id", default=None, help="Run identifier for artifact and W&B names.")
@click.option("--size", type=click.Choice(H100_LADDER_SIZES), default="d512", show_default=True)
@click.option("--dense/--moe", default=True, show_default=True, help="Select the dense or MoE model.")
@click.option("--match", type=click.Choice([mode.value for mode in MatchMode]), default="data", show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=None)
@click.option("--num-steps", type=click.IntRange(min=1), default=None)
@click.option("--seed", type=click.IntRange(min=0), default=0, show_default=True, help="Model initialization seed.")
@click.option("--data-seed", type=click.IntRange(min=0), default=0, show_default=True, help="Training data seed.")
@click.option("--repository", default=None, help="Hugging Face dataset repository.")
@click.option("--revision", default=None, help="Immutable Hugging Face commit hash.")
@click.option("--subset", default=None, help="Hugging Face dataset subset.")
@click.option("--split", default=None, help="Hugging Face split to stream.")
@click.option("--text-field", default=None, help="String field to tokenize.")
@click.option("--fraction", type=click.FloatRange(min=0, max=1, min_open=True, max_open=True), default=None)
@click.option(
    "--target-production-tokens",
    type=click.IntRange(min=1),
    default=TOTAL_TOKENS,
    show_default=True,
    help="Production token budget used to scale unique-token exposure.",
)
@click.option("--available-unique-tokens", type=click.IntRange(min=1), default=None)
@click.option("--max-rows", type=click.IntRange(min=1), default=None)
@click.option("--max-overshoot-tokens", type=click.IntRange(min=0), default=16_384, show_default=True)
@click.option("--prepare-token-cap", type=click.IntRange(min=1), default=None)
@click.option("--prepare-only", is_flag=True, help="Build only the bounded cache. Requires --prepare-token-cap.")
@click.option("--prepared-cache", default=None, help="Existing PreparedAddDatasetCache artifact path.")
@click.option("--baseline-artifact", default=None, help="Existing PreparedHeroSample artifact path.")
@build_options
def main(
    run_id: str | None,
    size: str,
    dense: bool,
    match: str,
    batch_size: int | None,
    num_steps: int | None,
    seed: int,
    data_seed: int,
    repository: str | None,
    revision: str | None,
    subset: str | None,
    split: str | None,
    text_field: str | None,
    fraction: float | None,
    target_production_tokens: int,
    available_unique_tokens: int | None,
    max_rows: int | None,
    max_overshoot_tokens: int,
    prepare_token_cap: int | None,
    prepare_only: bool,
    prepared_cache: str | None,
    baseline_artifact: str | None,
) -> ArtifactStep:
    if prepare_only and prepared_cache:
        raise click.UsageError("--prepare-only cannot be used with --prepared-cache")
    if prepare_only:
        if prepare_token_cap is None:
            raise click.UsageError("--prepare-token-cap is required with --prepare-only")
        prepared_token_cap = prepare_token_cap
        budget = None
        training_token_cap = None
    else:
        if not run_id or not run_id.strip():
            raise click.UsageError("--run-id is required for training")
        if fraction is None or available_unique_tokens is None:
            raise click.UsageError("training requires --fraction and --available-unique-tokens")
        if not baseline_artifact:
            raise click.UsageError("training requires --baseline-artifact from hero_sample_cli")
        budget = resolve_h100_ladder_budget(
            size=size,
            dense=dense,
            match=MatchMode(match),
            num_steps=num_steps,
            batch_size=batch_size,
        )
        if budget.token_count > target_production_tokens:
            raise click.UsageError("the training token budget must not exceed --target-production-tokens")
        training_token_cap = unique_token_sample_cap(
            target_production_tokens=target_production_tokens,
            fast_track_budget=budget.token_count,
            available_unique_tokens=available_unique_tokens,
            fraction=fraction,
            sequence_length=budget.sequence_length,
        )
        if training_token_cap < budget.sequence_length:
            if fraction * budget.token_count < budget.sequence_length:
                raise click.UsageError("--fraction yields fewer than one training sequence")
            minimum_unique_tokens = max(
                budget.sequence_length,
                (budget.sequence_length * target_production_tokens + budget.token_count - 1) // budget.token_count,
            )
            raise click.UsageError(
                f"--available-unique-tokens must be at least {minimum_unique_tokens:,} "
                "to supply one training sequence at this budget"
            )

    if prepared_cache:
        if any(value is not None for value in (repository, revision, subset, split, text_field, max_rows)):
            raise click.UsageError("--prepared-cache cannot be combined with Hugging Face source options")
        if prepare_only or prepare_token_cap is not None:
            raise click.UsageError("--prepared-cache cannot be combined with preparation options")
        prepared = PreparedAddDatasetCache.raw_load(prepared_cache)
        prefix = prepared.prefix
        cache_step = _adopt_artifact(
            prepared_cache,
            kind=PreparedAddDatasetCache,
            name="prepared-cache",
        )
        assert training_token_cap is not None
        if prepared.actual_num_tokens < training_token_cap:
            raise click.UsageError(
                f"--prepared-cache has {prepared.actual_num_tokens:,} tokens; "
                f"this rung requires {training_token_cap:,}"
            )
        click.echo(
            f"Cache has {prepared.actual_num_tokens:,} tokens. This rung can sample {training_token_cap:,} tokens."
        )
    else:
        required_source_options = [
            option
            for option, value in (
                ("--repository", repository),
                ("--revision", revision),
                ("--split", split),
                ("--text-field", text_field),
                ("--max-rows", max_rows),
            )
            if value is None
        ]
        if required_source_options:
            raise click.UsageError(f"Hugging Face preparation requires {', '.join(required_source_options)}")
        assert repository is not None and revision is not None and split is not None
        assert text_field is not None and max_rows is not None
        if not prepare_only:
            maximum_preparation_cap = unique_token_sample_cap(
                target_production_tokens=target_production_tokens,
                fast_track_budget=min(target_production_tokens, max(maximum_h100_ladder_tokens(), budget.token_count)),
                available_unique_tokens=available_unique_tokens,
                fraction=fraction,
                sequence_length=budget.sequence_length,
            )
            prepared_token_cap = maximum_preparation_cap if prepare_token_cap is None else prepare_token_cap
            assert training_token_cap is not None
            if prepared_token_cap < training_token_cap:
                raise click.UsageError("--prepare-token-cap must be at least the training sample cap")

        try:
            prefix = DatasetPrefix(
                repo=repository,
                revision=revision,
                subset=subset,
                split=split,
                text_field=text_field,
                tokenizer=V16384_TOKENIZER,
                tokenizer_hash=tokenizer_content_hash(V16384_TOKENIZER),
                sampling_policy=AddDatasetSamplingPolicy.PREFIX,
                max_rows=max_rows,
                max_overshoot_tokens=max_overshoot_tokens,
                requested_token_cap=prepared_token_cap,
            )
        except ValidationError as error:
            details = "\n".join(issue["msg"] for issue in error.errors())
            raise click.UsageError(details) from error

        preparation = AddDatasetPreparationConfig(prefix=prefix, output_path="<output_path>")
        cache_step = add_dataset_cache_step(config=preparation)
        if prepare_only:
            return cache_step
        click.echo(f"Prepare {prepared_token_cap:,} tokens. This rung can sample {training_token_cap:,} tokens.")

    assert budget is not None and fraction is not None and available_unique_tokens is not None
    assert baseline_artifact is not None
    baseline: TrainingSource = HeroTrainingSource(
        _adopt_artifact(baseline_artifact, kind=PreparedHeroSample, name="hero-baseline")
    )

    source = AddDatasetTrainingSource(
        config=AddDatasetConfig(
            token_cache=cache_step,
            prefix=prefix,
            fraction=fraction,
            target_production_tokens=target_production_tokens,
            available_unique_tokens=available_unique_tokens,
        ),
        baseline=baseline,
    )
    return build_h100_ladder_run(
        run_id=run_id,
        size=size,
        match=MatchMode(match),
        batch_size=budget.batch_size,
        num_steps=budget.num_steps,
        dense=dense,
        seed=seed,
        data_seed=data_seed,
        training_source=source,
    )


def _adopt_artifact(source: str, *, kind: type[Artifact], name: str) -> ArtifactStep:
    source_digest = hashlib.sha256(source.encode()).hexdigest()[:20]
    artifact_name = f"fast-track/add-dataset/{name}/{source_digest}"
    version = resolve_version(artifact_name, None)
    return ArtifactStep.adopt(
        user_namespaced_name(artifact_name, version),
        version,
        source,
        kind=kind,
    )


if __name__ == "__main__":
    main()
