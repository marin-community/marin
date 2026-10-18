# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a bounded Hugging Face prefix or add it to a fast-track run."""

import click
from levanter.tokenizers import tokenizer_content_hash
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from pydantic import ValidationError

from experiments.grug.fast_track.add_dataset import AddDatasetTrainingSource, add_dataset_cache_step
from experiments.grug.fast_track.contracts import (
    AddDatasetConfig,
    DatasetPrefix,
    PreparedAddDatasetCache,
    ResolvedTrainingBudget,
    adopt_artifact_path,
    unique_token_sample_cap,
)
from experiments.grug.fast_track.hero_sample import HeroTrainingSource, PreparedHeroSample
from experiments.grug.fast_track.launch import (
    V16384_TOKENIZER,
    MatchMode,
    build_h100_ladder_run,
    ladder_options,
    maximum_h100_ladder_tokens,
    resolve_h100_ladder_budget,
)
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import TOTAL_TOKENS


def _required[T](value: T | None, option: str) -> T:
    if value is None:
        raise click.UsageError(f"{option} is required in this mode")
    return value


def _hf_prefix(
    *,
    repository: str | None,
    revision: str | None,
    subset: str | None,
    split: str | None,
    text_field: str | None,
    max_rows: int | None,
    requested_token_cap: int,
) -> DatasetPrefix:
    try:
        return DatasetPrefix(
            repo=_required(repository, "--repository"),
            revision=_required(revision, "--revision"),
            subset=subset,
            split=_required(split, "--split"),
            text_field=_required(text_field, "--text-field"),
            tokenizer=V16384_TOKENIZER,
            tokenizer_hash=tokenizer_content_hash(V16384_TOKENIZER),
            max_rows=_required(max_rows, "--max-rows"),
            requested_token_cap=requested_token_cap,
        )
    except ValidationError as error:
        raise click.UsageError("\n".join(issue["msg"] for issue in error.errors())) from error


def _training_token_cap(
    budget: ResolvedTrainingBudget,
    *,
    fraction: float,
    available_unique_tokens: int,
    target_production_tokens: int,
) -> int:
    if budget.token_count > target_production_tokens:
        raise click.UsageError("the training token budget must not exceed --target-production-tokens")
    cap = unique_token_sample_cap(
        target_production_tokens=target_production_tokens,
        fast_track_budget=budget.token_count,
        available_unique_tokens=available_unique_tokens,
        fraction=fraction,
        sequence_length=budget.sequence_length,
    )
    if cap >= budget.sequence_length:
        return cap
    if fraction * budget.token_count < budget.sequence_length:
        raise click.UsageError("--fraction yields fewer than one training sequence")
    minimum_unique_tokens = (
        budget.sequence_length * target_production_tokens + budget.token_count - 1
    ) // budget.token_count
    raise click.UsageError(
        f"--available-unique-tokens must be at least {minimum_unique_tokens:,} "
        "to supply one training sequence at this budget"
    )


@click.command()
@click.option("--run-id", default=None, help="Run identifier for artifact and W&B names.")
@ladder_options
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
    data_seed: int | None,
    repository: str | None,
    revision: str | None,
    subset: str | None,
    split: str | None,
    text_field: str | None,
    fraction: float | None,
    target_production_tokens: int,
    available_unique_tokens: int | None,
    max_rows: int | None,
    prepare_token_cap: int | None,
    prepare_only: bool,
    prepared_cache: str | None,
    baseline_artifact: str | None,
) -> ArtifactStep:
    def hf_prefix(requested_token_cap: int) -> DatasetPrefix:
        return _hf_prefix(
            repository=repository,
            revision=revision,
            subset=subset,
            split=split,
            text_field=text_field,
            max_rows=max_rows,
            requested_token_cap=requested_token_cap,
        )

    if prepare_only:
        training_options = (run_id, fraction, available_unique_tokens, prepared_cache, baseline_artifact)
        if any(value is not None for value in training_options):
            raise click.UsageError("--prepare-only accepts only Hugging Face source options and --prepare-token-cap")
        return add_dataset_cache_step(prefix=hf_prefix(_required(prepare_token_cap, "--prepare-token-cap")))

    run_id = _required(run_id, "--run-id")
    fraction = _required(fraction, "--fraction")
    available_unique_tokens = _required(available_unique_tokens, "--available-unique-tokens")
    baseline = HeroTrainingSource(
        adopt_artifact_path(_required(baseline_artifact, "--baseline-artifact"), kind=PreparedHeroSample)
    )
    budget = resolve_h100_ladder_budget(
        size=size,
        dense=dense,
        match=MatchMode(match),
        num_steps=num_steps,
        batch_size=batch_size,
    )
    training_cap = _training_token_cap(
        budget,
        fraction=fraction,
        available_unique_tokens=available_unique_tokens,
        target_production_tokens=target_production_tokens,
    )

    if prepared_cache is not None:
        hf_options = (repository, revision, subset, split, text_field, max_rows, prepare_token_cap)
        if any(value is not None for value in hf_options):
            raise click.UsageError("--prepared-cache cannot be combined with Hugging Face source or preparation options")
        prepared = PreparedAddDatasetCache.raw_load(prepared_cache)
        if prepared.actual_num_tokens < training_cap:
            raise click.UsageError(
                f"--prepared-cache has {prepared.actual_num_tokens:,} tokens; this rung requires {training_cap:,}"
            )
        cache_step = adopt_artifact_path(prepared_cache, kind=PreparedAddDatasetCache)
        click.echo(f"Cache has {prepared.actual_num_tokens:,} tokens. This rung can sample {training_cap:,} tokens.")
    else:
        # Prepare enough for the largest default rung, so that one cache serves every smaller rung.
        prepare_cap = prepare_token_cap or unique_token_sample_cap(
            target_production_tokens=target_production_tokens,
            fast_track_budget=min(target_production_tokens, max(maximum_h100_ladder_tokens(), budget.token_count)),
            available_unique_tokens=available_unique_tokens,
            fraction=fraction,
            sequence_length=budget.sequence_length,
        )
        if prepare_cap < training_cap:
            raise click.UsageError("--prepare-token-cap must be at least the training sample cap")
        cache_step = add_dataset_cache_step(prefix=hf_prefix(prepare_cap))
        click.echo(f"Prepare {prepare_cap:,} tokens. This rung can sample {training_cap:,} tokens.")

    return build_h100_ladder_run(
        run_id=run_id,
        size=size,
        match=MatchMode(match),
        batch_size=budget.batch_size,
        num_steps=budget.num_steps,
        dense=dense,
        seed=seed,
        data_seed=data_seed,
        training_source=AddDatasetTrainingSource(
            config=AddDatasetConfig(
                token_cache=cache_step,
                fraction=fraction,
                target_production_tokens=target_production_tokens,
                available_unique_tokens=available_unique_tokens,
            ),
            baseline=baseline,
        ),
    )


if __name__ == "__main__":
    main()
