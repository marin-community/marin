# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a bounded Hugging Face prefix or add it to a fast-track run."""

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
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
    unique_token_sample_cap,
)
from experiments.grug.fast_track.launch import (
    H100_LADDER_SIZES,
    V16384_TOKENIZER,
    MatchMode,
    build_h100_ladder_run,
    resolve_h100_ladder_budget,
)


@click.command()
@click.option("--run-id", default=None, help="Run identifier for artifact and W&B names.")
@click.option("--size", type=click.Choice(H100_LADDER_SIZES), default="d512", show_default=True)
@click.option("--dense/--moe", default=True, show_default=True, help="Select the dense or MoE model.")
@click.option("--match", type=click.Choice([mode.value for mode in MatchMode]), default="data", show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=None)
@click.option("--num-steps", type=click.IntRange(min=1), default=None)
@click.option("--seed", type=click.IntRange(min=0), default=0, show_default=True, help="Model initialization seed.")
@click.option("--data-seed", type=click.IntRange(min=0), default=0, show_default=True, help="Training data seed.")
@click.option("--repository", required=True, help="Hugging Face dataset repository.")
@click.option("--revision", required=True, help="Immutable Hugging Face commit hash.")
@click.option("--subset", default=None, help="Hugging Face dataset subset.")
@click.option("--split", required=True, help="Hugging Face split to stream.")
@click.option("--text-field", required=True, help="String field to tokenize.")
@click.option("--fraction", type=click.FloatRange(min=0, max=1, min_open=True, max_open=True), default=None)
@click.option("--target-production-tokens", type=click.IntRange(min=1), default=None)
@click.option("--available-unique-tokens", type=click.IntRange(min=1), default=None)
@click.option("--max-rows", type=click.IntRange(min=1), required=True)
@click.option("--max-overshoot-tokens", type=click.IntRange(min=0), default=16_384, show_default=True)
@click.option("--prepare-token-cap", type=click.IntRange(min=1), default=None)
@click.option("--prepare-only", is_flag=True, help="Build only the bounded cache. Requires --prepare-token-cap.")
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
    repository: str,
    revision: str,
    subset: str | None,
    split: str,
    text_field: str,
    fraction: float | None,
    target_production_tokens: int | None,
    available_unique_tokens: int | None,
    max_rows: int,
    max_overshoot_tokens: int,
    prepare_token_cap: int | None,
    prepare_only: bool,
) -> ArtifactStep:
    if prepare_only:
        if prepare_token_cap is None:
            raise click.UsageError("--prepare-token-cap is required with --prepare-only")
        prepared_token_cap = prepare_token_cap
    else:
        if not run_id or not run_id.strip():
            raise click.UsageError("--run-id is required for training")
        missing = [
            option
            for option, value in (
                ("--fraction", fraction),
                ("--target-production-tokens", target_production_tokens),
                ("--available-unique-tokens", available_unique_tokens),
            )
            if value is None
        ]
        if missing:
            raise click.UsageError(f"training requires {', '.join(missing)}")
        assert fraction is not None
        assert target_production_tokens is not None
        assert available_unique_tokens is not None
        budget = resolve_h100_ladder_budget(
            size=size,
            dense=dense,
            match=MatchMode(match),
            num_steps=num_steps,
            batch_size=batch_size,
        )
        requested_token_cap = unique_token_sample_cap(
            target_production_tokens=target_production_tokens,
            fast_track_budget=budget.token_count,
            available_unique_tokens=available_unique_tokens,
            fraction=fraction,
            loader_unit=budget.batch_size * budget.sequence_length,
        )
        if requested_token_cap < budget.batch_size * budget.sequence_length:
            raise click.UsageError("dataset share yields fewer than one full training batch")
        prepared_token_cap = requested_token_cap if prepare_token_cap is None else prepare_token_cap
        if prepared_token_cap < requested_token_cap:
            raise click.UsageError("--prepare-token-cap must be at least the training sample cap")

    try:
        prefix = DatasetPrefix(
            repo=repository,
            revision=revision,
            subset=subset,
            split=split,
            text_field=text_field,
            tokenizer=V16384_TOKENIZER,
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

    source = AddDatasetTrainingSource(
        config=AddDatasetConfig(
            token_cache=cache_step,
            prefix=prefix,
            fraction=fraction,
            target_production_tokens=target_production_tokens,
            available_unique_tokens=available_unique_tokens,
        )
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


if __name__ == "__main__":
    main()
