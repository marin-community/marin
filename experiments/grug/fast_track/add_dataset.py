# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a bounded Hugging Face prefix for an add-dataset fast-track run."""

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import click
import numpy as np
from datasets import load_dataset
from levanter.data._preprocessor import BatchProcessor
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheLedger, write_levanter_cache
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.processing.tokenize.store_builder import write_stats_json
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.fast_track.contracts import (
    TRAIN_SPLIT,
    AddDatasetConfig,
    AddDatasetSamplingPolicy,
    DatasetPrefix,
    PreparedAddDatasetCache,
    unique_token_sample_cap,
)
from experiments.grug.fast_track.launch import (
    H100_LADDER_SIZES,
    V16384_TOKENIZER,
    AddDatasetTrainingSource,
    MatchMode,
    _h100_ladder_model,
    _h100_ladder_rung,
    build_h100_ladder_run,
    resolve_h100_ladder_budget,
)


@dataclasses.dataclass(frozen=True)
class AddDatasetPreparationConfig:
    """Dataset prefix configuration and its output path."""

    prefix: DatasetPrefix
    output_path: str


@dataclasses.dataclass
class _PrefixCounts:
    scanned_rows: int = 0
    total_tokens: int = 0


def _tokenized_prefix(
    rows: Iterable[Mapping[str, Any]],
    *,
    preprocessor: BatchProcessor[dict, dict],
    config: AddDatasetPreparationConfig,
    counts: _PrefixCounts,
) -> Iterable[dict[str, np.ndarray]]:
    row_iter = iter(rows)
    while counts.scanned_rows < config.prefix.max_rows and counts.total_tokens < config.prefix.requested_token_cap:
        try:
            row = next(row_iter)
        except StopIteration:
            break
        text = row.get(config.prefix.text_field)
        if not isinstance(text, str):
            raise ValueError(f"row {counts.scanned_rows} has no string field {config.prefix.text_field!r}")
        encoded = preprocessor([{config.prefix.text_field: text}])
        if not isinstance(encoded, list) or len(encoded) != 1:
            raise ValueError("text preprocessor must return one token record for each input row")
        input_ids = np.asarray(encoded[0]["input_ids"], dtype=np.int32)
        next_tokens = counts.total_tokens + len(input_ids)
        if next_tokens > config.prefix.requested_token_cap + config.prefix.max_overshoot_tokens:
            raise ValueError(
                f"row {counts.scanned_rows} exceeds the token cap by "
                f"{next_tokens - config.prefix.requested_token_cap:,} tokens; "
                f"the limit is {config.prefix.max_overshoot_tokens:,}"
            )
        counts.scanned_rows += 1
        counts.total_tokens = next_tokens
        yield {"input_ids": input_ids}

    if counts.total_tokens < config.prefix.requested_token_cap:
        raise ValueError(
            f"Hugging Face prefix produced {counts.total_tokens:,} tokens in {counts.scanned_rows:,} rows; "
            f"the cache requires {config.prefix.requested_token_cap:,} tokens"
        )


def prepare_add_dataset_cache(
    config: AddDatasetPreparationConfig,
    *,
    rows: Iterable[Mapping[str, Any]],
    tokenizer: MarinTokenizer,
) -> PreparedAddDatasetCache:
    """Tokenize and write a bounded row prefix to a Levanter cache."""
    preprocessor = TextLmDatasetFormat(text_key=config.prefix.text_field).build_preprocessor(tokenizer)
    counts = _PrefixCounts()
    train_path = prefix_join(config.output_path, TRAIN_SPLIT)
    write_levanter_cache(
        _tokenized_prefix(rows, preprocessor=preprocessor, config=config, counts=counts),
        train_path,
        metadata={
            "tokenizer": config.prefix.tokenizer,
            "format": "text",
            "text_field": config.prefix.text_field,
        },
    )
    ledger = CacheLedger.load(train_path)
    if ledger.total_num_rows != counts.scanned_rows or ledger.field_counts.get("input_ids", 0) != counts.total_tokens:
        raise ValueError("written cache counts do not match the prepared prefix counts")
    write_stats_json(train_path, ledger)
    return PreparedAddDatasetCache(
        cache_dir=config.output_path,
        prefix=config.prefix,
        actual_num_rows=ledger.total_num_rows,
        actual_num_tokens=ledger.field_counts["input_ids"],
    )


def _build_prepared_cache(config: AddDatasetPreparationConfig) -> PreparedAddDatasetCache:
    rows = load_dataset(
        config.prefix.repo,
        name=config.prefix.subset,
        revision=config.prefix.revision,
        split=config.prefix.split,
        streaming=True,
    )
    return prepare_add_dataset_cache(config, rows=rows, tokenizer=load_tokenizer(config.prefix.tokenizer))


def add_dataset_cache_step(
    *,
    config: AddDatasetPreparationConfig,
    version: str | None = None,
) -> ArtifactStep[PreparedAddDatasetCache]:
    """Build a cache handle whose name includes every preparation identity field."""
    identity = config.prefix.model_dump()
    identity_hash = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    name = f"fast-track/add-dataset/{identity_hash}"
    resolved_version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> AddDatasetPreparationConfig:
        return dataclasses.replace(config, output_path=ctx.output_path)

    return ArtifactStep(
        name=name,
        version=resolved_version,
        artifact_type=PreparedAddDatasetCache,
        run=_build_prepared_cache,
        build_config=build_config,
    )


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
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
@click.option("--fraction", type=click.FloatRange(min=0, max=1, min_open=True, max_open=True), required=True)
@click.option("--target-production-tokens", type=click.IntRange(min=1), required=True)
@click.option("--available-unique-tokens", type=click.IntRange(min=1), required=True)
@click.option("--max-rows", type=click.IntRange(min=1), required=True)
@click.option("--max-overshoot-tokens", type=click.IntRange(min=0), default=16_384, show_default=True)
@click.option("--prepare-token-cap", type=click.IntRange(min=1), default=None)
@build_options
def main(
    run_id: str,
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
    fraction: float,
    target_production_tokens: int,
    available_unique_tokens: int,
    max_rows: int,
    max_overshoot_tokens: int,
    prepare_token_cap: int | None,
) -> ArtifactStep:
    if not run_id.strip():
        raise click.UsageError("--run-id must not be empty")
    model = _h100_ladder_model(_h100_ladder_rung(size), dense=dense)
    budget = resolve_h100_ladder_budget(
        size=size,
        dense=dense,
        match=MatchMode(match),
        num_steps=num_steps,
        batch_size=batch_size,
        model=model,
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
    preparation = AddDatasetPreparationConfig(prefix=prefix, output_path="<output_path>")
    cache_step = add_dataset_cache_step(config=preparation)
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
