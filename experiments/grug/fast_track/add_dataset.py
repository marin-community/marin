# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a bounded Hugging Face prefix for an add-dataset fast-track run."""

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from datasets import load_dataset
from levanter.data._preprocessor import BatchProcessor
from levanter.data.text.datasets import DatasetComponent, DatasetComponentBase, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheLedger, write_levanter_cache
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.store_builder import write_stats_json
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.fast_track.contracts import (
    TRAIN_SPLIT,
    AddDatasetConfig,
    DatasetPrefix,
    FrozenBaselineManifest,
    PreparedAddDatasetCache,
    ResolvedTrainingBudget,
    add_dataset_mixture_weights,
    unique_token_sample_cap,
)
from experiments.grug.fast_track.launch import FROZEN_BASELINE, FlatCacheTrainingSource


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
    namespaced_name = user_namespaced_name(name, resolved_version)

    def build_config(ctx: StepContext) -> AddDatasetPreparationConfig:
        return dataclasses.replace(config, output_path=ctx.output_path)

    return ArtifactStep(
        name=namespaced_name,
        version=resolved_version,
        artifact_type=PreparedAddDatasetCache,
        run=_build_prepared_cache,
        build_config=build_config,
    )


def add_prepared_dataset_component(
    baseline: LmDataConfig,
    *,
    name: str,
    component: DatasetComponentBase,
    fraction: float,
    max_train_batches: int,
) -> LmDataConfig:
    """Add a prepared dataset at its token share without a second simulated slice."""
    weights = baseline.train_weights
    if not isinstance(weights, dict):
        raise ValueError("add-dataset training requires fixed dictionary weights")
    if name in baseline.components:
        raise ValueError(f"new dataset component {name!r} already exists")

    return replace(
        baseline,
        components={**baseline.components, name: component},
        train_weights=add_dataset_mixture_weights(weights, new_component=name, fraction=fraction),
        max_train_batches={name: max_train_batches},
        target_budget=None,
        experiment_budget=None,
    )


@dataclasses.dataclass(frozen=True)
class AddDatasetTrainingSource:
    """Add one prepared Hugging Face token cache to the frozen baseline."""

    config: AddDatasetConfig
    baseline: FrozenBaselineManifest = FROZEN_BASELINE

    def dependencies(self) -> tuple[ArtifactStep, ...]:
        return (self.config.token_cache,)

    def data_config(
        self,
        *,
        ctx: StepContext,
        validation: Sequence[ArtifactStep[TokenizedCache]],
        tokenizer: str,
        budget: ResolvedTrainingBudget,
    ) -> LmDataConfig:
        baseline = FlatCacheTrainingSource(manifest=self.baseline).data_config(
            ctx=ctx,
            validation=validation,
            tokenizer=tokenizer,
            budget=budget,
        )
        loader_unit = budget.batch_size * budget.sequence_length
        sample_cap = unique_token_sample_cap(
            target_production_tokens=self.config.target_production_tokens,
            fast_track_budget=budget.token_count,
            available_unique_tokens=self.config.available_unique_tokens,
            fraction=self.config.fraction,
            loader_unit=loader_unit,
        )
        if sample_cap < loader_unit:
            raise ValueError("add-dataset share yields fewer than one full training batch")

        if ctx.is_fingerprint:
            cache_dir = ctx.artifact_path(self.config.token_cache)
        else:
            token_cache = ctx.resolved(self.config.token_cache)
            if token_cache.prefix.tokenizer != tokenizer:
                raise ValueError(
                    f"add-dataset tokenizer {token_cache.prefix.tokenizer!r} does not match requested {tokenizer!r}"
                )
            if token_cache.prefix != self.config.prefix:
                expected = self.config.prefix.model_dump(mode="json")
                actual = token_cache.prefix.model_dump(mode="json")
                differences = {
                    field: {"expected": value, "actual": actual[field]}
                    for field, value in expected.items()
                    if value != actual[field]
                }
                raise ValueError(f"prepared add-dataset prefix differs from the training source: {differences}")
            if token_cache.actual_num_tokens < sample_cap:
                raise ValueError(
                    f"prepared add-dataset cache has {token_cache.actual_num_tokens:,} tokens; "
                    f"the run requires {sample_cap:,}"
                )
            cache_dir = token_cache.cache_dir

        return add_prepared_dataset_component(
            baseline,
            name="add-dataset",
            component=DatasetComponent(
                cache_dir=prefix_join(cache_dir, TRAIN_SPLIT),
                format=TextLmDatasetFormat(text_key=self.config.prefix.text_field),
                flat_cache=True,
            ),
            fraction=self.config.fraction,
            max_train_batches=sample_cap // loader_unit,
        )
