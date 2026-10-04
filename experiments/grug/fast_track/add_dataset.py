# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a bounded Hugging Face prefix for an add-dataset fast-track run."""

import dataclasses
import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from itertools import islice
from typing import Any

import numpy as np
from datasets import load_dataset
from fray.types import ResourceConfig
from levanter.data._preprocessor import BatchProcessor
from levanter.data.text._batch_tokenizer import BatchTokenizer
from levanter.data.text.datasets import DatasetComponent, DatasetComponentBase, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheLedger, write_levanter_cache
from levanter.tokenizers import load_tokenizer, tokenizer_content_hash
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.store_builder import write_stats_json
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.grug.fast_track.batching import bounded_batches
from experiments.grug.fast_track.contracts import (
    TOKENIZATION_CHUNK_CHARS,
    TOKENIZATION_MAX_DOCUMENT_BYTES,
    TOKENIZATION_POLICY,
    TRAIN_SPLIT,
    AddDatasetConfig,
    DatasetPrefix,
    PreparedAddDatasetCache,
    ResolvedTrainingBudget,
    add_dataset_mixture_weights,
    unique_token_sample_cap,
)
from experiments.grug.fast_track.launch import TrainingSource

PREFIX_TOKENIZE_BLOCK_ROWS = 2048
PREFIX_TOKENIZE_BLOCK_BYTES = 256 * 1024


@dataclasses.dataclass(frozen=True)
class AddDatasetPreparationConfig:
    """Dataset prefix configuration and its output path."""

    prefix: DatasetPrefix
    output_path: str


@dataclasses.dataclass
class _PrefixCounts:
    scanned_rows: int = 0
    total_tokens: int = 0


@dataclasses.dataclass(frozen=True)
class _PrefixText:
    value: object
    num_bytes: int


def _prefix_text_batches(rows: Iterable[Mapping[str, Any]], *, text_field: str) -> Iterator[list[_PrefixText]]:
    texts = (row.get(text_field) for row in rows)
    prefix_texts = (_PrefixText(text, len(text.encode("utf-8")) if isinstance(text, str) else 0) for text in texts)
    yield from bounded_batches(
        prefix_texts,
        max_rows=PREFIX_TOKENIZE_BLOCK_ROWS,
        max_bytes=PREFIX_TOKENIZE_BLOCK_BYTES,
        byte_size=lambda row: row.num_bytes,
    )


def _tokenized_prefix(
    rows: Iterable[Mapping[str, Any]],
    *,
    preprocessor: BatchProcessor[dict, dict],
    config: AddDatasetPreparationConfig,
    counts: _PrefixCounts,
) -> Iterable[dict[str, np.ndarray]]:
    """Read and tokenize bounded blocks in source order within one Zephyr task."""
    blocks = _prefix_text_batches(islice(rows, config.prefix.max_rows), text_field=config.prefix.text_field)
    for block in blocks:
        texts = [row.value for row in block]
        for offset, row in enumerate(block):
            if row.num_bytes > TOKENIZATION_MAX_DOCUMENT_BYTES:
                raise ValueError(
                    f"{config.prefix.repo} row {counts.scanned_rows + offset} has {row.num_bytes:,} UTF-8 bytes; "
                    f"{TOKENIZATION_POLICY} permits at most {TOKENIZATION_MAX_DOCUMENT_BYTES:,} bytes per document"
                )
        valid_rows = [{config.prefix.text_field: text} for text in texts if isinstance(text, str)]
        encoded = preprocessor(valid_rows) if valid_rows else []
        if not isinstance(encoded, list) or len(encoded) != len(valid_rows):
            raise ValueError("text preprocessor must return one token record for each input row")
        token_records = iter(encoded)
        for text in texts:
            row_count = counts.scanned_rows
            # An invalid look-ahead row must fail only if the requested prefix consumes it.
            if not isinstance(text, str):
                raise ValueError(f"row {row_count} has no string field {config.prefix.text_field!r}")
            input_ids = np.asarray(next(token_records)["input_ids"], dtype=np.int32)
            next_tokens = counts.total_tokens + len(input_ids)
            if next_tokens > config.prefix.requested_token_cap + config.prefix.max_overshoot_tokens:
                raise ValueError(
                    f"row {row_count} exceeds the token cap by "
                    f"{next_tokens - config.prefix.requested_token_cap:,} tokens; "
                    f"the limit is {config.prefix.max_overshoot_tokens:,}"
                )
            counts.scanned_rows += 1
            counts.total_tokens = next_tokens
            yield {"input_ids": input_ids}
            if counts.total_tokens >= config.prefix.requested_token_cap:
                break
        if counts.total_tokens >= config.prefix.requested_token_cap:
            break

    if counts.total_tokens < config.prefix.requested_token_cap:
        raise ValueError(
            f"Hugging Face prefix produced {counts.total_tokens:,} tokens in {counts.scanned_rows:,} rows; "
            f"the cache requires {config.prefix.requested_token_cap:,} tokens"
        )


def prepare_add_dataset_cache(
    config: AddDatasetPreparationConfig,
    *,
    rows: Iterable[Mapping[str, Any]],
    tokenizer: str,
) -> PreparedAddDatasetCache:
    """Tokenize and write a bounded row prefix to a Levanter cache."""
    actual_tokenizer_hash = tokenizer_content_hash(tokenizer)
    if actual_tokenizer_hash != config.prefix.tokenizer_hash:
        raise ValueError(
            f"tokenizer content differs from the dataset prefix: "
            f"expected {config.prefix.tokenizer_hash}, got {actual_tokenizer_hash}"
        )
    counts = _PrefixCounts()
    train_path = prefix_join(config.output_path, TRAIN_SPLIT)
    preprocessor = BatchTokenizer(
        load_tokenizer(tokenizer),
        text_field=config.prefix.text_field,
        long_string_workaround=True,
        _workaround_len=TOKENIZATION_CHUNK_CHARS,
    )
    write_levanter_cache(
        _tokenized_prefix(rows, preprocessor=preprocessor, config=config, counts=counts),
        train_path,
        metadata=preprocessor.metadata,
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
        tokenization_policy=TOKENIZATION_POLICY,
    )


def _prepare_hf_prefix(config: AddDatasetPreparationConfig) -> PreparedAddDatasetCache:
    rows = load_dataset(
        config.prefix.repo,
        name=config.prefix.subset,
        revision=config.prefix.revision,
        split=config.prefix.split,
        streaming=True,
    )
    return prepare_add_dataset_cache(config, rows=rows, tokenizer=config.prefix.tokenizer)


def _build_prepared_cache(config: AddDatasetPreparationConfig) -> PreparedAddDatasetCache:
    # A single streaming task preserves the bounded HF prefix without per-block job startup.
    with ZephyrContext(
        name="fast-track-add-dataset-tokenize",
        max_workers=1,
        resources=ResourceConfig(cpu=4, ram="8g", disk="8g"),
    ) as context:
        return context.execute(Dataset.from_list([config]).map(_prepare_hf_prefix)).results[0]


def add_dataset_cache_step(
    *,
    config: AddDatasetPreparationConfig,
    version: str | None = None,
) -> ArtifactStep[PreparedAddDatasetCache]:
    """Build a cache handle whose name includes every preparation identity field."""
    identity = config.prefix.model_dump() | {
        "tokenization_policy": TOKENIZATION_POLICY,
        "tokenization_chunk_chars": TOKENIZATION_CHUNK_CHARS,
        "tokenization_max_document_bytes": TOKENIZATION_MAX_DOCUMENT_BYTES,
    }
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
    max_train_sequences: int,
) -> LmDataConfig:
    """Add a prepared dataset at its token share without a second simulated slice."""
    weights = baseline.train_weights
    if not isinstance(weights, dict):
        raise ValueError("add-dataset training requires fixed dictionary weights")
    if name in baseline.components:
        raise ValueError(f"new dataset component {name!r} already exists")
    if fraction * baseline.mixture_block_size < 1:
        raise ValueError(
            f"add-dataset fraction must be at least {1 / baseline.mixture_block_size:.6g} "
            f"to select one sequence in a mixture block of {baseline.mixture_block_size}"
        )
    max_train_sequences_by_component = baseline.max_train_sequences or {}
    if name in max_train_sequences_by_component:
        raise ValueError(f"new dataset component {name!r} already has a sequence limit")

    return replace(
        baseline,
        components={**baseline.components, name: component},
        train_weights=add_dataset_mixture_weights(weights, new_component=name, fraction=fraction),
        max_train_sequences={**max_train_sequences_by_component, name: max_train_sequences},
        target_budget=None,
        experiment_budget=None,
    )


@dataclasses.dataclass(frozen=True)
class AddDatasetTrainingSource:
    """Add one prepared Hugging Face token cache to the frozen baseline."""

    config: AddDatasetConfig
    baseline: TrainingSource

    def dependencies(self) -> tuple[ArtifactStep, ...]:
        return (*self.baseline.dependencies(), self.config.token_cache)

    def data_config(
        self,
        *,
        ctx: StepContext,
        validation: Sequence[ArtifactStep[TokenizedCache]],
        tokenizer: str,
        budget: ResolvedTrainingBudget,
    ) -> LmDataConfig:
        baseline = self.baseline.data_config(
            ctx=ctx,
            validation=validation,
            tokenizer=tokenizer,
            budget=budget,
        )
        sample_cap = unique_token_sample_cap(
            target_production_tokens=self.config.target_production_tokens,
            fast_track_budget=budget.token_count,
            available_unique_tokens=self.config.available_unique_tokens,
            fraction=self.config.fraction,
            sequence_length=budget.sequence_length,
        )
        if sample_cap < budget.sequence_length:
            raise ValueError("add-dataset share yields fewer than one training sequence")

        if ctx.is_fingerprint:
            cache_dir = ctx.artifact_path(self.config.token_cache)
        else:
            token_cache = ctx.resolved(self.config.token_cache)
            if token_cache.tokenization_policy != TOKENIZATION_POLICY:
                raise ValueError(
                    f"prepared cache uses {token_cache.tokenization_policy!r}; expected {TOKENIZATION_POLICY!r}"
                )
            actual_tokenizer_hash = tokenizer_content_hash(tokenizer)
            if actual_tokenizer_hash != self.config.prefix.tokenizer_hash:
                raise ValueError(
                    f"tokenizer content differs from the prepared add-dataset prefix: "
                    f"expected {self.config.prefix.tokenizer_hash}, got {actual_tokenizer_hash}"
                )
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
            max_train_sequences=sample_cap // budget.sequence_length,
        )
