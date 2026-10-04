# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
from marin.execution.lazy import ArtifactStep, StepContext

from experiments.grug.fast_track.add_dataset import AddDatasetTrainingSource
from experiments.grug.fast_track.contracts import (
    AddDatasetConfig,
    AddDatasetSamplingPolicy,
    DatasetPrefix,
    FrozenBaselineComponent,
    FrozenBaselineManifest,
    PreparedAddDatasetCache,
    ResolvedTrainingBudget,
    unique_token_sample_cap,
)
from experiments.grug.fast_track.launch import V16384_TOKENIZER, FlatCacheTrainingSource


def _source(*, fraction: float, available_unique_tokens: int) -> AddDatasetTrainingSource:
    token_cache = ArtifactStep(
        name="prepared/hf-dataset",
        version="2026.10.03",
        artifact_type=PreparedAddDatasetCache,
        run=lambda _config: None,
        build_config=lambda _ctx: None,
    )
    config = AddDatasetConfig(
        token_cache=token_cache,
        prefix=DatasetPrefix(
            repo="org/dataset",
            revision="a" * 40,
            subset="default",
            split="train",
            text_field="text",
            tokenizer=V16384_TOKENIZER,
            tokenizer_hash="sha256:test-tokenizer",
            sampling_policy=AddDatasetSamplingPolicy.PREFIX,
            max_rows=1_000,
            max_overshoot_tokens=16_384,
            requested_token_cap=100_000,
        ),
        fraction=fraction,
        target_production_tokens=1_000,
        available_unique_tokens=available_unique_tokens,
    )
    baseline = FrozenBaselineManifest(
        tokenizer=V16384_TOKENIZER,
        components=(
            FrozenBaselineComponent(name="a", cache_dir="a", weight=0.75),
            FrozenBaselineComponent(name="b", cache_dir="b", weight=0.25),
        ),
    )
    return AddDatasetTrainingSource(config=config, baseline=FlatCacheTrainingSource(manifest=baseline))


def test_add_dataset_config_preserves_baseline_mix_and_caps_prepared_cache_once():
    source = _source(fraction=0.19, available_unique_tokens=300)
    budget = ResolvedTrainingBudget(batch_size=2, num_steps=10, sequence_length=10)
    context = StepContext.for_fingerprint((), source.dependencies())

    data = source.data_config(ctx=context, validation=(), tokenizer=V16384_TOKENIZER, budget=budget)

    assert data.train_weights == pytest.approx({"a": 0.6075, "b": 0.2025, "add-dataset": 0.19})
    assert data.max_train_sequences == {"add-dataset": 3}
    assert data.target_budget is None
    assert data.experiment_budget is None
    assert source.dependencies() == (source.config.token_cache,)
    assert data.components["add-dataset"].cache_dir == f"{context.artifact_path(source.config.token_cache)}/train"


def test_add_dataset_config_accepts_one_sequence_below_one_full_batch():
    source = _source(fraction=0.05, available_unique_tokens=900)
    budget = ResolvedTrainingBudget(batch_size=2, num_steps=10, sequence_length=10)
    context = StepContext.for_fingerprint((), source.dependencies())

    data = source.data_config(ctx=context, validation=(), tokenizer=V16384_TOKENIZER, budget=budget)

    assert data.max_train_sequences == {"add-dataset": 1}


def test_add_dataset_config_rejects_availability_below_one_sequence():
    source = _source(fraction=0.99, available_unique_tokens=100_000_000)
    source = dataclasses.replace(
        source,
        config=dataclasses.replace(source.config, target_production_tokens=18_750_000_000_000),
    )
    budget = ResolvedTrainingBudget(batch_size=2, num_steps=10, sequence_length=10)
    context = StepContext.for_fingerprint((), source.dependencies())

    with pytest.raises(ValueError, match="fewer than one training sequence"):
        source.data_config(ctx=context, validation=(), tokenizer=V16384_TOKENIZER, budget=budget)


def test_add_dataset_config_keeps_one_sequence_at_high_fraction_with_enough_unique_data():
    source = _source(fraction=0.99, available_unique_tokens=1_000_000_000_000)
    source = dataclasses.replace(
        source,
        config=dataclasses.replace(source.config, target_production_tokens=18_750_000_000_000),
    )
    budget = ResolvedTrainingBudget(batch_size=2, num_steps=10, sequence_length=10)
    context = StepContext.for_fingerprint((), source.dependencies())

    data = source.data_config(ctx=context, validation=(), tokenizer=V16384_TOKENIZER, budget=budget)

    assert data.max_train_sequences == {"add-dataset": 1}


def test_exposure_simulation_rejects_a_run_larger_than_its_production_target():
    with pytest.raises(ValueError, match="must not exceed the target production"):
        unique_token_sample_cap(
            target_production_tokens=100,
            fast_track_budget=200,
            available_unique_tokens=50,
            fraction=0.5,
            sequence_length=1,
        )
