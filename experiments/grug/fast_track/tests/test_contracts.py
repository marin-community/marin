# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.execution.lazy import ArtifactStep, StepContext

from experiments.grug.fast_track.contracts import (
    AddDatasetConfig,
    AddDatasetSamplingPolicy,
    DatasetPrefix,
    FrozenBaselineComponent,
    FrozenBaselineManifest,
    PreparedAddDatasetCache,
    ResolvedTrainingBudget,
)
from experiments.grug.fast_track.launch import V16384_TOKENIZER, AddDatasetTrainingSource


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
    return AddDatasetTrainingSource(config=config, baseline=baseline)


def test_add_dataset_config_preserves_baseline_mix_and_caps_prepared_cache_once():
    source = _source(fraction=0.19, available_unique_tokens=300)
    budget = ResolvedTrainingBudget(batch_size=2, num_steps=10, sequence_length=10)
    context = StepContext.for_fingerprint((), source.dependencies())

    data = source.data_config(ctx=context, validation=(), tokenizer=V16384_TOKENIZER, budget=budget)

    assert data.train_weights == pytest.approx({"a": 0.6075, "b": 0.2025, "add-dataset": 0.19})
    assert data.max_train_batches == {"add-dataset": 1}
    assert data.target_budget is None
    assert data.experiment_budget is None
    assert source.dependencies() == (source.config.token_cache,)
    assert data.components["add-dataset"].cache_dir == context.artifact_path(source.config.token_cache)


def test_add_dataset_config_rejects_share_below_one_full_batch():
    source = _source(fraction=0.05, available_unique_tokens=900)
    budget = ResolvedTrainingBudget(batch_size=2, num_steps=10, sequence_length=10)
    context = StepContext.for_fingerprint((), source.dependencies())

    with pytest.raises(ValueError, match="fewer than one full training batch"):
        source.data_config(ctx=context, validation=(), tokenizer=V16384_TOKENIZER, budget=budget)
