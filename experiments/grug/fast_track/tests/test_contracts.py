# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
from levanter.data.text.datasets import LmDataConfig
from marin.execution.lazy import ArtifactStep, StepContext

from experiments.grug.fast_track.add_dataset import AddDatasetTrainingSource
from experiments.grug.fast_track.contracts import (
    AddDatasetConfig,
    FrozenBaselineComponent,
    FrozenBaselineManifest,
    PreparedAddDatasetCache,
    ResolvedTrainingBudget,
)
from experiments.grug.fast_track.launch import V16384_TOKENIZER, FlatCacheTrainingSource

BUDGET = ResolvedTrainingBudget(batch_size=2, num_steps=10, sequence_length=10)


@dataclass(frozen=True)
class _UnlimitedBaseline(FlatCacheTrainingSource):
    """A flat-cache baseline that has no measured capacity."""

    def mixture_config(self, **kwargs) -> LmDataConfig:
        return self.data_config(**kwargs)

    def require_capacity(self, data: LmDataConfig, *, ctx: StepContext, budget: ResolvedTrainingBudget) -> None:
        del data, ctx, budget


def _source(*, fraction: float, available_unique_tokens: int, target_production_tokens: int) -> AddDatasetTrainingSource:
    token_cache = ArtifactStep(
        name="prepared/hf-dataset",
        version="2026.10.03",
        artifact_type=PreparedAddDatasetCache,
        run=lambda _config: None,
        build_config=lambda _ctx: None,
    )
    config = AddDatasetConfig(
        token_cache=token_cache,
        fraction=fraction,
        target_production_tokens=target_production_tokens,
        available_unique_tokens=available_unique_tokens,
    )
    baseline = FrozenBaselineManifest(
        tokenizer=V16384_TOKENIZER,
        components=(
            FrozenBaselineComponent(name="a", cache_dir="a", weight=0.75),
            FrozenBaselineComponent(name="b", cache_dir="b", weight=0.25),
        ),
    )
    return AddDatasetTrainingSource(config=config, baseline=_UnlimitedBaseline(manifest=baseline))


def _data(source: AddDatasetTrainingSource) -> LmDataConfig:
    context = StepContext.for_fingerprint((), source.dependencies())
    return source.data_config(ctx=context, validation=(), tokenizer=V16384_TOKENIZER, budget=BUDGET)


def test_add_dataset_config_preserves_baseline_mix_and_points_at_the_prepared_cache():
    source = _source(fraction=0.19, available_unique_tokens=300, target_production_tokens=1_000)

    data = _data(source)

    assert data.train_weights == pytest.approx({"a": 0.6075, "b": 0.2025, "add-dataset": 0.19})
    context = StepContext.for_fingerprint((), source.dependencies())
    assert data.components["add-dataset"].cache_dir == f"{context.artifact_path(source.config.token_cache)}/train"


@pytest.mark.parametrize(
    ("fraction", "available_unique_tokens", "target_production_tokens", "expected_sequences"),
    [
        # Share limit: 0.19 * 200 = 38 tokens.
        (0.19, 300, 1_000, 3),
        # Share limit below one full batch: 0.05 * 200 = 10 tokens.
        (0.05, 900, 1_000, 1),
        # Availability limit: 300 * 200 / 1000 = 60 tokens.
        (0.5, 300, 1_000, 6),
        # Availability limit at a high share: 1e12 * 200 / 18.75e12 = 10.7 tokens.
        (0.99, 1_000_000_000_000, 18_750_000_000_000, 1),
        # Availability below one sequence: 1e8 * 200 / 18.75e12 < 1 token.
        (0.99, 100_000_000, 18_750_000_000_000, None),
    ],
)
def test_add_dataset_sequence_cap_takes_the_smaller_exposure_limit(
    fraction, available_unique_tokens, target_production_tokens, expected_sequences
):
    source = _source(
        fraction=fraction,
        available_unique_tokens=available_unique_tokens,
        target_production_tokens=target_production_tokens,
    )

    if expected_sequences is None:
        with pytest.raises(ValueError, match="fewer than one training sequence"):
            _data(source)
    else:
        assert _data(source).max_train_sequences == {"add-dataset": expected_sequences}
