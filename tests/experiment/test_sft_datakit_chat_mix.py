# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.datakit.sft_sources import all_sft_sources

from experiments.june_tpu_67b_a2b.moe import sft_datakit_chat_mix as recipe
from experiments.june_tpu_67b_a2b.moe.sft_datakit_chat_mix import (
    _MIXTURE_BLOCK_SIZE,
    _PRETRAIN_FRACTION,
    _SFT_FRACTION,
    _pretrain_components,
    _sft_mixture,
)


def test_every_sft_source_gets_samples_in_each_mixture_block() -> None:
    mixture = _sft_mixture()

    assert set(mixture.components) == {f"sft/{name}" for name in all_sft_sources()}
    assert sum(mixture.weights.values()) == pytest.approx(_SFT_FRACTION)
    assert all(int(weight * _MIXTURE_BLOCK_SIZE) >= 1 for weight in mixture.weights.values())


def test_every_pretrain_bucket_gets_samples_in_each_mixture_block() -> None:
    components, weights = _pretrain_components()

    assert set(components) == set(weights)
    assert sum(weights.values()) == pytest.approx(_PRETRAIN_FRACTION)
    assert all(int(weight * _MIXTURE_BLOCK_SIZE) >= 1 for weight in weights.values())
    assert all(child.pack for component in components.values() for child in component.children.values())


def test_recipe_rejects_cross_region_output_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recipe, "marin_prefix", lambda: "gs://marin-eu-west4")

    with pytest.raises(ValueError, match="SFT storage must be in us-central2"):
        recipe.build()
