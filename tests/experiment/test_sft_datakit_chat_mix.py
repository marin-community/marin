# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from haliax import Axis
from levanter.data.text.datasets import dataset_for_component
from levanter.store.cache import TreeCache, write_levanter_cache
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


def test_recipe_rejects_cross_region_output_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recipe, "marin_prefix", lambda: "gs://marin-eu-west4")

    with pytest.raises(ValueError, match="SFT storage must be in us-central2"):
        recipe.build()


def test_replay_continues_tokens_across_document_boundaries(tmp_path: Path) -> None:
    cache_path = str(tmp_path / "replay")
    write_levanter_cache(
        [{"input_ids": np.arange(1, 7, dtype=np.int32)}, {"input_ids": np.arange(7, 13, dtype=np.int32)}],
        cache_path,
        metadata={},
    )
    cache = TreeCache.load(cache_path, exemplar={"input_ids": np.zeros(0, dtype=np.int32)})
    component = recipe._pretrain_child("c01q4", "lte_64k")
    dataset = dataset_for_component(
        component, Axis("position", 4), cache, eos_id=None, block_cross_document_attention=True
    ).as_sync_dataset()
    np.testing.assert_array_equal(np.asarray(dataset[0].tokens), [1, 2, 3, 4])
    np.testing.assert_array_equal(np.asarray(dataset[1].tokens), [5, 6, 7, 8])
