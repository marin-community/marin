# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig, dataset_for_component
from levanter.data.text.examples import GrugLmExample
from levanter.schedule import BatchSchedule
from levanter.store.cache import TreeCache, write_levanter_cache
from marin.datakit.sft_sources import all_sft_sources

from experiments.june_tpu_67b_a2b.moe import sft_datakit_chat_mix as recipe
from experiments.june_tpu_67b_a2b.moe.sft_datakit_chat_mix import (
    _MIXTURE_BLOCK_SIZE,
    _SFT_FRACTION,
    _pretrain_components,
    _sft_mixture,
)
from experiments.june_tpu_67b_a2b.moe.train import ReplayDataConfig, build_train_dataset


def test_every_sft_source_gets_samples_in_each_mixture_block() -> None:
    mixture = _sft_mixture()

    assert set(mixture.components) == {f"sft/{name}" for name in all_sft_sources()}
    assert sum(mixture.weights.values()) == pytest.approx(_SFT_FRACTION)
    assert all(int(weight * _MIXTURE_BLOCK_SIZE) >= 1 for weight in mixture.weights.values())


def test_replay_preserves_lcr_components_and_relative_weights() -> None:
    components, weights = _pretrain_components()

    assert set(components) == set(weights)
    assert sum(weights.values()) == pytest.approx(1.0)
    phase_weights = recipe._phase_weights(1)
    assert weights == {f"pretrain/{name}": weight for name, weight in phase_weights.items()}
    tail = components["pretrain/tail"]
    assert {name.split("/")[0] for name in tail.children} == set(recipe._TAIL_BUCKETS)
    for bucket in recipe._TAIL_BUCKETS:
        assert f"pretrain/{bucket}" not in components
        assert sum(name.startswith(f"{bucket}/gt_64k/") for name in tail.children) == (
            0 if bucket in recipe._TAIL_BUCKETS_WITHOUT_LONG else 4
        )


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


def test_nested_replay_preserves_rare_component_samples() -> None:
    components = {
        name: DirectDatasetComponent(
            datasets={"train": ListAsyncDataset([GrugLmExample.causal(jnp.full((4,), value, dtype=jnp.int32))])}
        )
        for name, value in [("sft", 0), ("common", 1), ("rare", 2)]
    }
    main = LmDataConfig(
        components={"sft": components["sft"]},
        tokenizer="passthrough",
        vocab_size=8,
        train_weights={"sft": 1.0},
        shuffle=False,
        mixture_block_size=16,
    )
    replay = LmDataConfig(
        components={name: components[name] for name in ["common", "rare"]},
        tokenizer="passthrough",
        vocab_size=8,
        train_weights={"common": 15 / 16, "rare": 1 / 16},
        shuffle=False,
        mixture_block_size=16,
    )
    dataset = build_train_dataset(
        main,
        max_seq_len=4,
        batch_schedule=BatchSchedule(4),
        key=jax.random.PRNGKey(0),
        replay=ReplayDataConfig(data=replay, fraction=0.25),
    ).as_sync_dataset()
    values = [int(np.asarray(example.tokens)[0]) for example in dataset.get_batch(list(range(64)))]
    assert [values.count(value) for value in range(3)] == [48, 15, 1]
