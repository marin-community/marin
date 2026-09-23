# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.execution.lazy import StepContext

from experiments.datakit.store.datakit_store import BucketCacheStats, ClusteredStoreData
from experiments.datakit.store.mixture import MixtureWeighting, store_mixture
from experiments.grug.fast_track.launch import build_h100_ladder_run


def _store(*buckets: BucketCacheStats) -> ClusteredStoreData:
    return ClusteredStoreData(
        cache_path="datakit/store",
        cluster_view=8,
        bucket_edges=[0.0, 1.0],
        split="train",
        buckets=list(buckets),
        source_names=["source"],
        tokenizer="hero-bpe-v16384",
        counters={},
    )


def _bucket(cluster: int, quality: int, tokens: int) -> BucketCacheStats:
    return BucketCacheStats(
        cluster_id=cluster,
        quality_bucket=quality,
        path=f"datakit/store/cluster={cluster}/quality={quality}",
        total_elements=3,
        total_tokens=tokens,
        n_shards=1,
    )


def test_store_mixture_includes_every_bucket_with_selected_weights():
    store = _store(_bucket(1, 0, 100), _bucket(7, 2, 300))

    proportional = store_mixture(store)
    uniform = store_mixture(store, weighting=MixtureWeighting.UNIFORM)

    assert set(proportional.components) == {"c01q0", "c07q2"}
    assert proportional.train_weights == {"c01q0": 100.0, "c07q2": 300.0}
    assert uniform.train_weights == {"c01q0": 1.0, "c07q2": 1.0}
    assert all(component.flat_cache for component in proportional.components.values())
    assert proportional.tokenizer == store.tokenizer
    assert not proportional.auto_build_caches


def test_store_mixture_rejects_missing_training_data():
    with pytest.raises(ValueError, match="produced no data"):
        store_mixture(_store())

    with pytest.raises(ValueError, match="expected > 0"):
        store_mixture(_store(_bucket(0, 0, 0)))


def test_store_mixture_excludes_buckets_shorter_than_one_sequence():
    store = _store(_bucket(1, 0, 4_095), _bucket(7, 2, 4_096))

    mixture = store_mixture(store, min_tokens_per_component=4_096)

    assert set(mixture.components) == {"c07q2"}
    assert mixture.train_weights == {"c07q2": 4_096.0}

    with pytest.raises(ValueError, match="no bucket with at least 8192 tokens"):
        store_mixture(store, min_tokens_per_component=8_192)


def test_fast_track_keeps_store_mixture_when_it_adds_validation_data():
    store = _store(_bucket(1, 0, 100), _bucket(7, 2, 300))
    training = build_h100_ladder_run(
        run_id="store-mixture-test",
        size="d512",
        num_steps=1,
        batch_size=8,
        version="test-dev",
        tokenizer=store.tokenizer,
        training_data=store_mixture(store),
        dense=True,
        no_eval=True,
    )

    context = StepContext.for_fingerprint(training.runtime_args.keys(), training.deps)
    config = training.build_config(context)

    weights = config.data.train_weights
    assert isinstance(weights, dict)
    assert weights["c01q0"] == 100.0
    assert weights["c07q2"] == 300.0
    assert all(
        weight == 0.0 for name, weight in weights.items() if name not in {"c01q0", "c07q2"}
    )


def test_fast_track_rejects_two_training_sources():
    store = _store(_bucket(1, 0, 100))

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_h100_ladder_run(
            run_id="store-mixture-test",
            size="d512",
            num_steps=1,
            batch_size=8,
            version="test-dev",
            tokenizer=store.tokenizer,
            train_cache_dir="other/cache",
            training_data=store_mixture(store),
            dense=True,
            no_eval=True,
        )
