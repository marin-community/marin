# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One packed SFT epoch with held-out examples and long-document replay."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent, DatasetComponentBase, LmDataConfig

CONTEXT = 262_144
SFT_PREFIX = "sft/source/"
SFT_POOL = "sft/pooled"
REPLAY_PREFIX = "pretrain/"


@dataclass(frozen=True)
class SftSource:
    cache_dir: str
    packed_sequences: int


@dataclass(frozen=True)
class ReplaySource:
    short_cache: str | None
    long_cache: str | None
    weight: float


@dataclass(frozen=True)
class SftMixture:
    data: LmDataConfig
    training_counts: dict[str, int]
    holdout_counts: dict[str, int]


def one_epoch_schedule(
    sft_counts: Mapping[str, int],
    replay_weights: Mapping[str, float],
    *,
    steps: int,
    batch_size: int,
    block_size: int,
) -> list[tuple[int, dict[str, float]]]:
    if steps * batch_size % block_size or block_size % batch_size:
        raise ValueError("Mixture blocks must divide the run and align with optimizer steps")
    blocks = steps * batch_size // block_size
    if any(count < blocks for count in sft_counts.values()):
        raise ValueError("Pool small SFT sources before allocating the epoch")
    replay_total = sum(replay_weights.values())
    schedule = []
    for block in range(blocks):
        counts = {name: (block + 1) * count // blocks - block * count // blocks for name, count in sft_counts.items()}
        replay_draws = block_size - sum(counts.values())
        targets = {name: replay_draws * weight / replay_total for name, weight in replay_weights.items()}
        replay_counts = {name: max(1, int(target)) for name, target in targets.items()}
        remainder = replay_draws - sum(replay_counts.values())
        if remainder < 0:
            raise ValueError("Too many replay components for the available block slots")
        for name in sorted(targets, key=lambda name: targets[name] % 1, reverse=True)[:remainder]:
            replay_counts[name] += 1
        counts.update(replay_counts)
        assert sum(counts.values()) == block_size
        schedule.append((block * block_size // batch_size, {name: count / block_size for name, count in counts.items()}))
    return schedule


def build_mixture(
    sft: Mapping[str, SftSource],
    replay: Mapping[str, ReplaySource],
    *,
    tokenizer: str,
    cache_catalog: str | None,
    steps: int,
    batch_size: int,
    block_size: int,
    heldout_sft_sources: tuple[str, ...],
    holdout_sequences: int,
) -> SftMixture:
    blocks = steps * batch_size // block_size
    components: dict[str, DatasetComponentBase] = {}
    counts = {}
    pooled = {}
    for name, source in sorted(sft.items()):
        component = DatasetComponent(cache_dir=source.cache_dir, flat_cache=True, pack=CONTEXT, tags=["sft"])
        if source.packed_sequences < blocks:
            pooled[name] = component
        else:
            components[SFT_PREFIX + name] = component
            counts[SFT_PREFIX + name] = source.packed_sequences
    if pooled:
        components[SFT_POOL] = ConcatDatasetComponent(children=pooled, tags=["sft"])
        counts[SFT_POOL] = sum(sft[name].packed_sequences for name in pooled)

    if missing := set(heldout_sft_sources) - sft.keys():
        raise ValueError(f"Unknown holdout sources: {sorted(missing)}")
    heldout_names = list(
        dict.fromkeys(SFT_POOL if name in pooled else SFT_PREFIX + name for name in heldout_sft_sources)
    )
    if pooled and SFT_POOL not in heldout_names:
        heldout_names.append(SFT_POOL)
    holdouts = {name: holdout_sequences for name in heldout_names}
    counts = {name: count - holdouts.get(name, 0) for name, count in counts.items()}
    replay_fraction = 1 - sum(counts.values()) / (steps * batch_size)
    if not 0 < replay_fraction < 1 or not math.isclose(sum(source.weight for source in replay.values()), 1):
        raise ValueError("Expected a positive replay allocation and normalized replay weights")

    replay_weights = {}
    replay_pool = {}
    pool_weight = 0.0
    for name, source in sorted(replay.items()):
        if source.weight == 0:
            continue
        children = {}
        for stratum, path, copies in (("short", source.short_cache, 1), ("long", source.long_cache, 8)):
            if path is not None:
                for copy in range(copies):
                    children[f"{name}/{stratum}/{copy}"] = DatasetComponent(cache_dir=path, flat_cache=True)
        if not children:
            raise ValueError(f"No replay cache for {name}")
        if source.weight * replay_fraction * block_size < 1.000001:
            replay_pool.update(children)
            pool_weight += source.weight
        else:
            components[REPLAY_PREFIX + name] = ConcatDatasetComponent(children=children, tags=["replay"])
            replay_weights[REPLAY_PREFIX + name] = source.weight
    if replay_pool:
        if pool_weight * replay_fraction * block_size < 1.000001:
            smallest = min(replay_weights, key=replay_weights.__getitem__)
            component = components.pop(smallest)
            assert isinstance(component, ConcatDatasetComponent)
            replay_pool.update(component.children)
            pool_weight += replay_weights.pop(smallest)
        components[REPLAY_PREFIX + "pooled"] = ConcatDatasetComponent(children=replay_pool, tags=["replay"])
        replay_weights[REPLAY_PREFIX + "pooled"] = pool_weight

    schedule = one_epoch_schedule(counts, replay_weights, steps=steps, batch_size=batch_size, block_size=block_size)
    data = LmDataConfig(
        tokenizer=tokenizer,
        cache_catalog=cache_catalog,
        components=components,
        train_weights=schedule,
        num_validation_sequences=holdouts,
        auto_build_caches=False,
        shuffle=True,
        block_cross_document_attention=True,
        mixture_block_size=block_size,
    )
    return SftMixture(data, counts, holdouts)
