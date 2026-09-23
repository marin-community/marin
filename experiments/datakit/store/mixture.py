# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a Levanter training mixture from a DataKit clustered store."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat

from experiments.datakit.store.datakit_store import ClusteredStoreData

logger = logging.getLogger(__name__)


class MixtureWeighting(StrEnum):
    """How to weight the non-empty DataKit store buckets."""

    TOKEN_PROPORTIONAL = "token_proportional"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class FlatCacheComponent:
    """One named flat cache and its fixed training weight."""

    cache_dir: str
    weight: float


def bucket_name(cluster_id: int, quality_bucket: int) -> str:
    """Use the DataKit store's ``cXXqY`` bucket name as the component key."""
    return f"c{cluster_id:02d}q{quality_bucket}"


def log_store_summary(store: ClusteredStoreData) -> None:
    """Log store totals and each non-empty bucket."""
    total_documents = sum(bucket.total_elements for bucket in store.buckets)
    total_tokens = sum(bucket.total_tokens for bucket in store.buckets)
    logger.info(
        "DataKit store: %d non-empty buckets, %d documents, %d tokens, sources=%s, tokenizer=%s, path=%s",
        len(store.buckets),
        total_documents,
        total_tokens,
        store.source_names,
        store.tokenizer,
        store.cache_path,
    )
    for bucket in sorted(store.buckets, key=lambda value: (value.cluster_id, value.quality_bucket)):
        logger.info(
            "  %s: documents=%d tokens=%d shards=%d",
            bucket_name(bucket.cluster_id, bucket.quality_bucket),
            bucket.total_elements,
            bucket.total_tokens,
            bucket.n_shards,
        )


def flat_cache_mixture(*, tokenizer: str, caches: Mapping[str, FlatCacheComponent]) -> LmDataConfig:
    """Build a fixed training mixture from named flat-cache directories."""
    components = {
        name: DatasetComponent(
            source=None,
            cache_dir=cache.cache_dir,
            format=TextLmDatasetFormat(),
            tags=[name],
            flat_cache=True,
        )
        for name, cache in caches.items()
    }
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        components=components,
        train_weights={name: cache.weight for name, cache in caches.items()},
        auto_build_caches=False,
    )


def store_mixture(
    store: ClusteredStoreData,
    *,
    weighting: MixtureWeighting = MixtureWeighting.TOKEN_PROPORTIONAL,
    min_tokens_per_component: int = 1,
) -> LmDataConfig:
    """Build a token-proportional or uniform mixture from usable store buckets.

    Raises:
        ValueError: If the store has no buckets, has invalid token counts, or
            has no bucket that meets ``min_tokens_per_component``.
    """
    if min_tokens_per_component < 1:
        raise ValueError(f"min_tokens_per_component must be positive, got {min_tokens_per_component}")
    if not store.buckets:
        raise ValueError(f"store at {store.cache_path} has no non-empty buckets; DataKit produced no data")

    for bucket in store.buckets:
        if bucket.total_tokens <= 0:
            name = bucket_name(bucket.cluster_id, bucket.quality_bucket)
            raise ValueError(f"bucket {name} at {bucket.path} has total_tokens={bucket.total_tokens}; expected > 0")

    usable_buckets = [bucket for bucket in store.buckets if bucket.total_tokens >= min_tokens_per_component]
    if not usable_buckets:
        raise ValueError(
            f"store at {store.cache_path} has no bucket with at least {min_tokens_per_component} tokens; "
            "each mixture component must meet the minimum token count"
        )

    caches: dict[str, FlatCacheComponent] = {}
    for bucket in usable_buckets:
        name = bucket_name(bucket.cluster_id, bucket.quality_bucket)
        if weighting is MixtureWeighting.TOKEN_PROPORTIONAL:
            weight = float(bucket.total_tokens)
        else:
            weight = 1.0
        # Loaded store artifacts resolve bucket.path to an absolute object-store path.
        caches[name] = FlatCacheComponent(cache_dir=bucket.path, weight=weight)

    logger.info(
        "store_mixture: %d of %d buckets, minimum_tokens=%d, %s weighting, tokenizer=%s",
        len(caches),
        len(store.buckets),
        min_tokens_per_component,
        weighting.value,
        store.tokenizer,
    )
    return flat_cache_mixture(tokenizer=store.tokenizer, caches=caches)
