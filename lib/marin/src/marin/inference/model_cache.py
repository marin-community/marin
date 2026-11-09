# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache identity shared by evaluation orchestration and serving workers."""

from levanter.model_cache import HfModelCacheEntry, cache_hf_model, hf_model_cache_entry

MODEL_CACHE_PREFIX = "quick-serve-models"


def model_cache_entry(model: str, cache_ttl_days: int, revision: str | None = None) -> HfModelCacheEntry | None:
    """Return the serving cache entry without downloading model weights."""
    if revision is not None:
        model = f"{model}@{revision}"
    return hf_model_cache_entry(model, cache_ttl_days=cache_ttl_days, cache_prefix=MODEL_CACHE_PREFIX)


def populate_model_cache(repo: str, revision: str, cache_ttl_days: int) -> str:
    """Mirror ``repo@revision`` into the serving cache of the storage this process runs against.

    Run it at the serving placement: the cache path depends on the ambient storage prefix, so a
    coordinator on another cluster would pick a different bucket than the accelerator worker.
    Returns immediately when the entry is already complete.
    """
    entry = model_cache_entry(repo, cache_ttl_days, revision)
    if entry is None:
        raise ValueError(f"{repo}@{revision} is not a cacheable Hugging Face model (cache_ttl_days={cache_ttl_days})")
    return cache_hf_model(entry.path, entry.repo, revision=entry.revision)
