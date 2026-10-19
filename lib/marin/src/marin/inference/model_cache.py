# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache identity shared by evaluation orchestration and serving workers."""

from levanter.model_cache import HfModelCacheEntry, hf_model_cache_entry

MODEL_CACHE_PREFIX = "quick-serve-models"


def model_cache_entry(model: str, cache_ttl_days: int, revision: str | None = None) -> HfModelCacheEntry | None:
    """Return the serving cache entry without downloading model weights."""
    if revision is not None:
        model = f"{model}@{revision}"
    return hf_model_cache_entry(model, cache_ttl_days=cache_ttl_days, cache_prefix=MODEL_CACHE_PREFIX)
