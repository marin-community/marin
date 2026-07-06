# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample documents from a datakit store's per-bucket Levanter caches.

The clustered store persists each ``cluster=<C>/quality=<Q>`` bucket as a Levanter
cache of ``input_ids`` (the caches drop ``id``/text). This reads those caches
directly and detokenizes -- the ground truth of what the store actually holds,
independent of ducky or the upstream stage parquet that
:meth:`~experiments.datakit.web_explorer.queries.WebExplorer.store_bucket_samples`
reconstructs a bucket from. The tokenizer and each bucket cache are loaded lazily
(only when first sampled), so this adds nothing to the dashboard's startup cost.
"""

from __future__ import annotations

import functools
import logging
import math
import random

import numpy as np
from levanter.store.cache import CacheMetadata, TreeCache
from levanter.tokenizers import load_tokenizer

logger = logging.getLogger(__name__)

# Detokenize at most this many tokens per sample for display -- long documents
# would otherwise bloat the JSON payload and the browser.
_MAX_DISPLAY_TOKENS = 6000
# Exemplar shape for the per-bucket Levanter cache: one jagged ``input_ids`` row.
_EXEMPLAR = {"input_ids": np.zeros((0,), dtype=np.int32)}


def _run_indices(total: int, n: int, runs: int, seed: int) -> list[int]:
    """Pick ``n`` row indices as ``runs`` random *contiguous* windows.

    Reading contiguous rows keeps each window inside one (or few) tensorstore
    data chunks, so a window of length ``L`` costs ~1 chunk fetch instead of
    ``L``. Several windows at random offsets restore the spread that a single
    contiguous slice would lose; ``get_batch_sync`` issues all their chunk reads
    in one ``ts.Batch`` (concurrently). Fewer ``runs`` => longer windows =>
    faster but less diverse; more ``runs`` => the opposite.
    """
    n = max(1, min(n, total))
    runs = max(1, min(runs, n))
    run_len = math.ceil(n / runs)
    rng = random.Random(seed)
    max_start = max(0, total - run_len)
    n_starts = min(runs, max_start + 1)
    starts = sorted(rng.sample(range(max_start + 1), n_starts))
    picked: list[int] = []
    seen: set[int] = set()
    for start in starts:
        for i in range(start, min(start + run_len, total)):
            if i not in seen:
                seen.add(i)
                picked.append(i)
            if len(picked) >= n:
                return sorted(picked)
    return sorted(picked)


class StoreCacheSampler:
    """Lazily reads per-(cluster, quality) bucket caches and detokenizes samples."""

    def __init__(self, store_path: str, tokenizer_name: str, bucket_keys: set[tuple[int, int]]):
        self._store_path = store_path.rstrip("/")
        self._tokenizer_name = tokenizer_name
        self._bucket_keys = set(bucket_keys)
        self._tokenizer = None

    def _tok(self):
        if self._tokenizer is None:
            logger.info("loading tokenizer %s", self._tokenizer_name)
            self._tokenizer = load_tokenizer(self._tokenizer_name)
        return self._tokenizer

    @functools.lru_cache(maxsize=64)  # noqa: B019 -- bounded per-bucket cache handles for the server lifetime
    def _cache(self, cluster_id: int, quality_bucket: int) -> TreeCache:
        path = f"{self._store_path}/cluster={cluster_id}/quality={quality_bucket}"
        logger.info("opening bucket cache %s", path)
        return TreeCache.load(path, _EXEMPLAR, CacheMetadata.empty())

    def samples(self, cluster_id: int, quality_bucket: int, n: int, seed: int, runs: int) -> list[dict]:
        if (cluster_id, quality_bucket) not in self._bucket_keys:
            raise ValueError(f"no bucket cluster={cluster_id} quality={quality_bucket}")
        cache = self._cache(cluster_id, quality_bucket)
        total = len(cache)
        # Contiguous windows fetched in one batched (parallel) read -- random
        # point reads over a remote jagged store are pathologically slow.
        indices = _run_indices(total, n=n, runs=runs, seed=seed)
        rows = cache.get_batch_sync(indices)
        tok = self._tok()
        out = []
        for idx, row in zip(indices, rows, strict=True):
            ids = np.asarray(row["input_ids"]).reshape(-1)
            n_tokens = int(ids.shape[0])
            out.append(
                {
                    "index": idx,
                    "n_tokens": n_tokens,
                    "truncated": n_tokens > _MAX_DISPLAY_TOKENS,
                    "text": tok.decode(ids[:_MAX_DISPLAY_TOKENS].tolist()),
                }
            )
        return out
