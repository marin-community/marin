# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local web visualizer for the datakit clustered-store output.

Reads a :class:`~experiments.datakit.store.datakit_store.ClusteredStoreData`
artifact (``<artifact>/artifact.json`` + per-bucket Levanter caches under
``cluster=<C>/quality=<Q>/``) and serves a single-page web UI that shows:

* per-(cluster, quality) bucket statistics (doc count, token count, avg
  tokens/doc, quality-score range) as a sortable heatmap, and
* on-demand samples of documents drawn from any bucket -- the bucket caches
  store only ``input_ids``, so samples are reconstructed by detokenizing with
  the artifact's tokenizer (a near-exact, slightly lossy roundtrip).

Run locally against a GCS (or local) artifact:

    uv run python experiments/datakit/viz/server.py \\
        --artifact gs://marin-<region>/datakit/store_<hash>

then open http://localhost:8000. Reads are lazy -- a bucket's cache is only
loaded the first time it is sampled, then cached in-process.
"""

from __future__ import annotations

import argparse
import functools
import logging
import random
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from levanter.store.cache import CacheMetadata, TreeCache
from levanter.tokenizers import load_tokenizer
from marin.execution.artifact import Artifact
from rigging.log_setup import configure_logging

from experiments.datakit.store.datakit_store import BucketCacheStats, ClusteredStoreData

logger = logging.getLogger(__name__)

# Detokenize at most this many tokens per sample for display -- long documents
# would otherwise bloat the JSON payload and the browser.
_MAX_DISPLAY_TOKENS = 6000
_INDEX_HTML = Path(__file__).with_name("index.html")
# Exemplar shape for the per-bucket Levanter cache: one jagged ``input_ids`` row.
_EXEMPLAR = {"input_ids": np.zeros((0,), dtype=np.int32)}


def _quality_range_label(quality_bucket: int, thresholds: list[float]) -> str:
    """Human-readable score interval for a quality bucket (``bisect_right`` cutoffs)."""
    lo = "0.0" if quality_bucket == 0 else f"{thresholds[quality_bucket - 1]:g}"
    hi = "1.0" if quality_bucket >= len(thresholds) else f"{thresholds[quality_bucket]:g}"
    upper = "]" if quality_bucket >= len(thresholds) else ")"
    return f"[{lo}, {hi}{upper}"


def _bucket_view(bucket: BucketCacheStats, thresholds: list[float]) -> dict:
    avg = bucket.total_tokens / bucket.total_elements if bucket.total_elements else 0.0
    return {
        "cluster_id": bucket.cluster_id,
        "quality_bucket": bucket.quality_bucket,
        "quality_range": _quality_range_label(bucket.quality_bucket, thresholds),
        "total_elements": bucket.total_elements,
        "total_tokens": bucket.total_tokens,
        "avg_tokens": round(avg, 1),
        "n_shards": bucket.n_shards,
    }


class StoreViz:
    """Holds the loaded artifact + lazily-loaded per-bucket caches and tokenizer."""

    def __init__(self, artifact_path: str, tokenizer_override: str | None):
        self.artifact_path = artifact_path.rstrip("/")
        logger.info("loading artifact metadata from %s", self.artifact_path)
        store = Artifact.from_path(self.artifact_path, ClusteredStoreData)
        assert isinstance(store, ClusteredStoreData), f"not a ClusteredStoreData artifact: {self.artifact_path}"
        self.store = store
        self._buckets = {(b.cluster_id, b.quality_bucket): b for b in store.buckets}
        tokenizer_name = tokenizer_override or store.tokenizer
        logger.info("loading tokenizer %s", tokenizer_name)
        self.tokenizer = load_tokenizer(tokenizer_name)

    def summary(self) -> dict:
        store = self.store
        buckets = [_bucket_view(b, store.quality_thresholds) for b in store.buckets]
        return {
            "artifact_path": self.artifact_path,
            "cluster_view": store.cluster_view,
            "quality_thresholds": store.quality_thresholds,
            "n_quality_buckets": len(store.quality_thresholds) + 1,
            "split": store.split,
            "tokenizer": store.tokenizer,
            "source_names": store.source_names,
            "counters": store.counters,
            "n_buckets": len(buckets),
            "n_clusters": len({b["cluster_id"] for b in buckets}),
            "total_elements": sum(b["total_elements"] for b in buckets),
            "total_tokens": sum(b["total_tokens"] for b in buckets),
            "buckets": buckets,
        }

    @functools.lru_cache(maxsize=64)  # noqa: B019  -- bounded per-bucket cache handles for the server lifetime
    def _cache(self, cluster_id: int, quality_bucket: int) -> TreeCache:
        path = f"{self.artifact_path}/cluster={cluster_id}/quality={quality_bucket}"
        logger.info("opening bucket cache %s", path)
        return TreeCache.load(path, _EXEMPLAR, CacheMetadata.empty())

    def samples(self, cluster_id: int, quality_bucket: int, n: int, seed: int) -> dict:
        key = (cluster_id, quality_bucket)
        if key not in self._buckets:
            raise HTTPException(status_code=404, detail=f"no bucket cluster={cluster_id} quality={quality_bucket}")
        cache = self._cache(cluster_id, quality_bucket)
        total = len(cache)
        n = max(1, min(n, total))
        indices = sorted(random.Random(seed).sample(range(total), n))
        rows = cache.get_batch_sync(indices)
        samples = []
        for idx, row in zip(indices, rows, strict=True):
            ids = np.asarray(row["input_ids"]).reshape(-1)
            n_tokens = int(ids.shape[0])
            display_ids = ids[:_MAX_DISPLAY_TOKENS].tolist()
            samples.append(
                {
                    "index": idx,
                    "n_tokens": n_tokens,
                    "truncated": n_tokens > _MAX_DISPLAY_TOKENS,
                    "text": self.tokenizer.decode(display_ids),
                }
            )
        return {
            "cluster_id": cluster_id,
            "quality_bucket": quality_bucket,
            "quality_range": _quality_range_label(quality_bucket, self.store.quality_thresholds),
            "total_elements": self._buckets[key].total_elements,
            "total_tokens": self._buckets[key].total_tokens,
            "seed": seed,
            "n_returned": len(samples),
            "samples": samples,
        }


def build_app(viz: StoreViz) -> FastAPI:
    app = FastAPI(title="datakit store viz")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_INDEX_HTML)

    @app.get("/api/summary")
    def summary() -> JSONResponse:
        return JSONResponse(viz.summary())

    @app.get("/api/bucket/{cluster_id}/{quality_bucket}/samples")
    def bucket_samples(cluster_id: int, quality_bucket: int, n: int = 20, seed: int = 0) -> JSONResponse:
        return JSONResponse(viz.samples(cluster_id, quality_bucket, n=n, seed=seed))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", required=True, help="Path to the clustered-store artifact (gs:// or local).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Override the artifact's tokenizer name (defaults to the one recorded in artifact.json).",
    )
    args = parser.parse_args()

    configure_logging(logging.INFO)
    viz = StoreViz(args.artifact, args.tokenizer)
    summary = viz.summary()
    logger.info(
        "serving %d buckets across %d clusters (%d docs, %d tokens) on http://%s:%d",
        summary["n_buckets"],
        summary["n_clusters"],
        summary["total_elements"],
        summary["total_tokens"],
        args.host,
        args.port,
    )
    uvicorn.run(build_app(viz), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
