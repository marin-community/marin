# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distributed Zephyr port of the 168-bucket Grug-MoE histogram build.

One Zephyr map task per bucket: sample 20k docs (seed 0, token-weighted) from the bucket's
levanter cache partition on the CoreWeave object-store mirror
``s3://marin-us-east-02a/marin/datakit/store_8ac06c74/``, embed with luxical (CPU torch),
assign against the FROZEN K=5000 centroids, and write ``part-<bucket>.parquet`` + ``<bucket>.npy``
(token-weighted RFF mean) to the CW output prefix. Same frozen basis as the local build
(``build_grug_histograms.py``) — same centroids, same int8 round-trip assignment, and the SAME
frozen RFF bandwidth (loaded from the shipped ``bandwidth.json``, never recomputed) — so this run's
features are byte-for-byte poolable with both the qsplit240 39-domain set and the local grug set.

The histogram math is reused verbatim from the local build (``build_domain_histograms`` +
``build_grug_histograms``); only the data path (CW S3 caches read via levanter, StoragePath I/O)
and the map/stage orchestration are new.

Design notes cribbed from ``experiments/datakit/embeddings/luxical/pipeline.py``:
- luxical weights are staged once on the driver to an in-region temp path and broadcast as a URL
  via ``ZephyrContext.put``; workers ``get_shared`` + download + ``Embedder.load`` (``@cache``),
  reused across tasks under ``InlineRunner``.
- the small frozen basis (centroids 3.8MB, bandwidth.json, lookups) is copied to the CW prefix
  first and its URLs broadcast the same way, so workers read the basis same-cloud.

Cross-cloud/creds: worker pods on cw-rno2a read+write ``s3://marin-us-east-02a`` same-cloud via the
cluster's task-env secret; ``StoragePath`` / levanter ``TreeCache.load`` resolve the s3 handle
(pyarrow-native S3 reads 400 against the CW store — always go through StoragePath/levanter).

Resource shaping: CPU-only tasks (``ResourceConfig(cpu=8, ram="16g")``, no device). Do NOT pin
``regions``/``--region RNO2A`` — the shared CPU capacity node lacks the ``iris.region`` label and a
pinned task hangs Pending forever; leave regions UNSET so it lands on spare cluster capacity.
"""

import json
import logging
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from huggingface_hub import hf_hub_download
from rigging.filesystem import StoragePath, marin_temp_bucket
from transformers import AutoTokenizer
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner
from zephyr.worker_context import zephyr_worker_ctx

# Import first: sets MARIN_I_WILL_PAY_FOR_ALL_FEES + LEVANTER_TS_CACHE_LIMIT before levanter.
from experiments.datakit.mixture_features import build_domain_histograms as bdh
from experiments.datakit.mixture_features.build_domain_histograms import DomainAccumulator, InsufficientSampleError
from experiments.datakit.mixture_features.build_grug_histograms import (
    SAMPLE_SIZE,
    TAIL_MIN_DOCS,
    bucket_key,
    read_docs,
)
from experiments.datakit.mixture_features.featurize import build_rff_map
from experiments.grug.moe.launch_datakit_moe_mix import _TAIL_BUCKETS

logger = logging.getLogger("grug_histograms_zephyr")

# CoreWeave object-store mirror of the datakit store + the run's output prefix.
CW_STORE = "s3://marin-us-east-02a/marin/datakit/store_8ac06c74"
CW_OUT = "s3://marin-us-east-02a/marin/user/rav/projects/mixing_via_embeddings/v0/grug_histograms_zephyr"

SEED = bdh.SEED
READ_THREADS = 16
MAX_WORKERS = 32

_LUXICAL_KEY = "luxical_npz_url"
_CENTROIDS_KEY = "centroids_url"
_BANDWIDTH_KEY = "rff_bandwidth"

REPO_ROOT = bdh.REPO_ROOT
# Frozen-basis inputs live in the bundled workspace (scratch/ is gitignored and would not ship to
# the cluster). centroids_5000.npy sha256 == the qsplit240 basis; bandwidth.json is the frozen RFF
# sigma copied from domain_histograms/ — never recomputed here.
INPUTS_DIR = os.path.join(REPO_ROOT, "experiments/datakit/mixture_features/grug_inputs")
CENTROIDS_LOCAL = os.path.join(INPUTS_DIR, "centroids_5000.npy")
BANDWIDTH_LOCAL = os.path.join(INPUTS_DIR, "bandwidth.json")


# --- bucket specs ---------------------------------------------------------------------


def cw_partition_path(bucket: str) -> str:
    cluster, quality = bucket_key(bucket)
    return f"{CW_STORE}/cluster={cluster}/quality={quality}"


def build_bucket_specs(buckets: list[str], partitions: dict) -> list[dict]:
    """One spec per bucket. Direct buckets carry a single partition; ``tail`` carries its 33
    children with per-child token counts for the proportional allocation."""
    specs = []
    for b in sorted(x for x in buckets if x != "tail"):
        cluster, tier = bucket_key(b)
        specs.append(
            {
                "bucket": b,
                "tier": tier,
                "is_tail": False,
                "sources": [{"path": cw_partition_path(b), "tokens": int(partitions[(cluster, tier)]["total_tokens"])}],
            }
        )
    tail_sources = [
        {"path": cw_partition_path(c), "tokens": int(partitions[bucket_key(c)]["total_tokens"]), "child": c}
        for c in _TAIL_BUCKETS
    ]
    specs.append({"bucket": "tail", "tier": -1, "is_tail": True, "sources": tail_sources})
    return specs


# --- worker-side shared artifacts -----------------------------------------------------


@cache
def _worker_embedder() -> Any:
    from luxical.embedder import Embedder  # noqa: PLC0415  # optional dep: luxical

    url = zephyr_worker_ctx().get_shared(_LUXICAL_KEY)
    fd, local = tempfile.mkstemp(prefix="luxical-", suffix=".npz")
    os.close(fd)
    StoragePath(url).download_to(local)
    return Embedder.load(local)


@cache
def _worker_centroids() -> np.ndarray:
    url = zephyr_worker_ctx().get_shared(_CENTROIDS_KEY)
    fd, local = tempfile.mkstemp(prefix="centroids-", suffix=".npy")
    os.close(fd)
    StoragePath(url).download_to(local)
    c = np.load(local).astype(np.float32)
    assert c.shape == (bdh.K, bdh.EMBED_DIM), c.shape
    return c


@cache
def _worker_tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(bdh.TOKENIZER_ID)


@cache
def _worker_rff_map():
    bandwidth: float = zephyr_worker_ctx().get_shared(_BANDWIDTH_KEY)
    return build_rff_map(bdh.RFF_DIM, bdh.EMBED_DIM, bdh.RFF_SEED, bandwidth)


# --- per-bucket sampling (CW caches) --------------------------------------------------


def _sample_one(path: str, n_docs: int, rng: np.random.Generator, pool: ThreadPoolExecutor):
    """Sample ~``n_docs`` docs from one CW partition cache via uniform-random contiguous ranges."""
    cache_obj = bdh.open_cache(path)  # levanter TreeCache.load on the s3:// path
    n = len(cache_obj)
    if n == 0:
        raise InsufficientSampleError(f"empty partition {path}")
    n_ranges = max(1, math.ceil(n_docs / bdh.RANGE_LEN))
    idx = bdh._range_indices(n, n_ranges, rng)
    docs = read_docs(cache_obj, idx, pool)
    if len(docs) > n_docs:
        keep = rng.choice(len(docs), size=n_docs, replace=False)
        docs = [docs[i] for i in keep]
    return docs, n


def _sample_bucket(spec: dict, pool: ThreadPoolExecutor) -> tuple[list, dict]:
    rng = bdh.domain_rng(spec["bucket"])
    if not spec["is_tail"]:
        docs, n_avail = _sample_one(spec["sources"][0]["path"], SAMPLE_SIZE, rng, pool)
        return docs, {"sources": [spec["sources"][0]["path"]]}

    total = sum(s["tokens"] for s in spec["sources"])
    child_seeds = rng.integers(0, 2**63, size=len(spec["sources"]))
    docs, children = [], {}
    for src, seed in zip(sorted(spec["sources"], key=lambda s: s["child"]), child_seeds, strict=True):
        if src["tokens"] == 0:
            continue
        want = max(TAIL_MIN_DOCS, round(SAMPLE_SIZE * src["tokens"] / total))
        child_docs, n_avail = _sample_one(src["path"], want, np.random.default_rng(int(seed)), pool)
        docs.extend(child_docs)
        children[src["child"]] = {
            "path": src["path"],
            "total_tokens": src["tokens"],
            "allocated_docs": int(want),
            "sampled_docs": len(child_docs),
            "docs_available": int(n_avail),
        }
    return docs, {"sources": [CW_STORE], "tail_children": children}


# --- the map task ---------------------------------------------------------------------


def process_bucket(spec: dict) -> dict:
    """Sample -> embed -> assign -> write ``part-<bucket>.parquet`` + ``<bucket>.npy`` to CW.

    Returns a small meta dict (no arrays) for driver-side assembly of ``_meta.json``.
    Idempotent: skips work if the parquet + npy already exist at the output prefix.
    """
    bucket = spec["bucket"]
    parquet_url = f"{CW_OUT}/part-{bucket}.parquet"
    rff_url = f"{CW_OUT}/rff/{bucket}.npy"
    if StoragePath(parquet_url).exists() and StoragePath(rff_url).exists():
        logger.info("%s: already present, skipping", bucket)
        counters.pipeline.update_counter("grug/buckets_skipped", 1)
        return {"bucket": bucket, "status": "skipped"}

    tokenizer = _worker_tokenizer()
    embedder = _worker_embedder()
    centroids = _worker_centroids()
    w_map, b_map = _worker_rff_map()

    with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
        docs, extra = _sample_bucket(spec, pool)
    norm_emb, token_lengths = bdh.embed_docs(docs, tokenizer, embedder)
    del docs
    cells = bdh.assign_cells(norm_emb, centroids)
    acc = DomainAccumulator(bucket, norm_emb, token_lengths, cells)
    counts = acc.cell_counts()
    total = sum(counts.values())
    cell_ids = sorted(counts)
    table = pa.table(
        {
            "domain": pa.array([bucket] * len(cell_ids), pa.string()),
            "cluster_id": pa.array(cell_ids, pa.int32()),
            "quality_bucket": pa.array([spec["tier"]] * len(cell_ids), pa.int8()),
            "token_count": pa.array([counts[c] for c in cell_ids], pa.int64()),
            "frac": pa.array([counts[c] / total for c in cell_ids], pa.float64()),
        }
    )
    rff = acc.rff_mean(w_map, b_map)

    _write_parquet_to_cw(table, parquet_url)
    _write_npy_to_cw(rff, rff_url)
    meta = {
        "bucket": bucket,
        "sample_size": int(norm_emb.shape[0]),
        "token_count": int(total),
        "occupied_cells_k5000": len(counts),
        "quality_bucket": spec["tier"],
        "mean_doc_tokens": float(token_lengths.mean()),
        **extra,
    }
    _write_json_to_cw(meta, f"{CW_OUT}/meta/{bucket}.json")
    counters.pipeline.update_counter("grug/buckets_done", 1)
    counters.pipeline.update_counter("grug/tokens", int(total))
    logger.info("%s: sampled=%d tokens=%d occupied_cells=%d", bucket, norm_emb.shape[0], total, len(counts))
    return {"bucket": bucket, "status": "done", "occupied_cells_k5000": len(counts), "token_count": int(total)}


def _write_parquet_to_cw(table: pa.Table, url: str) -> None:
    fd, local = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    pq.write_table(table, local)
    StoragePath(url).upload_from(local)
    os.unlink(local)


def _write_npy_to_cw(arr: np.ndarray, url: str) -> None:
    fd, local = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    np.save(local, arr)
    StoragePath(url).parent.mkdirs()
    StoragePath(url).upload_from(local)
    os.unlink(local)


def _write_json_to_cw(obj: dict, url: str) -> None:
    sp = StoragePath(url)
    sp.parent.mkdirs()
    sp.write_text(json.dumps(obj, indent=2))


# --- driver ---------------------------------------------------------------------------


def _stage_luxical() -> str:
    sanitized = bdh.LUXICAL_REPO.replace("/", "__")
    url = f"{marin_temp_bucket(ttl_days=1, prefix='luxical-staging')}/{sanitized}/{bdh.LUXICAL_WEIGHTS}"
    sp = StoragePath(url)
    if sp.exists():
        return url
    local = hf_hub_download(repo_id=bdh.LUXICAL_REPO, filename=bdh.LUXICAL_WEIGHTS)
    sp.parent.mkdirs()
    sp.upload_from(local)
    return url


def _stage_centroids_to_cw() -> str:
    """Copy the frozen centroids to the CW prefix so workers read the basis same-cloud."""
    url = f"{CW_OUT}/basis/centroids_5000.npy"
    sp = StoragePath(url)
    if not sp.exists():
        sp.parent.mkdirs()
        sp.upload_from(CENTROIDS_LOCAL)
    return url


def _frozen_bandwidth() -> float:
    info = bdh._read_json(BANDWIDTH_LOCAL)
    return float(info["rff_bandwidth"])


def run(buckets_json: str, store_meta: str, max_shards: int | None) -> None:
    buckets = bdh._read_json(buckets_json)
    assert len(buckets) == 168 and "tail" in buckets
    meta = bdh._read_json(store_meta)
    partitions = {(b["cluster_id"], b["quality_bucket"]): b for b in meta["buckets"]}
    specs = build_bucket_specs(buckets, partitions)
    if max_shards is not None:
        specs = specs[:max_shards]

    bandwidth = _frozen_bandwidth()
    luxical_url = _stage_luxical()
    centroids_url = _stage_centroids_to_cw()
    logger.info("staged luxical=%s centroids=%s bandwidth=%.6f", luxical_url, centroids_url, bandwidth)

    ds = Dataset.from_list(specs).map(process_bucket)
    ctx = ZephyrContext(
        # CPU-only; leave regions UNSET so the task lands on spare cw-rno2a capacity (pinning
        # RNO2A hangs Pending — the shared CPU node lacks the iris.region label).
        resources=ResourceConfig(cpu=8, ram="16g"),
        max_workers=min(MAX_WORKERS, len(specs)),
        name="grug-histograms",
        stage_runner_factory=InlineRunner,
    )
    ctx.put(_LUXICAL_KEY, luxical_url)
    ctx.put(_CENTROIDS_KEY, centroids_url)
    ctx.put(_BANDWIDTH_KEY, bandwidth)
    outcome = ctx.execute(ds, verbose=True)
    logger.info("counters: %s", dict(outcome.counters))
    return outcome
