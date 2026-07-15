# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build token-weighted content histograms for the 168 Grug-MoE mixing-swarm buckets.

Same frozen basis as the 39-domain qsplit240 build (``build_domain_histograms.py``): same
K=5000 centroids, same luxical embedder, same int8-round-trip assignment, and the SAME frozen
RFF bandwidth (loaded from the existing ``domain_histograms/bandwidth.json`` — never recomputed)
so the two sweeps' features are poolable.

Bucket -> data mapping (documented on issue #7067, canonical in
``experiments/grug/moe/launch_datakit_moe_mix.py``):

- direct bucket ``cNNqQ`` -> ``gs://marin-us-east5/datakit/store_8ac06c74/cluster=NN/quality=Q``
  (0-based, no zero padding on the GCS side); 167 buckets.
- ``tail`` -> token-count-weighted pooled sample over the 33 below-threshold partitions in
  ``_TAIL_BUCKETS`` (min 100 docs per nonzero child), one pooled histogram.

Each partition is a *sharded* levanter cache (top-level ``shard_ledger.json`` + many
``part-NNNNN-of-06301`` shard stores); ``TreeCache.load`` opens it directly, but the sync batch
path reads shards serially, so we fan sampled ranges out over a thread pool.

Resumable exactly like the original: per-bucket parquet + rff + meta + ``.parquet.done``
sentinel, ENOSPC retry on writes, and a hard pause (exit code 75) if free disk drops below
3GB so a shared-VM disk crunch never corrupts outputs.
"""

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# Import first: sets MARIN_I_WILL_PAY_FOR_ALL_FEES + LEVANTER_TS_CACHE_LIMIT before levanter.
from experiments.datakit.mixture_features import build_domain_histograms as bdh
from experiments.datakit.mixture_features.build_domain_histograms import (
    DomainAccumulator,
    InsufficientSampleError,
    _range_indices,
    _read_json,
    _write_with_retry,
    domain_rng,
)
from experiments.datakit.mixture_features.featurize import build_rff_map
from experiments.grug.moe.launch_datakit_moe_mix import _TAIL_BUCKETS

logger = logging.getLogger("build_grug_histograms")

STORE = "gs://marin-us-east5/datakit/store_8ac06c74"
SAMPLE_SIZE = bdh.SAMPLE_SIZE  # 20_000
RANGE_LEN = bdh.RANGE_LEN  # 100
SEED = bdh.SEED  # 0
TAIL_MIN_DOCS = 100
READ_THREADS = 16
DISK_FLOOR_BYTES = 3 * 1024**3
EXIT_LOW_DISK = 75

DEFAULT_OUT = os.path.join(bdh.REPO_ROOT, "scratch/mixture_features/grug_histograms")
DOMAIN_HIST_DIR = os.path.join(bdh.REPO_ROOT, "scratch/mixture_features/domain_histograms")


def bucket_key(name: str) -> tuple[int, int]:
    return int(name[1:3]), int(name[-1])


def partition_path(name: str) -> str:
    cluster, quality = bucket_key(name)
    return f"{STORE}/cluster={cluster}/quality={quality}"


def load_partitions(store_meta_path: str) -> dict[tuple[int, int], dict]:
    meta = _read_json(store_meta_path)
    assert meta["cache_path"] == STORE, meta["cache_path"]
    assert meta["tokenizer"] == bdh.TOKENIZER_ID, meta["tokenizer"]
    parts = {(b["cluster_id"], b["quality_bucket"]): b for b in meta["buckets"]}
    assert len(parts) == 200, len(parts)
    return parts


# --- sampling -------------------------------------------------------------------------


def read_docs(cache, idx: list[int], pool: ThreadPoolExecutor) -> list[tuple[np.ndarray, int]]:
    """Read docs at ``idx`` (contiguous ranges), truncating immediately; threaded over ranges.

    A single sharded TreeCache reads shards serially in its sync path; sampled ranges are
    shard-local, so mapping ranges over a thread pool parallelizes the shard fetches.
    """
    chunks = [idx[i : i + RANGE_LEN] for i in range(0, len(idx), RANGE_LEN)]

    def read(chunk: list[int]) -> list[tuple[np.ndarray, int]]:
        recs = cache.get_batch_sync(chunk)
        out = []
        for r in recs:
            arr = np.asarray(r["input_ids"])
            out.append((arr[: bdh.MAX_EMBED_TOKENS].astype(np.int32, copy=True), len(arr)))
        return out

    docs: list[tuple[np.ndarray, int]] = []
    for res in pool.map(read, chunks):
        docs.extend(res)
    return docs


def sample_partition(path: str, n_docs: int, rng: np.random.Generator, pool: ThreadPoolExecutor):
    """Sample ~``n_docs`` docs from one partition cache via uniform-random contiguous ranges."""
    cache = bdh.open_cache(path)
    n = len(cache)
    if n == 0:
        raise InsufficientSampleError(f"empty partition {path}")
    n_ranges = max(1, math.ceil(n_docs / RANGE_LEN))
    idx = _range_indices(n, n_ranges, rng)
    docs = read_docs(cache, idx, pool)
    if len(docs) > n_docs:
        keep = rng.choice(len(docs), size=n_docs, replace=False)
        docs = [docs[i] for i in keep]
    return docs, n


def sample_direct_bucket(bucket: str, pool: ThreadPoolExecutor) -> tuple[list, dict]:
    rng = domain_rng(bucket)
    docs, n_avail = sample_partition(partition_path(bucket), SAMPLE_SIZE, rng, pool)
    if len(docs) < bdh.MIN_SAMPLE and len(docs) < n_avail:
        raise InsufficientSampleError(f"{bucket}: only {len(docs)} of {n_avail} docs sampled")
    return docs, {"sources": [partition_path(bucket)]}


def tail_allocation(partitions: dict) -> dict[str, int]:
    """20k docs across the 33 tail children, proportional to child token counts, min 100."""
    tokens = {c: int(partitions[bucket_key(c)]["total_tokens"]) for c in _TAIL_BUCKETS}
    total = sum(tokens.values())
    return {c: max(TAIL_MIN_DOCS, round(SAMPLE_SIZE * t / total)) for c, t in tokens.items() if t > 0}


def sample_tail_bucket(partitions: dict, pool: ThreadPoolExecutor) -> tuple[list, dict]:
    rng = domain_rng("tail")
    alloc = tail_allocation(partitions)
    child_seeds = rng.integers(0, 2**63, size=len(alloc))
    docs: list[tuple[np.ndarray, int]] = []
    children_meta = {}
    for (child, want), seed in zip(sorted(alloc.items()), child_seeds, strict=True):
        child_rng = np.random.default_rng(int(seed))
        child_docs, n_avail = sample_partition(partition_path(child), want, child_rng, pool)
        docs.extend(child_docs)
        children_meta[child] = {
            "path": partition_path(child),
            "total_tokens": int(partitions[bucket_key(child)]["total_tokens"]),
            "allocated_docs": int(want),
            "sampled_docs": len(child_docs),
            "docs_available": int(n_avail),
        }
    return docs, {"sources": [STORE], "tail_children": children_meta}


# --- persistence ----------------------------------------------------------------------


def write_bucket_parquet(out_dir: str, bucket: str, tier: int, counts: dict[int, int]) -> tuple[str, int]:
    total = sum(counts.values())
    cells = sorted(counts)
    table = pa.table(
        {
            "domain": pa.array([bucket] * len(cells), pa.string()),
            "cluster_id": pa.array(cells, pa.int32()),
            "quality_bucket": pa.array([tier] * len(cells), pa.int8()),
            "token_count": pa.array([counts[c] for c in cells], pa.int64()),
            "frac": pa.array([counts[c] / total for c in cells], pa.float64()),
        }
    )
    path = os.path.join(out_dir, f"part-{bucket}.parquet")
    _write_with_retry(lambda: pq.write_table(table, path))
    return path, total


def bucket_done(out_dir: str, bucket: str) -> bool:
    return all(
        os.path.exists(os.path.join(out_dir, sub, f"{bucket}{ext}"))
        for sub, ext in (("", ".parquet.done"), ("rff", ".npy"), ("meta", ".json"))
    ) and os.path.exists(os.path.join(out_dir, f"part-{bucket}.parquet"))


def persist_bucket(out_dir: str, acc: DomainAccumulator, w_map, b_map, tier: int, avail_tokens: int, extra: dict):
    counts = acc.cell_counts()
    path, total = write_bucket_parquet(out_dir, acc.domain, tier, counts)
    rff = acc.rff_mean(w_map, b_map)
    _write_with_retry(lambda: np.save(os.path.join(out_dir, "rff", f"{acc.domain}.npy"), rff))
    meta = {
        "sample_size": int(acc.norm_emb.shape[0]),
        "seed": SEED,
        "token_count": int(total),
        "occupied_cells_k5000": len(counts),
        "quality_bucket": tier,
        "parquet": os.path.basename(path),
        "bucket_stats": {
            "total_tokens_available": int(avail_tokens),
            "mean_doc_tokens": float(acc.token_lengths.mean()),
            "duplicate_frac": None,
            "loss_masked_frac": 0.0,
        },
        **extra,
    }

    def _write_meta():
        with open(os.path.join(out_dir, "meta", f"{acc.domain}.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

    _write_with_retry(_write_meta)
    open(os.path.join(out_dir, f"{acc.domain}.parquet.done"), "w").close()


def write_buckets_table(out_dir: str, buckets: list[str], partitions: dict) -> None:
    rows = []
    for b in buckets:
        if b == "tail":
            total = sum(int(partitions[bucket_key(c)]["total_tokens"]) for c in _TAIL_BUCKETS)
            rows.append((b, -1, -1, total, STORE))
        else:
            cluster, tier = bucket_key(b)
            rows.append((b, cluster, tier, int(partitions[bucket_key(b)]["total_tokens"]), partition_path(b)))
    table = pa.table(
        {
            "bucket": pa.array([r[0] for r in rows], pa.string()),
            "cluster_id": pa.array([r[1] for r in rows], pa.int32()),
            "quality_tier": pa.array([r[2] for r in rows], pa.int8()),
            "total_tokens": pa.array([r[3] for r in rows], pa.int64()),
            "source_path": pa.array([r[4] for r in rows], pa.string()),
        }
    )
    _write_with_retry(lambda: pq.write_table(table, os.path.join(out_dir, "buckets_table.parquet")))


def assemble_outputs(out_dir: str, buckets: list[str], bw_info: dict, partitions: dict) -> None:
    per_bucket = {}
    rff = {}
    for b in buckets:
        per_bucket[b] = _read_json(os.path.join(out_dir, "meta", f"{b}.json"))
        rff[b] = np.load(os.path.join(out_dir, "rff", f"{b}.npy"))
    order = sorted(rff)
    np.savez(
        os.path.join(out_dir, "rff_means.npz"),
        domains=np.array(order),
        rff_means=np.stack([rff[b] for b in order], axis=0),
    )
    basis = {
        "embedder": bdh.EMBEDDER_ID,
        "tokenizer": bdh.TOKENIZER_ID,
        "centroids_path": bdh.CENTROIDS_GCS,
        "centroids_sha256": bdh.sha256_file(os.path.join(bdh.BASIS_DIR, "centroids_5000.npy")),
        "k": bdh.K,
        "view_paths": {str(v): bdh.VIEW_GCS[v] for v in bdh.VIEWS},
        "view_sha256": {
            str(v): bdh.sha256_file(os.path.join(bdh.BASIS_DIR, os.path.basename(bdh.VIEW_GCS[v]))) for v in bdh.VIEWS
        },
        "quality_scorer": None,
        "quality_scorer_sha256": None,
        "rff_dim": bdh.RFF_DIM,
        "rff_seed": bdh.RFF_SEED,
        "rff_bandwidth": bw_info["rff_bandwidth"],
    }
    meta = {
        "basis": basis,
        "sweep": {
            "name": "grug-moe-mix-swarm",
            "store": STORE,
            "n_buckets": len(buckets),
            "mapping": (
                "cNNqQ -> <store>/cluster=NN/quality=Q (0-based); tail = token-weighted concat sample over _TAIL_BUCKETS"
            ),
            "tail_children": list(_TAIL_BUCKETS),
        },
        "sampling": {
            "strategy": (
                f"{SAMPLE_SIZE} docs/bucket via contiguous ranges of {RANGE_LEN} docs at uniform-random "
                f"starts within the bucket's store partition (sharded levanter cache, threaded range "
                f"reads). tail allocates the {SAMPLE_SIZE} across its 33 children proportional to child "
                f"total_tokens (min {TAIL_MIN_DOCS} docs per nonzero child), single pooled histogram. "
                f"token weight = full stored doc length; embedding text truncated to first "
                f"{bdh.MAX_EMBED_TOKENS} tokens. per-bucket rng seeded from (seed={SEED}, bucket)."
            ),
            "sample_size": SAMPLE_SIZE,
            "range_len": RANGE_LEN,
            "seed": SEED,
            "quant_scale": bdh.QUANT_SCALE,
            "rff_bandwidth_calib": (
                "FROZEN — reused from the 39-domain qsplit240 build "
                "(scratch/mixture_features/domain_histograms/bandwidth.json); never recomputed"
            ),
            "cross_region_note": (
                "sampled from the us-east5 store on a us-west2 VM; MARIN_I_WILL_PAY_FOR_ALL_FEES disables "
                "the levanter full-store pre-charge (real sampled egress ~10-20GB within US)."
            ),
        },
        "buckets": per_bucket,
        "rff_means_file": "rff_means.npz",
        "buckets_table_file": "buckets_table.parquet",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": bdh.git_sha(),
    }
    with open(os.path.join(out_dir, "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    write_buckets_table(out_dir, buckets, partitions)
    logger.info("assembled %d buckets + rff_means.npz + _meta.json + buckets_table.parquet in %s", len(buckets), out_dir)


# --- driver ---------------------------------------------------------------------------


def ensure_frozen_bandwidth(out_dir: str) -> dict:
    """Copy the qsplit240 frozen bandwidth into the output dir; NEVER compute one here."""
    dst = os.path.join(out_dir, "bandwidth.json")
    src = os.path.join(DOMAIN_HIST_DIR, "bandwidth.json")
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)
    info = _read_json(dst)
    ref = _read_json(src)
    if info["rff_bandwidth"] != ref["rff_bandwidth"]:
        raise RuntimeError(f"bandwidth fork: {info['rff_bandwidth']} != frozen {ref['rff_bandwidth']}")
    logger.info("using frozen RFF bandwidth %.6f (qsplit240 basis)", info["rff_bandwidth"])
    return info


def check_disk(out_dir: str) -> None:
    free = shutil.disk_usage(out_dir).free
    if free < DISK_FLOOR_BYTES:
        logger.error(
            "PAUSED: free disk %.2fGB below %.1fGB floor — resume after cleanup",
            free / 1024**3,
            DISK_FLOOR_BYTES / 1024**3,
        )
        sys.exit(EXIT_LOW_DISK)


def verify_basis_identity() -> None:
    """The grug features must live in the exact frozen basis — fail fast on any drift."""
    ref = _read_json(os.path.join(DOMAIN_HIST_DIR, "_meta.json"))["basis"]
    local_sha = bdh.sha256_file(os.path.join(bdh.BASIS_DIR, "centroids_5000.npy"))
    if local_sha != ref["centroids_sha256"]:
        raise RuntimeError(f"centroids sha mismatch: {local_sha} != {ref['centroids_sha256']}")
    for v in bdh.VIEWS:
        sha = bdh.sha256_file(os.path.join(bdh.BASIS_DIR, os.path.basename(bdh.VIEW_GCS[v])))
        if sha != ref["view_sha256"][str(v)]:
            raise RuntimeError(f"view {v} sha mismatch")
    logger.info("basis identity verified against domain_histograms/_meta.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--buckets-json", required=True, help="path to grug_buckets.json (168 names incl. tail)")
    ap.add_argument("--store-meta", required=True, help="path to the store's .artifact.json copy")
    ap.add_argument("--limit", type=int, default=0, help="process at most N pending buckets (0 = all; for smoke tests)")
    args = ap.parse_args()

    buckets = _read_json(args.buckets_json)
    assert len(buckets) == 168 and "tail" in buckets, (len(buckets), "tail" in buckets)
    partitions = load_partitions(args.store_meta)
    direct = sorted(b for b in buckets if b != "tail")
    covered = {bucket_key(b) for b in direct} | {bucket_key(c) for c in _TAIL_BUCKETS}
    assert covered == set(partitions), "bucket mapping does not cover the store exactly"

    os.makedirs(os.path.join(args.out, "rff"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "meta"), exist_ok=True)
    verify_basis_identity()
    bw_info = ensure_frozen_bandwidth(args.out)
    w_map, b_map = build_rff_map(bdh.RFF_DIM, bdh.EMBED_DIM, bdh.RFF_SEED, bw_info["rff_bandwidth"])

    centroids = np.load(os.path.join(bdh.BASIS_DIR, "centroids_5000.npy")).astype(np.float32)
    assert centroids.shape == (bdh.K, bdh.EMBED_DIM), centroids.shape
    logger.info("loading tokenizer + luxical embedder")
    tokenizer = AutoTokenizer.from_pretrained(bdh.TOKENIZER_ID)
    from luxical.embedder import Embedder  # noqa: PLC0415

    embedder = Embedder.load(hf_hub_download(bdh.LUXICAL_REPO, bdh.LUXICAL_WEIGHTS))

    order = [*direct, "tail"]
    pending = [b for b in order if not bucket_done(args.out, b)]
    if args.limit:
        pending = pending[: args.limit]
    logger.info("%d/%d buckets pending", len(pending), len(order))

    def sample_bucket(bucket: str, pool: ThreadPoolExecutor):
        if bucket == "tail":
            return sample_tail_bucket(partitions, pool)
        return sample_direct_bucket(bucket, pool)

    with ThreadPoolExecutor(max_workers=READ_THREADS) as read_pool, ThreadPoolExecutor(max_workers=1) as prefetcher:
        future = prefetcher.submit(sample_bucket, pending[0], read_pool) if pending else None
        for i, bucket in enumerate(pending):
            check_disk(args.out)
            t0 = time.time()
            docs, extra = future.result()
            future = prefetcher.submit(sample_bucket, pending[i + 1], read_pool) if i + 1 < len(pending) else None
            t_read = time.time() - t0
            norm_emb, token_lengths = bdh.embed_docs(docs, tokenizer, embedder)
            del docs
            t_embed = time.time() - t0 - t_read
            cells = bdh.assign_cells(norm_emb, centroids)
            acc = DomainAccumulator(bucket, norm_emb, token_lengths, cells)
            if bucket == "tail":
                tier = -1
                avail = sum(int(partitions[bucket_key(c)]["total_tokens"]) for c in _TAIL_BUCKETS)
            else:
                tier = bucket_key(bucket)[1]
                avail = int(partitions[bucket_key(bucket)]["total_tokens"])
            persist_bucket(args.out, acc, w_map, b_map, tier, avail, extra)
            logger.info(
                "[%d/%d] %s: sampled=%d tokens=%d wait=%.1fs embed=%.1fs occupied_cells=%d",
                i + 1,
                len(pending),
                bucket,
                acc.norm_emb.shape[0],
                int(token_lengths.sum()),
                t_read,
                t_embed,
                len(np.unique(cells)),
            )
            del acc, norm_emb, token_lengths, cells

    if all(bucket_done(args.out, b) for b in order):
        assemble_outputs(args.out, order, bw_info, partitions)
    else:
        logger.warning("not all buckets done; pending=%s", [b for b in order if not bucket_done(args.out, b)])


if __name__ == "__main__":
    main()
