# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build token-weighted content histograms for the 39 qsplit240 domains.

For each domain we sample documents from its levanter tokenized cache(s), detokenize with the
marin tokenizer, embed with Luxical-One, assign each document to its K=5000 spherical-k-means
cell exactly as the datakit ``assign`` pipeline does (L2-normalize -> int8 quant round-trip ->
nearest centroid by squared L2), and accumulate cell mass weighted by the document's full token
length. In the same pass we accumulate a token-weighted random-Fourier-feature mean (cluster-free
arm) and a :class:`BucketStats` summary.

Streaming: raw text and embeddings are never persisted; only per-cell token counts and the RFF
running mean survive per document. Outputs (parquet per domain + ``rff_means.npz`` + ``_meta.json``)
land under ``--out``.

The RFF bandwidth (median heuristic) is computed ONCE from a 2k-document subsample pooled across
the first four processed domains, then frozen into the persisted :class:`MixtureBasis`.

Cross-region note: this VM is in us-central1 and the caches are in us-east5. Levanter charges a
tokenized store's *full* on-disk size against the cross-region transfer budget on open, which is a
wild over-estimate for the tiny random sample we actually read (zarr v3 sharding_indexed reads only
the touched inner chunks). We set ``MARIN_I_WILL_PAY_FOR_ALL_FEES`` to disable that pessimistic
pre-charge; the real sampled egress is a few GB within the US multi-region.
"""

import argparse
import errno
import hashlib
import json
import logging
import math
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

os.environ.setdefault("MARIN_I_WILL_PAY_FOR_ALL_FEES", "1")
# Cap tensorstore's in-memory chunk cache (default 1GB) — this container is memory-limited and we
# only stream a small random sample. Must be set before levanter.store is imported.
os.environ.setdefault("LEVANTER_TS_CACHE_LIMIT", str(128 * 1024 * 1024))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from levanter.store.cache import TreeCache
from transformers import AutoTokenizer

from experiments.datakit.mixture_features.featurize import (
    InsufficientSampleError,
    build_rff_map,
    rff_features,
)

logger = logging.getLogger("build_domain_histograms")

# --- Frozen basis / sampling constants ------------------------------------------------
K = 5000
VIEWS = (40, 1000)
RFF_DIM = 2048
RFF_SEED = 0
EMBED_DIM = 192
SAMPLE_SIZE = 20_000
RANGE_LEN = 100  # contiguous docs per sampled range
READ_CHUNK = 1000  # docs per get_batch_sync call (bounds peak memory of long-doc reads)
READ_THREADS = 6  # concurrent source reads for multi-source domains
MIN_SAMPLE = 10_000
CALIB_SIGMA_PER_DOMAIN = 500  # docs pooled per calib domain for the median heuristic
CALIB_DOMAINS = 4
EMBED_BATCH = 4096
MAX_EMBED_TOKENS = 2048  # truncate long docs for embedding; token weight uses full length
SEED = 0

QUANT_RANGE = 0.6
QUANT_SCALE = QUANT_RANGE / 127  # matches experiments/datakit/embeddings/luxical/pipeline.py

TOKENIZER_ID = "marin-community/marin-tokenizer"
EMBEDDER_ID = "luxical-one-rc4"
LUXICAL_REPO = "DatologyAI/luxical-one"
LUXICAL_WEIGHTS = "luxical_one_rc4.npz"

CACHE_EXEMPLAR = {"input_ids": np.zeros((0,), dtype=np.int32)}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASIS_DIR = os.path.join(REPO_ROOT, "scratch/mixture_features/basis")
DEFAULT_OUT = os.path.join(REPO_ROOT, "scratch/mixture_features/domain_histograms")
CENTROIDS_GCS = "gs://marin-eu-west4/datakit/cluster/train_centroids_22d1e89d/centroids_5000.npy"
VIEW_GCS = {
    1000: "gs://marin-eu-west4/datakit/cluster/train_centroids_22d1e89d/lookup_5000_to_1000.npy",
    40: "gs://marin-eu-west4/datakit/cluster/train_centroids_22d1e89d/lookup_5000_to_40.npy",
}


# --- helpers --------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()


def open_cache(train_path: str) -> TreeCache:
    return TreeCache.load(train_path, CACHE_EXEMPLAR)


def _range_indices(n_docs: int, n_ranges: int, rng: np.random.Generator) -> list[int]:
    """Pick ``n_ranges`` contiguous windows of ``RANGE_LEN`` docs; take all docs if the cache is tiny."""
    want = n_ranges * RANGE_LEN
    if n_docs <= want:
        return list(range(n_docs))
    starts = rng.integers(0, n_docs - RANGE_LEN, size=n_ranges)
    idx: list[int] = []
    for s in starts:
        idx.extend(range(int(s), int(s) + RANGE_LEN))
    return idx


def sample_domain_tokens(sources: list[str], sample_size: int, rng: np.random.Generator) -> list[tuple[np.ndarray, int]]:
    """Return sampled documents (token-id arrays) for a domain, pooled across its source caches.

    Documents are allocated to sources proportional to each source's document count, emulating a
    uniform draw from the domain's concatenated (merged) cache. Reads are threaded (IO-bound).
    """
    caches = [open_cache(p) for p in sources]
    counts = [len(c) for c in caches]
    total = sum(counts)
    if total == 0:
        raise InsufficientSampleError(f"no documents across sources {sources}")

    # ranges per source, proportional to doc count (at least covering the budget in aggregate).
    total_ranges = max(1, math.ceil(sample_size / RANGE_LEN))
    per_source_ranges = [max(0, round(total_ranges * c / total)) for c in counts]
    # ensure the biggest source gets at least one range if everything rounded to 0.
    if sum(per_source_ranges) == 0:
        per_source_ranges[int(np.argmax(counts))] = 1

    # Pre-seed one child rng per source so threaded reads stay deterministic and thread-safe
    # (numpy Generators are not safe to share across threads).
    child_seeds = rng.integers(0, 2**63, size=len(caches))

    def read_source(i: int) -> list[tuple[np.ndarray, int]]:
        if per_source_ranges[i] == 0:
            return []
        idx = _range_indices(counts[i], per_source_ranges[i], np.random.default_rng(int(child_seeds[i])))
        # Read in small chunks and truncate immediately: get_batch_sync materializes every doc's FULL
        # token array, and arxiv docs run ~18k tokens, so reading 20k at once peaks at >1GB. Truncating
        # to the embedding window per chunk keeps peak memory bounded (this container is 8GB-capped).
        out: list[tuple[np.ndarray, int]] = []
        for start in range(0, len(idx), READ_CHUNK):
            recs = caches[i].get_batch_sync(idx[start : start + READ_CHUNK])
            for r in recs:
                arr = np.asarray(r["input_ids"])
                out.append((arr[:MAX_EMBED_TOKENS].astype(np.int32, copy=True), len(arr)))
            del recs
        return out

    docs: list[tuple[np.ndarray, int]] = []
    with ThreadPoolExecutor(max_workers=min(READ_THREADS, len(caches))) as ex:
        for chunk in ex.map(read_source, range(len(caches))):
            docs.extend(chunk)

    if len(docs) > sample_size:
        keep = rng.choice(len(docs), size=sample_size, replace=False)
        docs = [docs[i] for i in keep]
    return docs


def embed_docs(docs: list[tuple[np.ndarray, int]], tokenizer, embedder) -> tuple[np.ndarray, np.ndarray]:
    """Detokenize (already truncated) and embed. Return ``(norm_embeddings (n,192) fp32, token_lengths)``.

    ``token_lengths`` is the FULL stored doc length (the loader's token measure); ``docs`` already
    carries only the first ``MAX_EMBED_TOKENS`` tokens per doc for the embedding text.
    """
    token_lengths = np.array([full_len for _ids, full_len in docs], dtype=np.int64)
    texts = tokenizer.batch_decode([ids.tolist() for ids, _ in docs], skip_special_tokens=True)
    out = np.empty((len(texts), EMBED_DIM), dtype=np.float32)
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        vecs = np.asarray(embedder(batch, progress_bars=False), dtype=np.float32)
        out[start : start + len(batch)] = vecs
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms, token_lengths


def assign_cells(norm_emb: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each embedding to its nearest centroid, replicating the datakit assign pipeline.

    L2-normalized embeddings are int8-quantized then dequantized (the on-disk round trip the assign
    pipeline sees), then matched to the nearest centroid by squared L2 (FAISS ``IndexFlatL2``).
    """
    q = np.clip(np.round(norm_emb / QUANT_SCALE), -127, 127).astype(np.int8)
    deq = q.astype(np.float32) * QUANT_SCALE
    c_sq = (centroids**2).sum(axis=1)  # (K,)
    out = np.empty(deq.shape[0], dtype=np.int32)
    for start in range(0, deq.shape[0], EMBED_BATCH):
        block = deq[start : start + EMBED_BATCH]
        # squared L2 = ||x||^2 - 2 x.c + ||c||^2 ; ||x||^2 is constant per row -> omit for argmin.
        d2 = c_sq[None, :] - 2.0 * (block @ centroids.T)
        out[start : start + block.shape[0]] = np.argmin(d2, axis=1).astype(np.int32)
    return out


def median_bandwidth(pool: np.ndarray) -> float:
    """Median pairwise Euclidean distance of ``pool`` (n,192) — the RFF bandwidth (sigma)."""
    g = pool @ pool.T
    sq = np.diag(g)
    d2 = np.clip(sq[:, None] + sq[None, :] - 2.0 * g, 0.0, None)
    iu = np.triu_indices(pool.shape[0], k=1)
    return float(np.median(np.sqrt(d2[iu])))


# --- per-domain accumulation ----------------------------------------------------------


class DomainAccumulator:
    """Holds a domain's sampled embeddings/cells so RFF means can be finished after sigma is frozen."""

    def __init__(self, domain: str, norm_emb: np.ndarray, token_lengths: np.ndarray, cells: np.ndarray):
        self.domain = domain
        self.norm_emb = norm_emb
        self.token_lengths = token_lengths
        self.cells = cells

    def cell_counts(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for cell, w in zip(self.cells, self.token_lengths, strict=True):
            counts[int(cell)] += int(w)
        return counts

    def rff_mean(self, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        phi = rff_features(self.norm_emb, w, b)  # (n, D)
        weights = self.token_lengths.astype(np.float64)
        return (weights[:, None] * phi).sum(axis=0) / weights.sum()


def domain_rng(domain: str) -> np.random.Generator:
    """Per-domain generator seeded from (SEED, domain) so samples are resume-invariant."""
    h = int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
    return np.random.default_rng([SEED, h])


def process_domain(domain: str, sources: list[str], centroids: np.ndarray, tokenizer, embedder) -> DomainAccumulator:
    rng = domain_rng(domain)
    t0 = time.time()
    docs = sample_domain_tokens(sources, SAMPLE_SIZE, rng)
    if len(docs) < MIN_SAMPLE:
        raise InsufficientSampleError(f"{domain}: only {len(docs)} docs sampled (< {MIN_SAMPLE})")
    t_read = time.time() - t0
    norm_emb, token_lengths = embed_docs(docs, tokenizer, embedder)
    t_embed = time.time() - t0 - t_read
    cells = assign_cells(norm_emb, centroids)
    logger.info(
        "%s: sampled=%d tokens=%d read=%.1fs embed=%.1fs occupied_cells=%d",
        domain,
        len(docs),
        int(token_lengths.sum()),
        t_read,
        t_embed,
        len(np.unique(cells)),
    )
    return DomainAccumulator(domain, norm_emb, token_lengths, cells)


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_with_retry(fn, *, tries: int = 40, wait: float = 15.0) -> None:
    """Run a write ``fn``; on ENOSPC wait and retry so a transient disk-full (shared VM) does not
    lose an already-computed domain to a crash-and-re-embed cycle."""
    for attempt in range(tries):
        try:
            fn()
            return
        except OSError as exc:
            if exc.errno != errno.ENOSPC or attempt == tries - 1:
                raise
            logger.warning("ENOSPC on write (attempt %d/%d); waiting %.0fs for disk", attempt + 1, tries, wait)
            time.sleep(wait)


def write_domain_parquet(out_dir: str, domain: str, counts: dict[int, int]) -> tuple[str, int]:
    total = sum(counts.values())
    cells = sorted(counts)
    table = pa.table(
        {
            "domain": pa.array([domain] * len(cells), pa.string()),
            "cluster_id": pa.array(cells, pa.int32()),
            "quality_bucket": pa.array([-1] * len(cells), pa.int8()),
            "token_count": pa.array([counts[c] for c in cells], pa.int64()),
            "frac": pa.array([counts[c] / total for c in cells], pa.float64()),
        }
    )
    fname = "part-" + domain.replace("/", "-") + ".parquet"
    path = os.path.join(out_dir, fname)
    _write_with_retry(lambda: pq.write_table(table, path))
    return path, total


# --- driver ---------------------------------------------------------------------------


def _safe(domain: str) -> str:
    return domain.replace("/", "-")


def _domain_done(out_dir: str, domain: str) -> bool:
    s = _safe(domain)
    return all(
        os.path.exists(os.path.join(out_dir, sub, f"{s}{ext}"))
        for sub, ext in (("", ".parquet.done"), ("rff", ".npy"), ("meta", ".json"))
    ) and os.path.exists(os.path.join(out_dir, f"part-{s}.parquet"))


def _persist_domain(out_dir: str, acc: DomainAccumulator, w_map, b_map, sources: list[str], avail: int) -> None:
    counts = acc.cell_counts()
    path, total = write_domain_parquet(out_dir, acc.domain, counts)
    rff = acc.rff_mean(w_map, b_map)
    _write_with_retry(lambda: np.save(os.path.join(out_dir, "rff", f"{_safe(acc.domain)}.npy"), rff))
    meta = {
        "sample_size": int(acc.norm_emb.shape[0]),
        "seed": SEED,
        "token_count": int(total),
        "occupied_cells_k5000": len(counts),
        "sources": sources,
        "parquet": os.path.basename(path),
        "bucket_stats": {
            "total_tokens_available": int(avail),
            "mean_doc_tokens": float(acc.token_lengths.mean()),
            "duplicate_frac": None,
            "loss_masked_frac": 0.0,
        },
    }

    def _write_meta():
        with open(os.path.join(out_dir, "meta", f"{_safe(acc.domain)}.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

    _write_with_retry(_write_meta)
    # sentinel written last: marks the domain fully persisted (crash-safe resume).
    open(os.path.join(out_dir, f"{_safe(acc.domain)}.parquet.done"), "w").close()


def _ensure_bandwidth(out_dir: str, calib_domains: list[str], sources_map, centroids, tokenizer, embedder) -> dict:
    """Load or compute+persist the frozen RFF bandwidth (median heuristic over the calib pool)."""
    bw_path = os.path.join(out_dir, "bandwidth.json")
    if os.path.exists(bw_path):
        info = _read_json(bw_path)
        logger.info("loaded frozen bandwidth=%.6f from %s", info["rff_bandwidth"], bw_path)
        return info

    sigma_pool = []
    calib_accs: dict[str, DomainAccumulator] = {}
    for d in calib_domains:
        acc = process_domain(d, sources_map[d], centroids, tokenizer, embedder)
        calib_accs[d] = acc
        rng = domain_rng(d + ":sigma")
        take = min(CALIB_SIGMA_PER_DOMAIN, acc.norm_emb.shape[0])
        pick = rng.choice(acc.norm_emb.shape[0], size=take, replace=False)
        sigma_pool.append(acc.norm_emb[pick])
    pool = np.concatenate(sigma_pool, axis=0)
    bandwidth = median_bandwidth(pool)
    logger.info("frozen RFF bandwidth (median heuristic over %d docs) = %.6f", pool.shape[0], bandwidth)
    info = {
        "rff_bandwidth": bandwidth,
        "calib_domains": calib_domains,
        "calib_pool_size": int(pool.shape[0]),
    }
    with open(bw_path, "w") as f:
        json.dump(info, f, indent=2)
    # Persist the calib domains immediately (they are already embedded).
    w_map, b_map = build_rff_map(RFF_DIM, EMBED_DIM, RFF_SEED, bandwidth)
    for d, acc in calib_accs.items():
        if not _domain_done(out_dir, d):
            _persist_domain(out_dir, acc, w_map, b_map, sources_map[d], token_avail_for(d))
    return info


# Set by build() so _ensure_bandwidth's calib persistence can read availability.
_TOKEN_AVAIL: dict[str, int] = {}


def token_avail_for(domain: str) -> int:
    return int(_TOKEN_AVAIL[domain])


def _assemble_outputs(out_dir: str, domains: list[str], bw_info: dict, sources_map) -> None:
    """Assemble rff_means.npz + _meta.json from the per-domain side files (all domains must be done)."""
    per_domain_meta = {}
    rff = {}
    for d in domains:
        per_domain_meta[d] = _read_json(os.path.join(out_dir, "meta", f"{_safe(d)}.json"))
        rff[d] = np.load(os.path.join(out_dir, "rff", f"{_safe(d)}.npy"))
    order = sorted(rff)
    np.savez(
        os.path.join(out_dir, "rff_means.npz"),
        domains=np.array(order),
        rff_means=np.stack([rff[d] for d in order], axis=0),
    )
    basis = {
        "embedder": EMBEDDER_ID,
        "tokenizer": TOKENIZER_ID,
        "centroids_path": CENTROIDS_GCS,
        "centroids_sha256": sha256_file(os.path.join(BASIS_DIR, "centroids_5000.npy")),
        "k": K,
        "view_paths": {str(v): VIEW_GCS[v] for v in VIEWS},
        "view_sha256": {str(v): sha256_file(os.path.join(BASIS_DIR, os.path.basename(VIEW_GCS[v]))) for v in VIEWS},
        "quality_scorer": None,
        "quality_scorer_sha256": None,
        "rff_dim": RFF_DIM,
        "rff_seed": RFF_SEED,
        "rff_bandwidth": bw_info["rff_bandwidth"],
    }
    meta = {
        "basis": basis,
        "sampling": {
            "strategy": (
                f"{SAMPLE_SIZE} docs/domain via contiguous ranges of {RANGE_LEN} docs at uniform-random "
                f"starts; multi-source (pool) domains allocate ranges across source caches proportional "
                f"to doc count (emulating a uniform draw from the concatenated merged cache). "
                f"token weight = full stored doc length; embedding text truncated to first "
                f"{MAX_EMBED_TOKENS} tokens. per-domain rng seeded from (seed={SEED}, domain)."
            ),
            "sample_size": SAMPLE_SIZE,
            "range_len": RANGE_LEN,
            "seed": SEED,
            "quant_scale": QUANT_SCALE,
            "rff_bandwidth_calib": (
                f"median pairwise Euclidean distance over {bw_info['calib_pool_size']} docs pooled from "
                f"calib domains ({', '.join(bw_info['calib_domains'])})"
            ),
            "cross_region_note": (
                "sampled from us-east5 caches on a us-central1 VM; MARIN_I_WILL_PAY_FOR_ALL_FEES disables "
                "the levanter full-store pre-charge (real sampled egress is a few GB within US)."
            ),
        },
        "domains": per_domain_meta,
        "rff_means_file": "rff_means.npz",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
    }
    with open(os.path.join(out_dir, "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("assembled %d domains + rff_means.npz + _meta.json in %s", len(per_domain_meta), out_dir)


def build(domains: list[str], out_dir: str, sources_map: dict[str, list[str]], token_avail: dict[str, int]) -> None:
    global _TOKEN_AVAIL
    _TOKEN_AVAIL = token_avail
    os.makedirs(os.path.join(out_dir, "rff"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meta"), exist_ok=True)
    centroids = np.load(os.path.join(BASIS_DIR, "centroids_5000.npy")).astype(np.float32)
    assert centroids.shape == (K, EMBED_DIM), centroids.shape

    logger.info("loading tokenizer + luxical embedder")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    from luxical.embedder import Embedder  # noqa: PLC0415

    embedder = Embedder.load(hf_hub_download(LUXICAL_REPO, LUXICAL_WEIGHTS))

    calib_domains = domains[: min(CALIB_DOMAINS, len(domains))]
    bw_info = _ensure_bandwidth(out_dir, calib_domains, sources_map, centroids, tokenizer, embedder)
    w_map, b_map = build_rff_map(RFF_DIM, EMBED_DIM, RFF_SEED, bw_info["rff_bandwidth"])

    for d in domains:
        if _domain_done(out_dir, d):
            logger.info("%s: already done, skipping", d)
            continue
        acc = process_domain(d, sources_map[d], centroids, tokenizer, embedder)
        _persist_domain(out_dir, acc, w_map, b_map, sources_map[d], token_avail[d])
        del acc

    if all(_domain_done(out_dir, d) for d in domains):
        _assemble_outputs(out_dir, domains, bw_info, sources_map)
    else:
        pending = [d for d in domains if not _domain_done(out_dir, d)]
        logger.warning("not all domains done; pending=%s", pending)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sources", default=os.path.join(REPO_ROOT, "scratch/mixture_features/cache_sources.json"))
    ap.add_argument("--token-avail", default=os.path.join(REPO_ROOT, "scratch/mixture_features/available_tokens.json"))
    ap.add_argument("--domains", default="", help="comma-separated subset; default = all 39 sorted")
    args = ap.parse_args()

    sources_map = _read_json(args.sources)
    token_avail = _read_json(args.token_avail)
    if args.domains:
        domains = args.domains.split(",")
    else:
        domains = sorted(sources_map)
    missing = [d for d in domains if d not in sources_map]
    if missing:
        raise SystemExit(f"unknown domains: {missing}")
    logger.info("building histograms for %d domains -> %s", len(domains), args.out)
    build(domains, args.out, sources_map, token_avail)


if __name__ == "__main__":
    main()
