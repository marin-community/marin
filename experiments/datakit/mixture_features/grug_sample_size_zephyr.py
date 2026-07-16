# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Histogram sample-size sensitivity for the grug featurization (V-column noise floor vs n).

Question: are 20k-doc bucket histograms enough, or does the seed-to-seed V-column noise
(cos_K5000 median 0.977, worst 0.809 for diffuse buckets at n=20k) shrink materially at 100k?

10 probe buckets spanning the K5000 occupied-cell concentration spectrum x sample sizes
{20k, 50k, 100k} x seeds {0, 1} = 60 zephyr map tasks on cw-rno2a, reusing the PROVEN
distributed histogram path (``grug_histograms_zephyr``) verbatim: same CW store mirror, same
frozen basis (centroids + FROZEN RFF bandwidth from ``grug_inputs/`` — never recomputed), same
int8-round-trip assignment, same contiguous-range sampler. Only the (n_docs, seed)
parametrization and the output prefix are new. ``(n=20k, seed=0)`` reproduces the production
sampling path exactly, giving a free determinism cross-check against the July histogram run.

The readout runs on the DRIVER (on-cluster: the entrypoint pod has CW creds): per bucket per n
the seed0-vs-seed1 cos/Hellinger over the K=5000 (and K=40 view) histogram — the noise floor at
that n — plus same-seed cross-n distances (bias/convergence check). The summary JSON is written
to ``<out>/summary.json`` on CW AND printed to the job log between
``SAMPLE_SIZE_SUMMARY_BEGIN/END`` markers so the (CW-credless) launching VM can pull it via
``iris rpc controller get-task-logs``.

Same cw-rno2a gotchas as the parent pipeline: CPU-only tasks, do NOT pin ``--region``;
StoragePath/levanter for all CW S3 I/O (pyarrow-native reads 400).
"""

import hashlib
import json
import logging
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem import StoragePath
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner

# Import first: sets MARIN_I_WILL_PAY_FOR_ALL_FEES + LEVANTER_TS_CACHE_LIMIT before levanter.
from experiments.datakit.mixture_features import build_domain_histograms as bdh
from experiments.datakit.mixture_features import grug_histograms_zephyr as ghz
from experiments.datakit.mixture_features.build_domain_histograms import DomainAccumulator
from experiments.datakit.mixture_features.build_grug_histograms import TAIL_MIN_DOCS

logger = logging.getLogger("grug_sample_size_zephyr")

CW_SS_OUT = "s3://marin-us-east-02a/marin/user/rav/projects/mixing_via_embeddings/v0/sample_size"

SAMPLE_SIZES = (20_000, 50_000, 100_000)
SEEDS = (0, 1)

# Probe buckets spanning the occupied-cell concentration spectrum (occ cells / est. docs
# available from the July grug_histograms metas). Mid buckets were all in the original n=20k
# noise-floor panel (noise_floor.json), giving a direct cross-check of the 20k floor.
# tail: only ~9e4 docs exist across its 33 children, so n=50k/100k progressively exhaust the
# population (seeds overlap) — the readout records expected_overlap so this is explicit.
CONCENTRATED = ("c38q0", "c13q0", "c12q4")  # 3 / 10 / 28 cells; 4.9e5 / 4.6e6 / 3.0e6 docs
MID = ("c11q2", "c15q4", "c27q2", "c08q3")  # 79 / 101 / 120 / 153 cells; 1.4e7-4.3e7 docs
DIFFUSE = ("c05q1", "c01q3", "tail")  # 700 / 905 / 290 cells; 3.4e8 / 2.0e8 / 8.9e4 docs
PROBE_BUCKETS = (*CONCENTRATED, *MID, *DIFFUSE)
BUCKET_CLASS = {b: c for c, bs in (("concentrated", CONCENTRATED), ("mid", MID), ("diffuse", DIFFUSE)) for b in bs}

MAX_WORKERS = 30
K40_LOOKUP_LOCAL = os.path.join(ghz.INPUTS_DIR, "lookup_5000_to_40.npy")

SUMMARY_BEGIN = "SAMPLE_SIZE_SUMMARY_BEGIN"
SUMMARY_END = "SAMPLE_SIZE_SUMMARY_END"


def bucket_rng(bucket: str, seed: int) -> np.random.Generator:
    """``bdh.domain_rng`` generalized to a caller-chosen seed; seed=0 reproduces production."""
    h = int.from_bytes(hashlib.sha256(bucket.encode()).digest()[:8], "big")
    return np.random.default_rng([seed, h])


def config_tag(bucket: str, n_docs: int, seed: int) -> str:
    return f"{bucket}-n{n_docs}-s{seed}"


# --- the map task -----------------------------------------------------------------------


def _sample_config(spec: dict, pool: ThreadPoolExecutor) -> tuple[list, dict]:
    """Sample ``spec['n_docs']`` docs with ``spec['seed']``; mirrors ``ghz._sample_bucket``."""
    n_docs, rng = spec["n_docs"], bucket_rng(spec["bucket"], spec["seed"])
    if not spec["is_tail"]:
        docs, n_avail = ghz._sample_one(spec["sources"][0]["path"], n_docs, rng, pool)
        return docs, {"docs_available": int(n_avail)}

    total = sum(s["tokens"] for s in spec["sources"])
    child_seeds = rng.integers(0, 2**63, size=len(spec["sources"]))
    docs, children = [], {}
    for src, seed in zip(sorted(spec["sources"], key=lambda s: s["child"]), child_seeds, strict=True):
        if src["tokens"] == 0:
            continue
        want = max(TAIL_MIN_DOCS, round(n_docs * src["tokens"] / total))
        child_docs, n_avail = ghz._sample_one(src["path"], want, np.random.default_rng(int(seed)), pool)
        docs.extend(child_docs)
        children[src["child"]] = {
            "allocated_docs": int(want),
            "sampled_docs": len(child_docs),
            "docs_available": int(n_avail),
        }
    return docs, {"tail_children": children}


def process_config(spec: dict) -> dict:
    """Sample -> embed -> assign -> write ``part-<bucket>-n<n>-s<seed>.parquet`` (+rff, +meta)."""
    bucket, n_docs, seed = spec["bucket"], spec["n_docs"], spec["seed"]
    tag = config_tag(bucket, n_docs, seed)
    parquet_url = f"{CW_SS_OUT}/part-{tag}.parquet"
    rff_url = f"{CW_SS_OUT}/rff/{tag}.npy"
    meta_url = f"{CW_SS_OUT}/meta/{tag}.json"
    if StoragePath(parquet_url).exists() and StoragePath(rff_url).exists() and StoragePath(meta_url).exists():
        logger.info("%s: already present, skipping", tag)
        counters.pipeline.update_counter("sample_size/configs_skipped", 1)
        return {"tag": tag, "status": "skipped"}

    tokenizer = ghz._worker_tokenizer()
    embedder = ghz._worker_embedder()
    centroids = ghz._worker_centroids()
    w_map, b_map = ghz._worker_rff_map()

    with ThreadPoolExecutor(max_workers=ghz.READ_THREADS) as pool:
        docs, extra = _sample_config(spec, pool)
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

    ghz._write_parquet_to_cw(table, parquet_url)
    ghz._write_npy_to_cw(rff, rff_url)
    meta = {
        "bucket": bucket,
        "requested_docs": n_docs,
        "seed": seed,
        "sample_size": int(norm_emb.shape[0]),
        "token_count": int(total),
        "occupied_cells_k5000": len(counts),
        "mean_doc_tokens": float(token_lengths.mean()),
        **extra,
    }
    ghz._write_json_to_cw(meta, meta_url)
    counters.pipeline.update_counter("sample_size/configs_done", 1)
    logger.info("%s: sampled=%d tokens=%d occupied_cells=%d", tag, norm_emb.shape[0], total, len(counts))
    return {"tag": tag, "status": "done", "occupied_cells_k5000": len(counts)}


# --- driver-side readout ----------------------------------------------------------------


def _load_hist(url: str) -> tuple[np.ndarray, int]:
    """Dense K=5000 frac vector + occupied-cell count from a part parquet on CW."""
    fd, local = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    StoragePath(url).download_to(local)
    t = pq.read_table(local, columns=["cluster_id", "frac"])
    os.unlink(local)
    v = np.zeros(bdh.K, dtype=np.float64)
    ids = t.column("cluster_id").to_numpy()
    v[ids] = t.column("frac").to_numpy()
    return v, len(ids)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, 1.0 - np.sqrt(p * q).sum())))


def _k40(v: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    out = np.zeros(int(lookup.max()) + 1, dtype=np.float64)
    np.add.at(out, lookup, v)
    return out


def _pair_metrics(a: np.ndarray, b: np.ndarray, lookup: np.ndarray | None) -> dict:
    m = {"cos_k5000": _cos(a, b), "hell_k5000": _hellinger(a, b)}
    if lookup is not None:
        a40, b40 = _k40(a, lookup), _k40(b, lookup)
        m["cos_k40"] = _cos(a40, b40)
        m["hell_k40"] = _hellinger(a40, b40)
    return m


def _expected_overlap(meta: dict) -> float:
    """Expected fraction of sampled docs shared between two independent seeds (population reuse)."""
    if "tail_children" in meta:
        kids = meta["tail_children"].values()
        tot = sum(k["sampled_docs"] for k in kids)
        return sum(k["sampled_docs"] * min(1.0, k["allocated_docs"] / max(1, k["docs_available"])) for k in kids) / max(
            1, tot
        )
    return min(1.0, meta["sample_size"] / max(1, meta["docs_available"]))


def _loglog_slope(ns: list[int], ys: list[float]) -> float | None:
    pts = [(math.log(n), math.log(y)) for n, y in zip(ns, ys, strict=True) if y > 0]
    if len(pts) < 2:
        return None
    xs, ys2 = zip(*pts, strict=True)
    return float(np.polyfit(xs, ys2, 1)[0])


def readout() -> dict:
    lookup = None
    if os.path.exists(K40_LOOKUP_LOCAL):
        lookup = np.load(K40_LOOKUP_LOCAL).astype(np.int64)

    per_bucket = []
    for bucket in PROBE_BUCKETS:
        hists: dict[tuple[int, int], np.ndarray] = {}
        cells: dict[tuple[int, int], int] = {}
        metas: dict[tuple[int, int], dict] = {}
        for n in SAMPLE_SIZES:
            for s in SEEDS:
                tag = config_tag(bucket, n, s)
                hists[(n, s)], cells[(n, s)] = _load_hist(f"{CW_SS_OUT}/part-{tag}.parquet")
                metas[(n, s)] = json.loads(StoragePath(f"{CW_SS_OUT}/meta/{tag}.json").read_text())

        floors = {}
        for n in SAMPLE_SIZES:
            floors[str(n)] = {
                **_pair_metrics(hists[(n, 0)], hists[(n, 1)], lookup),
                "cells_seed0": cells[(n, 0)],
                "cells_seed1": cells[(n, 1)],
                "expected_overlap": _expected_overlap(metas[(n, 0)]),
            }
        bias = {}
        for s in SEEDS:
            for lo, hi in ((20_000, 100_000), (20_000, 50_000), (50_000, 100_000)):
                bias[f"{lo}v{hi}_s{s}"] = _pair_metrics(hists[(lo, s)], hists[(hi, s)], lookup)

        # Determinism cross-check: (n=20k, seed=0) should reproduce the July production part.
        identity = None
        prod_url = f"{ghz.CW_OUT}/part-{bucket}.parquet"
        if StoragePath(prod_url).exists():
            prod, prod_cells = _load_hist(prod_url)
            identity = {
                "max_abs_frac_delta": float(np.abs(prod - hists[(20_000, 0)]).max()),
                "cells_prod": prod_cells,
                "cells_new": cells[(20_000, 0)],
            }

        hell_slope = _loglog_slope(list(SAMPLE_SIZES), [floors[str(n)]["hell_k5000"] for n in SAMPLE_SIZES])
        one_minus_cos_slope = _loglog_slope(
            list(SAMPLE_SIZES), [1.0 - floors[str(n)]["cos_k5000"] for n in SAMPLE_SIZES]
        )
        per_bucket.append(
            {
                "bucket": bucket,
                "class": BUCKET_CLASS[bucket],
                "floors": floors,
                "bias": bias,
                "identity_vs_production_20k_s0": identity,
                "hell_loglog_slope": hell_slope,
                "one_minus_cos_loglog_slope": one_minus_cos_slope,
            }
        )

    by_class = {}
    for cls in ("concentrated", "mid", "diffuse"):
        rows = [r for r in per_bucket if r["class"] == cls]
        by_class[cls] = {
            str(n): {
                "median_cos_k5000": float(np.median([r["floors"][str(n)]["cos_k5000"] for r in rows])),
                "median_hell_k5000": float(np.median([r["floors"][str(n)]["hell_k5000"] for r in rows])),
                "worst_cos_k5000": float(min(r["floors"][str(n)]["cos_k5000"] for r in rows)),
            }
            for n in SAMPLE_SIZES
        }
    return {
        "out_prefix": CW_SS_OUT,
        "sample_sizes": list(SAMPLE_SIZES),
        "seeds": list(SEEDS),
        "buckets": list(PROBE_BUCKETS),
        "per_bucket": per_bucket,
        "by_class": by_class,
    }


# --- driver -----------------------------------------------------------------------------


def build_task_specs(store_meta_path: str) -> list[dict]:
    meta = bdh._read_json(store_meta_path)
    partitions = {(b["cluster_id"], b["quality_bucket"]): b for b in meta["buckets"]}
    bucket_specs = {s["bucket"]: s for s in ghz.build_bucket_specs(list(PROBE_BUCKETS), partitions)}
    assert set(bucket_specs) == set(PROBE_BUCKETS), set(bucket_specs) ^ set(PROBE_BUCKETS)
    return [{**bucket_specs[b], "n_docs": n, "seed": s} for b in PROBE_BUCKETS for n in SAMPLE_SIZES for s in SEEDS]


def run(store_meta_path: str, readout_only: bool) -> None:
    specs = build_task_specs(store_meta_path)
    logger.info("%d configs over %d buckets", len(specs), len(PROBE_BUCKETS))

    if not readout_only:
        bandwidth = ghz._frozen_bandwidth()
        luxical_url = ghz._stage_luxical()
        centroids_url = ghz._stage_centroids_to_cw()
        logger.info("staged luxical=%s centroids=%s bandwidth=%.6f", luxical_url, centroids_url, bandwidth)
        ds = Dataset.from_list(specs).map(process_config)
        ctx = ZephyrContext(
            # CPU-only; regions UNSET on purpose (see grug_histograms_zephyr module docstring).
            resources=ResourceConfig(cpu=8, ram="16g"),
            max_workers=min(MAX_WORKERS, len(specs)),
            name="grug-sample-size",
            stage_runner_factory=InlineRunner,
        )
        ctx.put(ghz._LUXICAL_KEY, luxical_url)
        ctx.put(ghz._CENTROIDS_KEY, centroids_url)
        ctx.put(ghz._BANDWIDTH_KEY, bandwidth)
        outcome = ctx.execute(ds, verbose=True)
        logger.info("counters: %s", dict(outcome.counters))

    summary = readout()
    ghz._write_json_to_cw(summary, f"{CW_SS_OUT}/summary.json")
    text = json.dumps(summary, indent=1)
    print(SUMMARY_BEGIN, flush=True)
    print(text, flush=True)
    print(SUMMARY_END, flush=True)
    logger.info("summary written to %s/summary.json (%d bytes)", CW_SS_OUT, len(text))
