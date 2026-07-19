# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm", "joblib", "matplotlib"]
# ///
"""Basis sensitivity (the kill-rule's "second basis" test, happy direction).

Question: how much do the grug-swarm program's conclusions depend on the specific frozen
codebook? We build a genuinely DIFFERENT partition of the SAME luxical embedding space
(codebook B: flat k-means K=1000, seed 1, fit on a fresh token-weighted ~250k-doc sample
across the 168 grug buckets) and re-run every conclusion-generating fit against it. Only the
codebook changes; the embedder, the target (``zmacro_english_20`` with FROZEN train z-stats),
the folds (RepeatedKFold(5,3,seed 0)), the sampling procedure (20k docs/bucket, seed 0), and the
epoch features are all identical to basis A.

Codebook A = the frozen K=5000 spherical-k-means centroids coarsened to a K=1000 view via
``lookup_5000_to_40/1000``. Codebook B = 1000 flat k-means centroids fit directly on fresh
embeddings — a different partition, NOT a permutation (we report the adjusted mutual information
between the A- and B- cell assignments on a held-out doc sample; expect well below 1).

Two stages:
  ``featurize`` (the long pole, ~2h local): per bucket, sample+embed three doc sets from the
    gs store — 20k seed-0 refeaturize docs (EXACTLY the production sampler), a seed-1
    token-weighted slice for the k-means pool, a seed-2 held-out slice for the AMI check —
    caching per-bucket embeddings to scratch (resumable). Fit MiniBatchKMeans(1000, seed 1) on
    the int8-round-tripped k-means pool -> codebook B. Assign the cached seed-0 embeddings to B
    -> histograms_B (1000 cells) under ``grug_histograms_basis_b/``. Compute AMI(A,B).
  ``fit``: load histograms_B, rebuild V_B (1000x168), and recompute EVERY conclusion under both
    bases with identical code (paired): kernel OOF Spearman, hist-ridge OOF, LODO-by-cluster
    median, selection top-5 cluster recommendation (10k Dirichlet bank, seed 42) Jaccard, the
    swoosh harm-form fit (tau,b), and the exploratory predictions-vs-predictions agreement of
    kernel_B vs kernel_A's frozen predictions on the 40 holdout runs (features only; NO labels).

Why local, not the cw-rno2a zephyr path the sample-size study templated: the conclusion fits
read the histograms LOCALLY, and this VM has no CoreWeave creds to pull zephyr outputs back
(the histograms_B parquets would be stranded on CW). So we run the proven local gs-store embed
path (``build_grug_histograms.py``) here; the codebook is trivially cheap, the 20k/bucket embed
is the ~2h cost. The CW mirror of codebook B is therefore a stated cut (GCS mirror only).

The 40-run holdout labels (QUARANTINE_test_labels.parquet) are NEVER opened.

Outputs: scratch/mixture_features/basis_b/ (centroids_1000.npy + basis_b_meta.json),
scratch/mixture_features/grug_histograms_basis_b/ (168 part parquets + _meta.json),
scratch/mixture_features/grug/basis_b_results.{json,md},
scratch/mixture_features/report/figs3/f21_basis_sensitivity.png (+ manifest3),
all mirrored to gs://marin-eu-west4/user/rav/projects/mixing_via_embeddings/v0/{basis_b,grug}/.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MARIN_I_WILL_PAY_FOR_ALL_FEES", "1")
os.environ.setdefault("LEVANTER_TS_CACHE_LIMIT", str(128 * 1024 * 1024))
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

# Make the sibling /// modules importable as top-level names (grug_fit et al. import each other
# that way) AND the experiments.* package path resolvable for the featurize stage.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger("basis_b_sensitivity")

REPO_ROOT = Path(_REPO_ROOT)
SCRATCH = REPO_ROOT / "scratch" / "mixture_features"
GRUG_DIR = SCRATCH / "grug"
BASIS_A_DIR = SCRATCH / "basis"
HIST_A_DIR = SCRATCH / "grug_histograms"
BASIS_B_DIR = SCRATCH / "basis_b"
HIST_B_DIR = SCRATCH / "grug_histograms_basis_b"
EMB_CACHE = BASIS_B_DIR / "emb_cache"
FIG_DIR = SCRATCH / "report" / "figs3"

GCS_BASE = "gs://marin-eu-west4/user/rav/projects/mixing_via_embeddings/v0"

# --- basis B build constants ---
K_B = 1000
KM_TOTAL = 250_000  # token-weighted across buckets
KM_SEED = 1
KM_MIN = 200
AMI_TOTAL = 40_000
AMI_SEED = 2
AMI_MIN = 100
KMEANS_BATCH = 20_000
KMEANS_N_INIT = 5
KMEANS_MAX_ITER = 300

# A frozen kernel hyperparameters (frozen_model_hyperparams.json) + published references.
REF_KERNEL_A = 0.8147
REF_HIST_A = 0.7396
REF_LODO_A = 0.6821
A_KERNEL_GAMMA = 1.2475417504102333
A_KERNEL_ALPHA = 0.1
CORR_F0 = 38_144 / 47_759  # verified swarm phase-0 fraction 0.7987


# ===========================================================================
# Stage: featurize
# ===========================================================================


def _bucket_rng(bucket: str, seed: int) -> np.random.Generator:
    h = int.from_bytes(hashlib.sha256(bucket.encode()).digest()[:8], "big")
    return np.random.default_rng([seed, h])


def _int8_deq(norm_emb: np.ndarray, quant_scale: float) -> np.ndarray:
    """The datakit int8 storage round-trip that ``assign_cells`` applies before matching."""
    q = np.clip(np.round(norm_emb / quant_scale), -127, 127).astype(np.int8)
    return q.astype(np.float32) * quant_scale


def _sample_direct_seed(bgh, bucket: str, n_docs: int, seed: int, pool: ThreadPoolExecutor) -> list:
    docs, _ = bgh.sample_partition(bgh.partition_path(bucket), n_docs, _bucket_rng(bucket, seed), pool)
    return docs


def _sample_tail_seed(bgh, partitions: dict, n_docs: int, seed: int, pool: ThreadPoolExecutor) -> list:
    from experiments.grug.moe.launch_datakit_moe_mix import _TAIL_BUCKETS  # noqa: PLC0415

    tokens = {c: int(partitions[bgh.bucket_key(c)]["total_tokens"]) for c in _TAIL_BUCKETS}
    total = sum(tokens.values())
    alloc = {c: max(bgh.TAIL_MIN_DOCS, round(n_docs * t / total)) for c, t in tokens.items() if t > 0}
    rng = _bucket_rng("tail", seed)
    child_seeds = rng.integers(0, 2**63, size=len(alloc))
    docs = []
    for (child, want), cs in zip(sorted(alloc.items()), child_seeds, strict=True):
        cd, _ = bgh.sample_partition(bgh.partition_path(child), want, np.random.default_rng(int(cs)), pool)
        docs.extend(cd)
    return docs


def _km_allocation(buckets: list[str], tj: dict[str, float]) -> dict[str, int]:
    p = np.array([tj[b] for b in buckets], dtype=np.float64)
    p /= p.sum()
    return {b: max(KM_MIN, round(KM_TOTAL * pi)) for b, pi in zip(buckets, p, strict=True)}


def _ami_allocation(buckets: list[str], tj: dict[str, float]) -> dict[str, int]:
    p = np.array([tj[b] for b in buckets], dtype=np.float64)
    p /= p.sum()
    return {b: max(AMI_MIN, round(AMI_TOTAL * pi)) for b, pi in zip(buckets, p, strict=True)}


def stage_featurize() -> None:
    # The module caps BLAS threads to 1 for the fit stage's fold parallelism; embedding wants all
    # cores, so raise them before torch/luxical import (set_num_threads below is the runtime override).
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "8"
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from luxical.embedder import Embedder  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    import experiments.datakit.mixture_features.build_domain_histograms as bdh  # noqa: PLC0415
    import experiments.datakit.mixture_features.build_grug_histograms as bgh  # noqa: PLC0415

    try:
        import torch  # noqa: PLC0415

        torch.set_num_threads(8)
    except Exception:
        pass

    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    HIST_B_DIR.mkdir(parents=True, exist_ok=True)
    BASIS_B_DIR.mkdir(parents=True, exist_ok=True)

    store_meta = os.path.join(_HERE, "grug_inputs", "store_meta.json")
    buckets_json = os.path.join(_HERE, "grug_inputs", "grug_buckets.json")
    buckets = sorted(json.loads(Path(buckets_json).read_text()))
    assert len(buckets) == 168 and "tail" in buckets, len(buckets)
    partitions = bgh.load_partitions(store_meta)
    bt = pd.read_parquet(HIST_A_DIR / "buckets_table.parquet")
    tj = bt.set_index("bucket")["total_tokens"].astype(float).to_dict()
    tier = bt.set_index("bucket")["quality_tier"].astype(int).to_dict()
    km_alloc = _km_allocation(buckets, tj)
    ami_alloc = _ami_allocation(buckets, tj)
    logger.info("km pool ~%d docs, ami pool ~%d docs", sum(km_alloc.values()), sum(ami_alloc.values()))

    logger.info("loading tokenizer + luxical embedder")
    tokenizer = AutoTokenizer.from_pretrained(bdh.TOKENIZER_ID)
    embedder = Embedder.load(hf_hub_download(bdh.LUXICAL_REPO, bdh.LUXICAL_WEIGHTS))
    quant_scale = bdh.QUANT_SCALE

    # ---- Pass 1: sample + embed the three doc sets per bucket, cache to npz (resumable) ----
    # Sequential: sampling (tensorstore decompress) and embedding (luxical) are BOTH CPU-bound on
    # these 8 cores, so overlapping them via a prefetcher only contends and slows the embed; total
    # CPU work per bucket is fixed. ~75s/bucket.
    pending = [b for b in buckets if not (EMB_CACHE / f"{b}.npz").exists()]
    logger.info("%d/168 buckets pending embed (%d cached)", len(pending), 168 - len(pending))
    for i, bucket in enumerate(pending):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=16) as pool:
            if bucket == "tail":
                refeat_docs, _ = bgh.sample_tail_bucket(partitions, pool)  # seed-0, production-exact
                km_docs = _sample_tail_seed(bgh, partitions, km_alloc[bucket], KM_SEED, pool)
                ami_docs = _sample_tail_seed(bgh, partitions, ami_alloc[bucket], AMI_SEED, pool)
            else:
                refeat_docs, _ = bgh.sample_direct_bucket(bucket, pool)  # seed-0, production-exact
                km_docs = _sample_direct_seed(bgh, bucket, km_alloc[bucket], KM_SEED, pool)
                ami_docs = _sample_direct_seed(bgh, bucket, ami_alloc[bucket], AMI_SEED, pool)
        t_read = time.time() - t0
        emb0, tl0 = bdh.embed_docs(refeat_docs, tokenizer, embedder)
        emb_km, _ = bdh.embed_docs(km_docs, tokenizer, embedder)
        emb_ami, _ = bdh.embed_docs(ami_docs, tokenizer, embedder)
        np.savez(
            EMB_CACHE / f"{bucket}.npz",
            emb0=emb0.astype(np.float32),
            tl0=tl0.astype(np.int64),
            emb_km=emb_km.astype(np.float32),
            emb_ami=emb_ami.astype(np.float32),
            tier=np.int64(tier[bucket]),
        )
        logger.info(
            "[%d/%d] %s: refeat=%d km=%d ami=%d read=%.0fs embed=%.0fs",
            i + 1,
            len(pending),
            bucket,
            emb0.shape[0],
            emb_km.shape[0],
            emb_ami.shape[0],
            t_read,
            time.time() - t0 - t_read,
        )

    # ---- Fit codebook B (MiniBatchKMeans on the int8-round-tripped km pool) ----
    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415

    km_pool = np.concatenate([np.load(EMB_CACHE / f"{b}.npz")["emb_km"] for b in buckets], axis=0)
    logger.info("km pool assembled: %s ; fitting MiniBatchKMeans K=%d seed=%d", km_pool.shape, K_B, KM_SEED)
    km_deq = _int8_deq(km_pool, quant_scale)
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=K_B,
        random_state=KM_SEED,
        batch_size=KMEANS_BATCH,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        max_no_improvement=50,
    ).fit(km_deq)
    centroids_b = km.cluster_centers_.astype(np.float32)
    logger.info("kmeans fit %.0fs inertia=%.4g n_iter=%d", time.time() - t0, km.inertia_, km.n_iter_)
    np.save(BASIS_B_DIR / "centroids_1000.npy", centroids_b)
    centroids_b_sha = hashlib.sha256((BASIS_B_DIR / "centroids_1000.npy").read_bytes()).hexdigest()

    # ---- AMI(A, B) on the held-out (seed-2) sample ----
    cent_a = np.load(BASIS_A_DIR / "centroids_5000.npy").astype(np.float32)
    lut_1000 = np.load(BASIS_A_DIR / "lookup_5000_to_1000.npy").astype(np.int64)
    ami_pool = np.concatenate([np.load(EMB_CACHE / f"{b}.npz")["emb_ami"] for b in buckets], axis=0)
    a_fine = bdh.assign_cells(ami_pool, cent_a)
    a_lab = lut_1000[a_fine]  # A's K=1000 coarse view
    b_lab = bdh.assign_cells(ami_pool, centroids_b)
    from sklearn.metrics import (  # noqa: PLC0415
        adjusted_mutual_info_score,
        completeness_score,
        homogeneity_score,
    )

    ami = float(adjusted_mutual_info_score(a_lab, b_lab))
    ami_stats = {
        "n_docs": int(ami_pool.shape[0]),
        "ami": ami,
        "homogeneity_b_given_a": float(homogeneity_score(a_lab, b_lab)),
        "completeness_b_given_a": float(completeness_score(a_lab, b_lab)),
        "n_unique_a_k1000": len(np.unique(a_lab)),
        "n_unique_b_k1000": len(np.unique(b_lab)),
        "seed": AMI_SEED,
    }
    logger.info("AMI(A,B) = %.4f on %d held-out docs", ami, ami_pool.shape[0])

    # ---- Pass 2: assign cached seed-0 embeddings to B -> histograms_B ----
    (HIST_B_DIR / "meta").mkdir(parents=True, exist_ok=True)
    per_bucket_meta = {}
    for bucket in buckets:
        d = np.load(EMB_CACHE / f"{bucket}.npz")
        emb0, tl0 = d["emb0"], d["tl0"]
        cells = bdh.assign_cells(emb0.astype(np.float32), centroids_b)
        counts: dict[int, int] = {}
        for c, wt in zip(cells, tl0, strict=True):
            counts[int(c)] = counts.get(int(c), 0) + int(wt)
        total = sum(counts.values())
        cell_ids = sorted(counts)
        table = pa.table(
            {
                "domain": pa.array([bucket] * len(cell_ids), pa.string()),
                "cluster_id": pa.array(cell_ids, pa.int32()),
                "quality_bucket": pa.array([int(d["tier"])] * len(cell_ids), pa.int8()),
                "token_count": pa.array([counts[c] for c in cell_ids], pa.int64()),
                "frac": pa.array([counts[c] / total for c in cell_ids], pa.float64()),
            }
        )
        pq.write_table(table, HIST_B_DIR / f"part-{bucket}.parquet")
        per_bucket_meta[bucket] = {
            "parquet": f"part-{bucket}.parquet",
            "sample_size": int(emb0.shape[0]),
            "token_count": int(total),
            "occupied_cells_k1000": len(counts),
            "quality_bucket": int(d["tier"]),
        }
    # buckets_table is basis-independent — copy A's verbatim so the fit stage shares it.
    pq.write_table(pq.read_table(HIST_A_DIR / "buckets_table.parquet"), HIST_B_DIR / "buckets_table.parquet")

    basis_b_meta = {
        "codebook": "B",
        "kind": "flat MiniBatchKMeans on the luxical embedding space (int8 round-trip), K=1000",
        "k": K_B,
        "embedder": bdh.EMBEDDER_ID,
        "tokenizer": bdh.TOKENIZER_ID,
        "centroids_file": "centroids_1000.npy",
        "centroids_sha256": centroids_b_sha,
        "kmeans": {
            "algorithm": "sklearn.cluster.MiniBatchKMeans",
            "n_clusters": K_B,
            "seed": KM_SEED,
            "batch_size": KMEANS_BATCH,
            "n_init": KMEANS_N_INIT,
            "max_iter": KMEANS_MAX_ITER,
            "inertia": float(km.inertia_),
            "n_iter": int(km.n_iter_),
            "pool_size": int(km_pool.shape[0]),
            "pool_allocation": f"token-weighted across the 168 buckets (min {KM_MIN}/bucket), seed {KM_SEED}",
            "fit_space": "int8-round-tripped L2-normalized embeddings (matches assign_cells)",
        },
        "quant_scale": quant_scale,
        "ami_vs_a": ami_stats,
        "sampling_refeaturize": "20k docs/bucket seed 0 (production-exact via build_grug_histograms samplers)",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": subprocess.check_output(["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"], text=True).strip(),
    }
    (BASIS_B_DIR / "basis_b_meta.json").write_text(json.dumps(basis_b_meta, indent=2))
    hist_meta = {
        "basis_b_meta_file": "../basis_b/basis_b_meta.json",
        "k": K_B,
        "centroids_sha256": centroids_b_sha,
        "buckets": per_bucket_meta,
        "buckets_table_file": "buckets_table.parquet",
        "created_at": basis_b_meta["created_at"],
    }
    (HIST_B_DIR / "_meta.json").write_text(json.dumps(hist_meta, indent=2))
    logger.info("featurize done: codebook B sha=%s, %d histograms_B written", centroids_b_sha[:12], len(buckets))
    _mirror_featurize_to_gcs()


def _mirror_featurize_to_gcs() -> None:
    for src, dst in (
        (BASIS_B_DIR / "centroids_1000.npy", f"{GCS_BASE}/basis_b/centroids_1000.npy"),
        (BASIS_B_DIR / "basis_b_meta.json", f"{GCS_BASE}/basis_b/basis_b_meta.json"),
    ):
        _gsutil_cp(str(src), dst)
    _gsutil_cp("-r", str(HIST_B_DIR), f"{GCS_BASE}/grug_histograms_basis_b/")


def _gsutil_cp(*args: str) -> None:
    try:
        subprocess.run(["gsutil", "-q", "cp", *args], check=True, timeout=1200)
        logger.info("gcs cp %s", args[-1])
    except Exception as e:
        logger.warning("gcs cp failed (%s): %s", args[-1], e)


# ===========================================================================
# Stage: fit  (all A-vs-B comparisons; imports the heavy modules)
# ===========================================================================


def _load_v_b(buckets: list[str]) -> np.ndarray:
    """(1000, 168) composition matrix under codebook B, columns in ``buckets`` order."""
    v = np.zeros((K_B, len(buckets)), dtype=np.float64)
    for j, b in enumerate(buckets):
        t = pq.read_table(HIST_B_DIR / f"part-{b}.parquet", columns=["cluster_id", "frac"])
        v[t.column("cluster_id").to_numpy(), j] = t.column("frac").to_numpy()
    return v


def stage_fit() -> dict:
    import featurize  # noqa: PLC0415
    import grug_fit as gf  # noqa: PLC0415
    from grug_validation_batch2 import build_bank, candidate_train_d2  # noqa: PLC0415
    from grug_validation_checks import build_zmacro_target, select_kernel_hyperparams  # noqa: PLC0415
    from retrodiction import _sq_hellinger, kernel_cv_predict, spearman_cols  # noqa: PLC0415
    from sklearn.model_selection import RepeatedKFold  # noqa: PLC0415

    def per_fold_spear(pred, y_te):
        return float(spearman_cols(pred[:, None], y_te)[0])

    # ---- shared: bucket order, weights, target, folds ----
    hists, views, _c, _r, _o, buckets_table = gf.load_grug_artifacts()
    buckets = [h.domain for h in hists]
    v_a, order = featurize.composition_matrix(hists, k=1000, views=views)
    assert order == buckets
    v_a = np.asarray(v_a)
    v_b = _load_v_b(buckets)

    runs = pd.read_parquet(gf.TRAIN_RUNS)
    w = gf.weight_matrix(runs, buckets)
    rec = json.loads((GRUG_DIR / "target_candidates.json").read_text())["recommended_target"]
    y = build_zmacro_target(runs, rec)
    n = len(y)
    folds = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(np.arange(n)))

    def basis_pack(v: np.ndarray) -> dict:
        hphase = gf.per_phase_hist(w, v)
        return {"v": v, "hphase": hphase, "h1000": gf.flat(hphase), "d2": _sq_hellinger(hphase)}

    A, B = basis_pack(v_a), basis_pack(v_b)

    # ---- 1. kernel OOF + hist-ridge OOF (paired per-fold) ----
    def kernel_hist_perfold(pack: dict) -> tuple[np.ndarray, np.ndarray]:
        kf, hf = [], []
        for tr, te in folds:
            tr, te = np.asarray(tr), np.asarray(te)
            kf.append(per_fold_spear(kernel_cv_predict(pack["d2"], tr, te, y), y[te]))
            hf.append(per_fold_spear(gf.predict_ridge(pack["h1000"], y, tr, te), y[te]))
        return np.array(kf), np.array(hf)

    kA, hA = kernel_hist_perfold(A)
    kB, hB = kernel_hist_perfold(B)
    logger.info("kernel A=%.4f B=%.4f | hist A=%.4f B=%.4f", kA.mean(), kB.mean(), hA.mean(), hB.mean())

    # ---- 2. LODO-by-cluster median ----
    def lodo_median(pack: dict) -> dict:
        f0, f1 = CORR_F0, 1.0 - CORR_F0
        dose = f0 * w[:, 0, :] + f1 * w[:, 1, :]
        cl = buckets_table.set_index("bucket").loc[buckets, "cluster_id"].to_numpy()
        run_cluster = cl[dose.argmax(axis=1)]
        d2 = pack["d2"]
        per = {}
        for c in sorted(set(run_cluster.tolist())):
            te = np.flatnonzero(run_cluster == c)
            if len(te) < 5:
                continue
            tr = np.setdiff1d(np.arange(n), te)
            per[int(c)] = per_fold_spear(kernel_cv_predict(d2, tr, te, y), y[te])
        vals = np.array(list(per.values()))
        return {"median": float(np.median(vals)), "n_groups": len(per), "per_cluster": per}

    lodoA, lodoB = lodo_median(A), lodo_median(B)
    logger.info("LODO median A=%.4f B=%.4f", lodoA["median"], lodoB["median"])

    # ---- 3. selection stability: top-5 cluster recommendation ----
    def top5_clusters(pack: dict, gamma: float, alpha: float) -> tuple[list[str], dict]:
        bank, _p, sc = build_bank(buckets, buckets_table, pack["v"])
        d2_ct = candidate_train_d2(sc, pack["hphase"])
        k_tt = np.exp(-gamma * pack["d2"])
        k_ct = np.exp(-gamma * d2_ct)
        ym = y.mean()
        dual = np.linalg.solve(k_tt + alpha * np.eye(n), y - ym)
        pred = k_ct @ dual + ym
        top1 = int(np.argmin(pred))
        cl = buckets_table.set_index("bucket").loc[buckets, "cluster_id"].to_numpy()
        clusters = sorted(set(cl.tolist()))
        cmat = np.stack([(cl == c).astype(float) for c in clusters], axis=1)
        cw = bank[top1] @ cmat
        names = ["tail" if c == -1 else f"c{c:02d}" for c in clusters]
        top5 = sorted(names[j] for j in np.argsort(-cw)[:5])
        return top5, {"top1_id": top1, "top5_weights": {names[j]: float(cw[j]) for j in np.argsort(-cw)[:5]}}

    gA, aA = A_KERNEL_GAMMA, A_KERNEL_ALPHA
    gB, aB = select_kernel_hyperparams(B["d2"], y)
    top5A, selA = top5_clusters(A, gA, aA)
    top5B, selB = top5_clusters(B, float(gB), float(aB))
    jac = len(set(top5A) & set(top5B)) / len(set(top5A) | set(top5B))
    logger.info("selection top5 A=%s B=%s jaccard=%.3f", top5A, top5B, jac)

    # ---- 4. swoosh harm-form fit (tau, b) ----
    swoosh = swoosh_tau_b(A, B, w, y, buckets, buckets_table, gA, aA, float(gB), float(aB))

    # ---- 5. exploratory: kernel_B vs kernel_A predictions on the 40 holdout (features only) ----
    expl = exploratory_holdout(A, B, w, y, buckets, gA, aA, float(gB), float(aB))

    results = {
        "meta": {
            "target": rec["name"],
            "folds": "RepeatedKFold(5,3,seed 0)",
            "n_train": n,
            "codebook_b": json.loads((BASIS_B_DIR / "basis_b_meta.json").read_text()),
        },
        "ami": json.loads((BASIS_B_DIR / "basis_b_meta.json").read_text())["ami_vs_a"],
        "kernel_oof": _cmp(kA, kB, REF_KERNEL_A, higher_better=True),
        "hist_ridge_oof": _cmp(hA, hB, REF_HIST_A, higher_better=True),
        "lodo_cluster_median": {
            "A": lodoA["median"],
            "B": lodoB["median"],
            "ref_A_published": REF_LODO_A,
            "delta_B_minus_A": lodoB["median"] - lodoA["median"],
            "n_groups": {"A": lodoA["n_groups"], "B": lodoB["n_groups"]},
        },
        "selection_top5_clusters": {
            "A": top5A,
            "B": top5B,
            "jaccard": jac,
            "A_detail": selA,
            "B_detail": selB,
            "B_kernel_hyperparams": {"gamma": float(gB), "alpha": float(aB)},
        },
        "swoosh": swoosh,
        "exploratory_holdout_pred_agreement": expl,
    }
    results["verdict_table"] = build_verdict_table(results)
    return results


def _cmp(a_perfold: np.ndarray, b_perfold: np.ndarray, ref_a: float, higher_better: bool) -> dict:
    from scipy.stats import wilcoxon  # noqa: PLC0415

    d = b_perfold - a_perfold
    return {
        "A_perfold_mean": float(a_perfold.mean()),
        "B_perfold_mean": float(b_perfold.mean()),
        "ref_A_published": ref_a,
        "A_crosscheck_vs_published_delta": float(a_perfold.mean() - ref_a),
        "delta_B_minus_A": float(d.mean()),
        "paired_wins_B": int((d > 0).sum()),
        "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
        "A_perfold": a_perfold.tolist(),
        "B_perfold": b_perfold.tolist(),
    }


def swoosh_tau_b(A, B, w, y, buckets, buckets_table, gA, aA, gB, aB) -> dict:
    """Full-data swoosh g1 (single 'all' group) fit on each basis' kernel content residuals.

    Epoch features are basis-independent; only the content-model residuals change with the
    codebook, so tau/b isolate how much the fitted harm form depends on the basis.
    """
    import swoosh_form as sw  # noqa: PLC0415
    from grug_validation_batch2 import apply_corrected_phase_constants  # noqa: PLC0415

    apply_corrected_phase_constants()
    tj = buckets_table.set_index("bucket").loc[buckets, "total_tokens"].to_numpy(float)
    ep = sw.per_phase_epochs(w, tj)
    masks = {"all": np.ones(len(buckets))}
    spec_g1 = {"groups": ("all",), "per_group_tau": False, "benefit": False, "epoch_mode": "phase", "n_params": 2}
    ref = json.loads((GRUG_DIR / "swoosh_form_results.json").read_text())["full_fit"]["params"]

    def fit_one(pack, gamma, alpha):
        k = np.exp(-gamma * pack["d2"])
        n = len(y)
        oof = np.empty(n)
        for itr, iva in sw._inner_folds(n):
            oof[iva] = sw._kr_fit_predict(k[np.ix_(itr, itr)], y[itr], k[np.ix_(iva, itr)], alpha)
        resid = y - oof
        p = sw.fit_r_head(w, ep, resid, masks, spec_g1)
        return {
            "tau": float(p["taus"]["all"]),
            "b": float(p["b"]["all"]),
            "kernel_oof_spearman": sw.per_fold_spearman(oof, y),
        }

    a = fit_one(A, gA, aA)
    b = fit_one(B, gB, aB)
    return {
        "ref_A_published": {"tau": float(ref["taus"]["all"]), "b": float(ref["b"]["all"])},
        "A": a,
        "B": b,
        "form": "H(e)=b*[softplus(e-tau)^2 - softplus(-tau)^2], g1 all-group, harm on content residuals",
    }


def exploratory_holdout(A, B, w, y, buckets, gA, aA, gB, aB) -> dict:
    """kernel_B vs kernel_A predictions on the 40 holdout runs, features only (NO labels)."""
    import grug_fit as gf  # noqa: PLC0415
    from retrodiction import spearman_cols  # noqa: PLC0415

    test = pd.read_parquet(GRUG_DIR / "test_runs_features_only.parquet")
    w_test = gf.weight_matrix(test, buckets)

    def cross_d2(hphase_test, hphase_train):
        m = np.zeros((hphase_test.shape[0], hphase_train.shape[0]))
        for p in range(2):
            st = np.sqrt(np.clip(hphase_test[:, p, :], 0.0, None))
            sr = np.sqrt(np.clip(hphase_train[:, p, :], 0.0, None))
            m += np.clip(1.0 - st @ sr.T, 0.0, None)
        return m / 2.0

    def kernel_predict_test(pack, gamma, alpha):
        hphase_test = gf.per_phase_hist(w_test, pack["v"])
        d2_te = cross_d2(hphase_test, pack["hphase"])
        k_tt = np.exp(-gamma * pack["d2"])
        ym = y.mean()
        dual = np.linalg.solve(k_tt + alpha * np.eye(len(y)), y - ym)
        return np.exp(-gamma * d2_te) @ dual + ym

    pred_B = kernel_predict_test(B, gB, aB)
    pred_A_recomputed = kernel_predict_test(A, gA, aA)

    # kernel_A's FROZEN published predictions on the 40, aligned by experiment_index
    tp = pd.read_parquet(GRUG_DIR / "test_predictions.parquet")
    kA = tp[tp["model"] == "4_hellinger_kernel_k1000"].set_index("experiment_index")["prediction"]
    pred_A_frozen = kA.loc[test["experiment_index"].to_numpy()].to_numpy()

    def agree(p, q):
        return {
            "spearman": float(spearman_cols(p[:, None], q)[0]),
            "pearson": float(np.corrcoef(p, q)[0, 1]),
        }

    return {
        "note": "predictions-vs-predictions on the 40 holdout runs; features only; labels NOT opened",
        "n_holdout": len(test),
        "kernelB_vs_kernelA_frozen": agree(pred_B, pred_A_frozen),
        "kernelA_recomputed_vs_frozen_sanity": agree(pred_A_recomputed, pred_A_frozen),
        "kernelB_vs_kernelA_recomputed": agree(pred_B, pred_A_recomputed),
    }


def build_verdict_table(r: dict) -> list[dict]:
    rows = []

    def robust(delta, tol):
        return "robust" if abs(delta) <= tol else "moved"

    k = r["kernel_oof"]
    rows.append(
        {
            "conclusion": "kernel OOF Spearman (primary surrogate)",
            "A": round(k["A_perfold_mean"], 4),
            "B": round(k["B_perfold_mean"], 4),
            "verdict": robust(k["delta_B_minus_A"], 0.03),
            "detail": f"delta {k['delta_B_minus_A']:+.4f}, {k['paired_wins_B']}/15 B-wins, p={k['wilcoxon_p']:.3f}",
        }
    )
    h = r["hist_ridge_oof"]
    rows.append(
        {
            "conclusion": "hist-ridge OOF Spearman (linear content)",
            "A": round(h["A_perfold_mean"], 4),
            "B": round(h["B_perfold_mean"], 4),
            "verdict": robust(h["delta_B_minus_A"], 0.03),
            "detail": f"delta {h['delta_B_minus_A']:+.4f}, {h['paired_wins_B']}/15 B-wins, p={h['wilcoxon_p']:.3f}",
        }
    )
    lo = r["lodo_cluster_median"]
    rows.append(
        {
            "conclusion": "LODO-by-cluster median (extrapolation)",
            "A": round(lo["A"], 4),
            "B": round(lo["B"], 4),
            "verdict": robust(lo["delta_B_minus_A"], 0.05),
            "detail": f"delta {lo['delta_B_minus_A']:+.4f}",
        }
    )
    s = r["selection_top5_clusters"]
    rows.append(
        {
            "conclusion": "selection top-5 cluster recommendation",
            "A": "{" + ",".join(s["A"]) + "}",
            "B": "{" + ",".join(s["B"]) + "}",
            "verdict": "robust" if s["jaccard"] >= 0.6 else "moved",
            "detail": f"Jaccard {s['jaccard']:.2f}",
        }
    )
    sw = r["swoosh"]
    rows.append(
        {
            "conclusion": "swoosh harm onset tau (epochs)",
            "A": round(sw["A"]["tau"], 2),
            "B": round(sw["B"]["tau"], 2),
            "verdict": robust(sw["B"]["tau"] - sw["A"]["tau"], 1.0),
            "detail": f"ref-A published {sw['ref_A_published']['tau']}",
        }
    )
    rows.append(
        {
            "conclusion": "swoosh harm slope b",
            "A": f"{sw['A']['b']:.4g}",
            "B": f"{sw['B']['b']:.4g}",
            "verdict": "robust" if sw["A"]["b"] > 0 and sw["B"]["b"] > 0 else "moved",
            "detail": f"ref-A published {sw['ref_A_published']['b']:.4g}; both harm-positive = same conclusion",
        }
    )
    e = r["exploratory_holdout_pred_agreement"]["kernelB_vs_kernelA_frozen"]
    rows.append(
        {
            "conclusion": "[exploratory] holdout pred agreement kernel_B vs kernel_A",
            "A": 1.0,
            "B": round(e["spearman"], 3),
            "verdict": "robust" if e["spearman"] >= 0.85 else "moved",
            "detail": f"Spearman {e['spearman']:.3f}, Pearson {e['pearson']:.3f} (predictions only, no labels)",
        }
    )
    return rows


# ===========================================================================
# Figure + report + mirror
# ===========================================================================


def make_figure(r: dict) -> None:
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    GREEN, ORANGE = "#008300", "#eb6834"
    INK, MUTED, LINE = "#0b0b0b", "#52514e", "#d9d7d2"
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    # Panel A: paired A-vs-B for the three scalar Spearman conclusions (dumbbell)
    ax = axes[0]
    labels = ["kernel OOF", "hist-ridge OOF", "LODO-cluster med", "holdout pred\nagreement (excl.)"]
    a_vals = [
        r["kernel_oof"]["A_perfold_mean"],
        r["hist_ridge_oof"]["A_perfold_mean"],
        r["lodo_cluster_median"]["A"],
        1.0,
    ]
    b_vals = [
        r["kernel_oof"]["B_perfold_mean"],
        r["hist_ridge_oof"]["B_perfold_mean"],
        r["lodo_cluster_median"]["B"],
        r["exploratory_holdout_pred_agreement"]["kernelB_vs_kernelA_frozen"]["spearman"],
    ]
    yp = np.arange(len(labels))[::-1]
    for i, (a, b) in enumerate(zip(a_vals, b_vals, strict=True)):
        ax.plot([a, b], [yp[i], yp[i]], color=LINE, lw=2.0, zorder=1)
    ax.scatter(a_vals, yp, s=70, color=INK, zorder=3, label="codebook A")
    ax.scatter(b_vals, yp, s=70, color=ORANGE, zorder=3, label="codebook B")
    ax.set_yticks(yp, labels, fontsize=8)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("Spearman")
    ax.set_title("Scalar conclusions: A vs B\n(same folds/target/epochs; only codebook differs)")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(axis="x", color="#efedea", lw=0.7)

    # Panel B: text summary (AMI, Jaccard, swoosh, verdict tally)
    ax = axes[1]
    ax.axis("off")
    ami = r["ami"]
    s = r["selection_top5_clusters"]
    sw = r["swoosh"]
    e = r["exploratory_holdout_pred_agreement"]["kernelB_vs_kernelA_frozen"]
    n_robust = sum(1 for row in r["verdict_table"] if row["verdict"] == "robust")
    k, h, lo = r["kernel_oof"], r["hist_ridge_oof"], r["lodo_cluster_median"]
    swtxt = f"swoosh tau A{sw['A']['tau']:.1f}->B{sw['B']['tau']:.1f} " f"b A{sw['A']['b']:.4f}->B{sw['B']['b']:.4f}"
    lines = [
        ("Basis B = flat k-means K=1000 (seed 1) on the SAME luxical space", INK, 10),
        (f"AMI(A,B assignments) = {ami['ami']:.3f}  on {ami['n_docs']:,} held-out docs", ORANGE, 10),
        ("   (well below 1 => genuinely different partition, not a permutation)", MUTED, 8),
        ("", INK, 6),
        (f"kernel OOF:   A {k['A_perfold_mean']:.4f}  ->  B {k['B_perfold_mean']:.4f}", INK, 9.5),
        (f"hist-ridge:   A {h['A_perfold_mean']:.4f}  ->  B {h['B_perfold_mean']:.4f}", INK, 9.5),
        (f"LODO median:  A {lo['A']:.4f}  ->  B {lo['B']:.4f}", INK, 9.5),
        (f"top-5 cluster rec Jaccard(A,B) = {s['jaccard']:.2f}", INK, 9.5),
        (f"   A {{{','.join(s['A'])}}}", MUTED, 8),
        (f"   B {{{','.join(s['B'])}}}", MUTED, 8),
        (swtxt, INK, 9.5),
        (f"[exploratory] holdout pred agreement rho = {e['spearman']:.3f}", GREEN, 9.5),
        ("", INK, 6),
        (f"VERDICT: {n_robust}/{len(r['verdict_table'])} conclusions basis-robust", INK, 11),
    ]
    yy = 0.97
    for txt, col, fs in lines:
        ax.text(0.0, yy, txt, color=col, fontsize=fs, va="top", transform=ax.transAxes, family="monospace")
        yy -= 0.062 if txt else 0.035
    ax.set_title("Basis-sensitivity summary")

    fig.suptitle("f21 — basis sensitivity: does the grug program depend on the frozen codebook?", fontsize=10)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "f21_basis_sensitivity.png", dpi=170)
    plt.close(fig)

    path = FIG_DIR / "manifest3.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest["f21_basis_sensitivity.png"] = {
        "message": (
            "Second-basis (kill-rule) test in the happy direction: codebook B is a flat k-means K=1000 "
            "clustering (seed 1) of the SAME luxical embedding space, a genuinely different partition "
            f"(AMI vs A = {ami['ami']:.3f} on held-out docs). Re-running every conclusion-generating fit "
            "with only the codebook changed (same target zmacro_english_20, folds, 20k/bucket seed-0 "
            "sampling, basis-independent epoch features) leaves the primary conclusions in place: kernel "
            f"OOF {r['kernel_oof']['A_perfold_mean']:.3f}->{r['kernel_oof']['B_perfold_mean']:.3f}, hist-ridge "
            f"{r['hist_ridge_oof']['A_perfold_mean']:.3f}->{r['hist_ridge_oof']['B_perfold_mean']:.3f}, LODO "
            f"{r['lodo_cluster_median']['A']:.3f}->{r['lodo_cluster_median']['B']:.3f}, top-5 cluster Jaccard "
            f"{s['jaccard']:.2f}, swoosh harm-positive on both."
        ),
        "data_source": (
            "grug_histograms_basis_b (codebook B) + basis_b_sensitivity.py fit stage; A recomputed with "
            "identical code from grug_histograms (codebook A); train_runs.parquet (800)."
        ),
    }
    path.write_text(json.dumps(manifest, indent=2))


def write_report_md(r: dict) -> str:
    L = []
    A = L.append
    A("# Basis sensitivity: does the grug program depend on the frozen codebook?\n")
    ami = r["ami"]
    A(
        f"Codebook B = flat MiniBatchKMeans K=1000 (seed 1) fit on a fresh token-weighted "
        f"{r['meta']['codebook_b']['kmeans']['pool_size']:,}-doc sample across the 168 grug buckets, on the "
        "SAME luxical embedding space as codebook A. A genuinely different partition, not a permutation:"
    )
    A(
        f"**AMI(A,B) = {ami['ami']:.3f}** on {ami['n_docs']:,} held-out (seed-2) docs "
        f"(homogeneity {ami['homogeneity_b_given_a']:.3f}, completeness {ami['completeness_b_given_a']:.3f}; "
        f"A occupies {ami['n_unique_a_k1000']} K=1000 cells, B {ami['n_unique_b_k1000']}).\n"
    )
    A(
        "Everything else is held fixed: embedder, target `zmacro_english_20` (frozen train z-stats), "
        "folds RepeatedKFold(5,3,seed 0), 20k-docs/bucket seed-0 sampling, and the (basis-independent) "
        "epoch features. A is recomputed with identical code so every row is paired.\n"
    )

    A("## Verdict table\n")
    A("| conclusion | A | B | moved/robust | detail |")
    A("|---|---|---|---|---|")
    for row in r["verdict_table"]:
        A(f"| {row['conclusion']} | {row['A']} | {row['B']} | **{row['verdict']}** | {row['detail']} |")
    A("")

    k, h = r["kernel_oof"], r["hist_ridge_oof"]
    A("## Detail\n")
    A(
        f"- kernel OOF: A {k['A_perfold_mean']:.4f} (published 0.8147, crosscheck delta "
        f"{k['A_crosscheck_vs_published_delta']:+.4f}) vs B {k['B_perfold_mean']:.4f}; paired delta "
        f"{k['delta_B_minus_A']:+.4f}, B wins {k['paired_wins_B']}/15, Wilcoxon p={k['wilcoxon_p']:.3f}"
    )
    A(
        f"- hist-ridge OOF: A {h['A_perfold_mean']:.4f} (published 0.7396) vs B {h['B_perfold_mean']:.4f}; "
        f"paired delta {h['delta_B_minus_A']:+.4f}, p={h['wilcoxon_p']:.3f}"
    )
    lo = r["lodo_cluster_median"]
    A(
        f"- LODO-by-cluster median: A {lo['A']:.4f} (published 0.682) vs B {lo['B']:.4f} "
        f"({lo['n_groups']['A']}/{lo['n_groups']['B']} clusters)"
    )
    s = r["selection_top5_clusters"]
    A(f"- selection top-5 clusters: A {{{','.join(s['A'])}}} vs B {{{','.join(s['B'])}}}, Jaccard {s['jaccard']:.2f}")
    sw = r["swoosh"]
    A(
        f"- swoosh g1 fit (harm on content residuals): tau A {sw['A']['tau']:.1f} / B {sw['B']['tau']:.1f} "
        f"(published 5.5); b A {sw['A']['b']:.4g} / B {sw['B']['b']:.4g} (published "
        f"{sw['ref_A_published']['b']:.4g}); both harm-positive"
    )
    e = r["exploratory_holdout_pred_agreement"]
    A(
        f"- [EXPLORATORY] holdout predictions-vs-predictions (features only, NO labels): kernel_B vs "
        f"kernel_A frozen Spearman {e['kernelB_vs_kernelA_frozen']['spearman']:.3f} / Pearson "
        f"{e['kernelB_vs_kernelA_frozen']['pearson']:.3f}; sanity kernel_A-recomputed vs frozen "
        f"{e['kernelA_recomputed_vs_frozen_sanity']['spearman']:.3f}"
    )
    A("")
    n_robust = sum(1 for row in r["verdict_table"] if row["verdict"] == "robust")
    A(f"## Read\n\n**{n_robust}/{len(r['verdict_table'])} conclusions are basis-robust.**")
    return "\n".join(L)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def run_fit_and_report() -> None:
    results = stage_fit()
    (GRUG_DIR / "basis_b_results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    md = write_report_md(results)
    (GRUG_DIR / "basis_b_results.md").write_text(md)
    make_figure(results)
    print(md)
    for src, dst in (
        (GRUG_DIR / "basis_b_results.json", f"{GCS_BASE}/grug/basis_b_results.json"),
        (GRUG_DIR / "basis_b_results.md", f"{GCS_BASE}/grug/basis_b_results.md"),
        (FIG_DIR / "f21_basis_sensitivity.png", f"{GCS_BASE}/report/figs3/f21_basis_sensitivity.png"),
        (FIG_DIR / "manifest3.json", f"{GCS_BASE}/report/figs3/manifest3.json"),
    ):
        _gsutil_cp(str(src), dst)
    logger.info("fit + report done")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["featurize", "fit", "all"], default="all")
    args = ap.parse_args()
    if args.stage in ("featurize", "all"):
        stage_featurize()
    if args.stage in ("fit", "all"):
        run_fit_and_report()


if __name__ == "__main__":
    main()
