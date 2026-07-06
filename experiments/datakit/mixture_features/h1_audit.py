# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm"]
# ///
"""H1 information audit for the mixing-via-embeddings experiment.

Answers, on the frozen local artifacts (`scratch/mixture_features/`):

1. Rank / conditioning of the composition matrix V at K = 40 / 1000 / 5000.
2. Nearest-duplicate columns (cosine + Hellinger affinity), incl. the 13 dolma3_cc
   high/low quality-pair distances vs the cross-pair median.
3. Weight reconstruction from mixture histograms (h = V @ w, least-squares inverse).
4. Derivability honesty control: linear map raw weights -> per-phase K=40 histogram
   features must give R^2 ~ 1 by construction.
5. Nested-CV fit comparison on identical splits (5-fold x 3 repeats, per scale) of
   ridge/LightGBM on raw weights vs K=40 histogram features vs RFF mixture features,
   target `eval/uncheatable_eval/bpb`.

Outputs parquet + json under `scratch/mixture_features/h1/` and prints a report.
"""

import json
import logging
from pathlib import Path

import featurize
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def json_default(o):
    """Serialize numpy scalars in json payloads."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = REPO_ROOT / "scratch" / "mixture_features"
HIST_DIR = SCRATCH / "domain_histograms"
BASIS_DIR = SCRATCH / "basis"
OUT_DIR = SCRATCH / "h1"

SEED = 0
TARGET = "eval/uncheatable_eval/bpb"
GRANULARITIES = (40, 1000, 5000)
SV_RANK_TOL = 1e-10  # pre-registered: rank = #(sv > 1e-10 * sv_max)
N_RECON_DRAWS = 200
DUP_COS_FLAG = 0.99
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
LGBM_GRID = {"num_leaves": [7, 15, 31], "min_child_samples": [5, 10]}
N_SPLITS, N_REPEATS = 5, 3


# ---------------------------------------------------------------------------
# Artifact loading (local mirror of the persisted-shape contract in the spec)
# ---------------------------------------------------------------------------


def load_artifacts() -> tuple[list[featurize.DomainHistogram], dict[int, np.ndarray], np.ndarray]:
    """Rebuild DomainHistogram objects + coarsening views + centroids from scratch/."""
    meta = json.loads((HIST_DIR / "_meta.json").read_text())
    b = meta["basis"]
    basis = featurize.MixtureBasis(
        embedder=b["embedder"],
        tokenizer=b["tokenizer"],
        centroids_path=b["centroids_path"],
        centroids_sha256=b["centroids_sha256"],
        k=b["k"],
        view_paths={int(k): v for k, v in b["view_paths"].items()},
        view_sha256={int(k): v for k, v in b["view_sha256"].items()},
        quality_scorer=b["quality_scorer"],
        quality_scorer_sha256=b["quality_scorer_sha256"],
        rff_dim=b["rff_dim"],
        rff_seed=b["rff_seed"],
        rff_bandwidth=b["rff_bandwidth"],
    )
    npz = np.load(HIST_DIR / meta["rff_means_file"])
    rff_by_domain = dict(zip(npz["domains"].tolist(), npz["rff_means"], strict=True))

    hists = []
    for domain, dmeta in meta["domains"].items():
        df = pd.read_parquet(HIST_DIR / dmeta["parquet"])
        counts = {
            (int(c), int(q)): int(t)
            for c, q, t in zip(df["cluster_id"], df["quality_bucket"], df["token_count"], strict=True)
        }
        bs = dmeta["bucket_stats"]
        hists.append(
            featurize.DomainHistogram(
                domain=domain,
                basis=basis,
                sample_size=dmeta["sample_size"],
                token_count=dmeta["token_count"],
                seed=dmeta["seed"],
                counts=counts,
                rff_mean=tuple(np.asarray(rff_by_domain[domain], dtype=np.float64).tolist()),
                stats=featurize.BucketStats(
                    total_tokens_available=bs["total_tokens_available"],
                    mean_doc_tokens=bs["mean_doc_tokens"],
                    duplicate_frac=bs["duplicate_frac"],
                    loss_masked_frac=bs["loss_masked_frac"],
                ),
            )
        )
    hists.sort(key=lambda h: h.domain)
    views = {
        40: np.load(BASIS_DIR / "lookup_5000_to_40.npy"),
        1000: np.load(BASIS_DIR / "lookup_5000_to_1000.npy"),
    }
    centroids = np.load(BASIS_DIR / "centroids_5000.npy").astype(np.float64)
    return hists, views, centroids


def run_weight_matrix(runs: pd.DataFrame, domains: list[str]) -> np.ndarray:
    """(n_runs, 2, 39) per-phase weights in sorted-domain order."""
    w = np.stack(
        [runs[[f"phase_{p}_{d}" for d in domains]].to_numpy(dtype=np.float64) for p in (0, 1)],
        axis=1,
    )
    if not np.allclose(w.sum(axis=2), 1.0, atol=1e-6):
        raise ValueError("run phase weights do not sum to 1")
    return w


# ---------------------------------------------------------------------------
# 1. Rank / conditioning
# ---------------------------------------------------------------------------


def audit_spectrum(vs: dict[int, np.ndarray]) -> tuple[pd.DataFrame, list[dict]]:
    rows, summaries = [], []
    for k, v in vs.items():
        sv = v.diagnostics.singular_values
        rank_pre = int((sv > SV_RANK_TOL * sv.max()).sum())
        for i, s in enumerate(sv):
            rows.append({"k": k, "index": i, "singular_value": float(s)})
        summaries.append(
            {
                "k": k,
                "shape": list(v.diagnostics.shape),
                "numerical_rank_1e-10": rank_pre,
                "numerical_rank_eps": v.diagnostics.numerical_rank,
                "condition_number": v.diagnostics.condition_number,
                "top10_sv": [float(s) for s in sv[:10]],
                "bottom5_sv": [float(s) for s in sv[-5:]],
            }
        )
    return pd.DataFrame(rows), summaries


# ---------------------------------------------------------------------------
# 2. Nearest-duplicate columns + quality pairs
# ---------------------------------------------------------------------------


def column_cosines(v: np.ndarray) -> np.ndarray:
    vn = v / np.linalg.norm(v, axis=0, keepdims=True)
    return vn.T @ vn


def hellinger_affinity(v: np.ndarray) -> np.ndarray:
    """Bhattacharyya coefficient = cosine of sqrt-frac columns (unit L2 norm by construction)."""
    return column_cosines(np.sqrt(v))


def quality_topics(domains: list[str]) -> list[str]:
    highs = {d.removesuffix("_high") for d in domains if d.startswith("dolma3_cc/") and d.endswith("_high")}
    lows = {d.removesuffix("_low") for d in domains if d.startswith("dolma3_cc/") and d.endswith("_low")}
    topics = sorted(highs & lows)
    if len(topics) != 13:
        raise ValueError(f"expected 13 dolma3_cc high/low topics, got {len(topics)}")
    return topics


def audit_duplicates(vs: dict[int, np.ndarray], domains: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cos40, cos5000 = column_cosines(np.asarray(vs[40])), column_cosines(np.asarray(vs[5000]))
    bc40, bc5000 = hellinger_affinity(np.asarray(vs[40])), hellinger_affinity(np.asarray(vs[5000]))
    topics = quality_topics(domains)
    within = {(f"{t}_high", f"{t}_low") for t in topics}

    n = len(domains)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            pair = (domains[i], domains[j])
            is_q = pair in within or (pair[1], pair[0]) in within
            rows.append(
                {
                    "domain_a": domains[i],
                    "domain_b": domains[j],
                    "cos_k40": cos40[i, j],
                    "cos_k5000": cos5000[i, j],
                    "bc_k40": bc40[i, j],
                    "bc_k5000": bc5000[i, j],
                    "hellinger_k5000": float(np.sqrt(max(0.0, 1.0 - bc5000[i, j]))),
                    "is_quality_pair": is_q,
                }
            )
    pairs = pd.DataFrame(rows)

    off = ~np.eye(n, dtype=bool)
    max_cos = pd.DataFrame(
        {
            "domain": domains,
            "max_cos_k40": np.where(off, cos40, -np.inf).max(axis=1),
            "max_cos_k5000": np.where(off, cos5000, -np.inf).max(axis=1),
            "max_bc_k5000": np.where(off, bc5000, -np.inf).max(axis=1),
        }
    )
    flagged = pairs[pairs["cos_k5000"] > DUP_COS_FLAG].sort_values("cos_k5000", ascending=False)
    qp = pairs[pairs["is_quality_pair"]]
    cross = pairs[~pairs["is_quality_pair"]]
    summary = {
        "flagged_pairs_cos_k5000_gt_0.99": flagged[["domain_a", "domain_b", "cos_k5000", "bc_k5000"]].to_dict("records"),
        "quality_pairs": {
            "n": len(qp),
            "cosine_dist_k5000": {
                "median": float((1 - qp["cos_k5000"]).median()),
                "min": float((1 - qp["cos_k5000"]).min()),
                "max": float((1 - qp["cos_k5000"]).max()),
            },
            "hellinger_k5000": {
                "median": float(qp["hellinger_k5000"].median()),
                "min": float(qp["hellinger_k5000"].min()),
                "max": float(qp["hellinger_k5000"].max()),
            },
        },
        "cross_pairs": {
            "n": len(cross),
            "cosine_dist_k5000_median": float((1 - cross["cos_k5000"]).median()),
            "hellinger_k5000_median": float(cross["hellinger_k5000"].median()),
        },
    }
    return pairs, max_cos, summary


# ---------------------------------------------------------------------------
# 3. Reconstruction
# ---------------------------------------------------------------------------


def audit_reconstruction(vs: dict[int, np.ndarray], n_domains: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    w = rng.dirichlet(np.ones(n_domains), size=N_RECON_DRAWS)  # (200, 39)
    rows = []
    for k, v in vs.items():
        h = np.asarray(v) @ w.T  # (cells, 200)
        w_hat, *_ = np.linalg.lstsq(np.asarray(v), h, rcond=None)
        err = np.abs(w_hat.T - w)
        rows.append(
            {
                "k": k,
                "max_abs_err": float(err.max()),
                "median_abs_err": float(np.median(err)),
                "p99_abs_err": float(np.quantile(err, 0.99)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Derivability honesty control
# ---------------------------------------------------------------------------


def audit_derivability(w60: np.ndarray, v40: np.ndarray) -> dict:
    """R^2 of a linear map raw-weights (78) -> per-phase K=40 histogram (80), fit on all runs."""
    n = w60.shape[0]
    x = w60.reshape(n, -1)  # (n, 78)
    h = np.concatenate([w60[:, p, :] @ np.asarray(v40).T for p in (0, 1)], axis=1)  # (n, 80)
    x1 = np.concatenate([x, np.ones((n, 1))], axis=1)
    coef, *_ = np.linalg.lstsq(x1, h, rcond=None)
    resid = h - x1 @ coef
    ss_res = (resid**2).sum(axis=0)
    ss_tot = ((h - h.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.inf)
    return {
        "n_feature_dims": int(h.shape[1]),
        "mean_r2": float(r2.mean()),
        "median_r2": float(np.median(r2)),
        "min_r2": float(r2.min()),
    }


# ---------------------------------------------------------------------------
# 5. Nested-CV fit comparison
# ---------------------------------------------------------------------------


def make_ridge() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS))])


def make_lgbm() -> GridSearchCV:
    return GridSearchCV(
        lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=SEED, verbosity=-1),
        param_grid=LGBM_GRID,
        cv=KFold(n_splits=3, shuffle=True, random_state=SEED),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )


def nested_cv(x: np.ndarray, y: np.ndarray, kind: str) -> dict:
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    oof = np.full((N_REPEATS, len(y)), np.nan)
    for i, (tr, te) in enumerate(rkf.split(x)):
        est = make_ridge() if kind == "ridge" else make_lgbm()
        est.fit(x[tr], y[tr])
        oof[i // N_SPLITS, te] = est.predict(x[te])
    sp = np.array([spearmanr(oof[r], y).statistic for r in range(N_REPEATS)])
    rmse = np.sqrt(((oof - y) ** 2).mean(axis=1))
    return {
        "oof_spearman_mean": float(sp.mean()),
        "oof_spearman_std": float(sp.std(ddof=1)),
        "oof_rmse_mean": float(rmse.mean()),
        "oof_rmse_std": float(rmse.std(ddof=1)),
    }


def cv_comparison(runs: pd.DataFrame, domains: list[str], v40: np.ndarray, rff_means: np.ndarray) -> pd.DataFrame:
    rows = []
    for scale, sub in runs.groupby("scale"):
        sub = sub.reset_index(drop=True)
        w = run_weight_matrix(sub, domains)  # (n, 2, 39)
        y = sub[TARGET].to_numpy(dtype=np.float64)
        x_weights = w.reshape(len(sub), -1)  # 78
        x_h40 = np.concatenate([w[:, p, :] @ np.asarray(v40).T for p in (0, 1)], axis=1)  # 80
        x_rff = np.concatenate([w[:, p, :] @ rff_means for p in (0, 1)], axis=1)  # 4096
        specs = [
            ("ridge_weights", "ridge", x_weights),
            ("ridge_h40", "ridge", x_h40),
            ("lgbm_weights", "lgbm", x_weights),
            ("lgbm_h40", "lgbm", x_h40),
            ("ridge_rff", "ridge", x_rff),
        ]
        for name, kind, x in specs:
            res = nested_cv(x, y, kind)
            logger.info(
                "cv %s %s: spearman %.4f±%.4f rmse %.5f",
                scale,
                name,
                res["oof_spearman_mean"],
                res["oof_spearman_std"],
                res["oof_rmse_mean"],
            )
            rows.append({"scale": scale, "model": name, "n_features": x.shape[1], "n_runs": len(sub), **res})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hists, views, _centroids = load_artifacts()
    domains = [h.domain for h in hists]
    runs = pd.read_parquet(SCRATCH / "runs.parquet")

    vs: dict[int, featurize.CompositionMatrix] = {}
    orders: dict[int, list[str]] = {}
    for k in GRANULARITIES:
        vs[k], orders[k] = featurize.composition_matrix(hists, k=k, views=views)
    assert all(orders[k] == domains for k in GRANULARITIES)

    sv_df, spectrum = audit_spectrum(vs)
    pairs_df, max_cos_df, dup_summary = audit_duplicates(vs, domains)
    recon_df = audit_reconstruction(vs, len(domains))
    runs60 = runs[runs["scale"] == "60m_1p2b"].reset_index(drop=True)
    deriv = audit_derivability(run_weight_matrix(runs60, domains), np.asarray(vs[40]))

    npz = np.load(HIST_DIR / "rff_means.npz")
    rff_order = {d: i for i, d in enumerate(npz["domains"].tolist())}
    rff_means = np.asarray(npz["rff_means"], dtype=np.float64)[[rff_order[d] for d in domains]]
    cv_df = cv_comparison(runs, domains, vs[40], rff_means)

    sv_df.to_parquet(OUT_DIR / "singular_values.parquet", index=False)
    pairs_df.to_parquet(OUT_DIR / "pair_similarity.parquet", index=False)
    max_cos_df.to_parquet(OUT_DIR / "nearest_duplicate.parquet", index=False)
    recon_df.to_parquet(OUT_DIR / "reconstruction.parquet", index=False)
    cv_df.to_parquet(OUT_DIR / "cv_comparison.parquet", index=False)

    results = {
        "seed": SEED,
        "target": TARGET,
        "spectrum": spectrum,
        "duplicates": dup_summary,
        "reconstruction": recon_df.to_dict("records"),
        "derivability_control": deriv,
        "cv_comparison": cv_df.to_dict("records"),
    }
    (OUT_DIR / "h1_results.json").write_text(json.dumps(results, indent=2, default=json_default))

    print("\n=== H1: spectrum ===")
    for s in spectrum:
        print(
            f"K={s['k']:>4}: shape={tuple(s['shape'])} rank(1e-10)={s['numerical_rank_1e-10']} "
            f"rank(eps)={s['numerical_rank_eps']} cond={s['condition_number']:.3e}"
        )
        print(f"        top10 sv: {np.array2string(np.array(s['top10_sv']), precision=4)}")
        print(f"        bottom5 sv: {np.array2string(np.array(s['bottom5_sv']), precision=6)}")
    print("\n=== H1: duplicates ===")
    print(json.dumps(dup_summary, indent=2, default=json_default))
    print("\n=== H1: reconstruction (200 Dirichlet draws) ===")
    print(recon_df.to_string(index=False))
    print("\n=== H1: derivability control (weights -> h K=40, 60M runs) ===")
    print(deriv)
    print("\n=== H1: nested-CV comparison ===")
    print(cv_df.to_string(index=False))
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
