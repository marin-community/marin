# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm"]
# ///
"""H2a content->domain-response LODO gate for the mixing-via-embeddings experiment.

PRIMARY response definition: per scale, ridge (alpha by 10-fold CV) of
`eval/uncheatable_eval/bpb` on the 78 per-phase mixture weights; each domain's response
params are its two coefficients (beta_phase0, beta_phase1), with bootstrap SEs
(1000 run-resamples).

LODO: for each of the 39 domains, fit ridge (inner LOO-CV alpha) mapping the other 38
domains' content features -> their (beta_p0, beta_p1) and predict the holdout's params.
Featurizations: semantic K=40 / K=1000 fracs, KME (mass-weighted centroid mean), RFF
(2048-dim); controls: shuffled-columns and matched-random on the K=40 composition matrix
(20 seeds each). Headline: uncertainty-weighted (1/SE^2) Pearson + Spearman between
predicted and fitted betas across all 39 folds (pooled phases + per phase), and paired
bootstrap 95% CIs of (semantic - control), resampling domains.

Pre-registered GATE: PASS iff at least one semantic featurization beats BOTH controls
with CI excluding 0, in either the clustered or the RFF arm, at 60M (primary scale);
300M reported as replication. SECONDARY response definition (swarm-branch pre-fitted
per-domain DSP/GRP params) was searched for and does not exist as a per-domain table on
commit bf26b666a (GRP params are factor-level), so only PRIMARY runs.

Also runs the mandatory quality-pair diagnostic on the 13 dolma3_cc high/low topic
pairs (beta differences + sign predictability = the quality-blindness ceiling).

Outputs parquet + json under `scratch/mixture_features/h2a/` and prints a report.
"""

import json
import logging
from pathlib import Path

import featurize
import numpy as np
import pandas as pd
from h1_audit import json_default, load_artifacts, quality_topics, run_weight_matrix
from scipy.stats import binomtest, rankdata, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = REPO_ROOT / "scratch" / "mixture_features"
HIST_DIR = SCRATCH / "domain_histograms"
OUT_DIR = SCRATCH / "h2a"

SEED = 0
TARGET = "eval/uncheatable_eval/bpb"
PRIMARY_SCALE = "60m_1p2b"
N_BOOT = 1000
N_CONTROL_SEEDS = 20
RESPONSE_ALPHAS = np.logspace(-6, 3, 46)  # 10-fold CV grid for the response fit
LODO_ALPHAS = np.logspace(-3, 3, 25)  # pre-registered inner-CV grid
SEMANTIC = ("sem_k40", "sem_k1000", "kme", "rff")
CONTROLS = ("ctrl_shuffled", "ctrl_matched")
ARM = {"sem_k40": "clustered", "sem_k1000": "clustered", "kme": "clustered", "rff": "rff"}


# ---------------------------------------------------------------------------
# PRIMARY response fit: ridge betas on the 78 per-phase weights
# ---------------------------------------------------------------------------


def fit_response(sub: pd.DataFrame, domains: list[str]) -> dict:
    w = run_weight_matrix(sub, domains)  # (n, 2, 39)
    x = w.reshape(len(sub), -1)  # (n, 78): phase0 block then phase1 block
    y = sub[TARGET].to_numpy(dtype=np.float64)
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    gs = GridSearchCV(Ridge(), {"alpha": RESPONSE_ALPHAS}, cv=cv, scoring="neg_mean_squared_error")
    gs.fit(x, y)
    alpha = float(gs.best_params_["alpha"])
    if alpha in (RESPONSE_ALPHAS[0], RESPONSE_ALPHAS[-1]):
        logger.warning("response-fit alpha %g is at the grid boundary", alpha)
    oof = cross_val_predict(Ridge(alpha=alpha), x, y, cv=cv)
    oof_sp = float(spearmanr(oof, y).statistic)
    betas = Ridge(alpha=alpha).fit(x, y).coef_.reshape(2, 39)

    rng = np.random.default_rng(SEED)
    boot = np.empty((N_BOOT, 2, 39))
    n = len(y)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot[b] = Ridge(alpha=alpha).fit(x[idx], y[idx]).coef_.reshape(2, 39)
    se = boot.std(axis=0, ddof=1)
    return {
        "alpha": alpha,
        "oof_spearman": oof_sp,
        "oof_rmse": float(np.sqrt(((oof - y) ** 2).mean())),
        "betas": betas,  # (2, 39)
        "se": se,  # (2, 39)
        "boot": boot,  # (N_BOOT, 2, 39)
        "stability_mean_abs_beta_over_se": float((np.abs(betas) / se).mean()),
    }


# ---------------------------------------------------------------------------
# Featurizations (rows = domains, sorted order)
# ---------------------------------------------------------------------------


def domain_features(
    vs: dict[int, np.ndarray], centroids: np.ndarray, rff_means: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    v40, v1000, v5000 = (np.asarray(vs[k]) for k in (40, 1000, 5000))
    semantic = {
        "sem_k40": v40.T,
        "sem_k1000": v1000.T,
        "kme": v5000.T @ centroids,  # (39, 192) mass-weighted mean centroid vector
        "rff": rff_means,  # (39, 2048)
    }
    controls = {
        "ctrl_shuffled": [featurize.shuffled_columns_v(v40, seed=s).T for s in range(N_CONTROL_SEEDS)],
        "ctrl_matched": [featurize.matched_random_v(v40, seed=s).T for s in range(N_CONTROL_SEEDS)],
    }
    return semantic, controls


def lodo_predict(feats: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """LODO ridge (per-phase target, inner LOO-CV alpha) predictions, shape (39, 2)."""
    n = feats.shape[0]
    preds = np.empty((n, 2))
    for k in range(n):
        tr = np.arange(n) != k
        for p in range(2):
            model = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=LODO_ALPHAS))])
            model.fit(feats[tr], betas[p, tr])
            preds[k, p] = model.predict(feats[k : k + 1])[0]
    return preds


# ---------------------------------------------------------------------------
# Uncertainty-weighted correlations + paired domain bootstrap
# ---------------------------------------------------------------------------


def wpearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    mx, my = w @ x, w @ y
    cov = w @ ((x - mx) * (y - my))
    return float(cov / np.sqrt((w @ (x - mx) ** 2) * (w @ (y - my) ** 2)))


def wspearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return wpearson(rankdata(x), rankdata(y), w)


CORR_FNS = {"pearson": wpearson, "spearman": wspearman}


def pooled(arr: np.ndarray, dom_idx: np.ndarray) -> np.ndarray:
    """Stack phase0 then phase1 values for the selected domains: (2,39)[..., idx] -> (2*len(idx),)."""
    return np.concatenate([arr[0, dom_idx], arr[1, dom_idx]])


def scale_correlations(
    betas: np.ndarray,
    se: np.ndarray,
    sem_preds: dict[str, np.ndarray],
    ctrl_preds: dict[str, list[np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Headline correlations (pooled + per-phase) and paired-bootstrap CIs of semantic - control."""
    n_dom = betas.shape[1]
    weights = 1.0 / se**2
    all_idx = np.arange(n_dom)

    def corr_rows(name: str, preds: np.ndarray) -> list[dict]:
        rows = []
        for cname, fn in CORR_FNS.items():
            rows.append(
                {
                    "featurization": name,
                    "corr": cname,
                    "pooled": fn(pooled(preds.T, all_idx), pooled(betas, all_idx), pooled(weights, all_idx)),
                    "phase0": fn(preds[:, 0], betas[0], weights[0]),
                    "phase1": fn(preds[:, 1], betas[1], weights[1]),
                }
            )
        return rows

    rows = []
    for name, preds in sem_preds.items():
        rows.extend(corr_rows(name, preds))
    for name, seed_preds in ctrl_preds.items():
        per_seed = [corr_rows(f"{name}", p) for p in seed_preds]
        for i in range(len(per_seed[0])):
            agg = dict(per_seed[0][i])
            for col in ("pooled", "phase0", "phase1"):
                agg[col] = float(np.mean([ps[i][col] for ps in per_seed]))
            rows.append(agg)
    corr_df = pd.DataFrame(rows)

    # Paired bootstrap over domains (identical resamples across featurizations).
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, n_dom, (N_BOOT, n_dom))
    ci_rows = []
    for cname, fn in CORR_FNS.items():
        sem_boot = {}
        for name, preds in sem_preds.items():
            sem_boot[name] = np.array(
                [fn(pooled(preds.T, idx), pooled(betas, idx), pooled(weights, idx)) for idx in boot_idx]
            )
        for ctrl, seed_preds in ctrl_preds.items():
            ctrl_boot = np.mean(
                [
                    [fn(pooled(p.T, idx), pooled(betas, idx), pooled(weights, idx)) for idx in boot_idx]
                    for p in seed_preds
                ],
                axis=0,
            )
            for name in sem_preds:
                diff = sem_boot[name] - ctrl_boot
                lo, hi = np.percentile(diff, [2.5, 97.5])
                ci_rows.append(
                    {
                        "featurization": name,
                        "arm": ARM[name],
                        "control": ctrl,
                        "corr": cname,
                        "diff_mean": float(diff.mean()),
                        "ci_lo": float(lo),
                        "ci_hi": float(hi),
                        "beats_control": bool(lo > 0),
                    }
                )
    return corr_df, pd.DataFrame(ci_rows)


# ---------------------------------------------------------------------------
# Quality-pair diagnostic (13 dolma3_cc high/low topics)
# ---------------------------------------------------------------------------


def quality_pair_diagnostic(
    domains: list[str],
    fit: dict,
    sem_preds: dict[str, np.ndarray],
    ctrl_preds: dict[str, list[np.ndarray]],
    v5000: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    topics = quality_topics(domains)
    idx = {d: i for i, d in enumerate(domains)}
    hi = np.array([idx[f"{t}_high"] for t in topics])
    lo = np.array([idx[f"{t}_low"] for t in topics])

    vn = np.sqrt(np.asarray(v5000))  # for Hellinger; raw cos too
    v = np.asarray(v5000)
    cosv = (v / np.linalg.norm(v, axis=0)).T @ (v / np.linalg.norm(v, axis=0))

    betas, boot = fit["betas"], fit["boot"]
    rows = []
    for p in range(2):
        d = betas[p, hi] - betas[p, lo]
        d_se = (boot[:, p, hi] - boot[:, p, lo]).std(axis=0, ddof=1)
        for t, dt, dse, i, j in zip(topics, d, d_se, hi, lo, strict=True):
            rows.append(
                {
                    "topic": t.removeprefix("dolma3_cc/"),
                    "phase": p,
                    "beta_high": float(betas[p, i]),
                    "beta_low": float(betas[p, j]),
                    "diff_high_minus_low": float(dt),
                    "se_diff": float(dse),
                    "z": float(dt / dse),
                    "cos_k5000": float(cosv[i, j]),
                    "bc_k5000": float((vn[:, i] / np.linalg.norm(vn[:, i])) @ (vn[:, j] / np.linalg.norm(vn[:, j]))),
                }
            )
    pairs_df = pd.DataFrame(rows)

    sign_rows = []
    fitted_diff = betas[:, hi] - betas[:, lo]  # (2, 13)

    def sign_acc(preds: np.ndarray) -> list[float]:
        pd_ = preds.T[:, hi] - preds.T[:, lo]  # (2, 13)
        return [float((np.sign(pd_[p]) == np.sign(fitted_diff[p])).mean()) for p in range(2)]

    for name, preds in sem_preds.items():
        acc = sign_acc(preds)
        for p in range(2):
            k = round(acc[p] * len(topics))
            sign_rows.append(
                {
                    "featurization": name,
                    "phase": p,
                    "sign_accuracy": acc[p],
                    "n_pairs": len(topics),
                    "binom_p_vs_0.5": float(binomtest(k, len(topics), 0.5).pvalue),
                }
            )
    for name, seed_preds in ctrl_preds.items():
        accs = np.array([sign_acc(p) for p in seed_preds])  # (seeds, 2)
        for p in range(2):
            sign_rows.append(
                {
                    "featurization": name,
                    "phase": p,
                    "sign_accuracy": float(accs[:, p].mean()),
                    "n_pairs": len(topics),
                    "binom_p_vs_0.5": float("nan"),
                }
            )
    return pairs_df, pd.DataFrame(sign_rows)


# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hists, views, centroids = load_artifacts()
    domains = [h.domain for h in hists]
    runs = pd.read_parquet(SCRATCH / "runs.parquet")

    vs = {k: featurize.composition_matrix(hists, k=k, views=views)[0] for k in (40, 1000, 5000)}
    npz = np.load(HIST_DIR / "rff_means.npz")
    rff_order = {d: i for i, d in enumerate(npz["domains"].tolist())}
    rff_means = np.asarray(npz["rff_means"], dtype=np.float64)[[rff_order[d] for d in domains]]
    sem_feats, ctrl_feats = domain_features(vs, centroids, rff_means)

    betas_rows, lodo_rows, corr_frames, ci_frames, qp_frames, sign_frames = [], [], [], [], [], []
    fit_summaries = {}
    for scale in sorted(runs["scale"].unique()):
        sub = runs[runs["scale"] == scale].reset_index(drop=True)
        fit = fit_response(sub, domains)
        fit_summaries[scale] = {
            "n_runs": len(sub),
            "alpha": fit["alpha"],
            "oof_spearman": fit["oof_spearman"],
            "oof_rmse": fit["oof_rmse"],
            "stability_mean_abs_beta_over_se": fit["stability_mean_abs_beta_over_se"],
        }
        logger.info(
            "%s response fit: alpha=%g oof_spearman=%.3f stability=%.2f",
            scale,
            fit["alpha"],
            fit["oof_spearman"],
            fit["stability_mean_abs_beta_over_se"],
        )

        sem_preds = {name: lodo_predict(f, fit["betas"]) for name, f in sem_feats.items()}
        ctrl_preds = {name: [lodo_predict(f, fit["betas"]) for f in fs] for name, fs in ctrl_feats.items()}
        logger.info(
            "%s: LODO done (%d semantic + %d control fits)",
            scale,
            len(sem_preds),
            sum(len(v) for v in ctrl_preds.values()),
        )

        corr_df, ci_df = scale_correlations(fit["betas"], fit["se"], sem_preds, ctrl_preds)
        corr_df["scale"] = scale
        ci_df["scale"] = scale
        corr_frames.append(corr_df)
        ci_frames.append(ci_df)

        qp_df, sign_df = quality_pair_diagnostic(domains, fit, sem_preds, ctrl_preds, vs[5000])
        qp_df["scale"] = scale
        sign_df["scale"] = scale
        qp_frames.append(qp_df)
        sign_frames.append(sign_df)

        for p in range(2):
            for i, d in enumerate(domains):
                betas_rows.append(
                    {"scale": scale, "domain": d, "phase": p, "beta": fit["betas"][p, i], "se": fit["se"][p, i]}
                )
                row = {
                    "scale": scale,
                    "domain": d,
                    "phase": p,
                    "fitted_beta": fit["betas"][p, i],
                    "se": fit["se"][p, i],
                }
                for name, preds in sem_preds.items():
                    row[f"pred_{name}"] = preds[i, p]
                for name, seed_preds in ctrl_preds.items():
                    stack = np.array([sp[i, p] for sp in seed_preds])
                    row[f"pred_{name}_mean"] = stack.mean()
                    row[f"pred_{name}_std"] = stack.std(ddof=1)
                lodo_rows.append(row)

    corr_all = pd.concat(corr_frames, ignore_index=True)
    ci_all = pd.concat(ci_frames, ignore_index=True)
    qp_all = pd.concat(qp_frames, ignore_index=True)
    sign_all = pd.concat(sign_frames, ignore_index=True)

    pd.DataFrame(betas_rows).to_parquet(OUT_DIR / "betas.parquet", index=False)
    pd.DataFrame(lodo_rows).to_parquet(OUT_DIR / "lodo_predictions.parquet", index=False)
    corr_all.to_parquet(OUT_DIR / "correlations.parquet", index=False)
    ci_all.to_parquet(OUT_DIR / "bootstrap_cis.parquet", index=False)
    qp_all.to_parquet(OUT_DIR / "quality_pairs.parquet", index=False)
    sign_all.to_parquet(OUT_DIR / "quality_pair_sign_accuracy.parquet", index=False)

    # Pre-registered gate at the primary scale: any semantic featurization beating BOTH
    # controls with CI excluding 0 (evaluated per correlation metric, either arm).
    def gate_verdict(scale: str) -> dict:
        sub = ci_all[ci_all["scale"] == scale]
        passes = []
        for name in SEMANTIC:
            for cname in CORR_FNS:
                beats = sub[(sub["featurization"] == name) & (sub["corr"] == cname)]["beats_control"]
                if len(beats) == len(CONTROLS) and beats.all():
                    passes.append({"featurization": name, "arm": ARM[name], "corr": cname})
        return {"scale": scale, "pass": bool(passes), "passing_combinations": passes}

    gate = {
        "primary": gate_verdict(PRIMARY_SCALE),
        "replication": gate_verdict(next(s for s in runs["scale"].unique() if s != PRIMARY_SCALE)),
        "response_fits": fit_summaries,
        "secondary_response_definition": (
            "unavailable: no per-domain DSP/GRP param table for qsplit240 "
            "uncheatable bpb on swarm commit bf26b666a (GRP params are factor-level)"
        ),
        "n_boot": N_BOOT,
        "n_control_seeds": N_CONTROL_SEEDS,
        "seed": SEED,
    }
    (OUT_DIR / "gate.json").write_text(json.dumps(gate, indent=2, default=json_default))

    pd.set_option("display.width", 200)
    print("\n=== H2a: response fits ===")
    print(json.dumps(fit_summaries, indent=2))
    print("\n=== H2a: uncertainty-weighted correlations (controls seed-averaged) ===")
    print(corr_all.to_string(index=False))
    print("\n=== H2a: paired bootstrap CIs of (semantic - control) ===")
    print(ci_all.to_string(index=False))
    print("\n=== H2a GATE ===")
    print(json.dumps({k: gate[k] for k in ("primary", "replication")}, indent=2))
    print("\n=== quality pairs (fitted beta_high - beta_low) ===")
    print(qp_all.to_string(index=False))
    print("\n=== quality-pair sign accuracy (quality-blindness ceiling) ===")
    print(sign_all.to_string(index=False))
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
