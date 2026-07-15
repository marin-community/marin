# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scipy"]
# ///
"""Grug holdout test, PHASE 2: the pre-registered readout on the 40 quarantined runs.

Run ONLY after the predictions SHA-256 is published on issue #7067 (amendment comment).
Opens ``QUARANTINE_test_labels.parquet`` exactly once, computes the amended readout:

  R1  Spearman(primary predictions, realized zmacro_english_20) + permutation p
      (10k label permutations, seed 0). PASS: p < 0.001.
  R2  paired dSpearman primary - best weights baseline (best-on-test of weights-ridge /
      weights-LGBM, the conservative reading), 1000-resample paired bootstrap 95% CI.
      PASS: point estimate >= 0.
  R3  primary test Spearman >= train-CV - 0.15 = 0.668.
  R4  primary - matched-random control dSpearman + bootstrap CI (measured, no bar).

Plus RMSE for every frozen model and the realized-vs-predicted scatter parquet.
Verdict: PASS = R1 and R2 and R3. Anything beyond is labeled EXPLORATORY.

Outputs (scratch/mixture_features/grug/): holdout_readout.json, holdout_readout.md,
realized_vs_predicted.parquet.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

logger = logging.getLogger("grug_holdout_readout")

GRUG_DIR = Path(__file__).resolve().parents[3] / "scratch" / "mixture_features" / "grug"
PREDICTIONS = GRUG_DIR / "test_predictions.parquet"
QUARANTINE = GRUG_DIR / "QUARANTINE_test_labels.parquet"
MANIFEST = GRUG_DIR / "holdout_manifest.json"
TARGET_CANDIDATES = GRUG_DIR / "target_candidates.json"
OUT_JSON = GRUG_DIR / "holdout_readout.json"
OUT_MD = GRUG_DIR / "holdout_readout.md"
OUT_SCATTER = GRUG_DIR / "realized_vs_predicted.parquet"

PUBLISHED_PREDICTIONS_SHA = "0a4cc4e6430d92001e65b9ecf00873dd02b5666fb29da41c38d77d7d34c2474f"
PRIMARY = "4_hellinger_kernel_k1000"
WEIGHTS_BASELINES = ("1_weights_ridge", "2_weights_lgbm")
MATCHED_CONTROL = "ctrl_matched_mean10"
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 1_000
SEED = 0
R1_P_BAR = 0.001
R3_BAR = 0.8180054500085157 - 0.15


def sp(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)


def permutation_p(pred: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    obs = sp(pred, y)
    perm = np.array([sp(pred, rng.permutation(y)) for _ in range(N_PERMUTATIONS)])
    return {
        "observed_spearman": obs,
        "n_permutations": N_PERMUTATIONS,
        "p_one_sided": float((1 + (perm >= obs).sum()) / (1 + N_PERMUTATIONS)),
        "p_two_sided": float((1 + (np.abs(perm) >= abs(obs)).sum()) / (1 + N_PERMUTATIONS)),
        "perm_spearman_max": float(perm.max()),
    }


def paired_bootstrap_delta(pa: np.ndarray, pb: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """dSpearman = Spearman(pa,y) - Spearman(pb,y); percentile CI over run resamples."""
    point = sp(pa, y) - sp(pb, y)
    n = len(y)
    deltas = []
    while len(deltas) < N_BOOTSTRAP:
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 3:  # degenerate resample, cannot rank
            continue
        deltas.append(sp(pa[idx], y[idx]) - sp(pb[idx], y[idx]))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_spearman_point": point,
        "bootstrap_ci95": [float(lo), float(hi)],
        "bootstrap_mean": float(deltas.mean()),
        "n_bootstrap": N_BOOTSTRAP,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # --- integrity gates before opening anything ---
    pred_sha = hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest()
    assert pred_sha == PUBLISHED_PREDICTIONS_SHA, f"predictions SHA mismatch: {pred_sha}"
    manifest = json.loads(MANIFEST.read_text())
    quar_sha = hashlib.sha256(QUARANTINE.read_bytes()).hexdigest()
    assert quar_sha == manifest["sha256_test_labels"], f"quarantine SHA mismatch: {quar_sha}"

    cand = json.loads(TARGET_CANDIDATES.read_text())
    rec = cand["recommended_target"]
    task_list, z_mu, z_sd = rec["task_list"], rec["train_z_mu"], rec["train_z_sd"]

    ptbl = pq.read_table(PREDICTIONS)
    meta = json.loads(ptbl.schema.metadata[b"grug_holdout_phase1"])
    preds_long = ptbl.to_pandas()
    pw = preds_long.pivot(index="experiment_index", columns="model", values="prediction")

    # --- THE ONE QUARANTINE OPEN ---
    labels = pd.read_parquet(QUARANTINE)
    logger.info("quarantine opened once: %d rows, cols %s", len(labels), list(labels.columns))
    assert sorted(labels["experiment_index"].tolist()) == sorted(manifest["test_experiment_indices"])

    y_rows = []
    for _, r in labels.iterrows():
        ev = json.loads(r["evals"])
        zs = [(ev[t]["bpb"] - z_mu[t]) / z_sd[t] for t in task_list if t in ev and "bpb" in ev[t]]
        y_rows.append((int(r["experiment_index"]), float(np.mean(zs)), len(zs)))
    ydf = pd.DataFrame(y_rows, columns=["experiment_index", "realized", "n_tasks_observed"]).set_index(
        "experiment_index"
    )
    pw = pw.loc[ydf.index]
    y = ydf["realized"].to_numpy()
    logger.info(
        "realized %s: mean %.4f std %.4f, tasks observed %d-%d",
        rec["name"],
        y.mean(),
        y.std(),
        ydf["n_tasks_observed"].min(),
        ydf["n_tasks_observed"].max(),
    )

    models = sorted(pw.columns)
    spearman_all = {m: sp(pw[m].to_numpy(), y) if pw[m].std() > 0 else float("nan") for m in models}
    rmse_all = {m: float(np.sqrt(((pw[m].to_numpy() - y) ** 2).mean())) for m in models}

    # --- R1 ---
    rng = np.random.default_rng(SEED)
    p_primary = pw[PRIMARY].to_numpy()
    r1 = permutation_p(p_primary, y, rng)
    r1["pass"] = r1["p_one_sided"] < R1_P_BAR

    # --- R2 (vs best-on-test weights baseline; both deltas reported) ---
    best_weights = max(WEIGHTS_BASELINES, key=lambda m: spearman_all[m])
    r2 = {
        "best_weights_baseline": best_weights,
        "baseline_test_spearman": spearman_all[best_weights],
        **paired_bootstrap_delta(p_primary, pw[best_weights].to_numpy(), y, np.random.default_rng(SEED)),
        "delta_vs_all_weights_baselines": {
            m: paired_bootstrap_delta(p_primary, pw[m].to_numpy(), y, np.random.default_rng(SEED))
            for m in WEIGHTS_BASELINES
        },
        "delta_rmse_point": rmse_all[PRIMARY] - rmse_all[best_weights],
    }
    r2["pass"] = r2["delta_spearman_point"] >= 0.0

    # --- R3 ---
    r3 = {"primary_test_spearman": spearman_all[PRIMARY], "bar": R3_BAR, "pass": spearman_all[PRIMARY] >= R3_BAR}

    # --- R4 (measured, no bar) ---
    r4 = {
        "control": MATCHED_CONTROL,
        "control_test_spearman": spearman_all[MATCHED_CONTROL],
        **paired_bootstrap_delta(p_primary, pw[MATCHED_CONTROL].to_numpy(), y, np.random.default_rng(SEED)),
    }

    verdict = bool(r1["pass"] and r2["pass"] and r3["pass"])

    # --- scatter parquet ---
    scatter = preds_long.merge(ydf.reset_index(), on="experiment_index")
    scatter.to_parquet(OUT_SCATTER, index=False)

    readout = {
        "protocol": "test_protocol.md + amendment on issue #7067 (target zmacro_english_20)",
        "predictions_sha256": pred_sha,
        "quarantine_sha256": quar_sha,
        "git_sha_predictions": meta["git_sha"],
        "target": {"name": rec["name"], "n_tasks": len(task_list)},
        "n_test": len(y),
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "R4": r4,
        "verdict_pass": verdict,
        "spearman_all_models": spearman_all,
        "rmse_all_models": rmse_all,
        "realized_summary": {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        },
    }
    OUT_JSON.write_text(json.dumps(readout, indent=2))

    lines = [
        "# Grug holdout readout (pre-registered, amended target zmacro_english_20)\n",
        f"- predictions sha256 `{pred_sha}` (published before label opening)",
        f"- quarantine sha256 `{quar_sha}` == manifest",
        f"- n_test = {len(y)}; realized target mean {y.mean():.4f} std {y.std():.4f}\n",
        "## Pre-registered readout\n",
        f"- **R1** Spearman(primary, realized) = **{r1['observed_spearman']:.4f}**, permutation p "
        f"(10k, one-sided) = **{r1['p_one_sided']:.6f}** (two-sided {r1['p_two_sided']:.6f}; "
        f"max perm rho {r1['perm_spearman_max']:.3f}) -> {'PASS' if r1['pass'] else 'FAIL'} (bar p<0.001)",
        f"- **R2** vs best weights baseline ({best_weights}, test Spearman {r2['baseline_test_spearman']:.4f}): "
        f"dSpearman = **{r2['delta_spearman_point']:+.4f}**, bootstrap 95% CI "
        f"[{r2['bootstrap_ci95'][0]:+.4f}, {r2['bootstrap_ci95'][1]:+.4f}], dRMSE {r2['delta_rmse_point']:+.4f} "
        f"-> {'PASS' if r2['pass'] else 'FAIL'} (bar point >= 0)",
        f"- **R3** primary test Spearman {spearman_all[PRIMARY]:.4f} vs bar {R3_BAR:.4f} -> "
        f"{'PASS' if r3['pass'] else 'FAIL'}",
        f"- **R4** vs matched-random control ({MATCHED_CONTROL}, test Spearman "
        f"{r4['control_test_spearman']:.4f}): dSpearman = {r4['delta_spearman_point']:+.4f}, "
        f"95% CI [{r4['bootstrap_ci95'][0]:+.4f}, {r4['bootstrap_ci95'][1]:+.4f}] (measured, no bar)",
        f"\n## VERDICT: **{'PASS' if verdict else 'FAIL'}** (R1 & R2 & R3)\n",
        "## All frozen models on the 40 (Spearman / RMSE)\n",
        "| model | Spearman | RMSE |",
        "|-------|----------|------|",
    ]
    for m in models:
        lines.append(f"| {m} | {spearman_all[m]:+.4f} | {rmse_all[m]:.4f} |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
