# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-1 midtraining loss predictor validation.

Given completed midtraining runs with different LR factors, test whether a
schedule-aware power-law fit in cumulative-LR space can predict the final loss
from an early prefix.

Functional form:
    L(u) = L_inf + A * (U - u) ** c
where u = cumulative learning_rate up to step t, U = total cumulative LR at end.

Baselines compared:
    B0: last-value (y at prefix_end)
    B1: a + b / sqrt(t)  (schedule-unaware)
    B2: L_inf + A * (U - u)^c  (schedule-aware; the proposal)

Tests:
    1. Self-prefix: fit on first X% of a run, predict its own final.
    2. Cross-LR LOO: fit shared c on 2 of 3 runs, predict the third.

Data is pulled via the wandb API — tracker_metrics.jsonl at the GCS output path
only carries final summary, not a time series. Target quantity is the EMA over
the last 167 training steps (steps 4600..4767) to stay below the ~0.01 noise
floor established from original-vs-v2 rerun pairs.

Run: ``uv run python scripts/analysis/midtrain_loss_predictor.py``.

See ``.agents/logbooks/midtraining_delphi.md`` for the full plan and
evaluation criteria.
"""

import logging
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import wandb
from scipy.optimize import OptimizeWarning, curve_fit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical runs (clean W&B, lr0.67/0.83 are v2 reruns)
# ---------------------------------------------------------------------------

WANDB_PROJECT = "marin-community/marin"


@dataclass(frozen=True)
class RunSpec:
    name: str
    lr_factor: float
    wandb_name: str


RUNS_1E20: tuple[RunSpec, ...] = (
    RunSpec("lr=0.5", 0.5, "delphi-1e20-iso-d2048-L21-math-10b-lr0.5-4d19a2"),
    RunSpec("lr=0.67", 0.67, "delphi-1e20-iso-d2048-L21-math-10b-lr0.67-v2-a176ff"),
    RunSpec("lr=0.83", 0.83, "delphi-1e20-iso-d2048-L21-math-10b-lr0.83-v2-4487d2"),
)

METRICS_DECREASING = ("train/loss",)
METRICS_INCREASING = ("eval/paloma/c4_en/loss",)
ALL_METRICS = METRICS_DECREASING + METRICS_INCREASING

WARMUP_STEPS = 500  # skip in fits
TOTAL_STEPS = 4768
TARGET_WINDOW = (4600, 4767)  # EMA-average endpoint target
PREFIX_FRACS = (0.3, 0.5, 0.8)
EMA_HALFLIFE_TRAIN = 100  # steps; eval is already low-freq so we skip smoothing


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_run(spec: RunSpec) -> pd.DataFrame:
    """Fetch full step-level history from W&B, with separate queries for
    train and eval series (they're logged at different cadences so a single
    multi-key query would intersect down to only eval-step rows).

    Returns a DataFrame indexed by ``_step`` with columns for every metric
    in ``ALL_METRICS`` plus ``optim/learning_rate``. Eval metrics are NaN
    between eval steps.
    """
    logger.info("Loading %s (wandb=%s)", spec.name, spec.wandb_name)
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{spec.wandb_name}")

    # Two-phase fetch:
    #   Phase 1: every-step metrics (train/loss + learning_rate). ~4768 rows.
    #   Phase 2: eval-step metrics. ~48 rows.
    train_keys = ["_step", "optim/learning_rate", *METRICS_DECREASING]
    eval_keys = ["_step", *METRICS_INCREASING]

    train_rows = list(run.scan_history(keys=train_keys, page_size=2000))
    eval_rows = list(run.scan_history(keys=eval_keys, page_size=2000))
    if not train_rows:
        raise RuntimeError(f"empty train history for {spec.wandb_name}")

    df_train = pd.DataFrame(train_rows).sort_values("_step").reset_index(drop=True)
    df_eval = (
        pd.DataFrame(eval_rows).sort_values("_step").reset_index(drop=True) if eval_rows else pd.DataFrame({"_step": []})
    )

    df = df_train.merge(df_eval, on="_step", how="left")
    df["optim/learning_rate"] = df["optim/learning_rate"].astype(float)
    return df


def compute_cumulative_lr(df: pd.DataFrame) -> pd.Series:
    """u(t) = Σ_{s≤t} lr(s). Returns one row per entry in df in step order."""
    return df["optim/learning_rate"].fillna(0.0).cumsum()


def smooth_train_loss(df: pd.DataFrame) -> pd.DataFrame:
    """EMA-smooth the train/loss series in place; leave evals alone."""
    df = df.copy()
    df["train/loss_smooth"] = df["train/loss"].ewm(halflife=EMA_HALFLIFE_TRAIN).mean()
    return df


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------


def evaluate_final(df: pd.DataFrame, metric: str, window: tuple[int, int] = TARGET_WINDOW) -> float:
    """Target: mean of metric over [window[0], window[1]] inclusive, ignoring NaN."""
    mask = (df["_step"] >= window[0]) & (df["_step"] <= window[1])
    vals = df.loc[mask, metric].dropna()
    if len(vals) == 0:
        raise RuntimeError(f"no rows for metric {metric!r} in window {window}")
    return float(vals.mean())


# ---------------------------------------------------------------------------
# Baseline B0: last value in prefix
# ---------------------------------------------------------------------------


def fit_last_value(df: pd.DataFrame, prefix_frac: float, metric: str) -> float:
    cutoff = prefix_frac * TOTAL_STEPS
    mask = df["_step"] <= cutoff
    vals = df.loc[mask, metric].dropna()
    if len(vals) == 0:
        raise RuntimeError(f"no rows for metric {metric!r} under prefix {prefix_frac}")
    # Use last ~5 observations for stability rather than literal last row
    return float(vals.tail(5).mean())


# ---------------------------------------------------------------------------
# Baseline B1: L(t) = a + b / sqrt(t), raw step axis
# ---------------------------------------------------------------------------


def _fit_curve(model, xdata, ydata, p0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, _ = curve_fit(model, xdata, ydata, p0=p0, maxfev=5000)
    return popt


def fit_sqrt_t(df: pd.DataFrame, prefix_frac: float, metric: str) -> tuple[float, tuple]:
    """Fit L(t) = a + b/sqrt(t) on the second half of [WARMUP_STEPS, cutoff].

    Returns (prediction_at_TOTAL_STEPS, (a, b)).
    """
    cutoff = prefix_frac * TOTAL_STEPS
    fit_lo = max(WARMUP_STEPS, cutoff // 2)
    mask = (df["_step"] >= fit_lo) & (df["_step"] <= cutoff) & df[metric].notna()
    sub = df.loc[mask, ["_step", metric]]
    if len(sub) < 5:
        raise RuntimeError(f"B1 too few points: {len(sub)} at prefix {prefix_frac} for {metric!r}")
    t = sub["_step"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)

    def model(tt, a, b):
        return a + b / np.sqrt(tt)

    popt = _fit_curve(model, t, y, p0=[y[-1], 1.0])
    pred = model(TOTAL_STEPS, *popt)
    return float(pred), tuple(map(float, popt))


# ---------------------------------------------------------------------------
# Baseline B2: schedule-aware L(u) = L_inf + A * (U - u)^c
# ---------------------------------------------------------------------------


def fit_cumlr_power(df: pd.DataFrame, prefix_frac: float, metric: str) -> tuple[float, tuple]:
    """Fit L(u) = L_inf + A*(U - u)^c with bounds on c to avoid degenerate fits.

    Bounds: c ∈ [0.2, 3.0]. Without these, curve_fit tends toward (tiny_A, huge_c),
    which makes the model essentially a constant (L_inf) — a useless fit. Bounds
    also eliminate the negative-c blow-ups seen on Paloma at larger prefixes.

    Sign-handling: A is signed (positive for decreasing metrics, negative for
    rising metrics). Seeded from (y_early - y_late).

    Returns (prediction_at_U, (L_inf, A, c)).
    """
    u_full = compute_cumulative_lr(df)
    U = float(u_full.iloc[-1])
    cutoff = prefix_frac * TOTAL_STEPS
    fit_lo = max(WARMUP_STEPS, cutoff // 2)
    mask = (df["_step"] >= fit_lo) & (df["_step"] <= cutoff) & df[metric].notna()
    sub = df.loc[mask, ["_step", metric]].copy()
    sub["u"] = u_full.loc[mask].to_numpy()
    if len(sub) < 5:
        raise RuntimeError(f"B2 too few points: {len(sub)} at prefix {prefix_frac} for {metric!r}")
    u = sub["u"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)

    # Seed
    L_inf0 = float(y[-1]) * (0.95 if y[-1] > 0 else 1.05)
    A0 = float(y[0] - y[-1]) / max(abs(U - u[0]) ** 0.5, 1e-6)
    c0 = 0.5

    def model(uu, L_inf, A, c):
        rem = np.maximum(U - uu, 1e-12)
        return L_inf + A * np.power(rem, c)

    # Bounds: L_inf and A unbounded; c ∈ [0.2, 3.0]
    lo = [-np.inf, -np.inf, 0.2]
    hi = [np.inf, np.inf, 3.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, _ = curve_fit(model, u, y, p0=[L_inf0, A0, c0], bounds=(lo, hi), maxfev=10000)
    pred = model(U, *popt)
    return float(pred), tuple(map(float, popt))


def fit_cumlr_fixed_c(df: pd.DataFrame, prefix_frac: float, metric: str, c_fixed: float = 1.0) -> tuple[float, tuple]:
    """Fit L(u) = L_inf + A*(U - u)^c with c FIXED (default c=1.0, linear in
    remaining LR). Two free params (L_inf, A), same count as B1, but
    schedule-aware via the `u` axis.
    """
    u_full = compute_cumulative_lr(df)
    U = float(u_full.iloc[-1])
    cutoff = prefix_frac * TOTAL_STEPS
    fit_lo = max(WARMUP_STEPS, cutoff // 2)
    mask = (df["_step"] >= fit_lo) & (df["_step"] <= cutoff) & df[metric].notna()
    sub = df.loc[mask, ["_step", metric]].copy()
    sub["u"] = u_full.loc[mask].to_numpy()
    if len(sub) < 5:
        raise RuntimeError(f"B3 too few points: {len(sub)} at prefix {prefix_frac} for {metric!r}")
    u = sub["u"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)

    L_inf0 = float(y[-1]) * (0.95 if y[-1] > 0 else 1.05)
    A0 = float(y[0] - y[-1]) / max(abs(U - u[0]) ** c_fixed, 1e-6)

    def model(uu, L_inf, A):
        rem = np.maximum(U - uu, 1e-12)
        return L_inf + A * np.power(rem, c_fixed)

    popt = _fit_curve(model, u, y, p0=[L_inf0, A0])
    pred = model(U, *popt)
    return float(pred), tuple(map(float, popt))


# ---------------------------------------------------------------------------
# Shared-c cross-LR LOO fit
# ---------------------------------------------------------------------------


def fit_shared_c_jointly(
    runs_data: list[tuple[RunSpec, pd.DataFrame]],
    metric: str,
    prefix_frac: float,
) -> tuple[float, dict[str, tuple[float, float]]]:
    """Fit shared c across multiple runs with per-run (L_inf, A).

    Returns (c_shared, {run.name: (L_inf, A)}).
    """
    # Build packed x, y with run-indexing
    packed_u = []
    packed_y = []
    packed_runs = []
    per_run_U: dict[str, float] = {}

    for spec, df in runs_data:
        u_full = compute_cumulative_lr(df)
        U = float(u_full.iloc[-1])
        per_run_U[spec.name] = U
        cutoff = prefix_frac * TOTAL_STEPS
        fit_lo = max(WARMUP_STEPS, cutoff // 2)
        mask = (df["_step"] >= fit_lo) & (df["_step"] <= cutoff) & df[metric].notna()
        sub = df.loc[mask, ["_step", metric]].copy()
        sub["u"] = u_full.loc[mask].to_numpy()
        for _, row in sub.iterrows():
            packed_u.append(float(row["u"]))
            packed_y.append(float(row[metric]))
            packed_runs.append(spec.name)

    run_names = list(dict.fromkeys(packed_runs))  # preserve order, unique
    idx_of = {name: i for i, name in enumerate(run_names)}
    packed_idx = np.array([idx_of[name] for name in packed_runs])
    packed_u = np.asarray(packed_u, dtype=float)
    packed_y = np.asarray(packed_y, dtype=float)
    U_arr = np.asarray([per_run_U[run_names[i]] for i in packed_idx], dtype=float)

    def model(_u, *params):
        # params = [c_shared, L_inf_0, A_0, L_inf_1, A_1, ...]
        c = params[0]
        L_infs = np.asarray(params[1::2])
        A_s = np.asarray(params[2::2])
        rem = np.maximum(U_arr - packed_u, 1e-12)
        return L_infs[packed_idx] + A_s[packed_idx] * np.power(rem, c)

    # Initial guess: per-run fit first, then average c
    per_run_seeds: dict[str, tuple[float, float, float]] = {}
    for spec, df in runs_data:
        try:
            _, (L_inf, A, c) = fit_cumlr_power(df, prefix_frac, metric)
            per_run_seeds[spec.name] = (L_inf, A, c)
        except Exception as e:  # pragma: no cover
            logger.warning("seed fit failed for %s / %s: %s", spec.name, metric, e)
            per_run_seeds[spec.name] = (float(packed_y[-1]), 1.0, 0.5)

    c0 = float(np.mean([s[2] for s in per_run_seeds.values()]))
    p0 = [c0]
    for name in run_names:
        L_inf, A, _ = per_run_seeds[name]
        p0.extend([L_inf, A])
    p0 = np.asarray(p0, dtype=float)

    popt = _fit_curve(model, packed_u, packed_y, p0=list(p0))
    c_shared = float(popt[0])
    per_run_params = {name: (float(popt[1 + 2 * i]), float(popt[1 + 2 * i + 1])) for i, name in enumerate(run_names)}
    return c_shared, per_run_params


def predict_held_out_with_shared_c(
    held_out: tuple[RunSpec, pd.DataFrame],
    fit_runs: list[tuple[RunSpec, pd.DataFrame]],
    metric: str,
    prefix_frac: float,
) -> float:
    """Two-step cross-LR prediction:

    1. Fit shared c across fit_runs on the full training window (> warmup).
    2. Fit held_out's own (L_inf, A) with c FROZEN, using only held_out's first
       prefix_frac fraction.
    3. Predict held_out's final by evaluating the model at u = U_heldout.
    """
    # Step 1: shared c on full fit_runs (use a generous prefix = 1.0 to get the best c estimate)
    c_shared, _ = fit_shared_c_jointly(fit_runs, metric, prefix_frac=1.0)

    # Step 2: fit held_out with c frozen, using its early prefix
    spec, df = held_out
    u_full = compute_cumulative_lr(df)
    U = float(u_full.iloc[-1])
    cutoff = prefix_frac * TOTAL_STEPS
    fit_lo = max(WARMUP_STEPS, cutoff // 2)
    mask = (df["_step"] >= fit_lo) & (df["_step"] <= cutoff) & df[metric].notna()
    sub = df.loc[mask, ["_step", metric]].copy()
    sub["u"] = u_full.loc[mask].to_numpy()
    if len(sub) < 5:
        raise RuntimeError(f"cross-LR held-out too few points for {spec.name}/{metric}")
    u = sub["u"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)

    def model_frozen_c(uu, L_inf, A):
        rem = np.maximum(U - uu, 1e-12)
        return L_inf + A * np.power(rem, c_shared)

    L_inf0 = float(y[-1]) * (0.95 if y[-1] > 0 else 1.05)
    A0 = float(y[0] - y[-1])
    popt = _fit_curve(model_frozen_c, u, y, p0=[L_inf0, A0])
    return float(model_frozen_c(U, *popt))


# ---------------------------------------------------------------------------
# Experiment harness
# ---------------------------------------------------------------------------


def run_self_prefix(
    runs_data: list[tuple[RunSpec, pd.DataFrame]],
    metrics: tuple[str, ...],
    prefixes: tuple[float, ...],
) -> pd.DataFrame:
    rows = []
    for spec, df in runs_data:
        for metric in metrics:
            target = evaluate_final(df, metric)
            for frac in prefixes:
                for method_name, fit_fn in [
                    ("B0_last_value", lambda d, f, m: (fit_last_value(d, f, m), ())),
                    ("B1_sqrt_t", fit_sqrt_t),
                    ("B2_cumlr_power", fit_cumlr_power),
                    ("B3_cumlr_c=1", lambda d, f, m: fit_cumlr_fixed_c(d, f, m, c_fixed=1.0)),
                    ("B3_cumlr_c=0.5", lambda d, f, m: fit_cumlr_fixed_c(d, f, m, c_fixed=0.5)),
                ]:
                    try:
                        pred, params = fit_fn(df, frac, metric)
                    except Exception as e:  # pragma: no cover
                        logger.warning("fit fail %s/%s/%s: %s", spec.name, metric, method_name, e)
                        pred = float("nan")
                        params = ()
                    rows.append(
                        {
                            "run": spec.name,
                            "lr_factor": spec.lr_factor,
                            "metric": metric,
                            "prefix": frac,
                            "method": method_name,
                            "predicted": pred,
                            "target": target,
                            "abs_err": abs(pred - target) if not np.isnan(pred) else float("nan"),
                            "params": ";".join(f"{p:.4g}" for p in params),
                        }
                    )
    return pd.DataFrame(rows)


def run_cross_lr_loo(
    runs_data: list[tuple[RunSpec, pd.DataFrame]],
    metrics: tuple[str, ...],
    prefixes: tuple[float, ...],
) -> pd.DataFrame:
    rows = []
    for held_i in range(len(runs_data)):
        held_out = runs_data[held_i]
        fit_runs = [r for j, r in enumerate(runs_data) if j != held_i]
        spec, df = held_out
        for metric in metrics:
            target = evaluate_final(df, metric)
            for frac in prefixes:
                try:
                    pred = predict_held_out_with_shared_c(held_out, fit_runs, metric, frac)
                    err = abs(pred - target)
                except Exception as e:  # pragma: no cover
                    logger.warning("cross-LR fail %s/%s@%.2f: %s", spec.name, metric, frac, e)
                    pred = float("nan")
                    err = float("nan")
                rows.append(
                    {
                        "held_out": spec.name,
                        "lr_factor": spec.lr_factor,
                        "metric": metric,
                        "prefix": frac,
                        "predicted": pred,
                        "target": target,
                        "abs_err": err,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def c_stability_report(runs_data, metrics, prefixes):
    """Per-run c across prefixes. Large std => functional form unstable."""
    rows = []
    for spec, df in runs_data:
        for metric in metrics:
            cs = []
            for frac in prefixes:
                try:
                    _, (_, _, c) = fit_cumlr_power(df, frac, metric)
                    cs.append(c)
                except Exception:  # pragma: no cover
                    cs.append(float("nan"))
            rows.append(
                {
                    "run": spec.name,
                    "metric": metric,
                    "c_values": ",".join(f"{c:.3f}" for c in cs),
                    "c_mean": float(np.nanmean(cs)),
                    "c_std": float(np.nanstd(cs)),
                }
            )
    return pd.DataFrame(rows)


def summarize_mae(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols)["abs_err"]
        .agg(["mean", "median", "max"])
        .rename(columns={"mean": "MAE", "median": "MedAE", "max": "MaxAE"})
        .reset_index()
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Loading %d runs from W&B...", len(RUNS_1E20))

    runs_data: list[tuple[RunSpec, pd.DataFrame]] = []
    for spec in RUNS_1E20:
        df = load_run(spec)
        df = smooth_train_loss(df)
        runs_data.append((spec, df))

    logger.info("Loaded %d runs (rows: %s)", len(runs_data), [len(d) for _, d in runs_data])

    # For train loss we fit on the smoothed series
    train_metric = "train/loss_smooth"
    eval_metric = "eval/paloma/c4_en/loss"

    # Wire smoothed metric name into the harness
    metrics_to_run = (train_metric, eval_metric)

    logger.info("=" * 60)
    logger.info("Self-prefix test")
    logger.info("=" * 60)
    sp_df = run_self_prefix(runs_data, metrics_to_run, PREFIX_FRACS)
    print()
    print("Self-prefix: all predictions")
    print(sp_df.to_string(index=False))

    print()
    print("Self-prefix: MAE by (method, metric, prefix)")
    print(summarize_mae(sp_df, ["method", "metric", "prefix"]).to_string(index=False))

    logger.info("=" * 60)
    logger.info("c-stability across prefixes (B2 only)")
    logger.info("=" * 60)
    cs_df = c_stability_report(runs_data, metrics_to_run, PREFIX_FRACS)
    print()
    print("c stability")
    print(cs_df.to_string(index=False))

    logger.info("=" * 60)
    logger.info("Cross-LR LOO (shared c across the other 2 runs)")
    logger.info("=" * 60)
    cx_df = run_cross_lr_loo(runs_data, metrics_to_run, PREFIX_FRACS)
    print()
    print("Cross-LR LOO: all predictions")
    print(cx_df.to_string(index=False))

    print()
    print("Cross-LR LOO: MAE by (metric, prefix)")
    print(summarize_mae(cx_df, ["metric", "prefix"]).to_string(index=False))

    # Noise floor summary
    print()
    print("Reference noise floor: ~0.005-0.010 abs loss (from original-vs-v2 rerun pairs).")
    print("Success criteria:")
    print("  - B2 MAE < 2x noise floor at prefix=30% for train/loss (self-prefix)")
    print("  - B2 beats B1 by >20% on train/loss")
    print("  - c_std < 0.1 per run per metric")
    print("  - Cross-LR LOO MAE < 3x noise floor on train/loss at prefix=30%")

    # Dump full tables
    out_dir = os.path.dirname(os.path.abspath(__file__))
    sp_df.to_csv(os.path.join(out_dir, "midtrain_loss_predictor_self_prefix.csv"), index=False)
    cx_df.to_csv(os.path.join(out_dir, "midtrain_loss_predictor_cross_lr.csv"), index=False)
    cs_df.to_csv(os.path.join(out_dir, "midtrain_loss_predictor_c_stability.csv"), index=False)
    logger.info("Wrote CSVs to %s", out_dir)


if __name__ == "__main__":
    main()
