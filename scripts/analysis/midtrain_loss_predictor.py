# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-1 midtraining loss predictor validation.

Given completed midtraining runs with different LR factors, test whether a
schedule-aware power-law fit in cumulative-LR space can predict the final loss
from an early prefix.

Functional form:
    L(x) = L_inf + A * x ** c
where u = cumulative learning_rate up to step t, U = total cumulative LR at end.
The normalized remaining-LR coordinate is x = (U - u) / (U - u_fit_start),
so x=1 at the start of the fit window and x=0 at the final training step.

Baselines compared:
    B0: last-value (y at prefix_end)
    B1: a + b / sqrt(t)  (schedule-unaware)
    B2: profiled/validated c in L_inf + A*x^c (schedule-aware; the proposal)
    B2r: same as B2, with a weak log-c prior centered at c=1
    B3: fixed-c versions of the same schedule-aware model

Tests:
    1. Self-prefix: fit on first X% of a run, predict its own final.
    2. Cross-LR LOO: profile a shared c on 2 of 3 runs, predict the third.

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


@dataclass(frozen=True)
class CProfileSelection:
    c: float
    validation_rmse: float
    objective: float
    validation_mode: str
    c_at_grid_edge: bool


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
C_PROFILE_GRID = np.geomspace(0.1, 10.0, 241)
C_PRIOR = 1.0
C_LOG_PRIOR_STRENGTH = 0.25


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
# Baseline B2/B3: schedule-aware L(x) = L_inf + A * x^c
# ---------------------------------------------------------------------------


def normalized_remaining_lr(u: np.ndarray, U: float, fit_start_u: float) -> np.ndarray:
    """Map cumulative LR to x=(U-u)/(U-u_fit_start), where final training is x=0."""
    denom = max(U - fit_start_u, 1e-12)
    return np.maximum((U - u) / denom, 0.0)


def cumlr_fit_data(df: pd.DataFrame, prefix_frac: float, metric: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return normalized remaining-LR x, metric y, total U, and fit-window start u."""
    u_full = compute_cumulative_lr(df)
    U = float(u_full.iloc[-1])
    cutoff = prefix_frac * TOTAL_STEPS
    fit_lo = max(WARMUP_STEPS, cutoff // 2)
    mask = (df["_step"] >= fit_lo) & (df["_step"] <= cutoff) & df[metric].notna()
    sub = df.loc[mask, ["_step", metric]].copy()
    sub["u"] = u_full.loc[mask].to_numpy()
    if len(sub) < 4:
        raise RuntimeError(f"B2 too few points: {len(sub)} at prefix {prefix_frac} for {metric!r}")
    u = sub["u"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)
    fit_start_u = float(u[0])
    x = normalized_remaining_lr(u, U, fit_start_u)
    return x, y, U, fit_start_u


def solve_power_for_c(x: np.ndarray, y: np.ndarray, c: float) -> tuple[float, float, float]:
    """Least-squares fit of L_inf and A for a fixed exponent c."""
    basis = np.power(np.maximum(x, 0.0), c)
    design = np.column_stack([np.ones_like(basis), basis])
    L_inf, A = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = design @ np.asarray([L_inf, A]) - y
    mse = float(np.mean(np.square(residual)))
    return float(L_inf), float(A), mse


def power_prediction(x: np.ndarray, L_inf: float, A: float, c: float) -> np.ndarray:
    return L_inf + A * np.power(np.maximum(x, 0.0), c)


def _split_prefix_for_validation(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Chronological prefix split: fit early prefix points, validate on later prefix points."""
    n = len(y)
    if n < 8:
        return x, y, x, y, "in_sample_too_few_points"

    val_count = max(3, round(n * 0.25))
    if n - val_count < 5:
        val_count = max(1, n - 5)
    if val_count <= 0:
        return x, y, x, y, "in_sample_too_few_points"

    split = n - val_count
    return x[:split], y[:split], x[split:], y[split:], "tail_holdout"


def c_profile_records(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prior_c: float | None = None,
    prior_strength: float = C_LOG_PRIOR_STRENGTH,
    c_grid: np.ndarray = C_PROFILE_GRID,
) -> pd.DataFrame:
    """Profile c by solving the linear subproblem for every c on a grid.

    The optional prior is expressed as a penalty on log(c/prior_c), scaled by
    the observed validation variance so the weight is dimensionless.
    """
    x_fit, y_fit, x_val, y_val, validation_mode = _split_prefix_for_validation(x, y)
    val_scale = max(float(np.var(y_val)), 1e-8)
    rows = []
    for c in c_grid:
        L_inf, A, fit_mse = solve_power_for_c(x_fit, y_fit, float(c))
        val_pred = power_prediction(x_val, L_inf, A, float(c))
        val_mse = float(np.mean(np.square(val_pred - y_val)))
        penalty = 0.0
        if prior_c is not None:
            penalty = prior_strength * val_scale * float(np.log(float(c) / prior_c) ** 2)
        rows.append(
            {
                "c": float(c),
                "fit_mse": fit_mse,
                "validation_mse": val_mse,
                "validation_rmse": float(np.sqrt(val_mse)),
                "penalty": penalty,
                "objective": val_mse + penalty,
                "validation_mode": validation_mode,
            }
        )
    return pd.DataFrame(rows)


def select_profiled_c(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prior_c: float | None = None,
    prior_strength: float = C_LOG_PRIOR_STRENGTH,
    c_grid: np.ndarray = C_PROFILE_GRID,
) -> CProfileSelection:
    profile = c_profile_records(x, y, prior_c=prior_c, prior_strength=prior_strength, c_grid=c_grid)
    best_idx = int(profile["objective"].idxmin())
    row = profile.loc[best_idx]
    c = float(row["c"])
    return CProfileSelection(
        c=c,
        validation_rmse=float(row["validation_rmse"]),
        objective=float(row["objective"]),
        validation_mode=str(row["validation_mode"]),
        c_at_grid_edge=best_idx in {0, len(profile) - 1},
    )


def fit_cumlr_power(df: pd.DataFrame, prefix_frac: float, metric: str) -> tuple[float, tuple]:
    """Fit B2 by profiling c on normalized remaining LR and validating on the prefix tail.

    For a fixed c, the model is linear in (L_inf, A), so the fit uses exact
    least squares instead of bounded nonlinear optimization.
    """
    x, y, _, fit_start_u = cumlr_fit_data(df, prefix_frac, metric)
    selection = select_profiled_c(x, y)
    L_inf, A, _ = solve_power_for_c(x, y, selection.c)
    pred = L_inf  # x_final=0
    return pred, (L_inf, A, selection.c, fit_start_u, selection.validation_rmse)


def fit_cumlr_regularized_c(df: pd.DataFrame, prefix_frac: float, metric: str) -> tuple[float, tuple]:
    """Fit B2r: profiled c with a weak log-c prior centered at c=1."""
    x, y, _, fit_start_u = cumlr_fit_data(df, prefix_frac, metric)
    selection = select_profiled_c(x, y, prior_c=C_PRIOR)
    L_inf, A, _ = solve_power_for_c(x, y, selection.c)
    pred = L_inf
    return pred, (L_inf, A, selection.c, fit_start_u, selection.validation_rmse)


def fit_cumlr_fixed_c(df: pd.DataFrame, prefix_frac: float, metric: str, c_fixed: float = 1.0) -> tuple[float, tuple]:
    """Fit B3 with c fixed a priori. This is linear least squares in normalized x."""
    x, y, _, fit_start_u = cumlr_fit_data(df, prefix_frac, metric)
    L_inf, A, _ = solve_power_for_c(x, y, c_fixed)
    pred = L_inf
    return pred, (L_inf, A, c_fixed, fit_start_u)


# ---------------------------------------------------------------------------
# Shared-c cross-LR LOO fit
# ---------------------------------------------------------------------------


def fit_shared_c_jointly(
    runs_data: list[tuple[RunSpec, pd.DataFrame]],
    metric: str,
    prefix_frac: float,
) -> tuple[float, dict[str, tuple[float, float]]]:
    """Profile a shared c across multiple runs with per-run linear (L_inf, A).

    Returns (c_shared, {run.name: (L_inf, A)}).
    """
    prepared: list[tuple[str, np.ndarray, np.ndarray]] = []
    for spec, df in runs_data:
        x, y, _, _ = cumlr_fit_data(df, prefix_frac, metric)
        prepared.append((spec.name, x, y))

    best_c = float("nan")
    best_mse = float("inf")
    best_params: dict[str, tuple[float, float]] = {}
    for c in C_PROFILE_GRID:
        params: dict[str, tuple[float, float]] = {}
        weighted_sse = 0.0
        n_total = 0
        for name, x, y in prepared:
            L_inf, A, mse = solve_power_for_c(x, y, float(c))
            params[name] = (L_inf, A)
            weighted_sse += mse * len(y)
            n_total += len(y)
        mse = weighted_sse / max(n_total, 1)
        if mse < best_mse:
            best_mse = mse
            best_c = float(c)
            best_params = params
    return best_c, best_params


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
    _, df = held_out
    x, y, _, _ = cumlr_fit_data(df, prefix_frac, metric)
    L_inf, _, _ = solve_power_for_c(x, y, c_shared)
    return L_inf


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
                    ("B2_profiled_c", fit_cumlr_power),
                    ("B2r_profiled_c_logprior1", fit_cumlr_regularized_c),
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
                for method_name in ("shared_profiled_c", "fixed_c=1", "fixed_c=0.5"):
                    try:
                        if method_name == "shared_profiled_c":
                            pred = predict_held_out_with_shared_c(held_out, fit_runs, metric, frac)
                        elif method_name == "fixed_c=1":
                            pred = fit_cumlr_fixed_c(df, frac, metric, c_fixed=1.0)[0]
                        else:
                            pred = fit_cumlr_fixed_c(df, frac, metric, c_fixed=0.5)[0]
                        err = abs(pred - target)
                    except Exception as e:  # pragma: no cover
                        logger.warning("cross-LR fail %s/%s/%s@%.2f: %s", spec.name, metric, method_name, frac, e)
                        pred = float("nan")
                        err = float("nan")
                    rows.append(
                        {
                            "held_out": spec.name,
                            "lr_factor": spec.lr_factor,
                            "metric": metric,
                            "prefix": frac,
                            "method": method_name,
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
    """Per-run selected c across prefixes. Large std => functional form unstable."""
    rows = []
    for spec, df in runs_data:
        for metric in metrics:
            for method_name, prior_c in [("B2_profiled_c", None), ("B2r_profiled_c_logprior1", C_PRIOR)]:
                cs = []
                validation_rmses = []
                edge_hits = []
                for frac in prefixes:
                    try:
                        x, y, _, _ = cumlr_fit_data(df, frac, metric)
                        selection = select_profiled_c(x, y, prior_c=prior_c)
                        cs.append(selection.c)
                        validation_rmses.append(selection.validation_rmse)
                        edge_hits.append(selection.c_at_grid_edge)
                    except Exception:  # pragma: no cover
                        cs.append(float("nan"))
                        validation_rmses.append(float("nan"))
                        edge_hits.append(False)
                rows.append(
                    {
                        "run": spec.name,
                        "metric": metric,
                        "method": method_name,
                        "c_values": ",".join(f"{c:.3f}" for c in cs),
                        "validation_rmse_values": ",".join(f"{v:.4g}" for v in validation_rmses),
                        "edge_hits": ",".join(str(v) for v in edge_hits),
                        "c_mean": float(np.nanmean(cs)),
                        "c_std": float(np.nanstd(cs)),
                    }
                )
    return pd.DataFrame(rows)


def c_profile_report(runs_data, metrics, prefixes):
    rows = []
    for spec, df in runs_data:
        for metric in metrics:
            for frac in prefixes:
                x, y, _, _ = cumlr_fit_data(df, frac, metric)
                for method_name, prior_c in [("profiled_c", None), ("profiled_c_logprior1", C_PRIOR)]:
                    profile = c_profile_records(x, y, prior_c=prior_c)
                    profile = profile.assign(
                        run=spec.name,
                        lr_factor=spec.lr_factor,
                        metric=metric,
                        prefix=frac,
                        method=method_name,
                    )
                    rows.append(profile)
    return pd.concat(rows, ignore_index=True)


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
    logger.info("c-stability across prefixes")
    logger.info("=" * 60)
    cs_df = c_stability_report(runs_data, metrics_to_run, PREFIX_FRACS)
    print()
    print("c stability")
    print(cs_df.to_string(index=False))
    profile_df = c_profile_report(runs_data, metrics_to_run, PREFIX_FRACS)

    logger.info("=" * 60)
    logger.info("Cross-LR LOO (profiled shared c across the other 2 runs)")
    logger.info("=" * 60)
    cx_df = run_cross_lr_loo(runs_data, metrics_to_run, PREFIX_FRACS)
    print()
    print("Cross-LR LOO: all predictions")
    print(cx_df.to_string(index=False))

    print()
    print("Cross-LR LOO: MAE by (method, metric, prefix)")
    print(summarize_mae(cx_df, ["method", "metric", "prefix"]).to_string(index=False))

    # Noise floor summary
    print()
    print("Reference noise floor: ~0.005-0.010 abs loss (from original-vs-v2 rerun pairs).")
    print("Interpretation checks:")
    print("  - B2 no longer uses bounded nonlinear optimization; c is profiled on a normalized x axis.")
    print("  - Edge hits in c_stability mean the grid domain is too narrow or the data cannot identify c.")
    print("  - Prefer B3 c=1 when profiled-c validation is flat or unstable.")

    # Dump full tables
    out_dir = os.path.dirname(os.path.abspath(__file__))
    sp_df.to_csv(os.path.join(out_dir, "midtrain_loss_predictor_self_prefix.csv"), index=False)
    cx_df.to_csv(os.path.join(out_dir, "midtrain_loss_predictor_cross_lr.csv"), index=False)
    cs_df.to_csv(os.path.join(out_dir, "midtrain_loss_predictor_c_stability.csv"), index=False)
    profile_df.to_csv(os.path.join(out_dir, "midtrain_loss_predictor_c_profiles.csv"), index=False)
    logger.info("Wrote CSVs to %s", out_dir)


if __name__ == "__main__":
    main()
