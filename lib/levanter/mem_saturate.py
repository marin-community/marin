#!/usr/bin/env python3
"""
Memorization scaling law (saturating) fitter & visualizer.

Recommended model family:
1) Cloglog (no plateau):            M = 1 - exp( - k * epochs**a * seed**(-b) )
   -> log(-log(1 - M)) = log k + a*log(epochs) - b*log(seed)

2) Saturating with free M_max:      M = M_max * ( 1 - exp( - k * epochs**a * seed**(-b) ) )

Both are interpretable: a (>0) is elasticity to epochs (reuse), b (>0) is elasticity to seed size (breadth).
"""

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# SciPy is optional; only needed for the fully saturating fit with M_max
try:
    from scipy.optimize import least_squares
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ---------------------------
# Utilities
# ---------------------------

def parse_token_set(token_set_str: str) -> int:
    """
    Convert '1M' -> 1_000_000, '10M' -> 10_000_000, '100M' -> 100_000_000.
    """
    s = str(token_set_str).strip().upper().replace("TOKENS", "").replace("TOKEN", "")
    if s.endswith("M"):
        base = s[:-1]
        return int(base) * 1_000_000
    if s.endswith("K"):
        base = s[:-1]
        return int(base) * 1_000
    # Fall back: try int directly
    return int(float(s))


def safe_clip_m(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Clip M into (eps, 1 - eps) to avoid boundary issues for transforms.
    """
    return np.clip(M, eps, 1 - eps)


@dataclass
class CloglogFit:
    a: float
    b: float
    log_k: float   # log(k)
    r2_y: float    # R^2 in transformed space
    r2_M: float    # R^2 in original M space
    rmse_M: float
    mae_M: float


@dataclass
class PlateauFit:
    a: float
    b: float
    k: float
    M_max: float
    r2_M: float
    rmse_M: float
    mae_M: float
    success: bool
    message: str


@dataclass
class LogLinearFit:
    c1: float  # epochs coefficient
    c2: float  # seed coefficient
    c3: float  # intercept
    r2_log: float  # R^2 in log space
    r2_M: float    # R^2 in original M space
    rmse_M: float
    mae_M: float


def fit_cloglog(epochs: np.ndarray, seed_tokens: np.ndarray, M: np.ndarray) -> Tuple[CloglogFit, np.ndarray]:
    """
    Fit the cloglog-linearized model:
      y = log(-log(1 - M)) = log k + a*log(epochs) - b*log(seed)
    Return fit object and predictions in original space M_hat.
    """
    M_clip = safe_clip_m(M)
    y = np.log(-np.log(1.0 - M_clip))
    X = np.column_stack([np.log(epochs), np.log(seed_tokens)])
    lr = LinearRegression()
    lr.fit(X, y)
    # coeffs: [coef_epochs, coef_seed], intercept
    a = float(lr.coef_[0])
    coef_seed = float(lr.coef_[1])
    b = -coef_seed  # we report b > 0, but coef on log(seed) will be negative
    log_k = float(lr.intercept_)

    # predictions
    y_hat = lr.predict(X)
    M_hat = 1.0 - np.exp(-np.exp(y_hat))

    # metrics
    r2_y = r2_score(y, y_hat)
    r2_M = r2_score(M, M_hat)
    rmse_M = math.sqrt(mean_squared_error(M, M_hat))
    mae_M = mean_absolute_error(M, M_hat)

    return CloglogFit(a=a, b=b, log_k=log_k, r2_y=r2_y, r2_M=r2_M, rmse_M=rmse_M, mae_M=mae_M), M_hat


def _plateau_model(theta: np.ndarray, epochs: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """
    theta = [a, b, log_k, logit_Mmax]
    We fit in unconstrained params for stability:
      a = softplus(theta[0]), b = softplus(theta[1]), k = exp(theta[2]), M_max = sigmoid(theta[3])
    """
    a = np.log1p(np.exp(theta[0]))  # softplus for positivity
    b = np.log1p(np.exp(theta[1]))  # softplus for positivity
    k = np.exp(theta[2])            # positive
    M_max = 1.0 / (1.0 + np.exp(-theta[3]))  # (0,1)
    hazard = k * (epochs ** a) * (seed ** (-b))
    M_hat = M_max * (1.0 - np.exp(-hazard))
    return M_hat


def fit_plateau(epochs: np.ndarray,
                seed_tokens: np.ndarray,
                M: np.ndarray,
                init_from: Optional[CloglogFit] = None) -> Tuple[PlateauFit, np.ndarray]:
    """
    Fit the fully saturating model with free M_max using nonlinear least squares.
    Returns fit object and predictions M_hat.
    """
    if not SCIPY_AVAILABLE:
        return PlateauFit(a=np.nan, b=np.nan, k=np.nan, M_max=np.nan,
                          r2_M=np.nan, rmse_M=np.nan, mae_M=np.nan,
                          success=False, message="SciPy not available"), np.full_like(M, np.nan)

    M_clip = safe_clip_m(M)
    # Initialize from cloglog if provided; otherwise use generic guesses
    if init_from is not None:
        a0 = max(1e-3, init_from.a)
        b0 = max(1e-3, init_from.b)
        k0 = max(1e-12, math.exp(init_from.log_k))
    else:
        a0, b0, k0 = 1.0, 1.0, 1e-3
    Mmax0 = min(0.99, 1.05 * float(np.max(M_clip)))

    # Inverse transforms for initial theta
    # theta0 = [softplus^-1(a0), softplus^-1(b0), log(k0), logit(Mmax0)]
    def inv_softplus(x):
        # approximate inverse of softplus via log(exp(x)-1); guard very small x
        return np.log(np.expm1(max(x, 1e-12)))
    def logit(p):
        p = np.clip(p, 1e-8, 1-1e-8)
        return np.log(p/(1-p))

    theta0 = np.array([inv_softplus(a0), inv_softplus(b0), np.log(k0), logit(Mmax0)], dtype=float)

    def residuals(theta):
        M_hat = _plateau_model(theta, epochs, seed_tokens)
        return (M_hat - M_clip)

    res = least_squares(residuals, theta0, method="trf", max_nfev=10000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    M_hat = _plateau_model(res.x, epochs, seed_tokens)

    # Extract params
    theta = res.x
    a = float(np.log1p(np.exp(theta[0])))
    b = float(np.log1p(np.exp(theta[1])))
    k = float(np.exp(theta[2]))
    M_max = float(1.0 / (1.0 + np.exp(-theta[3])))

    r2_M = r2_score(M_clip, M_hat)
    rmse_M = math.sqrt(mean_squared_error(M_clip, M_hat))
    mae_M = mean_absolute_error(M_clip, M_hat)

    return PlateauFit(a=a, b=b, k=k, M_max=M_max,
                      r2_M=r2_M, rmse_M=rmse_M, mae_M=mae_M,
                      success=res.success, message=str(res.message)), M_hat


def fit_loglinear(epochs: np.ndarray, seed_tokens: np.ndarray, M: np.ndarray) -> Tuple[LogLinearFit, np.ndarray]:
    """
    Fit the standard log-linear model (like mem_scaling.py):
      log(M) = c1*log(epochs) + c2*log(seed) + c3
    Return fit object and predictions in original space M_hat.
    """
    M_clip = safe_clip_m(M)
    y = np.log(M_clip)
    X = np.column_stack([np.log(epochs), np.log(seed_tokens)])
    lr = LinearRegression()
    lr.fit(X, y)

    # coeffs: [coef_epochs, coef_seed], intercept
    c1 = float(lr.coef_[0])
    c2 = float(lr.coef_[1])
    c3 = float(lr.intercept_)

    # predictions
    y_hat = lr.predict(X)
    M_hat = np.exp(y_hat)

    # metrics
    r2_log = r2_score(y, y_hat)
    r2_M = r2_score(M, M_hat)
    rmse_M = math.sqrt(mean_squared_error(M, M_hat))
    mae_M = mean_absolute_error(M, M_hat)

    return LogLinearFit(c1=c1, c2=c2, c3=c3, r2_log=r2_log, r2_M=r2_M, rmse_M=rmse_M, mae_M=mae_M), M_hat


def load_and_prepare(csv_path: str,
                     finished_only: bool = True,
                     drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Load the runs CSV, parse seed tokens, filter, and return a tidy dataframe.
    """
    df = pd.read_csv(csv_path)

    if finished_only and "state" in df.columns:
        before = len(df)
        df = df[df["state"].astype(str).str.lower() == "finished"].copy()
        print(f"[info] Filtered to finished runs: {len(df)}/{before} rows")

    if "duplicate" in df.columns and drop_duplicates:
        before = len(df)
        df = df[df["duplicate"].astype(str).str.lower() == "no"].copy()
        print(f"[info] Dropped duplicates: {len(df)}/{before} rows")

    # Parse token set -> integer tokens
    if "seed_set_tokens" not in df.columns:
        if "token_set" in df.columns:
            df["seed_set_tokens"] = df["token_set"].apply(parse_token_set)
        elif "seed_tokens" in df.columns:
            df["seed_set_tokens"] = df["seed_tokens"]
        else:
            raise ValueError("CSV must contain 'token_set' or 'seed_tokens'")

    # Standard column names
    if "epochs" not in df.columns or "final_mean_pz" not in df.columns:
        raise ValueError("CSV must contain 'epochs' and 'final_mean_pz'")

    # Ensure numeric
    df["epochs"] = pd.to_numeric(df["epochs"])
    df["seed_set_tokens"] = pd.to_numeric(df["seed_set_tokens"])
    df["final_mean_pz"] = pd.to_numeric(df["final_mean_pz"])
    # Total tokens (useful for plots or slices)
    df["total_tokens"] = df["epochs"] * df["seed_set_tokens"]

    # Replace exact zeros with tiny epsilon for stability
    zero_mask = (df["final_mean_pz"] <= 0)
    if zero_mask.any():
        print(f"[warn] Found {zero_mask.sum()} rows with P(z) <= 0; replacing with epsilon")
        df.loc[zero_mask, "final_mean_pz"] = 1e-12

    return df


def add_predictions(df: pd.DataFrame, M_hat: np.ndarray, label: str) -> None:
    df[f"pred_{label}"] = M_hat
    df[f"residual_{label}"] = df["final_mean_pz"].values - M_hat


def plot_actual_vs_predicted(df: pd.DataFrame, ycol: str, yhat_col: str, out_path: str, model_name: str = "") -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(df[ycol], df[yhat_col], s=60, alpha=0.8)
    lo = float(min(df[ycol].min(), df[yhat_col].min()))
    hi = float(max(df[ycol].max(), df[yhat_col].max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
    plt.xlabel("Actual M")
    plt.ylabel("Predicted M")
    title = f"Actual vs Predicted"
    if model_name:
        title += f": {model_name}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[save] {out_path}")


def plot_residuals_vs_epochs(df: pd.DataFrame, residual_col: str, out_path: str, model_name: str = "") -> None:
    plt.figure(figsize=(7, 6))
    for token_set in sorted(df["token_set"].unique(), key=lambda x: int(str(x).rstrip("M"))):
        subset = df[df["token_set"] == token_set]
        plt.scatter(subset["epochs"], subset[residual_col], s=60, alpha=0.8, label=f"{token_set}")
    plt.axhline(0.0, linestyle="--", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Epochs (log)")
    plt.ylabel("Residual (Actual - Predicted)")
    title = "Residuals vs Epochs"
    if model_name:
        title += f": {model_name}"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[save] {out_path}")


def plot_scaling_curves(df: pd.DataFrame, ycol: str, yhat_col: str, out_path: str, model_name: str = "") -> None:
    plt.figure(figsize=(8, 6))
    for token_set in sorted(df["token_set"].unique(), key=lambda x: int(str(x).rstrip("M"))):
        subset = df[df["token_set"] == token_set].copy()
        subset.sort_values("epochs", inplace=True)
        plt.plot(subset["epochs"], subset[ycol], marker="o", linewidth=2, label=f"{token_set} (actual)")
        plt.plot(subset["epochs"], subset[yhat_col], linestyle="--", linewidth=2, label=f"{token_set} (fit)")
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Epochs (log)")
    plt.ylabel("M")
    title = "Scaling Curves by Seed Set"
    if model_name:
        title += f": {model_name}"
    plt.title(title)
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[save] {out_path}")


def plot_heatmap(df: pd.DataFrame, value_col: str, out_path: str) -> None:
    # Pivot to (token_set x epochs)
    pv = df.pivot_table(values=value_col, index="token_set", columns="epochs")
    pv = pv.reindex(sorted(pv.index, key=lambda x: int(str(x).rstrip("M"))))
    plt.figure(figsize=(10, 5))
    plt.imshow(pv.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label=value_col)
    plt.xticks(ticks=range(len(pv.columns)), labels=[str(int(e)) for e in pv.columns], rotation=45, ha="right")
    plt.yticks(ticks=range(len(pv.index)), labels=list(pv.index))
    plt.xlabel("Epochs")
    plt.ylabel("Token Set")
    plt.title(f"Heatmap: {value_col}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[save] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit saturating memorization scaling laws.")
    parser.add_argument("--csv", default="comma_150m_runs.csv", help="Path to runs CSV")
    parser.add_argument("--model", choices=["cloglog", "saturating", "loglinear", "all"], default="all",
                        help="Fit with cloglog-linear, saturating (with M_max), standard log-linear, or all models")
    parser.add_argument("--include-crashed", action="store_true",
                        help="Include crashed runs (default: only finished)")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Do not drop duplicate rows (default: drop duplicates)")
    parser.add_argument("--out-prefix", default="mem_scaling",
                        help="Prefix for saved CSV and figures")
    args = parser.parse_args()

    df = load_and_prepare(args.csv, finished_only=not args.include_crashed, drop_duplicates=not args.no_dedupe)

    # Core variables
    epochs = df["epochs"].to_numpy(dtype=float)
    seed = df["seed_set_tokens"].to_numpy(dtype=float)
    M = df["final_mean_pz"].to_numpy(dtype=float)

    # Determine which models to fit
    fit_models = [args.model] if args.model != "all" else ["loglinear", "cloglog", "saturating"]

    # Always fit cloglog first (used to seed plateau fit)
    cloglog_fit, M_hat_clog = fit_cloglog(epochs, seed, M)
    add_predictions(df, M_hat_clog, "cloglog")

    if "cloglog" in fit_models or args.model == "all":
        print("\n=== Cloglog fit (no plateau) ===")
        print(f"Model: M = 1 - exp(-k * epochs^a * seed^(-b))")
        print(f"a (epochs elasticity):       {cloglog_fit.a: .6f}")
        print(f"b (seed elasticity):         {cloglog_fit.b: .6f}")
        print(f"log k:                       {cloglog_fit.log_k: .6f}")
        print(f"R^2 (transformed y):         {cloglog_fit.r2_y: .4f}")
        print(f"R^2 (M space):               {cloglog_fit.r2_M: .4f}")
        print(f"RMSE (M space):              {cloglog_fit.rmse_M: .6f}")
        print(f"MAE  (M space):              {cloglog_fit.mae_M: .6f}")

    # Fit log-linear model (standard scaling law)
    if "loglinear" in fit_models or args.model == "all":
        loglinear_fit, M_hat_loglinear = fit_loglinear(epochs, seed, M)
        add_predictions(df, M_hat_loglinear, "loglinear")

        print("\n=== Log-linear fit (standard scaling law) ===")
        print(f"Model: log(M) = c1*log(epochs) + c2*log(seed) + c3")
        print(f"c1 (epochs coefficient):     {loglinear_fit.c1: .6f}")
        print(f"c2 (seed coefficient):       {loglinear_fit.c2: .6f}")
        print(f"c3 (intercept):              {loglinear_fit.c3: .6f}")
        print(f"R^2 (log space):             {loglinear_fit.r2_log: .4f}")
        print(f"R^2 (M space):               {loglinear_fit.r2_M: .4f}")
        print(f"RMSE (M space):              {loglinear_fit.rmse_M: .6f}")
        print(f"MAE  (M space):              {loglinear_fit.mae_M: .6f}")
        print(f"\nInterpretation:")
        print(f"  - Each 10x increase in epochs multiplies M by {10**loglinear_fit.c1:.2f}x")
        print(f"  - Each 10x increase in seed multiplies M by {10**loglinear_fit.c2:.2f}x")

    # Fit saturating model
    if "saturating" in fit_models or args.model == "all":
        plateau_fit, M_hat_plateau = fit_plateau(epochs, seed, M, init_from=cloglog_fit)
        add_predictions(df, M_hat_plateau, "saturating")

        print("\n=== Saturating fit (free M_max) ===")
        print(f"Model: M = M_max * (1 - exp(-k * epochs^a * seed^(-b)))")
        print(f"success:                     {plateau_fit.success}  ({plateau_fit.message})")
        print(f"a (epochs elasticity):       {plateau_fit.a: .6f}")
        print(f"b (seed elasticity):         {plateau_fit.b: .6f}")
        print(f"k (scale):                   {plateau_fit.k: .6e}")
        print(f"M_max (plateau):             {plateau_fit.M_max: .6f}")
        print(f"R^2 (M space):               {plateau_fit.r2_M: .4f}")
        print(f"RMSE (M space):              {plateau_fit.rmse_M: .6f}")
        print(f"MAE  (M space):              {plateau_fit.mae_M: .6f}")

    # Save per-row predictions
    out_csv = f"{args.out_prefix}_{args.model}_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[save] wrote {out_csv}")

    # Generate visualizations for each model
    for model_label in fit_models:
        if model_label == "all":
            continue

        model_display_name = {
            "cloglog": "Cloglog",
            "saturating": "Saturating",
            "loglinear": "Log-linear"
        }[model_label]

        print(f"\n[viz] Generating plots for {model_display_name} model...")

        # 1) Actual vs Predicted
        plot_actual_vs_predicted(df, "final_mean_pz", f"pred_{model_label}",
                                f"{args.out_prefix}_{model_label}_actual_vs_pred.png",
                                model_name=model_display_name)
        # 2) Residuals vs epochs
        plot_residuals_vs_epochs(df, f"residual_{model_label}",
                                f"{args.out_prefix}_{model_label}_residuals_vs_epochs.png",
                                model_name=model_display_name)
        # 3) Scaling curves by seed set
        plot_scaling_curves(df, "final_mean_pz", f"pred_{model_label}",
                           f"{args.out_prefix}_{model_label}_scaling_curves.png",
                           model_name=model_display_name)
        # 4) Heatmap of predicted M (only once per run)
        if model_label == fit_models[0]:
            plot_heatmap(df, "final_mean_pz", f"{args.out_prefix}_heatmap_actual.png")
        plot_heatmap(df, f"pred_{model_label}", f"{args.out_prefix}_{model_label}_heatmap_pred.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
