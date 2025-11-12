#!/usr/bin/env python3
"""
Fit BOTH memorization models and generate clearly titled plots:

(1) Power-law (unbounded in M):
    log M = c1*log(epochs) + c2*log(seed) + c3

(2) Saturating hazard model via cloglog link (plateau=1):
    M = 1 - exp(-k * epochs^a * seed^{-b})
    => log(-log(1 - M)) = log k + a*log(epochs) - b*log(seed)

Optionally, you can also fit the fully saturating plateau model:
    M = M_max * (1 - exp(-k * epochs^a * seed^{-b}))  [--also-plateau]
"""

import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from scipy.optimize import least_squares
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ---------------------------
# Utilities
# ---------------------------

def parse_token_set(token_set_str: str) -> int:
    s = str(token_set_str).strip().upper().replace("TOKENS", "").replace("TOKEN", "")
    if s.endswith("M"):
        return int(s[:-1]) * 1_000_000
    if s.endswith("K"):
        return int(s[:-1]) * 1_000
    return int(float(s))


def safe_clip_m(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(M, eps, 1 - eps)


# ---------------------------
# Models / Fits
# ---------------------------

@dataclass
class PowerlawFit:
    c1: float  # coef on log(epochs)
    c2: float  # coef on log(seed)
    c3: float  # intercept
    r2_y: float
    r2_M: float
    rmse_M: float
    mae_M: float


def fit_powerlaw(epochs: np.ndarray, seed_tokens: np.ndarray, M: np.ndarray) -> Tuple[PowerlawFit, np.ndarray]:
    """
    Original (unbounded) power-law in log-space:
      log(M) = c1*log(epochs) + c2*log(seed) + c3
    Predictions are exp of the linear predictor and may exceed 1.
    """
    M_eps = np.clip(M, 1e-12, None)  # avoid -inf
    y = np.log(M_eps)
    X = np.column_stack([np.log(epochs), np.log(seed_tokens)])
    lr = LinearRegression()
    lr.fit(X, y)
    y_hat = lr.predict(X)
    M_hat = np.exp(y_hat)

    r2_y = r2_score(y, y_hat)
    r2_M = r2_score(M, M_hat)
    rmse_M = math.sqrt(mean_squared_error(M, M_hat))
    mae_M = mean_absolute_error(M, M_hat)

    return PowerlawFit(c1=float(lr.coef_[0]), c2=float(lr.coef_[1]), c3=float(lr.intercept_),
                       r2_y=r2_y, r2_M=r2_M, rmse_M=rmse_M, mae_M=mae_M), M_hat


@dataclass
class CloglogFit:
    a: float
    b: float
    log_k: float   # log(k)
    r2_y: float    # R^2 in transformed space
    r2_M: float    # R^2 in original M space
    rmse_M: float
    mae_M: float


def fit_cloglog(epochs: np.ndarray, seed_tokens: np.ndarray, M: np.ndarray) -> Tuple[CloglogFit, np.ndarray]:
    """
    Cloglog-linearized model:
      y = log(-log(1 - M)) = log k + a*log(epochs) - b*log(seed)
    """
    M_clip = safe_clip_m(M)
    y = np.log(-np.log(1.0 - M_clip))
    X = np.column_stack([np.log(epochs), np.log(seed_tokens)])
    lr = LinearRegression()
    lr.fit(X, y)
    a = float(lr.coef_[0])
    b = -float(lr.coef_[1])        # report b > 0
    log_k = float(lr.intercept_)

    y_hat = lr.predict(X)
    M_hat = 1.0 - np.exp(-np.exp(y_hat))

    r2_y = r2_score(y, y_hat)
    r2_M = r2_score(M, M_hat)
    rmse_M = math.sqrt(mean_squared_error(M, M_hat))
    mae_M = mean_absolute_error(M, M_hat)

    return CloglogFit(a=a, b=b, log_k=log_k, r2_y=r2_y, r2_M=r2_M, rmse_M=rmse_M, mae_M=mae_M), M_hat


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


def _plateau_model(theta: np.ndarray, epochs: np.ndarray, seed: np.ndarray) -> np.ndarray:
    a = np.log1p(np.exp(theta[0]))  # softplus
    b = np.log1p(np.exp(theta[1]))
    k = np.exp(theta[2])
    M_max = 1.0 / (1.0 + np.exp(-theta[3]))
    hazard = k * (epochs ** a) * (seed ** (-b))
    return M_max * (1.0 - np.exp(-hazard))


def fit_plateau(epochs: np.ndarray, seed_tokens: np.ndarray, M: np.ndarray, init_from: Optional[CloglogFit]) -> Tuple[PlateauFit, np.ndarray]:
    if not SCIPY_AVAILABLE:
        return PlateauFit(a=np.nan, b=np.nan, k=np.nan, M_max=np.nan, r2_M=np.nan,
                          rmse_M=np.nan, mae_M=np.nan, success=False, message="SciPy not available"), np.full_like(M, np.nan)

    M_clip = safe_clip_m(M)

    if init_from is not None:
        a0 = max(1e-3, init_from.a)
        b0 = max(1e-3, init_from.b)
        k0 = max(1e-12, math.exp(init_from.log_k))
    else:
        a0, b0, k0 = 1.0, 1.0, 1e-3
    Mmax0 = min(0.99, 1.05 * float(np.max(M_clip)))

    def inv_softplus(x):
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

    theta = res.x
    a = float(np.log1p(np.exp(theta[0])))
    b = float(np.log1p(np.exp(theta[1])))
    k = float(np.exp(theta[2]))
    M_max = float(1.0 / (1.0 + np.exp(-theta[3])))

    r2_M = r2_score(M_clip, M_hat)
    rmse_M = math.sqrt(mean_squared_error(M_clip, M_hat))
    mae_M = mean_absolute_error(M_clip, M_hat)

    return PlateauFit(a=a, b=b, k=k, M_max=M_max, r2_M=r2_M, rmse_M=rmse_M, mae_M=mae_M,
                      success=res.success, message=str(res.message)), M_hat


# ---------------------------
# Plotting
# ---------------------------

def plot_actual_vs_predicted(df: pd.DataFrame, ycol: str, yhat_col: str, out_path: str, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(df[ycol], df[yhat_col], s=60, alpha=0.8)
    lo = float(min(df[ycol].min(), df[yhat_col].min()))
    hi = float(max(df[ycol].max(), df[yhat_col].max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
    plt.xlabel("Actual M")
    plt.ylabel("Predicted M")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[save] {out_path}")


def plot_residuals_vs_epochs(df: pd.DataFrame, residual_col: str, out_path: str, title: str) -> None:
    plt.figure(figsize=(7, 6))
    for token_set in sorted(df["token_set"].unique(), key=lambda x: int(str(x).rstrip("M"))):
        subset = df[df["token_set"] == token_set]
        plt.scatter(subset["epochs"], subset[residual_col], s=60, alpha=0.8, label=f"{token_set}")
    plt.axhline(0.0, linestyle="--", linewidth=2)
    plt.xscale("log")
    plt.xlabel("Epochs (log)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[save] {out_path}")


def plot_scaling_curves(df: pd.DataFrame, ycol: str, yhat_col: str, out_path: str, title: str) -> None:
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
    plt.title(title)
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[save] {out_path}")


def plot_heatmap(df: pd.DataFrame, value_col: str, out_path: str) -> None:
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


# ---------------------------
# Data loading
# ---------------------------

def load_and_prepare(csv_path: str,
                     finished_only: bool = True,
                     drop_duplicates: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if finished_only and "state" in df.columns:
        before = len(df)
        df = df[df["state"].astype(str).str.lower() == "finished"].copy()
        print(f"[info] Filtered to finished runs: {len(df)}/{before} rows")

    if "duplicate" in df.columns and drop_duplicates:
        before = len(df)
        df = df[df["duplicate"].astype(str).str.lower() == "no"].copy()
        print(f"[info] Dropped duplicates: {len(df)}/{before} rows")

    if "seed_set_tokens" not in df.columns:
        if "token_set" in df.columns:
            df["seed_set_tokens"] = df["token_set"].apply(parse_token_set)
        elif "seed_tokens" in df.columns:
            df["seed_set_tokens"] = df["seed_tokens"]
        else:
            raise ValueError("CSV must contain 'token_set' or 'seed_tokens'")

    if "epochs" not in df.columns or "final_mean_pz" not in df.columns:
        raise ValueError("CSV must contain 'epochs' and 'final_mean_pz'")

    df["epochs"] = pd.to_numeric(df["epochs"])
    df["seed_set_tokens"] = pd.to_numeric(df["seed_set_tokens"])
    df["final_mean_pz"] = pd.to_numeric(df["final_mean_pz"])
    df["total_tokens"] = df["epochs"] * df["seed_set_tokens"]

    zero_mask = (df["final_mean_pz"] <= 0)
    if zero_mask.any():
        print(f"[warn] Found {zero_mask.sum()} rows with P(z) <= 0; replacing with epsilon")
        df.loc[zero_mask, "final_mean_pz"] = 1e-12

    return df


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Fit BOTH power-law and saturating (cloglog) memorization laws.")
    parser.add_argument("--csv", default="comma_150m_runs.csv", help="Path to runs CSV")
    parser.add_argument("--include-crashed", action="store_true", help="Include crashed runs")
    parser.add_argument("--no-dedupe", action="store_true", help="Do not drop duplicate rows")
    parser.add_argument("--out-prefix", default="mem_scaling", help="Prefix for saved CSV/figures")
    parser.add_argument("--also-plateau", action="store_true", help="Also fit free-plateau saturating model (needs SciPy)")
    args = parser.parse_args()

    df = load_and_prepare(args.csv, finished_only=not args.include_crashed, drop_duplicates=not args.no_dedupe)

    epochs = df["epochs"].to_numpy(dtype=float)
    seed = df["seed_set_tokens"].to_numpy(dtype=float)
    M = df["final_mean_pz"].to_numpy(dtype=float)

    # --- Power-law fit ---
    pw_fit, M_hat_pw = fit_powerlaw(epochs, seed, M)
    df["pred_powerlaw"] = M_hat_pw
    df["residual_powerlaw"] = df["final_mean_pz"].values - M_hat_pw
    print("\n=== Power-law fit (unbounded in M) ===")
    print("Form: log M = c1*log(epochs) + c2*log(seed) + c3")
    print(f"c1 (epochs):                 {pw_fit.c1: .6f}")
    print(f"c2 (seed):                   {pw_fit.c2: .6f}")
    print(f"c3 (intercept):              {pw_fit.c3: .6f}")
    print(f"R^2 (log-space):             {pw_fit.r2_y: .4f}")
    print(f"R^2 (M space):               {pw_fit.r2_M: .4f}")
    print(f"RMSE (M space):              {pw_fit.rmse_M: .6f}")
    print(f"MAE  (M space):              {pw_fit.mae_M: .6f}")

    # --- Cloglog fit ---
    clog_fit, M_hat_cl = fit_cloglog(epochs, seed, M)
    df["pred_cloglog"] = M_hat_cl
    df["residual_cloglog"] = df["final_mean_pz"].values - M_hat_cl
    print("\n=== Cloglog fit (saturating, plateau=1) ===")
    print("Form: log(-log(1 - M)) = log k + a*log(epochs) - b*log(seed)")
    print(f"a (epochs elasticity):       {clog_fit.a: .6f}")
    print(f"b (seed elasticity):         {clog_fit.b: .6f}")
    print(f"log k:                       {clog_fit.log_k: .6f}")
    print(f"R^2 (transformed y):         {clog_fit.r2_y: .4f}")
    print(f"R^2 (M space):               {clog_fit.r2_M: .4f}")
    print(f"RMSE (M space):              {clog_fit.rmse_M: .6f}")
    print(f"MAE  (M space):              {clog_fit.mae_M: .6f}")

    # Optional plateau fit
    if args.also_plateau:
        plateau_fit, M_hat_pl = fit_plateau(epochs, seed, M, init_from=clog_fit)
        df["pred_plateau"] = M_hat_pl
        df["residual_plateau"] = df["final_mean_pz"].values - M_hat_pl
        print("\n=== Saturating fit with free M_max ===")
        print("Form: M = M_max * (1 - exp(-k * epochs^a * seed^{-b}))")
        print(f"success:                     {plateau_fit.success}  ({plateau_fit.message})")
        print(f"a (epochs elasticity):       {plateau_fit.a: .6f}")
        print(f"b (seed elasticity):         {plateau_fit.b: .6f}")
        print(f"k (scale):                   {plateau_fit.k: .6e}")
        print(f"M_max (plateau):             {plateau_fit.M_max: .6f}")
        print(f"R^2 (M space):               {plateau_fit.r2_M: .4f}")
        print(f"RMSE (M space):              {plateau_fit.rmse_M: .6f}")
        print(f"MAE  (M space):              {plateau_fit.mae_M: .6f}")

    out_csv = f"{args.out_prefix}_both_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[save] wrote {out_csv}")

    # --- Plots: POWER-LAW ---
    plot_actual_vs_predicted(
        df, "final_mean_pz", "pred_powerlaw",
        f"{args.out_prefix}_powerlaw_actual_vs_pred.png",
        title="Actual vs Predicted — Power-law: log M = c1 log epochs + c2 log seed + c3"
    )
    plot_residuals_vs_epochs(
        df, "residual_powerlaw",
        f"{args.out_prefix}_powerlaw_residuals_vs_epochs.png",
        title="Residuals vs Epochs — Power-law"
    )
    plot_scaling_curves(
        df, "final_mean_pz", "pred_powerlaw",
        f"{args.out_prefix}_powerlaw_scaling_curves.png",
        title="Scaling Curves by Seed Set — log M = c1 log epochs + c2 log seed + c3"
    )
    plot_heatmap(df, "pred_powerlaw", f"{args.out_prefix}_heatmap_pred_powerlaw.png")

    # --- Plots: CLOGLOG ---
    plot_actual_vs_predicted(
        df, "final_mean_pz", "pred_cloglog",
        f"{args.out_prefix}_cloglog_actual_vs_pred.png",
        title="Actual vs Predicted — Saturating: M = 1 - exp(-k * epochs^a * seed^{-b})"
    )
    plot_residuals_vs_epochs(
        df, "residual_cloglog",
        f"{args.out_prefix}_cloglog_residuals_vs_epochs.png",
        title="Residuals vs Epochs — Saturating (cloglog)"
    )
    plot_scaling_curves(
        df, "final_mean_pz", "pred_cloglog",
        f"{args.out_prefix}_cloglog_scaling_curves.png",
        title="Scaling Curves by Seed Set — M = 1 - exp(-k * epochs^a * seed^{-b})"
    )
    plot_heatmap(df, "pred_cloglog", f"{args.out_prefix}_heatmap_pred_cloglog.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
