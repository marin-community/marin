# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ISOFlop curve fitting for moe_iteration_01 (V2 sweep).

Loads the 15 runs from W&B, fits parabolic isoflop curves per budget,
extrapolates optimal loss to higher FLOP budgets, and saves the plot.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb

from experiments.grug.moe_scaling.launch import (
    BUDGETS,
    HIDDEN_DIMS,
    _build_model_config,
    _compute_flops_per_token,
    _compute_tokens_and_batch,
)
from experiments.grug.moe_scaling.model import GrugModelConfig

# ============================================================
# Config
# ============================================================

WANDB_PROJECT = "dial_moe"
WANDB_GROUP = "isoflop-moe-v2"
METRIC_KEY = "eval/uncheatable_eval/macro_loss"
OUTPUT_DIR = Path(__file__).parent


# ============================================================
# Fetch runs from W&B
# ============================================================


def fetch_runs() -> list[dict]:
    """Fetch the 15 isoflop-moe-v2 runs from W&B."""
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT, filters={"group": WANDB_GROUP, "state": "finished"})

    rows = []
    for r in runs:
        s = r.summary
        loss = s.get(METRIC_KEY)
        if loss is None:
            continue

        # Parse budget and hidden_dim from run name: isoflop-moe-v2-{budget}-d{dim}
        name = r.name
        parts = name.split("-")
        # e.g. ['isoflop', 'moe', 'v2', '1e+18', 'd512']
        budget_str = parts[3]  # '1e+18'
        dim_str = parts[4]  # 'd512'
        budget = float(budget_str)
        hidden_dim = int(dim_str[1:])

        model_cfg = _build_model_config(hidden_dim)
        fpt = _compute_flops_per_token(model_cfg)
        tokens, batch_size, train_steps = _compute_tokens_and_batch(budget, fpt)

        rows.append(
            {
                "name": name,
                "budget": budget,
                "hidden_dim": hidden_dim,
                "tokens": tokens,
                "batch_size": batch_size,
                "train_steps": train_steps,
                "loss": loss,
                "flops": budget,
                "params": s.get("parameter_count"),
            }
        )

    return rows


# ============================================================
# Parabolic fit per budget
# ============================================================


@dataclass
class IsoFlopFit:
    budget: float
    a: float
    b: float
    c: float
    logT_star: float
    T_star: float
    loss_star: float


def fit_parabola(budget_rows: list[dict]) -> IsoFlopFit:
    """Fit loss = a*(log10 T)^2 + b*(log10 T) + c for a single FLOP budget."""
    T = np.array([r["tokens"] for r in budget_rows], dtype=float)
    y = np.array([r["loss"] for r in budget_rows], dtype=float)
    logT = np.log10(T)

    a, b, c = np.polyfit(logT, y, 2)
    logT_star = float(np.clip(-b / (2 * a), logT.min(), logT.max()))
    T_star = 10**logT_star
    loss_star = a * logT_star**2 + b * logT_star + c

    budget = budget_rows[0]["budget"]
    return IsoFlopFit(
        budget=budget,
        a=a,
        b=b,
        c=c,
        logT_star=logT_star,
        T_star=T_star,
        loss_star=float(loss_star),
    )


# ============================================================
# Frontier extrapolation
# ============================================================


def fit_frontier(fits: list[IsoFlopFit]) -> tuple[np.ndarray, np.ndarray]:
    """Fit log10(loss*) = m * log10(T*) + k across budget optima.

    Returns (loss_coeffs, token_coeffs) for extrapolation:
      log10(T*) = alphaT + betaT * log10(C)
      loss* = polyval(loss_coeffs, log10(T*))
    """
    logC = np.array([np.log10(f.budget) for f in fits])
    logT = np.array([f.logT_star for f in fits])
    losses = np.array([f.loss_star for f in fits])

    # Token scaling: log10(T*) vs log10(C)
    token_coeffs = np.polyfit(logC, logT, 1)  # [betaT, alphaT]

    # Loss frontier: loss* vs log10(T*)
    loss_coeffs = np.polyfit(logT, losses, 1)

    return loss_coeffs, token_coeffs


def fit_param_frontier(fits: list[IsoFlopFit]) -> np.ndarray:
    """Fit log10(N*) vs log10(C) for optimal model size extrapolation.

    For each budget, compute optimal params from: fpt* = C / (3 * T*),
    then interpolate in the hidden_dim grid to find the corresponding param count.
    """
    logC = np.array([np.log10(f.budget) for f in fits])
    logN = np.array([np.log10(_optimal_params_for_budget(f.budget, f.T_star)) for f in fits])
    return np.polyfit(logC, logN, 1)


def _optimal_params_for_budget(budget: float, T_star: float) -> float:
    """Interpolate optimal total params from optimal tokens at a given budget."""
    fpt_star = budget / (3 * T_star)
    # Build (fpt, params) pairs from the hidden_dim grid
    fpts, params = [], []
    for dim in HIDDEN_DIMS:
        cfg = _build_model_config(dim)
        fpts.append(_compute_flops_per_token(cfg))
        # Estimate total params from the config
        params.append(_estimate_total_params(cfg))
    # Interpolate in log space
    log_fpts = np.log10(fpts)
    log_params = np.log10(params)
    return float(10 ** np.interp(np.log10(fpt_star), log_fpts, log_params))


def _estimate_total_params(cfg: GrugModelConfig) -> int:
    """Rough total param estimate from config dimensions."""
    d = cfg.hidden_dim
    V = cfg.vocab_size
    L = cfg.num_layers
    Ld = cfg.num_dense_layers
    Lm = L - Ld
    h = cfg.num_heads
    kv = cfg.num_kv_heads
    head_dim = d // h
    # Embedding
    embed = V * d
    # Attention per layer: Q, K, V projections + output
    attn = L * (d * d + 2 * kv * head_dim * d + d * d)
    # Dense MLP layers (GLU): gate + up + down
    dense_mlp = Ld * (2 * d * cfg.dense_intermediate_dim + cfg.dense_intermediate_dim * d)
    # MoE layers: experts + shared expert
    expert_mlp = Lm * cfg.num_experts * (2 * d * cfg.intermediate_dim + cfg.intermediate_dim * d)
    shared_mlp = Lm * (2 * d * cfg.shared_expert_intermediate_dim + cfg.shared_expert_intermediate_dim * d)
    # RMSNorm: 2 per layer + final
    norms = (2 * L + 1) * d
    return embed + attn + dense_mlp + expert_mlp + shared_mlp + norms


def predict_optimal(
    token_coeffs: np.ndarray,
    loss_coeffs: np.ndarray,
    budget: float,
    param_coeffs: np.ndarray | None = None,
) -> tuple[float, float, float | None]:
    """Predict optimal tokens, loss, and optionally model params at a given FLOP budget."""
    logC = np.log10(budget)
    logT_star = np.polyval(token_coeffs, logC)
    loss_star = np.polyval(loss_coeffs, logT_star)
    params_star = None
    if param_coeffs is not None:
        logN_star = np.polyval(param_coeffs, logC)
        params_star = float(10**logN_star)
    return 10**logT_star, float(loss_star), params_star


# ============================================================
# Plot
# ============================================================

PALETTE = ["#1877F2", "#F0701A", "#5A24C7", "#E42C97", "#00487C"]


def fmt_sci(x: float) -> str:
    exp = math.floor(math.log10(abs(x)))
    coeff = x / 10**exp
    if coeff == 1.0:
        return f"1e{exp}"
    return f"{coeff:.0f}e{exp}"


def make_plot(rows: list[dict], fits: list[IsoFlopFit], token_coeffs, loss_coeffs):
    fig = go.Figure()

    budget_colors = {b: PALETTE[i] for i, b in enumerate(BUDGETS)}
    extrapolation_budgets = [1e18, 1e19, 1e20, 1e21]

    # Scatter + parabolic fits per budget
    for fit in fits:
        budget = fit.budget
        color = budget_colors.get(budget, "#888")
        label = fmt_sci(budget)
        sub = [r for r in rows if r["budget"] == budget]

        tokens = [r["tokens"] for r in sub]
        losses = [r["loss"] for r in sub]
        dims = [r["hidden_dim"] for r in sub]

        fig.add_trace(
            go.Scatter(
                x=tokens,
                y=losses,
                mode="markers+text",
                marker=dict(size=8, color=color),
                text=[f"d{d}" for d in dims],
                textposition="top center",
                textfont=dict(size=9),
                name=f"{label} FLOPs",
                legendgroup=label,
                hovertemplate="d=%{text}<br>tokens=%{x:.3e}<br>loss=%{y:.4f}<extra></extra>",
            )
        )

        # Parabolic fit curve
        logT = np.log10(np.array(tokens))
        grid = np.linspace(logT.min(), logT.max(), 200)
        yhat = fit.a * grid**2 + fit.b * grid + fit.c

        fig.add_trace(
            go.Scatter(
                x=10**grid,
                y=yhat,
                mode="lines",
                line=dict(width=2, color=color),
                name=f"{label} fit",
                legendgroup=label,
                showlegend=False,
            )
        )

        # Mark optimum
        fig.add_trace(
            go.Scatter(
                x=[fit.T_star],
                y=[fit.loss_star],
                mode="markers",
                marker=dict(symbol="x", size=14, color=color, line=dict(width=2)),
                name=f"{label} optimum",
                legendgroup=label,
                showlegend=False,
                hovertemplate=f"Optimum @ {label}<br>T*=%{{x:.3e}}<br>loss*=%{{y:.4f}}<extra></extra>",
            )
        )

    # Frontier line through optima + extrapolations
    all_logT = []
    all_loss = []
    for b in extrapolation_budgets:
        T_star, loss_star, _ = predict_optimal(token_coeffs, loss_coeffs, b)
        all_logT.append(np.log10(T_star))
        all_loss.append(loss_star)

    grid_logT = np.linspace(min(all_logT), max(all_logT), 200)
    grid_loss = np.polyval(loss_coeffs, grid_logT)

    fig.add_trace(
        go.Scatter(
            x=10**grid_logT,
            y=grid_loss,
            mode="lines",
            line=dict(width=2, dash="dash", color="gray"),
            name="Optimal frontier",
        )
    )

    # Mark extrapolated optima
    for b in extrapolation_budgets:
        T_star, loss_star, _ = predict_optimal(token_coeffs, loss_coeffs, b)
        fig.add_trace(
            go.Scatter(
                x=[T_star],
                y=[loss_star],
                mode="markers",
                marker=dict(symbol="diamond", size=10, color="gray", line=dict(width=1, color="black")),
                name=f"Predicted @ {fmt_sci(b)}",
                hovertemplate=f"Predicted @ {fmt_sci(b)}<br>T*=%{{x:.3e}}<br>loss*=%{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_xaxes(type="log", title_text="Training tokens")
    fig.update_yaxes(title_text=METRIC_KEY)
    fig.update_layout(
        template="plotly_white",
        title="moe_iteration_01 ISOFlop curves (uncheatable_eval/macro_loss)",
        width=1000,
        height=600,
    )

    return fig


# ============================================================
# Training curves for a single run
# ============================================================

TRAINING_CURVE_METRICS = [
    ("train/cross_entropy_loss", "Cross Entropy Loss"),
    ("train/router/load_balancing_loss", "Load Balancing Loss"),
    ("train/router/router_z_loss", "Router Z Loss"),
    ("grad/norm/total", "Grad Norm"),
    ("moe_bias/layer_1/expert_10", "MoE Bias L1/E10"),
    ("moe_bias/layer_4/expert_10", "MoE Bias L4/E10"),
    ("moe_bias/layer_6/expert_10", "MoE Bias L6/E10"),
]


def make_training_curves(run_name: str = "isoflop-moe-v2-1e+19-d1024"):
    """Fetch training history and plot key metrics as subplots."""
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT, filters={"group": WANDB_GROUP, "display_name": run_name})
    r = runs[0]

    keys = ["_step"] + [m[0] for m in TRAINING_CURVE_METRICS]
    print(f"Fetching training history for {run_name}...")
    rows = list(r.scan_history(keys=keys, page_size=10000))
    print(f"  Got {len(rows)} rows")

    n = len(TRAINING_CURVE_METRICS)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[m[1] for m in TRAINING_CURVE_METRICS],
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )

    for i, (key, title) in enumerate(TRAINING_CURVE_METRICS):
        row = i // ncols + 1
        col = i % ncols + 1
        steps = [d["_step"] for d in rows if d.get(key) is not None]
        vals = [d[key] for d in rows if d.get(key) is not None]
        fig.add_trace(
            go.Scatter(x=steps, y=vals, mode="lines", line=dict(width=1.5), name=title, showlegend=False),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Step", row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        title=f"{run_name} Training Curves",
        width=1200,
        height=300 * nrows + 100,
    )

    img_path = OUTPUT_DIR / f"training_curves_{run_name.replace('+', '')}.png"
    fig.write_image(str(img_path), scale=2)
    print(f"  Saved to {img_path}")


# ============================================================
# Main
# ============================================================


def main():
    print("Fetching runs from W&B...")
    rows = fetch_runs()
    print(f"Found {len(rows)} finished runs")

    # Group by budget and fit
    budget_groups = {}
    for r in rows:
        budget_groups.setdefault(r["budget"], []).append(r)

    fits = []
    for budget in sorted(budget_groups.keys()):
        group = budget_groups[budget]
        if len(group) < 3:
            print(f"  Skipping budget {budget:.0e}: only {len(group)} runs")
            continue
        fit = fit_parabola(group)
        fits.append(fit)
        print(f"  {fmt_sci(budget)}: optimal tokens={fit.T_star:.3e}, loss*={fit.loss_star:.4f}")

    loss_coeffs, token_coeffs = fit_frontier(fits)
    param_coeffs = fit_param_frontier(fits)

    print("\n=== Predicted optimal uncheatable_eval/macro_loss ===")
    for budget in [1e18, 1e19, 1e20, 1e21]:
        T_star, loss_star, N_star = predict_optimal(token_coeffs, loss_coeffs, budget, param_coeffs)
        print(f"  {fmt_sci(budget)} FLOPs: T*={T_star:.3e}, N*={N_star:.3e}, loss*={loss_star:.4f}")

    fig = make_plot(rows, fits, token_coeffs, loss_coeffs)

    img_path = OUTPUT_DIR / "isoflop_curve.png"
    fig.write_image(str(img_path), scale=2)
    print(f"\nPlot saved to {img_path}")

    # Training curves for the best 1e19 run
    best_run_name = "isoflop-moe-v2-1e+19-d1024"
    make_training_curves(best_run_name)

    # Write summary JSON
    write_summary_json(rows, fits, token_coeffs, loss_coeffs, param_coeffs, best_run_name)


def _compute_active_params(cfg, total_params: int) -> int:
    """Active params = total - inactive expert params (E-K experts per MoE layer)."""
    d = cfg.hidden_dim
    i = cfg.intermediate_dim
    E = cfg.num_experts
    K = cfg.num_experts_per_token
    nm = cfg.num_layers - cfg.num_dense_layers
    # GLU expert: gate(d,i) + up(d,i) + down(i,d)
    per_expert = 2 * d * i + i * d
    inactive = nm * (E - K) * per_expert
    return total_params - inactive


def write_summary_json(rows, fits, token_coeffs, loss_coeffs, param_coeffs, best_run_name):
    """Write summary results to JSON."""
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT, filters={"group": WANDB_GROUP, "display_name": best_run_name})
    r = runs[0]
    s = r.summary

    # Best run info
    best_row = next(row for row in rows if row["name"] == best_run_name)
    cfg = _build_model_config(best_row["hidden_dim"])
    total_params = s.get("parameter_count")
    active_params = _compute_active_params(cfg, total_params)

    # Predictions with optimal model size
    predictions = {}
    for budget in [1e18, 1e19, 1e20, 1e21]:
        T_star, loss_star, N_star = predict_optimal(token_coeffs, loss_coeffs, budget, param_coeffs)
        predictions[fmt_sci(budget)] = {
            "optimal_tokens": T_star,
            "optimal_model_params": N_star,
            "predicted_loss": loss_star,
        }

    result = {
        "model": "moe_iteration_01",
        "description": (
            "DeepSeek-style MoE with AF load balancing (bias_rate=0.01) + 0.001 aux loss,"
            " AdamW, 2 leading dense layers, 1 shared expert at hidden_dim, k=4, e=64,"
            " expert_dim=hidden_dim/2, expert_lr_mul=1.0, warmup=0.1, cooldown=0.2,"
            " wd=0.1, router_z_loss_coef=0.001, z_loss_weight=1e-4."
        ),
        "metric": METRIC_KEY,
        "predictions": predictions,
        "best_1e19_run": {
            "name": best_run_name,
            "hidden_dim": best_row["hidden_dim"],
            "mean_mfu": s.get("throughput/mean_mfu"),
            "tokens": best_row["tokens"],
            "tokens_per_second": s.get("throughput/tokens_per_second"),
            "total_params": total_params,
            "active_params": active_params,
            "runtime_seconds": s.get("_runtime"),
            "resource_type": "v5p-8",
            "hardware_flops_per_second": s.get("throughput/theoretical_flops"),
        },
    }

    json_path = OUTPUT_DIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"\nJSON saved to {json_path}")


if __name__ == "__main__":
    main()
