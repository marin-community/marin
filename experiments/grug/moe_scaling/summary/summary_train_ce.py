# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ISOFlop curve fitting for moe_iteration_01 (V2 sweep) using avg train/cross_entropy_loss over last 20 steps."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import wandb

from experiments.grug.moe_scaling.launch import (
    BUDGETS,
    HIDDEN_DIMS,
    _build_model_config,
    _compute_flops_per_token,
    _compute_tokens_and_batch,
)
from experiments.grug.moe_scaling.model import GrugModelConfig

WANDB_PROJECT = "dial_moe"
WANDB_GROUP = "isoflop-moe-v2"
METRIC_KEY = "train/cross_entropy_loss"
TAIL_STEPS = 20
OUTPUT_DIR = Path(__file__).parent


def fetch_runs() -> list[dict]:
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT, filters={"group": WANDB_GROUP, "state": "finished"})

    rows = []
    for r in runs:
        name = r.name
        parts = name.split("-")
        budget_str = parts[3]
        dim_str = parts[4]
        budget = float(budget_str)
        hidden_dim = int(dim_str[1:])

        model_cfg = _build_model_config(hidden_dim)
        fpt = _compute_flops_per_token(model_cfg)
        tokens, batch_size, train_steps = _compute_tokens_and_batch(budget, fpt)

        # Average train CE over last TAIL_STEPS steps
        history = list(
            r.scan_history(
                keys=[METRIC_KEY, "_step"],
                min_step=max(0, train_steps - TAIL_STEPS),
                page_size=TAIL_STEPS + 10,
            )
        )
        vals = [h[METRIC_KEY] for h in history if h.get(METRIC_KEY) is not None]
        if not vals:
            continue
        loss = sum(vals) / len(vals)

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
                "params": r.summary.get("parameter_count"),
            }
        )

    return rows


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
    T = np.array([r["tokens"] for r in budget_rows], dtype=float)
    y = np.array([r["loss"] for r in budget_rows], dtype=float)
    logT = np.log10(T)

    a, b, c = np.polyfit(logT, y, 2)
    logT_star = float(np.clip(-b / (2 * a), logT.min(), logT.max()))
    T_star = 10**logT_star
    loss_star = a * logT_star**2 + b * logT_star + c

    budget = budget_rows[0]["budget"]
    return IsoFlopFit(budget=budget, a=a, b=b, c=c, logT_star=logT_star, T_star=T_star, loss_star=float(loss_star))


def fit_frontier(fits: list[IsoFlopFit]) -> tuple[np.ndarray, np.ndarray]:
    logC = np.array([np.log10(f.budget) for f in fits])
    logT = np.array([f.logT_star for f in fits])
    losses = np.array([f.loss_star for f in fits])

    token_coeffs = np.polyfit(logC, logT, 1)
    loss_coeffs = np.polyfit(logT, losses, 1)

    return loss_coeffs, token_coeffs


def fit_param_frontier(fits: list[IsoFlopFit]) -> np.ndarray:
    logC = np.array([np.log10(f.budget) for f in fits])
    logN = np.array([np.log10(_optimal_params_for_budget(f.budget, f.T_star)) for f in fits])
    return np.polyfit(logC, logN, 1)


def _optimal_params_for_budget(budget: float, T_star: float) -> float:
    fpt_star = budget / (3 * T_star)
    fpts, params = [], []
    for dim in HIDDEN_DIMS:
        cfg = _build_model_config(dim)
        fpts.append(_compute_flops_per_token(cfg))
        params.append(_estimate_total_params(cfg))
    log_fpts = np.log10(fpts)
    log_params = np.log10(params)
    return float(10 ** np.interp(np.log10(fpt_star), log_fpts, log_params))


def _estimate_total_params(cfg: GrugModelConfig) -> int:
    d = cfg.hidden_dim
    V = cfg.vocab_size
    L = cfg.num_layers
    Ld = cfg.num_dense_layers
    Lm = L - Ld
    h = cfg.num_heads
    kv = cfg.num_kv_heads
    head_dim = d // h
    embed = V * d
    attn = L * (d * d + 2 * kv * head_dim * d + d * d)
    dense_mlp = Ld * (2 * d * cfg.dense_intermediate_dim + cfg.dense_intermediate_dim * d)
    expert_mlp = Lm * cfg.num_experts * (2 * d * cfg.intermediate_dim + cfg.intermediate_dim * d)
    shared_mlp = Lm * (2 * d * cfg.shared_expert_intermediate_dim + cfg.shared_expert_intermediate_dim * d)
    norms = (2 * L + 1) * d
    return embed + attn + dense_mlp + expert_mlp + shared_mlp + norms


def predict_optimal(token_coeffs, loss_coeffs, budget, param_coeffs=None):
    logC = np.log10(budget)
    logT_star = np.polyval(token_coeffs, logC)
    loss_star = np.polyval(loss_coeffs, logT_star)
    params_star = None
    if param_coeffs is not None:
        logN_star = np.polyval(param_coeffs, logC)
        params_star = float(10**logN_star)
    return 10**logT_star, float(loss_star), params_star


def fmt_sci(x: float) -> str:
    exp = math.floor(math.log10(abs(x)))
    coeff = x / 10**exp
    if coeff == 1.0:
        return f"1e{exp}"
    return f"{coeff:.0f}e{exp}"


PALETTE = ["#1877F2", "#F0701A", "#5A24C7", "#E42C97", "#00487C"]


def make_plot(rows, fits, token_coeffs, loss_coeffs):
    fig = go.Figure()
    budget_colors = {b: PALETTE[i] for i, b in enumerate(BUDGETS)}
    extrapolation_budgets = [1e18, 3e18, 1e19, 1e20, 1e21]

    for fit in fits:
        budget = fit.budget
        color = budget_colors.get(budget, "#888")
        label = fmt_sci(budget)
        sub = [r for r in rows if r["budget"] == budget]

        tokens = [r["tokens"] for r in sub]
        losses = [r["loss"] for r in sub]
        dims = [r["hidden_dim"] for r in sub]

        fig.add_trace(go.Scatter(
            x=tokens, y=losses, mode="markers+text",
            marker=dict(size=8, color=color),
            text=[f"d{d}" for d in dims], textposition="top center", textfont=dict(size=9),
            name=f"{label} FLOPs", legendgroup=label,
            hovertemplate="d=%{text}<br>tokens=%{x:.3e}<br>loss=%{y:.4f}<extra></extra>",
        ))

        logT = np.log10(np.array(tokens))
        grid = np.linspace(logT.min(), logT.max(), 200)
        yhat = fit.a * grid**2 + fit.b * grid + fit.c
        fig.add_trace(go.Scatter(
            x=10**grid, y=yhat, mode="lines", line=dict(width=2, color=color),
            name=f"{label} fit", legendgroup=label, showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=[fit.T_star], y=[fit.loss_star], mode="markers",
            marker=dict(symbol="x", size=14, color=color, line=dict(width=2)),
            name=f"{label} optimum", legendgroup=label, showlegend=False,
        ))

    # Frontier line
    all_logT, all_loss = [], []
    for b in extrapolation_budgets:
        T_star, loss_star, _ = predict_optimal(token_coeffs, loss_coeffs, b)
        all_logT.append(np.log10(T_star))
        all_loss.append(loss_star)

    grid_logT = np.linspace(min(all_logT), max(all_logT), 200)
    grid_loss = np.polyval(loss_coeffs, grid_logT)
    fig.add_trace(go.Scatter(
        x=10**grid_logT, y=grid_loss, mode="lines",
        line=dict(width=2, dash="dash", color="gray"), name="Optimal frontier",
    ))

    for b in extrapolation_budgets:
        T_star, loss_star, _ = predict_optimal(token_coeffs, loss_coeffs, b)
        fig.add_trace(go.Scatter(
            x=[T_star], y=[loss_star], mode="markers",
            marker=dict(symbol="diamond", size=10, color="gray", line=dict(width=1, color="black")),
            name=f"Predicted @ {fmt_sci(b)}",
        ))

    fig.update_xaxes(type="log", title_text="Training tokens")
    fig.update_yaxes(title_text=f"avg {METRIC_KEY} (last {TAIL_STEPS} steps)")
    fig.update_layout(
        template="plotly_white",
        title=f"moe_iteration_01 ISOFlop curves (avg train CE, last {TAIL_STEPS} steps)",
        width=1000, height=600,
    )
    return fig


def main():
    print(f"Fetching runs from W&B (metric: avg {METRIC_KEY}, last {TAIL_STEPS} steps)...")
    rows = fetch_runs()
    print(f"Found {len(rows)} finished runs")

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

    print(f"\n=== Predicted optimal {METRIC_KEY} ===")
    predictions = {}
    for budget in [1e18, 3e18, 1e19, 1e20, 1e21]:
        T_star, loss_star, N_star = predict_optimal(token_coeffs, loss_coeffs, budget, param_coeffs)
        print(f"  {fmt_sci(budget)} FLOPs: T*={T_star:.3e}, N*={N_star:.3e}, loss*={loss_star:.4f}")
        predictions[fmt_sci(budget)] = {
            "optimal_tokens": T_star,
            "optimal_model_params": N_star,
            "predicted_loss": loss_star,
        }

    # Plot
    fig = make_plot(rows, fits, token_coeffs, loss_coeffs)
    img_path = OUTPUT_DIR / "isoflop_curve_train_ce.png"
    fig.write_image(str(img_path), scale=2)
    print(f"\nPlot saved to {img_path}")

    result = {
        "model": "moe_iteration_01",
        "metric": f"avg_{METRIC_KEY}_last{TAIL_STEPS}",
        "predictions": predictions,
    }

    json_path = OUTPUT_DIR / "summary_train_ce.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
