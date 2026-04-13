# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate isoflop curves from v16 sweep data.

For each compute budget, plots BPB vs total (active) parameters across model sizes,
fits a log-space parabola to find the optimal model size, and fits an asymptotic
scaling law: loss*(C) = L_inf + A * C^{-alpha}.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = "experiments/grug/moe_apr2/new_plots"


# ============================================================
# 1) Load data from wandb
# ============================================================


def fetch_v16_data():
    """Fetch finished v16 isoflop runs from wandb."""
    import wandb

    api = wandb.Api()
    runs = api.runs("marin-community/dial_moe", filters={"group": "isoflop-moe-v16", "state": "finished"})

    rows = []
    for r in runs:
        bpb = r.summary.get("eval/paloma/c4_en/bpb")
        macro_loss = r.summary.get("eval/paloma/macro_loss")
        if bpb is None:
            continue
        parts = r.name.split("-")
        budget_str = parts[3]  # e.g. '1e+18'
        dim = int(parts[4][1:])  # e.g. 'd768' -> 768
        budget = float(budget_str)
        rows.append({"budget": budget, "dim": dim, "bpb": bpb, "macro_loss": macro_loss, "name": r.name})

    return rows


def compute_active_params(dim):
    """Active parameters (lm_head, no embed) for a given hidden_dim."""
    # From heuristic
    # hidden_head_ratio = 128
    vocab_size = 128256
    # num_heads = max(1, dim // hidden_head_ratio)

    # num_layers from heuristic formula
    hs_pow = math.log2(dim)
    num_layers = round(dim / (64 + (hs_pow * 4.0) - 9))

    intermediate_dim = math.ceil(dim / 2 / 128) * 128
    shared_ffn = dim
    k = 4  # experts per token

    attn = 4 * dim * dim
    active_moe = k * 3 * dim * intermediate_dim
    shared = 3 * dim * shared_ffn
    router = dim * 64
    per_layer = attn + active_moe + shared + router

    lm_head = dim * vocab_size
    return num_layers * per_layer + lm_head


def compute_active_params_no_lmhead(dim):
    """Active parameters excluding lm_head."""
    return compute_active_params(dim) - dim * 128256


def compute_fpt(dim):
    """FLOPs per token (no lm_head)."""
    hidden_head_ratio = 128
    num_heads = max(1, dim // hidden_head_ratio)

    hs_pow = math.log2(dim)
    num_layers = round(dim / (64 + (hs_pow * 4.0) - 9))

    intermediate_dim = math.ceil(dim / 2 / 128) * 128
    shared_ffn = dim
    k = 4

    attn = 2 * dim * (dim + dim) + 2 * dim * (num_heads * (dim // num_heads)) * 2
    moe_mlp = k * 3 * 2 * dim * intermediate_dim
    shared_mlp = 3 * 2 * dim * shared_ffn
    per_layer = attn + moe_mlp + shared_mlp
    return num_layers * per_layer


# ============================================================
# 2) Fit parabolas per budget and find optima
# ============================================================


def fit_isoflop_parabola(dims, bpbs):
    """Fit bpb = a*log10(params)^2 + b*log10(params) + c. Return optimal params and bpb."""
    params = np.array([compute_active_params(d) for d in dims])
    log_params = np.log10(params)
    bpbs = np.array(bpbs)

    coeffs = np.polyfit(log_params, bpbs, 2)
    if coeffs[0] <= 0:
        # No minimum — use the actual minimum point
        idx = np.argmin(bpbs)
        return params[idx], bpbs[idx], coeffs, params

    opt_log = -coeffs[1] / (2 * coeffs[0])
    opt_bpb = np.polyval(coeffs, opt_log)
    opt_params = 10**opt_log

    return opt_params, opt_bpb, coeffs, params


def fit_asymptotic_powerlaw(C, loss, n_grid=400):
    """Fit loss(C) = L_inf + A * C^{-alpha}."""
    C = np.asarray(C, float)
    y = np.asarray(loss, float)
    idx = np.argsort(C)
    C, y = C[idx], y[idx]

    if len(C) < 2:
        return None

    y_min = float(np.min(y))
    Linf_grid = np.linspace(0, 0.95 * y_min, n_grid)
    logC = np.log(C)

    best = None
    for Linf in Linf_grid:
        z = y - Linf
        if np.any(z <= 0):
            continue
        logz = np.log(z)
        b1, b0 = np.polyfit(logC, logz, 1)
        alpha = -b1
        if not np.isfinite(alpha) or alpha <= 0:
            continue
        A = np.exp(b0)
        sse = float(np.mean((logz - (b0 + b1 * logC)) ** 2))
        if best is None or sse < best["sse"]:
            best = {"L_inf": float(Linf), "A": float(A), "alpha": float(alpha), "sse": sse}

    return best


# ============================================================
# 3) Generate plots
# ============================================================


def main():
    # Load data
    cache_path = f"{PLOT_DIR}/v16_isoflop_data.json"
    try:
        with open(cache_path) as f:
            rows = json.load(f)
        print(f"Loaded {len(rows)} runs from cache")
    except FileNotFoundError:
        rows = fetch_v16_data()
        with open(cache_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Fetched {len(rows)} runs from wandb")

    # Group by budget, exclude d640 and d896
    excluded_dims = {640, 896}
    by_budget = {}
    for r in rows:
        if r["dim"] in excluded_dims:
            continue
        by_budget.setdefault(r["budget"], []).append(r)

    target_budgets = [1e18, 3e18, 1e19, 3e19, 1e20, 3e20]
    colors = {1e18: "C0", 3e18: "C1", 1e19: "C2", 3e19: "C3", 1e20: "C4", 3e20: "C5"}

    # ---- Plot 1: IsoFLOP curves (BPB vs active params) ----
    fig, ax = plt.subplots(figsize=(10, 6))

    optima = []
    for budget in target_budgets:
        runs = by_budget.get(budget, [])
        if not runs:
            continue

        dims = [r["dim"] for r in runs]
        bpbs = [r["bpb"] for r in runs]
        params = [compute_active_params(d) for d in dims]

        # Sort by params
        order = np.argsort(params)
        params_sorted = np.array(params)[order]
        bpbs_sorted = np.array(bpbs)[order]
        dims_sorted = np.array(dims)[order]

        color = colors.get(budget, "gray")
        label = f"{budget:.0e} FLOPs"

        # Scatter
        ax.plot(params_sorted, bpbs_sorted, "o", color=color, label=label, markersize=6)

        # Fit parabola
        opt_params, opt_bpb, coeffs, _ = fit_isoflop_parabola(dims_sorted, bpbs_sorted)

        if coeffs[0] > 0:
            log_grid = np.linspace(np.log10(params_sorted.min()) - 0.1, np.log10(params_sorted.max()) + 0.1, 200)
            ax.plot(10**log_grid, np.polyval(coeffs, log_grid), "--", color=color, alpha=0.8)
            ax.plot(opt_params, opt_bpb, "*", color=color, markersize=15)

        # Compute FLOPs and no-lmhead params for the optimal
        best_dim = dims_sorted[np.argmin(bpbs_sorted)]
        optima.append(
            {
                "budget": budget,
                "opt_params": opt_params,
                "opt_params_no_lm": compute_active_params_no_lmhead(best_dim),
                "opt_bpb": opt_bpb,
                "opt_tokens": budget / (3 * compute_fpt(best_dim)),
            }
        )

    ax.set_xscale("log")
    ax.set_xlabel("Active Parameters (no embed)")
    ax.set_ylabel("c4_en/bpb")
    ax.set_title("IsoFLOP Curves (v16, GQA 4:1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/isoflop_curves.png", dpi=150)
    plt.close(fig)
    print("Saved isoflop_curves.png")

    # ---- Plot 1b: IsoFLOP curves (BPB vs tokens) ----
    fig1b, ax1b = plt.subplots(figsize=(10, 6))

    for budget in target_budgets:
        runs = by_budget.get(budget, [])
        if not runs:
            continue

        dims = [r["dim"] for r in runs]
        bpbs = [r["bpb"] for r in runs]
        tokens = [budget / (3 * compute_fpt(d)) for d in dims]

        order = np.argsort(tokens)
        tokens_sorted = np.array(tokens)[order]
        bpbs_sorted = np.array(bpbs)[order]
        dims_sorted = np.array(dims)[order]

        color = colors.get(budget, "gray")
        label = f"{budget:.0e} FLOPs"

        ax1b.plot(tokens_sorted, bpbs_sorted, "o", color=color, label=label, markersize=6)

        # Fit parabola in log(tokens)
        log_tokens = np.log10(tokens_sorted)
        coeffs = np.polyfit(log_tokens, bpbs_sorted, 2)
        if coeffs[0] > 0:
            log_grid = np.linspace(log_tokens.min() - 0.1, log_tokens.max() + 0.1, 200)
            ax1b.plot(10**log_grid, np.polyval(coeffs, log_grid), "--", color=color, alpha=0.8)
            opt_log = -coeffs[1] / (2 * coeffs[0])
            opt_bpb = np.polyval(coeffs, opt_log)
            ax1b.plot(10**opt_log, opt_bpb, "*", color=color, markersize=15)

        # Label dims
        for t, b, d in zip(tokens_sorted, bpbs_sorted, dims_sorted, strict=True):
            ax1b.annotate(f"d{d}", (t, b), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.7)

    ax1b.set_xscale("log")
    ax1b.set_xlabel("Training Tokens")
    ax1b.set_ylabel("c4_en/bpb")
    ax1b.set_title("IsoFLOP Curves vs Tokens (v16, GQA 4:1)")
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)
    fig1b.tight_layout()
    fig1b.savefig(f"{PLOT_DIR}/isoflop_curves_tokens.png", dpi=150)
    plt.close(fig1b)
    print("Saved isoflop_curves_tokens.png")

    # ---- Plot 1c: IsoFLOP curves (macro_loss vs tokens) ----
    fig1c, ax1c = plt.subplots(figsize=(10, 6))

    for budget in target_budgets:
        runs = by_budget.get(budget, [])
        if not runs:
            continue

        dims = [r["dim"] for r in runs]
        losses = [r.get("macro_loss") for r in runs]
        if any(l is None for l in losses):
            continue
        tokens = [budget / (3 * compute_fpt(d)) for d in dims]

        order = np.argsort(tokens)
        tokens_sorted = np.array(tokens)[order]
        losses_sorted = np.array(losses)[order]
        dims_sorted = np.array(dims)[order]

        color = colors.get(budget, "gray")
        label = f"{budget:.0e} FLOPs"

        ax1c.plot(tokens_sorted, losses_sorted, "o", color=color, label=label, markersize=6)

        log_tokens = np.log10(tokens_sorted)
        coeffs = np.polyfit(log_tokens, losses_sorted, 2)
        if coeffs[0] > 0:
            log_grid = np.linspace(log_tokens.min() - 0.1, log_tokens.max() + 0.1, 200)
            ax1c.plot(10**log_grid, np.polyval(coeffs, log_grid), "--", color=color, alpha=0.8)
            opt_log = -coeffs[1] / (2 * coeffs[0])
            opt_loss = np.polyval(coeffs, opt_log)
            ax1c.plot(10**opt_log, opt_loss, "*", color=color, markersize=15)

        for t, l, d in zip(tokens_sorted, losses_sorted, dims_sorted, strict=True):
            ax1c.annotate(f"d{d}", (t, l), textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.7)

    ax1c.set_xscale("log")
    ax1c.set_xlabel("Training Tokens")
    ax1c.set_ylabel("Paloma macro_loss")
    ax1c.set_title("IsoFLOP Curves — macro_loss vs Tokens (v16, GQA 4:1)")
    ax1c.legend()
    ax1c.grid(True, alpha=0.3)
    fig1c.tight_layout()
    fig1c.savefig(f"{PLOT_DIR}/isoflop_curves_tokens_macro_loss.png", dpi=150)
    plt.close(fig1c)
    print("Saved isoflop_curves_tokens_macro_loss.png")

    # ---- Compute macro_loss optima and optimal tokens ----
    macro_optima = []
    for budget in target_budgets:
        runs = by_budget.get(budget, [])
        if not runs:
            continue
        dims = [r["dim"] for r in runs]
        macros = [r.get("macro_loss") for r in runs]
        if any(m is None for m in macros):
            continue
        params = [compute_active_params(d) for d in dims]
        order = np.argsort(params)
        params_sorted = np.array(params)[order]
        macros_sorted = np.array(macros)[order]
        dims_sorted = np.array(dims)[order]

        log_params = np.log10(params_sorted)
        coeffs = np.polyfit(log_params, macros_sorted, 2)
        if coeffs[0] > 0:
            opt_log = -coeffs[1] / (2 * coeffs[0])
            opt_macro = np.polyval(coeffs, opt_log)
            opt_params = 10**opt_log
        else:
            idx = np.argmin(macros_sorted)
            opt_params = params_sorted[idx]
            opt_macro = macros_sorted[idx]

        # Find optimal tokens from FPT at nearest dim
        best_dim = dims_sorted[np.argmin(macros_sorted)]
        opt_tokens = budget / (3 * compute_fpt(best_dim))

        macro_optima.append(
            {"budget": budget, "opt_params": opt_params, "opt_macro": opt_macro, "opt_tokens": opt_tokens}
        )

    # ---- Plot 2: Optimal macro_loss vs Compute (scaling law) ----
    if len(macro_optima) >= 2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        C_arr = np.array([o["budget"] for o in macro_optima])
        macro_arr = np.array([o["opt_macro"] for o in macro_optima])

        ax2.plot(C_arr, macro_arr, "o", color="C0", markersize=10)

        fit = fit_asymptotic_powerlaw(C_arr, macro_arr)
        if fit:
            C_grid = np.logspace(np.log10(C_arr.min()) - 0.3, np.log10(C_arr.max()) + 0.5, 200)
            yhat = fit["L_inf"] + fit["A"] * C_grid ** (-fit["alpha"])
            ax2.plot(
                C_grid,
                yhat,
                "--",
                color="C0",
                alpha=0.5,
                label=f"L∞={fit['L_inf']:.3f} + {fit['A']:.2f}·C^(-{fit['alpha']:.3f})",
            )
            ax2.legend()
            print(f"Scaling law (macro): L_inf={fit['L_inf']:.4f}, A={fit['A']:.3f}, alpha={fit['alpha']:.4f}")

        for o in macro_optima:
            ax2.annotate(
                f"{o['budget']:.0e}",
                (o["budget"], o["opt_macro"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax2.set_xscale("log")
        ax2.set_xlabel("Compute (FLOPs, excl lm_head)")
        ax2.set_ylabel("Optimal paloma macro_loss")
        ax2.set_title("Optimal macro_loss vs Compute")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(f"{PLOT_DIR}/isoflop_scaling_law.png", dpi=150)
        plt.close(fig2)
        print("Saved isoflop_scaling_law.png")

    # ---- Plot 2b: Scaling law with L_inf pinned at 1.6, projected to 1e23 ----
    if len(macro_optima) >= 2:
        fig2b, ax2b = plt.subplots(figsize=(10, 6))

        C_arr = np.array([o["budget"] for o in macro_optima])
        macro_arr = np.array([o["opt_macro"] for o in macro_optima])

        # Fit with fixed L_inf = 1.6: log(y - 1.6) = log(A) - alpha * log(C)
        L_inf_fixed = 1.6
        z = macro_arr - L_inf_fixed
        logC = np.log(C_arr)
        logz = np.log(z)
        b1, b0 = np.polyfit(logC, logz, 1)
        alpha = -b1
        A = np.exp(b0)

        # Project from 1e17 to 1e23
        C_grid = np.logspace(17, 23, 500)
        yhat = L_inf_fixed + A * C_grid ** (-alpha)

        ax2b.plot(C_arr, macro_arr, "o", color="C0", markersize=10, zorder=5, label="v16 sweep optima")
        ax2b.plot(
            C_grid,
            yhat,
            "--",
            color="C0",
            alpha=0.6,
            label=f"L∞=1.6 + {A:.2f}\u00b7C^(-{alpha:.4f})",
        )

        # Mark projections at 1e21 and 1e23
        for C_proj in [1e21, 1e23]:
            y_proj = L_inf_fixed + A * C_proj ** (-alpha)
            ax2b.plot(C_proj, y_proj, "D", color="C3", markersize=10, zorder=5)
            ax2b.annotate(
                f"{C_proj:.0e} → {y_proj:.3f}",
                (C_proj, y_proj),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                fontweight="bold",
                color="C3",
            )

        for o in macro_optima:
            ax2b.annotate(
                f"{o['budget']:.0e}",
                (o["budget"], o["opt_macro"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax2b.set_xscale("log")
        ax2b.set_xlabel("Compute (FLOPs)")
        ax2b.set_ylabel("Optimal paloma macro_loss")
        ax2b.set_title("Scaling Law Projection (L∞ = 1.6)")
        ax2b.legend()
        ax2b.grid(True, alpha=0.3)
        fig2b.tight_layout()
        fig2b.savefig(f"{PLOT_DIR}/isoflop_scaling_law_pinned.png", dpi=150)
        plt.close(fig2b)
        print(f"Pinned scaling law: L∞=1.6, A={A:.4f}, alpha={alpha:.4f}")
        print(f"  1e21 projection: {L_inf_fixed + A * 1e21 ** (-alpha):.4f}")
        print(f"  1e23 projection: {L_inf_fixed + A * 1e23 ** (-alpha):.4f}")
        print("Saved isoflop_scaling_law_pinned.png")

    # ---- Plot 3: N*(C) and T*(C) on separate subplots ----
    if len(optima) >= 2:
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

        C_arr = np.array([o["budget"] for o in optima])
        T_arr = np.array([o["opt_tokens"] for o in optima])
        N_arr = np.array([o["opt_params_no_lm"] for o in optima])

        # Fit power laws
        a_t, b_t = np.polyfit(np.log10(C_arr), np.log10(T_arr), 1)
        a_n, b_n = np.polyfit(np.log10(C_arr), np.log10(N_arr), 1)
        C_grid = np.logspace(np.log10(C_arr.min()) - 0.3, np.log10(C_arr.max()) + 0.5, 200)

        # Left: N*(C)
        ax3a.plot(C_arr, N_arr, "s", color="C0", markersize=10)
        ax3a.plot(
            C_grid,
            10 ** (a_n * np.log10(C_grid) + b_n),
            "--",
            color="C0",
            alpha=0.5,
            label=f"N* = {10**b_n:.2e}·C^{{{a_n:.3f}}}",
        )
        for o in optima:
            ax3a.annotate(
                f"{o['budget']:.0e}",
                (o["budget"], o["opt_params_no_lm"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        ax3a.set_xscale("log")
        ax3a.set_yscale("log")
        ax3a.set_xlabel("Compute (FLOPs, excl lm_head)")
        ax3a.set_ylabel("Optimal Active Params (no lm_head)")
        ax3a.set_title("N*(C)")
        ax3a.legend()
        ax3a.grid(True, alpha=0.3)

        # Right: T*(C)
        ax3b.plot(C_arr, T_arr, "o", color="C1", markersize=10)
        ax3b.plot(
            C_grid,
            10 ** (a_t * np.log10(C_grid) + b_t),
            "--",
            color="C1",
            alpha=0.5,
            label=f"T* = {10**b_t:.2e}·C^{{{a_t:.3f}}}",
        )
        for o in optima:
            ax3b.annotate(
                f"{o['budget']:.0e}",
                (o["budget"], o["opt_tokens"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        ax3b.set_xscale("log")
        ax3b.set_yscale("log")
        ax3b.set_xlabel("Compute (FLOPs, excl lm_head)")
        ax3b.set_ylabel("Optimal Tokens")
        ax3b.set_title("T*(C)")
        ax3b.legend()
        ax3b.grid(True, alpha=0.3)

        fig3.suptitle(f"Scaling Laws (exponent sum = {a_n + a_t:.3f})", fontsize=13)
        fig3.tight_layout()
        fig3.savefig(f"{PLOT_DIR}/isoflop_nt_scaling.png", dpi=150)
        plt.close(fig3)
        print(f"T*(C) = {10**b_t:.2e} · C^{a_t:.3f}")
        print(f"N*(C) = {10**b_n:.2e} · C^{a_n:.3f}")
        print("Saved isoflop_nt_scaling.png")

    # ---- Print summary table ----
    print("\nOptima:")
    print(f"{'Budget':>10s}  {'Opt Params':>12s}  {'Opt BPB':>10s}  {'Opt Tokens':>12s}  {'Opt Macro':>10s}")
    print("-" * 60)
    for o, m in zip(optima, macro_optima, strict=True):
        row = (
            f"{o['budget']:>10.0e}  {o['opt_params']:>12.2e}  "
            f"{o['opt_bpb']:>10.4f}  {o['opt_tokens']:>12.2e}  {m['opt_macro']:>10.4f}"
        )
        print(row)


if __name__ == "__main__":
    main()
