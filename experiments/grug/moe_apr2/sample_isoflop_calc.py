# ============================================================
# 2) Fit parabolas + compute optima per (quantizer, gflops bucket)
# ============================================================
print(df)
gflop_groups = sorted(df["gflops_q"].unique())
g2c = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(gflop_groups)}

unique_quantizers = sorted((df["quantizer"]).unique().tolist())

fits = {}
optima_rows = []

max_log_star_T = 0

for q in unique_quantizers:
    for g in gflop_groups:
        sub = df[(df["quantizer"] == q) & (df["gflops_q"] == g)].copy()
        if len(sub) < 3:
            continue

        f = fit_parabola(sub)
        fits[(q, g)] = f

        optima_rows.append({
            "quantizer": q,
            "gflops_bucket": g,
            "C_flops": g * 1e9,   # GFLOPs → FLOPs
            "T_star": f.T_star,
            "P_star": f.P_star,
            "loss_star": f.loss_star,
            "quad_a": f.a,
            "quad_b": f.b,
            "quad_c": f.c,
        })

optima = (
    pd.DataFrame(optima_rows)
    .sort_values(["quantizer", "gflops_bucket"])
    .reset_index(drop=True)
)

# ============================================================
# 3) Chinchilla Approach #2 regression (log-log)
# ============================================================

reg = pd.DataFrame([
    chinchilla_approach2(optima, q) for q in unique_quantizers
])

print("\n=== Chinchilla Approach #2 (log-log regression) ===")
print(reg)

# ------------------------------------------------------------
# Add regression-predicted optimal tokens
# log10(T*) = alpha + beta * log10(C)
# ------------------------------------------------------------

reg_map = reg.set_index("quantizer").to_dict(orient="index")

optima = optima.copy()
optima["logC"] = np.log10(optima["C_flops"].astype(float))
optima["logT_hat"] = optima.apply(
    lambda r: reg_map[r["quantizer"]]["alphaT"]
            + reg_map[r["quantizer"]]["betaT"] * r["logC"],
    axis=1,
)
optima["T_hat"] = 10 ** optima["logT_hat"]

def fit_optimal_frontier(optima: pd.DataFrame, quantizer: str, degree=1):
    sub = optima[optima["quantizer"] == quantizer].copy()
    #V = 128_000          # vocab size
    #L0 = np.log(V)       # uniform-prediction loss (nats)

    x = np.concatenate([
        #np.array([0.0]),                             # log10(T) at T=1 token (or conceptual zero)
        np.log10(sub["T_star"].to_numpy(float)),
    ])

    y = np.concatenate([
        #np.array([L0]),                              # loss ceiling
        sub["loss_star"].to_numpy(float),
    ])
    coeff = np.polyfit(x, y, degree)
    return coeff

frontier_coeffs = {
    q: fit_optimal_frontier(optima, q, degree=1)  # degree=2 if you have many buckets
    for q in unique_quantizers
}

# ============================================================
# 4) Plotly: scatter + parabolic fits + dashed regression line
# ============================================================
num_rows = (len(unique_quantizers)//3)+(1 if len(unique_quantizers)%3 != 0 else 0)
fig = make_subplots(
    rows=num_rows,
    cols=3,
    shared_yaxes=True,
    subplot_titles=unique_quantizers,
)

for col_idx, q in enumerate(unique_quantizers, start=1):
    row_idx = (col_idx - 1) // 3 + 1
    col_idx = (col_idx - 1) % 3 + 1
    # --------------------------------------------------------
    # Scatter + quadratic fits per compute bucket
    # --------------------------------------------------------
    for g in gflop_groups:
        sub = df[(df["quantizer"] == q) & (df["gflops_q"] == g)].copy()
        if len(sub) == 0:
            continue

        color = g2c[g]
        label = fmt_sci(g)

        # Scatter points
        fig.add_trace(
            go.Scatter(
                x=sub["tokens"],
                y=sub["loss"],
                mode="markers",
                marker=dict(size=7, color=color),
                name=label,
                legendgroup=label,
                customdata=sub["wandb_url"],
                showlegend = (row_idx == 1 and col_idx == 1),
                hovertemplate=(
                    "tokens=%{x:.3e}<br>"
                    "loss=%{y:.4f}<br>"
                    f"gflops_q={label}<br>"
                    f"quantizer={q}<br>"
                    "<a href='%{customdata}' target='_blank'>Open in W&amp;B</a>"
                    "<extra></extra>"
                ),
            ),
            row=row_idx,
            col=col_idx,
        )

        # Quadratic fit (log(tokens))
        key = (q, g)
        if key in fits:
            f = fits[key]
            logT = np.log10(sub["tokens"].to_numpy(float))
            grid = np.linspace(logT.min(), logT.max(), 300)

            yhat = f.a * grid**2 + f.b * grid + f.c
            xgrid = 10 ** grid

            fig.add_trace(
                go.Scatter(
                    x=xgrid,
                    y=yhat,
                    mode="lines",
                    line=dict(width=3, color=color),
                    name=label,
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=(
                        "parabolic fit<br>"
                        "tokens=%{x:.3e}<br>"
                        "loss=%{y:.4f}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=col_idx,
            )

            # Mark optimum
            fig.add_trace(
                go.Scatter(
                    x=[f.T_star],
                    y=[f.loss_star],
                    mode="markers",
                    marker=dict(symbol="x", size=12, color=color),
                    name=label,
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=(
                        "optimum<br>"
                        f"T*={f.T_star:.3e}<br>"
                        f"loss*={f.loss_star:.4f}<br>"
                        f"P*\u2248{f.P_star:.3e}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=col_idx,
            )

        coeff = frontier_coeffs[q]

        # # --------------------------------------------------------
        # # Mark predicted optimum at C = 1e23 FLOPs
        # # --------------------------------------------------------
        # C_target = 1e21
        # logC_target = np.log10(C_target)

        # # Predicted optimal tokens from Chinchilla #2
        # alphaT = reg_map[q]["alphaT"]
        # betaT  = reg_map[q]["betaT"]
        # logT_star_23 = alphaT + betaT * logC_target
        # if logT_star_23 > max_log_star_T:
        #     max_log_star_T = logT_star_23
        # T_star_23 = 10 ** logT_star_23


        # # Predicted optimal loss from frontier regression
        # coeff = frontier_coeffs[q]
        # loss_star_23 = np.polyval(coeff, logT_star_23)

        # fig.add_trace(
        #     go.Scatter(
        #         x=[T_star_23],
        #         y=[loss_star_23],
        #         mode="markers",
        #         marker=dict(
        #             symbol="x",
        #             size=12,
        #             color="rgba(140,140,140,0.9)",
        #         ),
        #         name=f"Predicted optimum @ {C_target} FLOPs",
        #         legendgroup="frontier",
        #         showlegend=False,
        #         hovertemplate=(
        #             f"Predicted optimum (C={C_target})<br>"
        #             "T*=%{x:.3e}<br>"
        #             "loss*=%{y:.4f}<extra></extra>"
        #         ),
        #     ),
        #     row=row_idx,
        #     col=col_idx,
        # )


        # logT_grid = np.linspace(9.0, logT_star_23, 300)
        # T_grid = 10 ** logT_grid
        # loss_grid = np.polyval(coeff, logT_grid)

        # fig.add_trace(
        #     go.Scatter(
        #         x=T_grid,
        #         y=loss_grid,
        #         mode="lines",
        #         line=dict(
        #             width=2,
        #             dash="dash",
        #             color="rgba(140,140,140,0.9)",
        #         ),
        #         name="Optimal frontier (regressed)",
        #         legendgroup="frontier",
        #         showlegend=False,
        #         hovertemplate=(
        #             "optimal frontier<br>"
        #             "T=%{x:.3e}<br>"
        #             "loss=%{y:.4f}<extra></extra>"
        #         ),
        #     ),
        #     row=row_idx,
        #     col=col_idx,
        # )



# ============================================================
# Layout
# ============================================================

fig.update_xaxes(type="log", title_text="Training tokens")#, range=[9, max_log_star_T*1.1])
fig.update_yaxes(title_text="Paloma macro loss", row=1, col=1)

fig.update_layout(
    template=PLOTLY_TEMPLATE,
    title="isoFLOP Side by Side AdamH v.s. AdamW",
    legend_title_text="FLOP Count",
    legend=dict(
        x=0.8,
        y=0.8,
        xanchor="right",
        yanchor="top",
    ),
    width=1100,
    height=300*(num_rows)+180,
)

fig.show()

# ============================================================
# 6) Add an asymptote term (per-optimizer) so gaps don't have to vanish
#     Fit on first 3 observed optima only:
#         loss*(C) = L_inf + A * C^{-alpha}
#     (grid-search L_inf, then linear regression in log-space)
#     Plot first 3 points as train, remaining points as test.
# ============================================================

import numpy as np
import plotly.graph_objects as go

marker_symbol = {
    "train": "circle",
    "test": "diamond",
}

def fit_asymptotic_powerlaw(C, loss, n_grid=400):
    """
    Fit: loss(C) = L_inf + A * C^{-alpha}, with alpha>0, A>0.
    We grid-search L_inf, then do linear regression on:
        log(loss - L_inf) = log A - alpha * log C

    Returns dict with L_inf, A, alpha, and diagnostics.
    """
    C = np.asarray(C, float)
    y = np.asarray(loss, float)

    # sort by compute
    idx = np.argsort(C)
    C = C[idx]
    y = y[idx]

    if len(C) < 2:
        return {
            "L_inf": 0.0,
            "A": np.nan,
            "alpha": np.nan,
            "sse_log": np.inf,
            "ok": False,
        }

    y_min = float(np.min(y))
    Linf_lo = 0
    Linf_hi = 0.95 * y_min

    if Linf_hi <= Linf_lo:
        return {
            "L_inf": 0.0,
            "A": np.nan,
            "alpha": np.nan,
            "sse_log": np.inf,
            "ok": False,
        }

    Linf_grid = np.linspace(Linf_lo, Linf_hi, n_grid)
    logC = np.log(C)

    best = None
    for Linf in Linf_grid:
        z = y - Linf
        if np.any(z <= 0):
            continue

        logz = np.log(z)

        # Fit logz = b0 + b1 * logC
        b1, b0 = np.polyfit(logC, logz, 1)
        alpha = -b1
        if not np.isfinite(alpha) or alpha <= 0:
            continue

        A = np.exp(b0)
        pred_logz = b0 + b1 * logC
        sse_log = float(np.mean((logz - pred_logz) ** 2))

        if (best is None) or (sse_log < best["sse_log"]):
            best = {
                "L_inf": float(Linf),
                "A": float(A),
                "alpha": float(alpha),
                "sse_log": sse_log,
                "ok": True,
            }

    if best is None:
        return {
            "L_inf": 0.0,
            "A": np.nan,
            "alpha": np.nan,
            "sse_log": np.inf,
            "ok": False,
        }

    return best


# Fit asymptotic scaling laws for each optimizer using only the first 3 points
q_order = sorted(optima["quantizer"].unique().tolist())
q_color = {q: PALETTE[i % len(PALETTE)] for i, q in enumerate(q_order)}

asym_fit = {}
train_points = {}
test_points = {}

for q in q_order:
    sub_all = (
        optima[optima["quantizer"] == q]
        .sort_values("C_flops")
        .copy()
    )

    sub_train = sub_all.iloc[:5].copy()
    sub_test = sub_all.iloc[6:].copy()

    train_points[q] = sub_train
    test_points[q] = sub_test

    asym_fit[q] = fit_asymptotic_powerlaw(
        sub_train["C_flops"].to_numpy(float),
        sub_train["loss_star"].to_numpy(float),
    )

print("=== Asymptotic fits on first 3 points: loss*(C) = L_inf + A*C^{-alpha} ===")
for q in q_order:
    d = asym_fit[q]
    if d["ok"]:
        print(
            f"{q:8s}  "
            f"L_inf={d['L_inf']:.4f}  "
            f"A={d['A']:.3e}  "
            f"alpha={d['alpha']:.3f}  "
            f"(log-MSE={d['sse_log']:.3e})"
        )
    else:
        print(f"{q:8s}  fit failed")


# Plot: train/test points + asymptotic fits
C_min = float(optima["C_flops"].min())
C_max = max(float(optima["C_flops"].max()), 1e23)
C_grid = np.logspace(np.log10(C_min), np.log10(C_max), 400)

fig3 = go.Figure()

for q in q_order:
    sub_train = train_points[q]
    sub_test = test_points[q]
    d = asym_fit[q]

    # training points = first 3 points used for fitting
    if len(sub_train) > 0:
        fig3.add_trace(
            go.Scatter(
                x=sub_train["C_flops"],
                y=sub_train["loss_star"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=q_color[q],
                    symbol=marker_symbol["train"],
                ),
                name=f"{q} (train: first 3 pts)",
                hovertemplate=(
                    f"{q} (train)<br>"
                    "C=%{x:.3e} FLOPs<br>"
                    "loss*=%{y:.4f}<extra></extra>"
                ),
            )
        )

    # test points = remaining points not used for fitting
    if len(sub_test) > 0:
        fig3.add_trace(
            go.Scatter(
                x=sub_test["C_flops"],
                y=sub_test["loss_star"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=q_color[q],
                    symbol=marker_symbol["test"],
                ),
                name=f"{q} (test: remaining pts)",
                hovertemplate=(
                    f"{q} (test)<br>"
                    "C=%{x:.3e} FLOPs<br>"
                    "loss*=%{y:.4f}<extra></extra>"
                ),
            )
        )

    # fitted curve from first 3 points
    if d["ok"]:
        yhat = d["L_inf"] + d["A"] * (C_grid ** (-d["alpha"]))
        fig3.add_trace(
            go.Scatter(
                x=C_grid,
                y=yhat,
                mode="lines",
                line=dict(width=3, color=q_color[q]),
                name=f"{q} (fit: first 3 pts)",
                hovertemplate=(
                    f"{q} (asymptotic fit)<br>"
                    "C=%{x:.3e} FLOPs<br>"
                    "loss=%{y:.4f}<extra></extra>"
                ),
            )
        )

fig3.update_xaxes(type="log", title_text="Compute (FLOPs)")
fig3.update_yaxes(title_text="Paloma macro loss (optimal)")

fig3.update_layout(
    template=PLOTLY_TEMPLATE,
    title="Scaling law comparison with asymptotes: loss*(C) = L∞ + A·C^{-α}",
    width=950,
    height=540,
    legend_title_text="Optimizer / role",
)

fig3.show()