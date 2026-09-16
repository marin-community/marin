# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl

    return go, mo, np, pl


@app.cell
def _():
    PALETTE = [
        "#1877F2",
        "#F0701A",
        "#5A24C7",
        "#E42C97",
        "#00487C",
        "#0EAC96",
        "#AB76FF",
        "#B50550",
        "#0099E6",
        "#22085F",
        "#783301",
    ]
    PLOTLY_TEMPLATE = "plotly_white"

    def style_fig(fig, title=None, legend_below=True):
        _layout = dict(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=60, r=30, t=70, b=100 if legend_below else 60),
        )
        if title is not None:
            _layout["title"] = dict(
                text=title,
                font=dict(size=18),
                x=0.5,
                xanchor="center",
            )
        if legend_below:
            _layout["legend"] = dict(
                font=dict(size=11),
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(200,200,200,0.5)",
                borderwidth=1,
            )
        fig.update_layout(**_layout)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(220,220,220,0.5)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(220,220,220,0.5)")
        return fig

    return PALETTE, style_fig


@app.cell
def _(mo):
    mo.md(
        r"""
    # Aggregate capability via non-negative k-factor analysis on per-task bpb

    For continuous outcomes (bpb), classical IRT becomes a k-factor model:

    $$ z_{r,t} = \sum_{j=1}^{k} \lambda_{t,j} \theta_{r,j} + \varepsilon_t,
    \quad \theta_{r,\cdot} \sim \mathcal{N}(0, I_k),\; \varepsilon_t \sim \mathcal{N}(0, \psi_t),\;
    \lambda_{t,j} \ge 0 $$

    - **Inputs:** every `/bpb` column under `lm_eval/`, `mcq_smooth/`, `teacher_forced/` (per-subject MMLU dropped, 5-shot aggregates kept). All `eval/*` (paloma, uncheatable_eval) **excluded** from the IRT input so they can serve as held-out validation.
    - **Sign + scale:** bpb negated so higher = better, then z-scored per task.
    - **k chosen by Horn's parallel analysis** in the cell above.
    - **Non-negativity** on $\Lambda$ encodes the prior that all tasks measure capability in the same direction. Fit by projected-EM (~25 lines below).
    - **Outputs:** $\theta_{r,j}$ — run-level scores on each latent factor; $\lambda_{t,j}$ — how strongly task $t$ loads on factor $j$; $h^2_t$ — communality, the share of task $t$'s variance explained by all $k$ factors combined.
    """
    )
    return


@app.cell
def _(np, pl):

    raw_fa = pl.read_csv("raw_metric_matrix_300m.csv", infer_schema_length=500).filter(pl.col("status") == "completed")
    _MMLU_KEEP = {
        "lm_eval/mmlu_5shot/bpb",
        "lm_eval/mmlu_sl_verb_5shot/bpb",
    }
    _AGG_DROP = {
        "eval/bpb",
        "eval/macro_bpb",
        "eval/paloma/bpb",
        "eval/paloma/macro_bpb",
        "eval/uncheatable_eval/bpb",
        "eval/uncheatable_eval/macro_bpb",
    }
    _TASK_DROP = {
        "teacher_forced/gsm8k_5shot_answer_hash/bpb",
        "mcq_smooth/sciq_5shot/bpb",
    }

    def _keep(c):
        if not c.endswith("/bpb"):
            return False
        if c in _AGG_DROP or c in _TASK_DROP:
            return False
        if not c.startswith(
            (
                "eval/paloma/",
                "eval/uncheatable_eval/",
                "lm_eval/",
                "mcq_smooth/",
                "teacher_forced/",
            )
        ):
            return False
        if c.startswith("lm_eval/mmlu_") and c not in _MMLU_KEEP:
            return False
        return True

    _all = [c for c in raw_fa.columns if _keep(c)]
    _X_bpb = raw_fa.select(_all).to_numpy().astype(np.float64)
    _X_signed = -_X_bpb
    _stds = _X_signed.std(axis=0)
    _mask = _stds > 1e-12
    task_cols = [c for c, k in zip(_all, _mask) if k]
    _Xk = _X_signed[:, _mask]
    Z = (_Xk - _Xk.mean(axis=0)) / _Xk.std(axis=0)
    return Z, raw_fa, task_cols


@app.cell
def _(Z, go, np):
    _n, _p = Z.shape
    _real = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.T)))[::-1]
    _rng = np.random.default_rng(42)
    _N_MC = 500
    _rand = np.empty((_N_MC, _p))
    for _i in range(_N_MC):
        _Zr = _rng.standard_normal((_n, _p))
        _Zr = (_Zr - _Zr.mean(axis=0)) / _Zr.std(axis=0)
        _rand[_i] = np.sort(np.linalg.eigvalsh(np.corrcoef(_Zr.T)))[::-1]
    _p95 = np.percentile(_rand, 95, axis=0)
    _mean = _rand.mean(axis=0)
    k_horn = int(np.sum(_real > _p95))
    _k_mean = int(np.sum(_real > _mean))
    _ranks = np.arange(1, _p + 1)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_ranks, y=_real, name="real eigenvalues", mode="lines+markers", marker_color="#4C78A8"))
    _fig.add_trace(
        go.Scatter(
            x=_ranks,
            y=_p95,
            name="random 95th pct",
            mode="lines+markers",
            marker_color="#E45756",
            line=dict(dash="dash"),
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_ranks, y=_mean, name="random mean", mode="lines+markers", marker_color="#888", line=dict(dash="dot")
        )
    )
    _fig.add_vline(
        x=k_horn + 0.5,
        line_dash="dot",
        line_color="black",
        annotation_text=f"Horn k={k_horn}",
        annotation_position="top",
    )
    _fig.update_layout(
        title=(
            f"Horn's parallel analysis: k = {k_horn} (95th pct cutoff), "
            f"{_k_mean} (mean cutoff). p = {_p} tasks, n = {_n} runs, "
            f"{_N_MC} Monte Carlo draws."
        ),
        xaxis_title="component rank",
        yaxis_title="eigenvalue of correlation matrix",
        height=450,
        width=950,
    )
    _fig
    return (k_horn,)


@app.cell
def _(Z, k_horn, noise_share, np):
    n, p = Z.shape
    K = k_horn
    psi_anchor = np.where(np.isnan(noise_share), np.nan, np.clip(noise_share, 1e-3, 0.999))
    psi_fixed = ~np.isnan(psi_anchor)
    rng = np.random.default_rng(0)
    lam_mat = np.abs(rng.normal(scale=0.1, size=(p, K)))
    psi = np.where(psi_fixed, psi_anchor, 1.0)
    for _ in range(5000):
        _lam_psi_inv = lam_mat / psi[:, None]
        _v_post = np.linalg.inv(np.eye(K) + lam_mat.T @ _lam_psi_inv)
        _theta_hat = Z @ _lam_psi_inv @ _v_post
        _s_thth = n * _v_post + _theta_hat.T @ _theta_hat
        _zt_th = Z.T @ _theta_hat
        _lam_new = _zt_th @ np.linalg.inv(_s_thth)
        _lam_new = np.clip(_lam_new, 0.0, None)
        _psi_free = (
            (Z**2).mean(axis=0)
            - 2 * (_zt_th * _lam_new).sum(axis=1) / n
            + ((_lam_new @ _s_thth) * _lam_new).sum(axis=1) / n
        )
        _psi_free = np.clip(_psi_free, 1e-6, None)
        _psi_new = np.where(psi_fixed, psi_anchor, _psi_free)
        if np.max(np.abs(_lam_new - lam_mat)) < 1e-7:
            lam_mat, psi = _lam_new, _psi_new
            break
        lam_mat, psi = _lam_new, _psi_new
    _order = np.argsort(-(lam_mat**2).sum(axis=0))
    lam_mat = lam_mat[:, _order]
    _lam_psi_inv = lam_mat / psi[:, None]
    _v_post = np.linalg.inv(np.eye(K) + lam_mat.T @ _lam_psi_inv)
    theta_mat = Z @ _lam_psi_inv @ _v_post
    communality = (lam_mat**2).sum(axis=1) / ((lam_mat**2).sum(axis=1) + psi)
    return communality, lam_mat, psi, theta_mat


@app.cell
def _(communality, go, lam_mat, np, task_cols):
    _dom = lam_mat.argmax(axis=1)
    _order = np.lexsort((-lam_mat.max(axis=1), _dom))
    _labels = [f"{task_cols[i].removesuffix('/bpb')}  (h²={communality[i]:.2f})" for i in _order]
    _Z = lam_mat[_order]
    _fig = go.Figure(
        data=go.Heatmap(
            z=_Z,
            x=[f"factor {k + 1}" for k in range(lam_mat.shape[1])],
            y=_labels,
            colorscale="Viridis",
            zmin=0.0,
            colorbar=dict(title="λ"),
            hovertemplate="%{y}<br>%{x}: λ=%{z:.3f}<extra></extra>",
        )
    )
    _fig.update_layout(
        title=(
            "Loadings (non-negative 3-factor IRT). Tasks grouped by their "
            "dominant factor, sorted by max λ within group. h² = communality."
        ),
        height=max(500, 26 * len(_labels)),
        width=900,
        margin=dict(l=350),
        yaxis=dict(tickfont=dict(size=10)),
    )
    _fig
    return


@app.cell
def _(np, pl, task_cols):
    noise_df = pl.read_csv("noise_baseline_run00097_300m.csv", infer_schema_length=200)
    sweep_df = pl.read_csv("raw_metric_matrix_300m.csv", infer_schema_length=500).filter(pl.col("status") == "completed")
    _noise_cols = set(noise_df.columns)
    _has = np.array([c in _noise_cols for c in task_cols])
    _present = [c for c in task_cols if c in _noise_cols]
    _noise_X = noise_df.select(_present).to_numpy().astype(np.float64)
    _sweep_X = sweep_df.select(_present).to_numpy().astype(np.float64)
    _ns_p = _noise_X.std(axis=0, ddof=1)
    _ss_p = _sweep_X.std(axis=0, ddof=1)
    noise_sd = np.full(len(task_cols), np.nan)
    sweep_sd = np.full(len(task_cols), np.nan)
    noise_sd[_has] = _ns_p
    sweep_sd[_has] = _ss_p
    noise_share = (noise_sd / sweep_sd) ** 2
    h2_ceiling = np.where(np.isnan(noise_share), np.nan, np.clip(1.0 - noise_share, 0.0, 1.0))
    return h2_ceiling, noise_share


@app.cell
def _(communality, go, h2_ceiling, np, task_cols):
    _has_ceil = ~np.isnan(h2_ceiling)
    _idx = np.where(_has_ceil)[0]
    _idx = _idx[np.argsort(-h2_ceiling[_idx])]
    _names = [task_cols[i].removesuffix("/bpb") for i in _idx]
    _ceil = h2_ceiling[_idx]
    _act = np.minimum(communality[_idx], _ceil)
    _gap = _ceil - _act
    _over = np.maximum(communality[_idx] - _ceil, 0.0)
    _fig = go.Figure()
    _fig.add_trace(
        go.Bar(
            x=_act,
            y=_names,
            orientation="h",
            name="h² (factor model, capped at ceiling)",
            marker_color="#4C78A8",
        )
    )
    _fig.add_trace(
        go.Bar(
            x=_gap,
            y=_names,
            orientation="h",
            name="ceiling − h² (room left)",
            marker_color="#E0E0E0",
        )
    )
    if (_over > 0).any():
        _fig.add_trace(
            go.Bar(
                x=_over,
                y=_names,
                orientation="h",
                name="h² > ceiling (overfit / noise estimate too low)",
                marker_color="#E45756",
            )
        )
    _fig.add_trace(
        go.Scatter(
            x=_ceil,
            y=_names,
            mode="markers",
            marker=dict(symbol="line-ns-open", size=14, color="#000", line=dict(width=2)),
            name="h² ceiling = 1 − Var_noise / Var_sweep",
        )
    )
    _missing = [task_cols[i].removesuffix("/bpb") for i in range(len(task_cols)) if not _has_ceil[i]]
    _title = (
        "h² achieved vs noise-floor ceiling per task. "
        "Ceiling = 1 − Var(noise baseline, n=10 same-mix runs) / Var(241-run sweep). "
        f"Tasks missing from noise baseline (no ceiling): {', '.join(_missing) if _missing else 'none'}."
    )
    _fig.update_layout(
        barmode="stack",
        title=_title,
        xaxis=dict(title="share of sweep variance", range=[0, 1.05]),
        height=max(450, 26 * len(_names)),
        width=1200,
        margin=dict(l=320),
    )
    _fig
    return


@app.cell
def _(go, np, raw_fa, theta_mat):
    from plotly.subplots import make_subplots as _msp

    _x = raw_fa["eval/uncheatable_eval/macro_loss"].to_numpy()
    _names = raw_fa["run_name"].to_list()
    _is_baseline = np.array(["baseline" in n for n in _names])
    _K = theta_mat.shape[1]
    _rs = [float(np.corrcoef(_x, theta_mat[:, k])[0, 1]) for k in range(_K)]
    _fig = _msp(
        rows=1,
        cols=_K,
        subplot_titles=[f"factor {k + 1}: r = {_rs[k]:+.3f}" for k in range(_K)],
        horizontal_spacing=0.08,
    )
    for _k in range(_K):
        _y = theta_mat[:, _k]
        _fig.add_trace(
            go.Scatter(
                x=_x[~_is_baseline],
                y=_y[~_is_baseline],
                mode="markers",
                text=[n for n, b in zip(_names, _is_baseline) if not b],
                name="sweep",
                showlegend=(_k == 0),
                marker=dict(size=5, color="#4C78A8", opacity=0.65),
            ),
            row=1,
            col=_k + 1,
        )
        _fig.add_trace(
            go.Scatter(
                x=_x[_is_baseline],
                y=_y[_is_baseline],
                mode="markers",
                text=[n for n, b in zip(_names, _is_baseline) if b],
                name="baselines",
                showlegend=(_k == 0),
                marker=dict(size=11, color="#E45756", symbol="star"),
            ),
            row=1,
            col=_k + 1,
        )
        _fig.update_xaxes(title_text="uncheatable_eval/macro_loss", row=1, col=_k + 1)
        _fig.update_yaxes(title_text=f"θ_{_k + 1}", row=1, col=_k + 1)
    _fig.update_layout(
        title="Internal consistency: each factor vs uncheatable_eval/macro_loss (per-split bpbs are now in the IRT input; macro is approximately a weighted sum, so high r is expected)",
        height=450,
        width=1400,
    )
    _fig
    return


@app.cell
def _(go, np, raw_fa, theta_mat):
    from plotly.subplots import make_subplots as _msp

    _x = raw_fa["eval/paloma/macro_loss"].to_numpy()
    _names = raw_fa["run_name"].to_list()
    _is_baseline = np.array(["baseline" in n for n in _names])
    _K = theta_mat.shape[1]
    _rs = [float(np.corrcoef(_x, theta_mat[:, k])[0, 1]) for k in range(_K)]
    _fig = _msp(
        rows=1,
        cols=_K,
        subplot_titles=[f"factor {k + 1}: r = {_rs[k]:+.3f}" for k in range(_K)],
        horizontal_spacing=0.08,
    )
    for _k in range(_K):
        _y = theta_mat[:, _k]
        _fig.add_trace(
            go.Scatter(
                x=_x[~_is_baseline],
                y=_y[~_is_baseline],
                mode="markers",
                text=[n for n, b in zip(_names, _is_baseline) if not b],
                name="sweep",
                showlegend=(_k == 0),
                marker=dict(size=5, color="#4C78A8", opacity=0.65),
            ),
            row=1,
            col=_k + 1,
        )
        _fig.add_trace(
            go.Scatter(
                x=_x[_is_baseline],
                y=_y[_is_baseline],
                mode="markers",
                text=[n for n, b in zip(_names, _is_baseline) if b],
                name="baselines",
                showlegend=(_k == 0),
                marker=dict(size=11, color="#E45756", symbol="star"),
            ),
            row=1,
            col=_k + 1,
        )
        _fig.update_xaxes(title_text="paloma/macro_loss", row=1, col=_k + 1)
        _fig.update_yaxes(title_text=f"θ_{_k + 1}", row=1, col=_k + 1)
    _fig.update_layout(
        title="Internal consistency: each factor vs paloma/macro_loss (per-domain bpbs are now in the IRT input; macro is approximately a weighted sum, so high r is expected)",
        height=450,
        width=1400,
    )
    _fig
    return


@app.cell
def _(go, lam_mat, np, pl, psi, raw_fa, task_cols, theta_mat):
    _X_signed = -raw_fa.select(task_cols).to_numpy().astype(np.float64)
    _mu = _X_signed.mean(axis=0)
    _sd = _X_signed.std(axis=0)
    _noise_df = pl.read_csv("noise_baseline_run00097_300m.csv", infer_schema_length=200)
    _ncols = set(_noise_df.columns)
    _has = np.array([c in _ncols for c in task_cols])
    _present = [c for c in task_cols if c in _ncols]
    _nX = np.zeros((_noise_df.height, len(task_cols)))
    _nX[:, _has] = -_noise_df.select(_present).to_numpy().astype(np.float64)
    _nX[:, _has] = (_nX[:, _has] - _mu[_has]) / _sd[_has]

    _K = lam_mat.shape[1]
    _proj = (lam_mat / psi[:, None]) @ np.linalg.inv(np.eye(_K) + lam_mat.T @ (lam_mat / psi[:, None]))
    _theta_noise = _nX @ _proj

    _agg_sweep = theta_mat.mean(axis=1)
    _agg_noise = _theta_noise.mean(axis=1)
    _vs = float(_agg_sweep.var(ddof=1))
    _vn = float(_agg_noise.var(ddof=1))
    _per_factor_snr = []
    for _k in range(_K):
        _vsk = float(theta_mat[:, _k].var(ddof=1))
        _vnk = float(_theta_noise[:, _k].var(ddof=1))
        _per_factor_snr.append((_vsk, _vnk))

    _names = [f"factor {k + 1}" for k in range(_K)] + ["aggregate (mean)"]
    _snrs = [(s - n) / n for s, n in _per_factor_snr] + [(_vs - _vn) / _vn]
    _shares = [n / s for s, n in _per_factor_snr] + [_vn / _vs]

    _fig = go.Figure()
    _fig.add_trace(
        go.Bar(
            x=_snrs,
            y=_names,
            orientation="h",
            text=[f"{s:.0f}× (noise={p:.1%})" for s, p in zip(_snrs, _shares)],
            textposition="outside",
            marker_color=["#4C78A8"] * _K + ["#54A24B"],
        )
    )
    _fig.update_layout(
        title=(
            f"Signal-to-noise ratio per factor and on the uniform-mean aggregate. "
            f"Noise variance from {_noise_df.height} same-mix seed runs in "
            f"noise_baseline_run00097_300m.csv (n=10 → 95% CI ≈ [0.5×, 3.3×] of point estimate). "
            f"Missing tasks (28/{len(task_cols)}) zeroed → noise is a lower bound."
        ),
        xaxis=dict(title="(Var_sweep − Var_noise) / Var_noise", type="log"),
        height=80 + 35 * len(_names),
        width=1100,
        margin=dict(l=120),
    )
    _fig
    return


@app.cell
def _(pl, raw_fa, theta_mat):
    aggregate = theta_mat.mean(axis=1)
    leaderboard = (
        raw_fa.select(
            [
                "run_name",
                "is_qsplit240_core",
                "eval/paloma/macro_loss",
                "eval/uncheatable_eval/macro_loss",
            ]
        )
        .with_columns(
            [pl.Series("aggregate", aggregate)]
            + [pl.Series(f"theta_{k + 1}", theta_mat[:, k]) for k in range(theta_mat.shape[1])]
        )
        .sort("aggregate", descending=True)
    )
    leaderboard
    return


if __name__ == "__main__":
    app.run()
