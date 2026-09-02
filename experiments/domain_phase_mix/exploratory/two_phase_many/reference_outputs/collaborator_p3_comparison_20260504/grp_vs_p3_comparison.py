# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    GRP_CODE = "/tmp/scaling_packet/collaborator_scaling_data_packet_20260430/standalone_code"
    GRP_DATA = "/tmp/scaling_packet/collaborator_scaling_data_packet_20260430/data/grp_no_l2"
    if GRP_CODE not in sys.path:
        sys.path.insert(0, GRP_CODE)
    import grp_no_l2_exact as grp

    return (
        GRP_CODE,
        GRP_DATA,
        Path,
        go,
        grp,
        make_subplots,
        mo,
        np,
        pd,
        pl,
        shutil,
        sys,
        tempfile,
    )


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

    def style_fig(fig, title=None):
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=60, r=30, t=80, b=80),
            legend=dict(
                font=dict(size=11),
                orientation="h",
                yanchor="top",
                y=-0.10,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(200,200,200,0.5)",
                borderwidth=1,
            ),
        )
        if title is not None:
            fig.update_layout(title=dict(text=title, font=dict(size=17), x=0.5, xanchor="center"))
        fig.update_xaxes(showgrid=True, gridcolor="rgba(220,220,220,0.5)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(220,220,220,0.5)")
        return fig

    return PALETTE, style_fig


@app.cell
def _(mo):
    mo.md(
        """
    # GRP vs P3 — head-to-head optimal mixture comparison

    Both models are fit on the **same 241-run swarm** and the **same IRT aggregate** as their target. Then each model's optimizer is run on the simplex to produce its recommended mixture. Side-by-side bars show what each one ends up with, in weight space and epoch space.

    **GRP** — Calvin's full power-family-penalty form from `grp_no_l2_exact.py`: 9 nonlinear params (η, λ, β, per-family `a_f`, per-family `τ_f`, ridge α=0), NNLS linear head with 32 columns (singletons + CC pairs + family totals + per-family penalty groups). Refit here with the IRT aggregate as target; nonlinear params kept at the included `power_family_penalty_no_l2_retune_best` checkpoint, only the linear head is re-NNLS'd. Optimum found via L-BFGS on logits.

    **P3** — our simplified form from `grp_pipeline_300m.py`: 3 nonlinear params (η, a, p), 39 ridge signal coefs + 2 per-phase penalty coefs (γ_0, γ_1). Optimum is the **mean over 200 bootstrap simplex argmaxes via Frank-Wolfe** (Thompson sampling).
    """
    )
    return


@app.cell
def _(np, pl):
    raw = pl.read_csv("raw_metric_matrix_300m.csv", infer_schema_length=500).filter(pl.col("status") == "completed")
    noise_df = pl.read_csv("noise_baseline_run00097_300m.csv", infer_schema_length=200)
    meta = pl.read_csv("two_phase_many_epoch_metadata.csv")

    _MMLU_KEEP = {"lm_eval/mmlu_5shot/bpb", "lm_eval/mmlu_sl_verb_5shot/bpb"}
    _AGG_DROP = {
        "eval/bpb",
        "eval/macro_bpb",
        "eval/paloma/bpb",
        "eval/paloma/macro_bpb",
        "eval/uncheatable_eval/bpb",
        "eval/uncheatable_eval/macro_bpb",
    }
    _TASK_DROP = {"teacher_forced/gsm8k_5shot_answer_hash/bpb", "mcq_smooth/sciq_5shot/bpb"}

    def _keep(c):
        if not c.endswith("/bpb"):
            return False
        if c in _AGG_DROP or c in _TASK_DROP:
            return False
        if not c.startswith(("eval/uncheatable_eval/", "lm_eval/", "mcq_smooth/", "teacher_forced/")):
            return False
        if c.startswith("lm_eval/mmlu_") and c not in _MMLU_KEEP:
            return False
        return True

    _candidates = [c for c in raw.columns if _keep(c)]
    _raw_cols = set(raw.columns)
    task_cols, task_signs = [], []
    for _c in _candidates:
        _base = _c.removesuffix("/bpb")
        if _base.startswith(("lm_eval/", "mcq_smooth/")):
            _alt = _base + "/choice_logprob"
            if _alt in _raw_cols:
                task_cols.append(_alt)
                task_signs.append(+1.0)
                continue
        task_cols.append(_c)
        task_signs.append(-1.0)
    task_signs = np.array(task_signs)

    domains = sorted(c.removeprefix("phase_0_") for c in raw.columns if c.startswith("phase_0_"))
    _row = {r["domain_name"]: r for r in meta.iter_rows(named=True)}
    c0 = np.array([_row[d]["phase_0_epoch_multiplier"] for d in domains], dtype=np.float64)
    c1 = np.array([_row[d]["phase_1_epoch_multiplier"] for d in domains], dtype=np.float64)
    return c0, c1, domains, noise_df, raw, task_cols, task_signs


@app.cell
def _(noise_df, np, raw, task_cols, task_signs):
    X = raw.select(task_cols).to_numpy().astype(np.float64) * task_signs[None, :]
    swarm_mu = X.mean(axis=0)
    swarm_sd = X.std(axis=0)
    Z = (X - swarm_mu) / swarm_sd

    _noise_cols = set(noise_df.columns)
    _has_noise = np.array([c in _noise_cols for c in task_cols])
    _present = [c for c in task_cols if c in _noise_cols]
    _signs_present = task_signs[_has_noise]
    _nX = noise_df.select(_present).to_numpy().astype(np.float64) * _signs_present[None, :]
    noise_share = np.full(len(task_cols), np.nan)
    noise_share[_has_noise] = (_nX.std(axis=0, ddof=1) / X[:, _has_noise].std(axis=0, ddof=1)) ** 2
    return Z, noise_share


@app.cell
def _(Z, noise_share, np):
    _n, _p = Z.shape
    _real = np.sort(np.linalg.eigvalsh(np.corrcoef(Z.T)))[::-1]
    _rng = np.random.default_rng(42)
    _N_MC = 300
    _rand = np.empty((_N_MC, _p))
    for _i in range(_N_MC):
        _Zr = _rng.standard_normal((_n, _p))
        _rand[_i] = np.sort(np.linalg.eigvalsh(np.corrcoef(_Zr.T)))[::-1]
    _q95 = np.percentile(_rand, 95, axis=0)
    k_horn = max(1, int((_real > _q95).sum()))

    _psi_anchor = np.where(np.isnan(noise_share), np.nan, np.clip(noise_share, 1e-3, 0.999))
    _psi_fixed = ~np.isnan(_psi_anchor)
    _rng2 = np.random.default_rng(0)
    Lam = np.abs(_rng2.normal(scale=0.1, size=(_p, k_horn)))
    Psi = np.where(_psi_fixed, _psi_anchor, 1.0)
    for _ in range(3000):
        _Lpi = Lam / Psi[:, None]
        _V = np.linalg.inv(np.eye(k_horn) + Lam.T @ _Lpi)
        _Th = Z @ _Lpi @ _V
        _S = _n * _V + _Th.T @ _Th
        _ZtT = Z.T @ _Th
        _Ln = _ZtT @ np.linalg.inv(_S)
        _Ln = np.clip(_Ln, 0.0, None)
        _Pf = (Z**2).mean(0) - 2 * (_ZtT * _Ln).sum(1) / _n + ((_Ln @ _S) * _Ln).sum(1) / _n
        _Pf = np.clip(_Pf, 1e-6, None)
        _Pn = np.where(_psi_fixed, _psi_anchor, _Pf)
        if np.max(np.abs(_Ln - Lam)) < 1e-7:
            Lam, Psi = _Ln, _Pn
            break
        Lam, Psi = _Ln, _Pn
    _Lpi = Lam / Psi[:, None]
    _V = np.linalg.inv(np.eye(k_horn) + Lam.T @ _Lpi)
    proj = (_Lpi @ _V).mean(axis=1)
    return (proj,)


@app.cell
def _(Z, proj):
    agg = Z @ proj
    return (agg,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Step A — Run GRP, refit on IRT aggregate

    GRP's `optimize_model` minimizes the predicted target. Our IRT aggregate is "higher is better," so we feed `−agg` as the target — the resulting optimum maximizes IRT.

    Nonlinear params come from Calvin's `power_family_penalty_no_l2_retune_best` checkpoint (the one fit on `eval/uncheatable_eval/bpb`); the NNLS linear head is refit fresh on our IRT-aggregate target. Full Powell retune of nonlinear params would be more principled but takes longer and the swarm + structure are the same so the linear refit alone gives an apples-ish comparison.
    """
    )
    return


@app.cell
def _(GRP_DATA, Path, agg, c0, c1, domains, grp, np, pd, raw, tempfile):
    _raw_pd = raw.to_pandas()
    _agg_by_run = dict(zip(_raw_pd["run_name"], agg.tolist()))

    _grp_csv = pd.read_csv(Path(GRP_DATA) / "two_phase_many.csv")
    _grp_csv["irt_neg"] = _grp_csv["run_name"].map(lambda n: -_agg_by_run.get(n, np.nan))

    tmpdir = Path(tempfile.mkdtemp(prefix="grp_irt_"))
    _grp_csv.to_csv(tmpdir / "two_phase_many.csv", index=False)
    _meta_path = Path(GRP_DATA) / "two_phase_many_epoch_metadata.csv"
    pd.read_csv(_meta_path).to_csv(tmpdir / "two_phase_many_epoch_metadata.csv", index=False)
    for _f in ("grp_power_family_penalty_no_l2_retune_best.csv", "grp_penalty_calibration_variants_best.csv"):
        pd.read_csv(Path(GRP_DATA) / _f).to_csv(tmpdir / _f, index=False)

    packet_grp = grp.load_packet(tmpdir, target="irt_neg")
    _params = grp.included_no_l2_best_params(tmpdir)
    model_grp = grp.build_model(packet_grp, _params)
    model_grp.fit(packet_grp.base.w, packet_grp.base.y)

    _preds = model_grp.predict(packet_grp.base.w)
    _y = packet_grp.base.y
    grp_in_r2 = float(1 - ((_y - _preds) ** 2).sum() / ((_y - _y.mean()) ** 2).sum())

    _, p0_grp, p1_grp = grp.optimize_model(packet_grp, model_grp, n_random=5, seed=0)

    _grp_domains = packet_grp.base.domain_names
    grp_to_local = np.array([_grp_domains.index(d) for d in domains])
    p0_grp_aligned = p0_grp[grp_to_local]
    p1_grp_aligned = p1_grp[grp_to_local]
    grp_eps_p0 = p0_grp_aligned * c0
    grp_eps_p1 = p1_grp_aligned * c1
    return (
        grp_eps_p0,
        grp_eps_p1,
        grp_in_r2,
        model_grp,
        p0_grp_aligned,
        p1_grp_aligned,
        packet_grp,
        tmpdir,
    )


@app.cell
def _(grp_in_r2, mo):
    mo.md(
        f"**GRP refit on IRT aggregate: in-sample R² = `{grp_in_r2:.4f}`** "
        f"(NNLS linear head, included nonlinear params)."
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Step B — P3 with Thompson sampling on the simplex

    Same fit as the main `grp_pipeline_300m.py` notebook, replicated here so the comparison is self-contained: 3 nonlinear params (η, a, p) tuned via nested 5×5 CV from `(η ∈ {0.5, 1, 2, 5, 10, 15, 20})`, `(a ∈ linspace(0.5, 2.0, 8))`, `(p ∈ {0.5, 1, 1.5, 2, 2.5, 3, 4})`; ridge α from `(1, 3.16, 10, 31.6, ..., 10000)`; 39 signal coefs + 2 per-phase penalty coefs.
    """
    )
    return


@app.cell
def _(agg, c0, c1, domains, np, raw):
    _w0 = raw.select([f"phase_0_{d}" for d in domains]).to_numpy().astype(np.float64)
    _w1 = raw.select([f"phase_1_{d}" for d in domains]).to_numpy().astype(np.float64)
    w0_swarm = _w0 / _w0.sum(axis=1, keepdims=True)
    w1_swarm = _w1 / _w1.sum(axis=1, keepdims=True)
    EPS = 1e-4

    def build_p3_design(w0, w1, eta, a, p):
        E = np.maximum(w0 + eta * w1, EPS)
        sig = np.power(E, a)
        ne0 = np.maximum(w0 * c0[None, :], EPS)
        ne1 = np.maximum(w1 * c1[None, :], EPS)
        pen0 = np.power(ne0, p).sum(axis=1, keepdims=True)
        pen1 = np.power(ne1, p).sum(axis=1, keepdims=True)
        return np.column_stack([sig, -pen0, -pen1])

    def _ridge_predict(M_train, y_train, M_test, alpha):
        _msk = M_train.std(0) > 1e-10
        M_train = M_train[:, _msk]
        M_test = M_test[:, _msk]
        _mu = M_train.mean(0)
        _sd = M_train.std(0)
        _Ms_tr = (M_train - _mu) / _sd
        _Ms_te = (M_test - _mu) / _sd
        _yc = y_train - y_train.mean()
        _bb = np.linalg.solve(_Ms_tr.T @ _Ms_tr + alpha * np.eye(_Ms_tr.shape[1]), _Ms_tr.T @ _yc)
        return _Ms_te @ _bb + y_train.mean()

    def _ridge_inner_cv(M, y, fold_seed, alphas):
        _msk = M.std(0) > 1e-10
        if not _msk.any():
            return -np.inf, None
        M = M[:, _msk]
        _mu = M.mean(0)
        _sd = M.std(0)
        _Ms = (M - _mu) / _sd
        _p = _Ms.shape[1]
        _rng = np.random.default_rng(fold_seed)
        _idx = _rng.permutation(len(y))
        _folds = np.array_split(_idx, 5)
        _best, _best_a = -np.inf, None
        for _alpha in alphas:
            _preds = np.zeros(len(y))
            for _f in range(5):
                _t = _folds[_f]
                _tr = np.concatenate([_folds[i] for i in range(5) if i != _f])
                _Mt = _Ms[_tr]
                _yt = y[_tr] - y[_tr].mean()
                _bb = np.linalg.solve(_Mt.T @ _Mt + _alpha * np.eye(_p), _Mt.T @ _yt)
                _preds[_t] = _Ms[_t] @ _bb + y[_tr].mean()
            _r2 = 1 - ((y - _preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            if _r2 > _best:
                _best, _best_a = _r2, _alpha
        return _best, _best_a

    ETA_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
    A_GRID = tuple(np.linspace(0.5, 2.0, 8))
    P_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    ALPHA_GRID = tuple(np.logspace(0, 4, 9))

    _best, _combo = -np.inf, None
    for _eta in ETA_GRID:
        for _a in A_GRID:
            for _p in P_GRID:
                _M = build_p3_design(w0_swarm, w1_swarm, _eta, _a, _p)
                _r2, _alpha = _ridge_inner_cv(_M, agg, fold_seed=42, alphas=ALPHA_GRID)
                if _r2 > _best:
                    _best, _combo = _r2, (_eta, _a, _p, _alpha)
    eta_p3, a_p3, p_p3, alpha_p3 = _combo

    _D_full = build_p3_design(w0_swarm, w1_swarm, eta_p3, a_p3, p_p3)
    _msk = _D_full.std(0) > 1e-10
    _D = _D_full[:, _msk]
    _mu = _D.mean(0)
    _sd = _D.std(0)
    _Ds = (_D - _mu) / _sd
    _yc = agg - agg.mean()
    coef_std = np.linalg.solve(_Ds.T @ _Ds + alpha_p3 * np.eye(_Ds.shape[1]), _Ds.T @ _yc)
    p3_in_r2 = float(1 - ((agg - (_Ds @ coef_std + agg.mean())) ** 2).sum() / ((agg - agg.mean()) ** 2).sum())
    p3_design_mu = _mu
    p3_design_sd = _sd
    return (
        a_p3,
        alpha_p3,
        build_p3_design,
        coef_std,
        eta_p3,
        p3_design_mu,
        p3_design_sd,
        p3_in_r2,
        p_p3,
        w0_swarm,
        w1_swarm,
    )


@app.cell
def _(
    a_p3,
    agg,
    alpha_p3,
    build_p3_design,
    coef_std,
    eta_p3,
    p3_design_mu,
    p3_design_sd,
    p_p3,
    w0_swarm,
    w1_swarm,
    np,
):
    _D_full = build_p3_design(w0_swarm, w1_swarm, eta_p3, a_p3, p_p3)
    _msk = _D_full.std(0) > 1e-10
    _D = _D_full[:, _msk]
    _mu = _D.mean(0)
    _sd = _D.std(0)
    _Ds = (_D - _mu) / _sd

    N_BOOT = 200
    _rng = np.random.default_rng(7)
    _p = _Ds.shape[1]
    boot_coefs = np.zeros((N_BOOT, _p))
    boot_intercepts = np.zeros(N_BOOT)
    for _b in range(N_BOOT):
        _idx = _rng.integers(0, len(agg), len(agg))
        _Mt = _Ds[_idx]
        _yt = agg[_idx]
        _yc = _yt - _yt.mean()
        boot_coefs[_b] = np.linalg.solve(_Mt.T @ _Mt + alpha_p3 * np.eye(_p), _Mt.T @ _yc)
        boot_intercepts[_b] = float(_yt.mean())
    return boot_coefs, boot_intercepts


@app.cell
def _(
    a_p3,
    boot_coefs,
    c0,
    c1,
    domains,
    eta_p3,
    np,
    p3_design_sd,
    p_p3,
):
    _D = len(domains)
    _coef_natural = boot_coefs / p3_design_sd[None, :]
    _beta_signal = _coef_natural[:, :_D]
    _beta_pen0 = _coef_natural[:, _D]
    _beta_pen1 = _coef_natural[:, _D + 1]
    _eps = 1e-12
    _T = 300
    _B = boot_coefs.shape[0]

    ts_w0 = np.empty((_B, _D))
    ts_w1 = np.empty((_B, _D))
    for _i in range(_B):
        _bs = _beta_signal[_i]
        _bp0 = _beta_pen0[_i]
        _bp1 = _beta_pen1[_i]
        _w0 = np.full(_D, 1.0 / _D)
        _w1 = np.full(_D, 1.0 / _D)
        for _t in range(_T):
            _E = np.maximum(_w0 + eta_p3 * _w1, _eps)
            _sig_grad = a_p3 * np.power(_E, a_p3 - 1.0)
            _ne0 = np.maximum(c0 * _w0, _eps)
            _ne1 = np.maximum(c1 * _w1, _eps)
            _pg0 = _bp0 * p_p3 * np.power(_ne0, p_p3 - 1.0)
            _pg1 = _bp1 * p_p3 * np.power(_ne1, p_p3 - 1.0)
            _g0 = _bs * _sig_grad - c0 * _pg0
            _g1 = _bs * eta_p3 * _sig_grad - c1 * _pg1
            _gamma = 2.0 / (_t + 2)
            _w0 = (1 - _gamma) * _w0
            _w0[int(np.argmax(_g0))] += _gamma
            _w1 = (1 - _gamma) * _w1
            _w1[int(np.argmax(_g1))] += _gamma
        ts_w0[_i] = _w0
        ts_w1[_i] = _w1

    p0_p3 = ts_w0.mean(axis=0)
    p1_p3 = ts_w1.mean(axis=0)
    return p0_p3, p1_p3


@app.cell
def _(a_p3, alpha_p3, eta_p3, mo, p3_in_r2, p_p3):
    mo.md(
        f"**P3 fit:** η = {eta_p3:.0f}, a = {a_p3:.3f}, p = {p_p3}, ridge α = {alpha_p3:.0f}, in-sample R² = `{p3_in_r2:.4f}`."
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Step C — Side-by-side: optimal mixtures

    Left panel: phase-0 weight per domain, **GRP** (gray) vs **P3 Thompson mean** (blue). Right panel: same in epoch space (`w · c` per phase). Dotted vertical line at `8 epochs` for reference (the "one-pass-is-fine, multi-pass-might-hurt" rule of thumb).
    """
    )
    return


@app.cell
def _(
    PALETTE,
    c0,
    c1,
    domains,
    go,
    grp_eps_p0,
    grp_eps_p1,
    make_subplots,
    np,
    p0_grp_aligned,
    p0_p3,
    p1_grp_aligned,
    p1_p3,
    style_fig,
):
    _comp = [d[:42] for d in domains]
    _total_p3 = p0_p3 + p1_p3
    _total_grp = p0_grp_aligned + p1_grp_aligned
    _order = np.argsort(-(_total_p3 + _total_grp))
    _names = [_comp[i] for i in _order]

    _ep0_p3 = p0_p3 * c0
    _ep1_p3 = p1_p3 * c1
    _ep_total_p3 = _ep0_p3 + _ep1_p3
    _ep_total_grp = grp_eps_p0 + grp_eps_p1

    _fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=False,
        horizontal_spacing=0.18,
        subplot_titles=(
            "weight space — total weight (phase 0 + phase 1)",
            "epoch space — total epochs over full source",
        ),
    )

    _fig.add_trace(
        go.Bar(
            x=_total_grp[_order],
            y=_names,
            orientation="h",
            marker=dict(color="rgba(140,140,140,0.65)", line=dict(width=0)),
            name="GRP (NNLS, refit on IRT)",
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Bar(
            x=_total_p3[_order],
            y=_names,
            orientation="h",
            marker=dict(color=PALETTE[0], line=dict(width=0)),
            name="P3 (ridge + Thompson mean)",
        ),
        row=1,
        col=1,
    )

    _fig.add_trace(
        go.Bar(
            x=_ep_total_grp[_order],
            y=_names,
            orientation="h",
            marker=dict(color="rgba(140,140,140,0.65)", line=dict(width=0)),
            name="GRP epochs",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _fig.add_trace(
        go.Bar(
            x=_ep_total_p3[_order],
            y=_names,
            orientation="h",
            marker=dict(color=PALETTE[0], line=dict(width=0)),
            name="P3 epochs",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _fig.add_vline(x=8.0, line_dash="dot", line_color="black", line_width=1, row=1, col=2)

    _fig.update_xaxes(title_text="total weight (sum of phases)", row=1, col=1)
    _fig.update_xaxes(title_text="total epochs over full source", row=1, col=2)
    _fig.update_yaxes(tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)")
    _fig.update_layout(height=max(500, 26 * len(_names)), width=1500, barmode="group", bargap=0.15)
    style_fig(
        _fig,
        title=(
            "Optimal data mixture: GRP vs P3 (both fit and optimized on the same IRT aggregate target)<br>"
            "<sub>sorted by combined attention from both methods · dotted line at 8 epochs</sub>"
        ),
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Step D — agreement table (top-15 by combined weight)

    Pearson correlation between the two recommendations (across 39 domains, total weight per phase) reported below. High correlation → both methods agree on the shape of the optimum; low correlation → they diverge meaningfully.
    """
    )
    return


@app.cell
def _(
    c0,
    c1,
    domains,
    grp_eps_p0,
    grp_eps_p1,
    mo,
    np,
    p0_grp_aligned,
    p0_p3,
    p1_grp_aligned,
    p1_p3,
    pl,
):
    _total_p3 = p0_p3 + p1_p3
    _total_grp = p0_grp_aligned + p1_grp_aligned
    _r_w = float(np.corrcoef(_total_p3, _total_grp)[0, 1])
    _r_w0 = float(np.corrcoef(p0_p3, p0_grp_aligned)[0, 1])
    _r_w1 = float(np.corrcoef(p1_p3, p1_grp_aligned)[0, 1])
    _ep_p3 = p0_p3 * c0 + p1_p3 * c1
    _ep_grp = grp_eps_p0 + grp_eps_p1
    _r_e = float(np.corrcoef(_ep_p3, _ep_grp)[0, 1])

    _order = np.argsort(-(_total_p3 + _total_grp))
    _table = pl.DataFrame(
        {
            "domain": [domains[i] for i in _order[:15]],
            "P3_w_total": [round(_total_p3[i], 3) for i in _order[:15]],
            "GRP_w_total": [round(_total_grp[i], 3) for i in _order[:15]],
            "P3_epochs": [round(_ep_p3[i], 2) for i in _order[:15]],
            "GRP_epochs": [round(_ep_grp[i], 2) for i in _order[:15]],
        }
    )

    mo.md(
        f"### Agreement metrics\n\n"
        f"- Pearson(total weight): **{_r_w:+.3f}**\n"
        f"- Pearson(phase-0 weight): **{_r_w0:+.3f}**\n"
        f"- Pearson(phase-1 weight): **{_r_w1:+.3f}**\n"
        f"- Pearson(total epochs): **{_r_e:+.3f}**\n\n"
        f"**Top-15 domains by combined weight (P3 mean + GRP):**"
    )
    return (_table,)


@app.cell
def _(_table):
    _table
    return
