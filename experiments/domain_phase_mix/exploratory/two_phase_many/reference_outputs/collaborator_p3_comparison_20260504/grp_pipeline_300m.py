import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np, pl


@app.cell
def _():
    PALETTE = [
        "#1877F2", "#F0701A", "#5A24C7", "#E42C97", "#00487C", "#0EAC96",
        "#AB76FF", "#B50550", "#0099E6", "#22085F", "#783301",
    ]
    PLOTLY_TEMPLATE = "plotly_white"

    def style_fig(fig, title=None):
        _layout = dict(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=60, r=30, t=70, b=100),
            legend=dict(
                font=dict(size=11),
                orientation="h",
                yanchor="top", y=-0.15,
                xanchor="center", x=0.5,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(200,200,200,0.5)",
                borderwidth=1,
            ),
        )
        if title is not None:
            _layout["title"] = dict(
                text=title, font=dict(size=18),
                x=0.5, xanchor="center",
            )
        fig.update_layout(**_layout)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(220,220,220,0.5)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(220,220,220,0.5)")
        return fig

    return PALETTE, style_fig


@app.cell
def _(mo):
    mo.md("""
    # 300M data-mixture pipeline: IRT aggregate → P3 mixture model → in-distribution candidate search

    Five-step pipeline:

    1. **IRT features.** Non-negative k-factor analysis on per-task metrics. MCQ tasks use `choice_logprob`; everything else uses `bpb` (sign-flipped). Define `aggregate(metric_row)` as the uniform mean of the k posterior factor scores.
    2. **Noise σ²** of the aggregate, measured on 10 same-mix seed-only runs.
    3. **P3 mixture model** — a power-law on combined exposure plus *per-phase* concentration penalties on epoch counts with a shared learnable exponent:

       ```
       ŷ = α₀ + Σ_d β_d · (w₀,d + η · w₁,d)ᵃ
              − γ₀ · Σ_d (c_0,d · w₀,d)ᵖ
              − γ₁ · Σ_d (c_1,d · w₁,d)ᵖ
       ```

       Three nonlinear knobs (η, a, p) selected by **nested 5×5 CV**; ridge fits independent γ₀ and γ₁ so phase-0 and phase-1 over-epoching can be priced differently. The single-γ form let the optimizer dump phase-1 mass on small high-β domains (η = 5 amplifies phase-1 signal but the combined penalty wasn't η-aware). Splitting per-phase fixes that asymmetry.
    4. **Bootstrap ridge coefficients** (200 resamples) to get a posterior over the linear head, hence per-candidate predictive uncertainty.
    5. **Score 200k Dirichlet-jittered candidates** sampled around observed sweep mixtures, rank by **LCB = predicted_mean − κ · σ_bootstrap** (κ=1).
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Functional-form comparison: full GRP vs P3

    **Full GRP** (`grp_no_l2_exact.py` in the scaling packet) — 9 nonlinear params, 32 linear coefs, NNLS.

    Per-domain effective exposure (mass × epoch multiplier, with retained-exposure decay):

    ```
    x_d = exp(−λ(1−w₁,d)) · c₀,d · w₀,d  +  η · c₁,d · w₁,d
    ```

    Predicted aggregate has four additive blocks — singleton signals, CC high/low pair signals, family-total signals, and family concentration penalties:

    ```
    ŷ = α₀
        − Σ_{d∈S}      α^sgn_d                · x_d^{a_f(d)}
        − Σ_{(h,ℓ)∈P}  α^pair_{(h,ℓ)}        · (x_h + β · x_ℓ)^{(a_f(h)+a_f(ℓ))/2}
        − Σ_{f∈F}      α^fam_f                · (Σ_{d∈f} x_d)^{a_f}
        + Σ_{f∈F}      α^pen_f · Σ_{g∈G_f}  softplus(log(1+T_g) − τ_f)²
    ```

    with all α ≥ 0 enforced by non-negative least squares, families F = {broad_text, tech_code, reasoning}, CC pair set P, singletons S, family-groups G_f.

    **P3** — 3 nonlinear params (η, a, p), 39 + 2 linear coefs, ridge:

    ```
    ŷ = α₀ + Σ_d β_d · (w₀,d + η · w₁,d)^a
           − γ_0 · Σ_d (c_0,d · w_0,d)^p
           − γ_1 · Σ_d (c_1,d · w_1,d)^p
    ```

    with β_d, γ_0, γ_1 all unconstrained. `p` is the learnable above-threshold exponent (1=linear, 2=Herfindahl, larger=more concentrated). T_phase_0 ≈ 5.06T, T_phase_1 ≈ 1.27T (80/20 of 6.33T total). N_d is the full source-dataset token count. Splitting the penalty per-phase is necessary because the signal term has an η multiplier that asymmetrically rewards phase-1 mass — without a matching η in the penalty, the optimizer exploits cheap phase-1 budget on small-N domains.

    **What's dropped, term by term:**

    | GRP piece | P3 status |
    |---|---|
    | `exp(−λ(1−w₁,d))` retained-exposure decay | **dropped** (λ → 0 in any retune anyway) |
    | `c₀,d`, `c₁,d` per-domain epoch multipliers | **kept inside the penalty only** (used to derive `T_phase` and `N_d`, giving `num_epochs_d` in interpretable epoch units); dropped from the signal term — absorbed into β_d after standardization |
    | `η` phase-1 weight | **kept** |
    | `a_f(d)` per-family power exponent | **collapsed to a single global `a`** |
    | `β` low-quality CC discount | **dropped** (CC pair aggregation gone) |
    | CC high/low pair aggregation `(x_h + β · x_ℓ)^a` | **dropped** (each domain treated as a standalone feature) |
    | Family-total signal `(Σ_{d∈f} x_d)^{a_f}` | **dropped** |
    | Family concentration penalty `Σ_g softplus(log(1+T_g)−τ_f)²` | **kept** as two phase-specific terms: `γ_0·Σ_d (c_0,d·w_0,d)^p + γ_1·Σ_d (c_1,d·w_1,d)^p`, shared learnable exponent p |
    | `τ_f` per-family penalty thresholds | **dropped** — replaced by a single learnable exponent `p` instead of per-family thresholds |
    | NNLS sign constraint `α ≥ 0` | **dropped** — replaced by ridge with unconstrained β_d (some domains genuinely hurt the IRT aggregate at the margin on a simplex; NNLS forbids that) |

    **Why these specific cuts.** An incremental ablation under **honest nested CV** (no hyperparameter leakage) showed the only two pieces that earned their R² were combined exposure with a single η and the global power transform E^a. Per-family curvature, family penalties, family-total signals, CC pair aggregation, and NNLS each cost 0–2pp rather than helping when the leakage from grid-tuning on the validation folds was removed.

    Result: 9 nonlinear knobs → **3** (η, a, p); 32 NNLS linear coefs → **39 ridge linear coefs + 2 penalty coefs (γ_0, γ_1)**. About one-third of the parameter budget for similar predictive performance.
    """)
    return


@app.cell
def _(np, pl):
    raw = pl.read_csv("raw_metric_matrix_300m.csv", infer_schema_length=500).filter(
        pl.col("status") == "completed"
    )
    noise_df = pl.read_csv("noise_baseline_run00097_300m.csv", infer_schema_length=200)

    _MMLU_KEEP = {
        "lm_eval/mmlu_5shot/bpb",
        "lm_eval/mmlu_sl_verb_5shot/bpb",
    }
    _AGG_DROP = {
        "eval/bpb", "eval/macro_bpb",
        "eval/paloma/bpb", "eval/paloma/macro_bpb",
        "eval/uncheatable_eval/bpb", "eval/uncheatable_eval/macro_bpb",
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
        if not c.startswith((
            "eval/uncheatable_eval/",
            "lm_eval/",
            "mcq_smooth/",
            "teacher_forced/",
        )):
            return False
        if c.startswith("lm_eval/mmlu_") and c not in _MMLU_KEEP:
            return False
        return True

    _candidates = [c for c in raw.columns if _keep(c)]
    _raw_cols = set(raw.columns)
    task_cols = []
    task_signs = []
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

    _meta = pl.read_csv("two_phase_many_epoch_metadata.csv")
    _row = {row["domain_name"]: row for row in _meta.iter_rows(named=True)}
    _N = np.array([_row[d]["token_count"] for d in domains], dtype=np.float64)
    _c0 = np.array([_row[d]["phase_0_epoch_multiplier"] for d in domains], dtype=np.float64)
    _c1 = np.array([_row[d]["phase_1_epoch_multiplier"] for d in domains], dtype=np.float64)
    # phase_p_fraction is the training-budget fraction (80/20), not source partition.
    # Hence T_phase_p = c_p,d · N_d (constant across d) and num_epochs_d = c_p,d · w_p,d.
    T_phase_0 = float((_c0 * _N).mean())
    T_phase_1 = float((_c1 * _N).mean())
    epochs_per_unit_w0 = _c0
    epochs_per_unit_w1 = _c1
    return (
        domains,
        epochs_per_unit_w0,
        epochs_per_unit_w1,
        noise_df,
        raw,
        task_cols,
        task_signs,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Step 1 — IRT features and aggregate function
    """)
    return


@app.cell
def _(noise_df, np, raw, task_cols, task_signs):
    X_signed_swarm = raw.select(task_cols).to_numpy().astype(np.float64) * task_signs[None, :]
    _stds = X_signed_swarm.std(axis=0)
    _mask = _stds > 1e-12
    if not _mask.all():
        raise RuntimeError("Zero-variance task encountered; would need to drop")
    swarm_mu = X_signed_swarm.mean(axis=0)
    swarm_sd = X_signed_swarm.std(axis=0)
    Z_swarm = (X_signed_swarm - swarm_mu) / swarm_sd

    _noise_cols = set(noise_df.columns)
    _has_noise = np.array([c in _noise_cols for c in task_cols])
    _present = [c for c in task_cols if c in _noise_cols]
    _signs_present = task_signs[_has_noise]
    _noise_X = noise_df.select(_present).to_numpy().astype(np.float64) * _signs_present[None, :]
    _ns_p = _noise_X.std(axis=0, ddof=1)
    _ss_p = X_signed_swarm[:, _has_noise].std(axis=0, ddof=1)
    noise_share = np.full(len(task_cols), np.nan)
    noise_share[_has_noise] = (_ns_p / _ss_p) ** 2
    return Z_swarm, noise_share, swarm_mu, swarm_sd


@app.cell
def _(Z_swarm, go, np, style_fig):
    _n, _p = Z_swarm.shape
    _real = np.sort(np.linalg.eigvalsh(np.corrcoef(Z_swarm.T)))[::-1]
    _rng = np.random.default_rng(42)
    _N_MC = 500
    _rand = np.empty((_N_MC, _p))
    for _i in range(_N_MC):
        _Zr = _rng.standard_normal((_n, _p))
        _Zr = (_Zr - _Zr.mean(axis=0)) / _Zr.std(axis=0)
        _rand[_i] = np.sort(np.linalg.eigvalsh(np.corrcoef(_Zr.T)))[::-1]
    _p95 = np.percentile(_rand, 95, axis=0)
    k_horn = int(np.sum(_real > _p95))

    _ranks = np.arange(1, _p + 1)
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=_ranks, y=_real, name="real", mode="lines+markers"))
    _fig.add_trace(go.Scatter(x=_ranks, y=_p95, name="random p95", mode="lines+markers", line=dict(dash="dash")))
    _fig.add_vline(x=k_horn + 0.5, line_dash="dot", annotation_text=f"k = {k_horn}")
    _fig.update_layout(xaxis_title="component rank", yaxis_title="correlation-matrix eigenvalue", height=400, width=800)
    style_fig(_fig, title=f"Horn parallel analysis: k = {k_horn}")
    _fig
    return (k_horn,)


@app.cell
def _(Z_swarm, k_horn, noise_share, np):
    _n, _p = Z_swarm.shape
    K = k_horn
    _psi_anchor = np.where(np.isnan(noise_share), np.nan, np.clip(noise_share, 1e-3, 0.999))
    _psi_fixed = ~np.isnan(_psi_anchor)
    _rng = np.random.default_rng(0)
    Lam = np.abs(_rng.normal(scale=0.1, size=(_p, K)))
    Psi = np.where(_psi_fixed, _psi_anchor, 1.0)
    for _ in range(5000):
        _Lpi = Lam / Psi[:, None]
        _V = np.linalg.inv(np.eye(K) + Lam.T @ _Lpi)
        _Th = Z_swarm @ _Lpi @ _V
        _S = _n * _V + _Th.T @ _Th
        _ZtT = Z_swarm.T @ _Th
        _Ln = _ZtT @ np.linalg.inv(_S)
        _Ln = np.clip(_Ln, 0.0, None)
        _Pf = (
            (Z_swarm ** 2).mean(axis=0)
            - 2 * (_ZtT * _Ln).sum(axis=1) / _n
            + ((_Ln @ _S) * _Ln).sum(axis=1) / _n
        )
        _Pf = np.clip(_Pf, 1e-6, None)
        _Pn = np.where(_psi_fixed, _psi_anchor, _Pf)
        if np.max(np.abs(_Ln - Lam)) < 1e-7:
            Lam, Psi = _Ln, _Pn
            break
        Lam, Psi = _Ln, _Pn
    _order = np.argsort(-(Lam ** 2).sum(axis=0))
    Lam = Lam[:, _order]

    _Lpi = Lam / Psi[:, None]
    _V = np.linalg.inv(np.eye(K) + Lam.T @ _Lpi)
    proj_to_aggregate = (_Lpi @ _V).mean(axis=1)
    communality = (Lam ** 2).sum(axis=1) / ((Lam ** 2).sum(axis=1) + Psi)
    return Lam, communality, proj_to_aggregate


@app.cell
def _(Lam, communality, go, np, style_fig, task_cols):
    _dom_factor = Lam.argmax(axis=1)
    _order = np.lexsort((-Lam.max(axis=1), _dom_factor))
    _labels = [
        f"{task_cols[i].removesuffix('/bpb').removesuffix('/choice_logprob')}  (h²={communality[i]:.2f})"
        for i in _order
    ]
    _Z = Lam[_order]
    _fig = go.Figure(
        data=go.Heatmap(
            z=_Z,
            x=[f"factor {k + 1}" for k in range(Lam.shape[1])],
            y=_labels,
            colorscale="Viridis",
            zmin=0.0,
            colorbar=dict(title="λ"),
            hovertemplate="%{y}<br>%{x}: λ=%{z:.3f}<extra></extra>",
        )
    )
    _fig.update_layout(
        height=max(500, 26 * len(_labels)),
        width=900,
        margin=dict(l=350),
        yaxis=dict(tickfont=dict(size=10)),
    )
    style_fig(
        _fig,
        title=(
            "Per-task IRT loadings (non-negative k-factor model)<br>"
            "<sub>tasks grouped by their dominant factor; sorted by max λ within group · "
            "h² = share of task variance explained by all factors combined</sub>"
        ),
    )
    _fig
    return


@app.cell
def _(Z_swarm, proj_to_aggregate):
    agg_swarm = Z_swarm @ proj_to_aggregate
    return (agg_swarm,)


@app.cell
def _(np, proj_to_aggregate, swarm_mu, swarm_sd, task_signs):
    def aggregate_from_row(metric_row):
        """Map a single run's per-task metric vector (in `task_cols` order) to the IRT aggregate scalar.
        bpb columns are negated, choice_logprob kept as-is, then standardized and projected."""
        _x = np.asarray(metric_row, dtype=float) * task_signs
        _z = (_x - swarm_mu) / swarm_sd
        return float(_z @ proj_to_aggregate)

    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 2 — Noise σ² of the aggregate
    """)
    return


@app.cell
def _(
    mo,
    noise_df,
    np,
    proj_to_aggregate,
    swarm_mu,
    swarm_sd,
    task_cols,
    task_signs,
):
    _noise_cols = set(noise_df.columns)
    _has_noise = np.array([c in _noise_cols for c in task_cols])
    _present = [c for c in task_cols if c in _noise_cols]
    _signs_present = task_signs[_has_noise]
    _nX = np.zeros((noise_df.height, len(task_cols)))
    _nX[:, _has_noise] = noise_df.select(_present).to_numpy().astype(np.float64) * _signs_present[None, :]
    _nX[:, _has_noise] = (_nX[:, _has_noise] - swarm_mu[_has_noise]) / swarm_sd[_has_noise]
    aggs_noise = _nX @ proj_to_aggregate
    sigma2_noise = float(aggs_noise.var(ddof=1))
    sigma_noise = float(np.sqrt(sigma2_noise))
    mo.md(
        f"**Empirical noise on aggregate, n = {noise_df.height} same-mix seed runs:**  \n"
        f"σ²_noise = `{sigma2_noise:.5f}`  →  σ_noise ≈ `{sigma_noise:.4f}` (in aggregate units).  \n"
        f"The {len(task_cols) - int(_has_noise.sum())} task columns missing from the noise CSV are zero-imputed (lower bound)."
    )
    return sigma2_noise, sigma_noise


@app.cell
def _(mo):
    mo.md("""
    ## Step 3 — P3 mixture regression with nested CV

    Functional form:

    ```
    ŷ = α₀ + Σ_d β_d · (w₀,d + η · w₁,d)^a
           − γ_0 · Σ_d (c_0,d · w_0,d)^p
           − γ_1 · Σ_d (c_1,d · w_1,d)^p
    ```

    Hyperparameters `(η, a, p)` selected by **nested 5×5 cross-validation**. γ_0 and γ_1 are independent ridge coefficients fit by the linear head; per-phase split lets phase-1 over-epoching be priced separately from phase-0 (necessary because η multiplies phase-1 in the signal term but not in the penalty otherwise). `p` controls penalty shape (shared across phases): p=1 linear, p=2 Herfindahl, larger p concentrates penalty on the worst-offending domain.
    """)
    return


@app.cell
def _(agg_swarm, domains, epochs_per_unit_w0, epochs_per_unit_w1, np, raw):
    _w0 = raw.select([f"phase_0_{d}" for d in domains]).to_numpy().astype(np.float64)
    _w1 = raw.select([f"phase_1_{d}" for d in domains]).to_numpy().astype(np.float64)
    w0_swarm = _w0 / _w0.sum(axis=1, keepdims=True)
    w1_swarm = _w1 / _w1.sum(axis=1, keepdims=True)
    EPS = 1e-4

    def build_p3_design(w0, w1, eta, a, p):
        E = w0 + eta * w1
        sig = np.power(np.maximum(E, EPS), a)
        # Per-phase concentration penalties: γ_0 · Σ_d (c_0,d · w_0,d)^p + γ_1 · Σ_d (c_1,d · w_1,d)^p.
        # Two coefs let phase-0 and phase-1 over-epoching be priced independently.
        ne0 = np.maximum(w0 * epochs_per_unit_w0[None, :], EPS)
        ne1 = np.maximum(w1 * epochs_per_unit_w1[None, :], EPS)
        pen0 = np.power(ne0, p).sum(axis=1, keepdims=True)
        pen1 = np.power(ne1, p).sum(axis=1, keepdims=True)
        return np.column_stack([sig, -pen0, -pen1])

    def _ridge_inner_cv_score(M, y, fold_seed, alphas):
        _msk = M.std(0) > 1e-10
        if not _msk.any():
            return -np.inf
        M = M[:, _msk]
        _mu = M.mean(0)
        _sd = M.std(0)
        Ms = (M - _mu) / _sd
        _p = Ms.shape[1]
        _rng = np.random.default_rng(fold_seed)
        _idx = _rng.permutation(len(y))
        _folds = np.array_split(_idx, 5)
        _best = -np.inf
        for _alpha in alphas:
            _preds = np.zeros(len(y))
            for _f in range(5):
                _t = _folds[_f]
                _tr = np.concatenate([_folds[i] for i in range(5) if i != _f])
                _Mt = Ms[_tr]
                _yt = y[_tr] - y[_tr].mean()
                _bb = np.linalg.solve(_Mt.T @ _Mt + _alpha * np.eye(_p), _Mt.T @ _yt)
                _preds[_t] = Ms[_t] @ _bb + y[_tr].mean()
            _r2 = 1 - ((y - _preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            if _r2 > _best:
                _best = _r2
        return _best

    def _ridge_predict(M_train, y_train, M_test, alpha):
        _msk = M_train.std(0) > 1e-10
        M_train = M_train[:, _msk]
        M_test = M_test[:, _msk]
        _mu = M_train.mean(0)
        _sd = M_train.std(0)
        Ms_tr = (M_train - _mu) / _sd
        Ms_te = (M_test - _mu) / _sd
        _p = Ms_tr.shape[1]
        _yc = y_train - y_train.mean()
        _bb = np.linalg.solve(Ms_tr.T @ Ms_tr + alpha * np.eye(_p), Ms_tr.T @ _yc)
        return Ms_te @ _bb + y_train.mean()

    def _ridge_pick_alpha(M, y, fold_seed, alphas):
        _msk = M.std(0) > 1e-10
        M = M[:, _msk]
        _mu = M.mean(0)
        _sd = M.std(0)
        Ms = (M - _mu) / _sd
        _p = Ms.shape[1]
        _rng = np.random.default_rng(fold_seed)
        _idx = _rng.permutation(len(y))
        _folds = np.array_split(_idx, 5)
        _best, _best_alpha = -np.inf, None
        for _alpha in alphas:
            _preds = np.zeros(len(y))
            for _f in range(5):
                _t = _folds[_f]
                _tr = np.concatenate([_folds[i] for i in range(5) if i != _f])
                _Mt = Ms[_tr]
                _yt = y[_tr] - y[_tr].mean()
                _bb = np.linalg.solve(_Mt.T @ _Mt + _alpha * np.eye(_p), _Mt.T @ _yt)
                _preds[_t] = Ms[_t] @ _bb + y[_tr].mean()
            _r2 = 1 - ((y - _preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            if _r2 > _best:
                _best, _best_alpha = _r2, _alpha
        return _best_alpha

    ETA_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
    A_GRID = tuple(np.linspace(0.5, 2.0, 8))
    P_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    ALPHA_GRID = tuple(np.logspace(0, 4, 9))

    OUTER_FOLDS = 5
    _rng = np.random.default_rng(0)
    _idx = _rng.permutation(len(agg_swarm))
    outer_folds = np.array_split(_idx, OUTER_FOLDS)

    oof_p3 = np.zeros(len(agg_swarm))
    chosen_per_fold = []
    for _o in range(OUTER_FOLDS):
        _test = outer_folds[_o]
        _train = np.concatenate([outer_folds[i] for i in range(OUTER_FOLDS) if i != _o])
        _best_score, _best_combo = -np.inf, None
        for _eta in ETA_GRID:
            for _a in A_GRID:
                for _p in P_GRID:
                    _Mtr = build_p3_design(w0_swarm[_train], w1_swarm[_train], _eta, _a, _p)
                    _score = _ridge_inner_cv_score(_Mtr, agg_swarm[_train], fold_seed=99 + _o, alphas=ALPHA_GRID)
                    if _score > _best_score:
                        _best_score, _best_combo = _score, (_eta, _a, _p)
        _eta_best, _a_best, _p_best = _best_combo
        _Mtr_best = build_p3_design(w0_swarm[_train], w1_swarm[_train], _eta_best, _a_best, _p_best)
        _Mte_best = build_p3_design(w0_swarm[_test], w1_swarm[_test], _eta_best, _a_best, _p_best)
        _alpha_best = _ridge_pick_alpha(_Mtr_best, agg_swarm[_train], fold_seed=99 + _o, alphas=ALPHA_GRID)
        oof_p3[_test] = _ridge_predict(_Mtr_best, agg_swarm[_train], _Mte_best, _alpha_best)
        chosen_per_fold.append({
            "fold": _o, "eta": _eta_best, "a": _a_best, "p_pen": _p_best,
            "alpha": _alpha_best, "inner_cv_r2": _best_score,
        })

    r2_cv_p3 = float(1 - ((agg_swarm - oof_p3) ** 2).sum() / ((agg_swarm - agg_swarm.mean()) ** 2).sum())

    # Final fit on the full data: pick (eta, a, p, alpha) by full-data inner CV
    _best_score, _best_combo = -np.inf, None
    for _eta in ETA_GRID:
        for _a in A_GRID:
            for _p in P_GRID:
                _M = build_p3_design(w0_swarm, w1_swarm, _eta, _a, _p)
                _score = _ridge_inner_cv_score(_M, agg_swarm, fold_seed=42, alphas=ALPHA_GRID)
                if _score > _best_score:
                    _best_score, _best_combo = _score, (_eta, _a, _p)
    eta_p3, a_p3, p_pen_p3 = _best_combo
    D_full = build_p3_design(w0_swarm, w1_swarm, eta_p3, a_p3, p_pen_p3)
    alpha_p3 = _ridge_pick_alpha(D_full, agg_swarm, fold_seed=42, alphas=ALPHA_GRID)

    _msk = D_full.std(0) > 1e-10
    _D = D_full[:, _msk]
    _mu = _D.mean(0)
    _sd = _D.std(0)
    _Ds = (_D - _mu) / _sd
    _yc = agg_swarm - agg_swarm.mean()
    coef_p3_std = np.linalg.solve(_Ds.T @ _Ds + alpha_p3 * np.eye(_Ds.shape[1]), _Ds.T @ _yc)
    yhat_in_p3 = _Ds @ coef_p3_std + agg_swarm.mean()
    r2_in_p3 = float(1 - ((agg_swarm - yhat_in_p3) ** 2).sum() / ((agg_swarm - agg_swarm.mean()) ** 2).sum())

    p3_design_mu = _mu
    p3_design_sd = _sd
    p3_design_mask = _msk
    return (
        D_full,
        a_p3,
        alpha_p3,
        build_p3_design,
        chosen_per_fold,
        coef_p3_std,
        eta_p3,
        oof_p3,
        p3_design_mask,
        p3_design_mu,
        p3_design_sd,
        p_pen_p3,
        r2_cv_p3,
        r2_in_p3,
        w0_swarm,
        w1_swarm,
        yhat_in_p3,
    )


@app.cell
def _(
    PALETTE,
    a_p3,
    agg_swarm,
    alpha_p3,
    chosen_per_fold,
    eta_p3,
    go,
    np,
    oof_p3,
    p_pen_p3,
    r2_cv_p3,
    r2_in_p3,
    raw,
    sigma_noise,
    style_fig,
    yhat_in_p3,
):
    _names = raw["run_name"].to_list()
    _is_baseline = np.array(["baseline" in n for n in _names])
    _lo = min(agg_swarm.min(), oof_p3.min(), yhat_in_p3.min())
    _hi = max(agg_swarm.max(), oof_p3.max(), yhat_in_p3.max())
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=[_lo, _hi], y=[_lo, _hi], mode="lines", name="y = x",
                              line=dict(color="black", dash="dot", width=1)))
    _fig.add_trace(go.Scatter(x=agg_swarm[~_is_baseline], y=yhat_in_p3[~_is_baseline],
                              mode="markers", name=f"in-sample (R² = {r2_in_p3:.3f})",
                              text=[n for n, b in zip(_names, _is_baseline) if not b],
                              marker=dict(size=5, color="#A0CBE8", opacity=0.6)))
    _fig.add_trace(go.Scatter(x=agg_swarm[~_is_baseline], y=oof_p3[~_is_baseline],
                              mode="markers", name=f"nested 5×5 CV (R² = {r2_cv_p3:.3f})",
                              text=[n for n, b in zip(_names, _is_baseline) if not b],
                              marker=dict(size=6, color=PALETTE[0], opacity=0.85)))
    _fig.add_trace(go.Scatter(x=agg_swarm[_is_baseline], y=oof_p3[_is_baseline],
                              mode="markers+text",
                              text=[n.replace("baseline_", "") for n, b in zip(_names, _is_baseline) if b],
                              textposition="top center", name="baselines (CV)",
                              marker=dict(size=14, color=PALETTE[1], symbol="star")))
    _fig.update_layout(xaxis_title="actual aggregate", yaxis_title="predicted aggregate",
                       height=600, width=900)
    _per_fold_table = " · ".join(
        f"f{c['fold']}: η={c['eta']}, a={c['a']:.2f}, p={c['p_pen']:.1f}"
        for c in chosen_per_fold
    )
    style_fig(
        _fig,
        title=(
            f"P3 + per-phase epoch penalty on the IRT aggregate (300M / 6.3T sweep)<br>"
            f"<sub>final fit (η = {eta_p3}, a = {a_p3:.2f}, p = {p_pen_p3:.1f}, α = {alpha_p3:.2f}) · "
            f"in-sample R² = {r2_in_p3:.3f} · nested 5×5 CV R² = {r2_cv_p3:.3f} · noise σ ≈ {sigma_noise:.3f}<br>"
            f"per-fold (η, a, p): {_per_fold_table}</sub>"
        ),
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 4 — Bootstrap ridge coefficients (200 resamples)
    """)
    return


@app.cell
def _(
    D_full,
    agg_swarm,
    alpha_p3,
    np,
    p3_design_mask,
    p3_design_mu,
    p3_design_sd,
):
    N_BOOT = 200
    _rng = np.random.default_rng(7)
    _D = D_full[:, p3_design_mask]
    _Ds = (_D - p3_design_mu) / p3_design_sd
    _p = _Ds.shape[1]
    boot_coefs = np.zeros((N_BOOT, _p))
    boot_intercepts = np.zeros(N_BOOT)
    for _b in range(N_BOOT):
        _idx = _rng.integers(0, len(agg_swarm), len(agg_swarm))
        _Mt = _Ds[_idx]
        _yt = agg_swarm[_idx]
        _yc = _yt - _yt.mean()
        boot_coefs[_b] = np.linalg.solve(_Mt.T @ _Mt + alpha_p3 * np.eye(_p), _Mt.T @ _yc)
        boot_intercepts[_b] = float(_yt.mean())
    return boot_coefs, boot_intercepts


@app.cell
def _(
    D_full,
    PALETTE,
    boot_coefs,
    boot_intercepts,
    go,
    np,
    p3_design_mask,
    p3_design_mu,
    p3_design_sd,
    style_fig,
):
    _D = D_full[:, p3_design_mask]
    _Ds = (_D - p3_design_mu) / p3_design_sd
    _preds = _Ds @ boot_coefs.T + boot_intercepts[None, :]
    _per_row_std = _preds.std(axis=1, ddof=1)
    _fig = go.Figure()
    _fig.add_trace(go.Histogram(x=_per_row_std, nbinsx=40, marker_color=PALETTE[0]))
    _fig.update_layout(xaxis_title="per-row predictive std (across bootstraps)",
                       yaxis_title="count", height=400, width=800)
    style_fig(
        _fig,
        title=(
            f"Bootstrap predictive uncertainty across {len(_per_row_std)} swarm rows<br>"
            f"<sub>median per-row σ = {np.median(_per_row_std):.4f} · max = {_per_row_std.max():.4f} · "
            f"coef matrix shape {boot_coefs.shape}</sub>"
        ),
    )
    _fig
    return


@app.cell
def _(
    D_full,
    a_p3,
    alpha_p3,
    boot_coefs,
    boot_intercepts,
    coef_p3_std,
    domains,
    eta_p3,
    np,
    p3_design_mask,
    p3_design_mu,
    p3_design_sd,
    p_pen_p3,
    pl,
    r2_cv_p3,
    r2_in_p3,
    sigma2_noise,
    sigma_noise,
):
    import json as _json

    # Design columns are [D domain signal columns, 1 penalty column].
    _all_labels = list(domains) + [
        f"__penalty_phase0_sum_epochs_pow_{p_pen_p3:g}__",
        f"__penalty_phase1_sum_epochs_pow_{p_pen_p3:g}__",
    ]
    _kept_labels = [_all_labels[i] for i in range(len(_all_labels)) if p3_design_mask[i]]
    _coef_natural = coef_p3_std / p3_design_sd
    _coef_intercept_adj = -float(p3_design_mu @ _coef_natural)
    _coef_std_natural = boot_coefs.std(axis=0, ddof=1) / p3_design_sd
    _coef_p5 = np.percentile(boot_coefs, 5, axis=0)
    _coef_p95 = np.percentile(boot_coefs, 95, axis=0)
    _intercept_std = float(boot_intercepts.std(ddof=1))

    p3_fit = {
        "form": "y = α₀ + Σ_d β_d · (w_0,d + η · w_1,d)^a − γ_0 · Σ_d (c_0,d·w_0,d)^p − γ_1 · Σ_d (c_1,d·w_1,d)^p; phase-0 and phase-1 epoching priced by independent γ coefs sharing exponent p (1=linear, 2=Herfindahl).",
        "nonlinear_params": {
            "eta": float(eta_p3),
            "a": float(a_p3),
            "p_penalty_exponent": float(p_pen_p3),
            "ridge_alpha": float(alpha_p3),
        },
        "linear_intercept": {
            "value": float(_coef_intercept_adj),
            "bootstrap_std": _intercept_std,
        },
        "linear_coefs": [
            {
                "feature": _kept_labels[_i],
                "value_natural_units": float(_coef_natural[_i]),
                "value_standardized": float(coef_p3_std[_i]),
                "bootstrap_std_natural": float(_coef_std_natural[_i]),
                "bootstrap_p5_standardized": float(_coef_p5[_i]),
                "bootstrap_p95_standardized": float(_coef_p95[_i]),
            }
            for _i in range(len(_kept_labels))
        ],
        "fit_metrics": {
            "n_swarm_rows": int(D_full.shape[0]),
            "n_features_kept": int(p3_design_mask.sum()),
            "in_sample_R2": float(r2_in_p3),
            "nested_cv_R2": float(r2_cv_p3),
            "noise_sigma": float(sigma_noise),
            "noise_share_of_aggregate_variance": float(sigma2_noise) if sigma2_noise > 0 else None,
            "bootstrap_resamples": int(boot_coefs.shape[0]),
        },
        "notes": (
            "Design columns: first len(domains) are signal features E_d^a where E_d = w_0,d + η · w_1,d; "
            "then two penalty columns, −Σ_d (c_0,d·w_0,d)^p (phase 0) and −Σ_d (c_1,d·w_1,d)^p (phase 1). "
            "Each phase's concentration is priced by its own ridge coefficient (γ_0, γ_1), sharing the "
            "learnable exponent p. T_phase ≈ (5.06T, 1.27T) tokens. A positive ridge coefficient on a "
            "penalty column means predicted aggregate goes down as that phase's concentration grows. "
            "Bootstrap std is the marginal uncertainty in the linear head conditional on "
            "(η, a, p, ridge_alpha) held at their nested-CV-best values."
        ),
    }
    with open("p3_fit_300m.json", "w") as _f:
        _json.dump(p3_fit, _f, indent=2)

    p3_fit_table = pl.DataFrame({
        "feature": _kept_labels,
        "value": [float(v) for v in _coef_natural],
        "bootstrap_std": [float(s) for s in _coef_std_natural],
    }).sort("value", descending=True)
    p3_fit_table
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5 — Score 200k Dirichlet-jittered candidates ranked by LCB

    Sample candidates around observed sweep mixtures (Dirichlet jitter at concentration α = 200), score with the P3 point-estimate model, compute predictive σ from the Step-4 bootstrap, rank by **LCB = mean − κ · σ** (κ = 1).
    """)
    return


@app.cell
def _(np, w0_swarm, w1_swarm):
    N_CAND = 200_000
    DIRICHLET_CONCENTRATION = 200.0
    _rng = np.random.default_rng(11)
    _row_p0 = _rng.integers(0, w0_swarm.shape[0], N_CAND)
    _row_p1 = _rng.integers(0, w1_swarm.shape[0], N_CAND)
    _alpha0 = w0_swarm[_row_p0] * DIRICHLET_CONCENTRATION + 1e-3
    _alpha1 = w1_swarm[_row_p1] * DIRICHLET_CONCENTRATION + 1e-3
    _g0 = _rng.gamma(_alpha0, 1.0)
    _g1 = _rng.gamma(_alpha1, 1.0)
    cand_w0 = _g0 / _g0.sum(axis=1, keepdims=True)
    cand_w1 = _g1 / _g1.sum(axis=1, keepdims=True)
    return DIRICHLET_CONCENTRATION, N_CAND, cand_w0, cand_w1


@app.cell
def _(
    a_p3,
    agg_swarm,
    boot_coefs,
    boot_intercepts,
    build_p3_design,
    cand_w0,
    cand_w1,
    coef_p3_std,
    eta_p3,
    np,
    p3_design_mask,
    p3_design_mu,
    p3_design_sd,
    p_pen_p3,
):
    _D_cand = build_p3_design(cand_w0, cand_w1, eta_p3, a_p3, p_pen_p3)[:, p3_design_mask]
    _Ds_cand = (_D_cand - p3_design_mu) / p3_design_sd
    cand_pred = _Ds_cand @ coef_p3_std + agg_swarm.mean()
    KAPPA = 1.0
    _N = _Ds_cand.shape[0]
    _CHUNK = 4096
    cand_pred_std = np.empty(_N)
    for _i in range(0, _N, _CHUNK):
        _pp = boot_coefs @ _Ds_cand[_i:_i + _CHUNK].T + boot_intercepts[:, None]
        cand_pred_std[_i:_i + _CHUNK] = _pp.std(axis=0, ddof=1)
    cand_lcb = cand_pred - KAPPA * cand_pred_std
    top_idx = np.argsort(-cand_lcb)[:5]
    return KAPPA, cand_lcb, cand_pred, cand_pred_std, top_idx


@app.cell
def _(
    DIRICHLET_CONCENTRATION,
    KAPPA,
    N_CAND,
    PALETTE,
    agg_swarm,
    cand_lcb,
    cand_pred,
    cand_pred_std,
    go,
    style_fig,
    top_idx,
):
    _fig = go.Figure()
    _fig.add_trace(go.Histogram(
        x=cand_pred, name=f"Dirichlet candidates: predictive mean (n={N_CAND})",
        marker_color=PALETTE[2], opacity=0.45, nbinsx=80,
    ))
    _fig.add_trace(go.Histogram(
        x=cand_lcb, name=f"Dirichlet candidates: LCB = mean − {KAPPA}·σ",
        marker_color=PALETTE[1], opacity=0.45, nbinsx=80,
    ))
    _fig.add_trace(go.Histogram(
        x=agg_swarm, name="actual sweep aggregates (n=241)",
        marker_color=PALETTE[0], opacity=0.85, nbinsx=40,
    ))
    for _r, _i in enumerate(top_idx):
        _fig.add_vline(
            x=float(cand_lcb[_i]), line_dash="dot",
            line_color=PALETTE[1],
            annotation_text=f"top-{_r + 1} LCB", annotation_position="top",
        )
    _fig.update_layout(xaxis_title="aggregate", height=500, width=1100, barmode="overlay")
    style_fig(
        _fig,
        title=(
            f"Dirichlet-jittered candidates ranked by LCB = mean − κ·σ (κ = {KAPPA}; bootstrap σ from 200 ridge resamples)<br>"
            f"<sub>top-1 predictive mean = {float(cand_pred[top_idx[0]]):+.3f}, σ = {float(cand_pred_std[top_idx[0]]):.3f}, "
            f"LCB = {float(cand_lcb[top_idx[0]]):+.3f} · "
            f"sweep max = {float(agg_swarm.max()):+.3f} · sweep mean = {float(agg_swarm.mean()):+.3f} · "
            f"jitter concentration α = {DIRICHLET_CONCENTRATION}</sub>"
        ),
    )
    _fig
    return


@app.cell
def _(
    KAPPA,
    PALETTE,
    cand_lcb,
    cand_pred,
    cand_pred_std,
    cand_w0,
    cand_w1,
    domains,
    epochs_per_unit_w0,
    epochs_per_unit_w1,
    go,
    make_subplots,
    np,
    pl,
    raw,
    style_fig,
    top_idx,
):
    _i = int(top_idx[0])
    best_p0_cand = cand_w0[_i]
    best_p1_cand = cand_w1[_i]
    best_pred_cand = float(cand_pred[_i])
    best_std_cand = float(cand_pred_std[_i])
    best_lcb_cand = float(cand_lcb[_i])

    _br = raw.filter(pl.col("run_name") == "baseline_proportional").row(0, named=True)
    _bp0 = np.array([_br[f"phase_0_{d}"] for d in domains])
    _bp1 = np.array([_br[f"phase_1_{d}"] for d in domains])
    _comp = [d[:45] for d in domains]
    _order = np.argsort(-_bp0)
    _names = [_comp[i] for i in _order]

    _ep0_base = _bp0 * epochs_per_unit_w0
    _ep1_base = _bp1 * epochs_per_unit_w1
    _ep0_cand = best_p0_cand * epochs_per_unit_w0
    _ep1_cand = best_p1_cand * epochs_per_unit_w1

    _fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False, horizontal_spacing=0.18,
        subplot_titles=("weight space", "epoch space (full-source passes)"),
    )

    _fig.add_trace(go.Bar(
        x=_bp0[_order], y=_names, orientation="h",
        name="proportional baseline (weight)",
        marker=dict(color="rgba(140,140,140,0.55)", line=dict(width=0)),
        legendgroup="baseline_w",
    ), row=1, col=1)
    _fig.add_trace(go.Bar(
        x=best_p0_cand[_order], y=_names, orientation="h",
        name="phase 0 (top candidate)",
        marker=dict(color=PALETTE[0], line=dict(width=0)),
        legendgroup="cand_p0",
    ), row=1, col=1)
    _fig.add_trace(go.Bar(
        x=best_p1_cand[_order], y=_names, orientation="h",
        name="phase 1 (top candidate)",
        marker=dict(color=PALETTE[1], line=dict(width=0)),
        legendgroup="cand_p1",
    ), row=1, col=1)

    _fig.add_trace(go.Bar(
        x=(_ep0_base + _ep1_base)[_order], y=_names, orientation="h",
        name="proportional baseline (total epochs)",
        marker=dict(color="rgba(140,140,140,0.55)", line=dict(width=0)),
        legendgroup="baseline_e",
    ), row=1, col=2)
    _fig.add_trace(go.Bar(
        x=_ep0_cand[_order], y=_names, orientation="h",
        name="phase 0 epochs (top candidate)",
        marker=dict(color=PALETTE[0], line=dict(width=0)),
        legendgroup="cand_p0_e",
    ), row=1, col=2)
    _fig.add_trace(go.Bar(
        x=_ep1_cand[_order], y=_names, orientation="h",
        name="phase 1 epochs (top candidate)",
        marker=dict(color=PALETTE[1], line=dict(width=0)),
        legendgroup="cand_p1_e",
    ), row=1, col=2)

    _fig.update_xaxes(title_text="weight", row=1, col=1)
    _fig.update_xaxes(title_text="epochs over full source", row=1, col=2)
    _fig.update_yaxes(tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)")
    _fig.update_layout(
        height=max(500, 26 * len(_names)),
        width=1500,
        barmode="group",
        bargap=0.18,
    )
    style_fig(
        _fig,
        title=(
            f"Top candidate by LCB = mean − {KAPPA}·σ: predicted aggregate {best_pred_cand:+.3f} ± {best_std_cand:.3f} (LCB {best_lcb_cand:+.3f}) vs proportional baseline<br>"
            f"<sub>left: weight space (per-phase mass) · right: epoch space (mass × c, full-source passes) · sorted by baseline phase-0 weight</sub>"
        ),
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 6 — Thompson sampling on the simplex via the bootstrap

    For each of the 200 bootstrap resamples of the ridge head, find the simplex (w₀, w₁) point that argmaxes predicted aggregate under that posterior sample. Frank-Wolfe stays exactly on the simplex (no projection step). Two phases handled independently.

    Result: a posterior distribution of optimal mixtures. Domains with negative β get pushed to 0 in most samples; domains with high β + adequate `N_d` get high weight; per-domain spread captures coefficient uncertainty.
    """)
    return


@app.cell
def _(
    a_p3,
    boot_coefs,
    domains,
    epochs_per_unit_w0,
    epochs_per_unit_w1,
    eta_p3,
    np,
    p3_design_sd,
    p_pen_p3,
):
    _D = len(domains)
    _coef_natural_boot = boot_coefs / p3_design_sd[None, :]
    _beta_signal = _coef_natural_boot[:, :_D]
    _beta_pen0 = _coef_natural_boot[:, _D]
    _beta_pen1 = _coef_natural_boot[:, _D + 1]
    _c0 = epochs_per_unit_w0
    _c1 = epochs_per_unit_w1
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
            _ne0 = np.maximum(_c0 * _w0, _eps)
            _ne1 = np.maximum(_c1 * _w1, _eps)
            _pg0 = _bp0 * p_pen_p3 * np.power(_ne0, p_pen_p3 - 1.0)
            _pg1 = _bp1 * p_pen_p3 * np.power(_ne1, p_pen_p3 - 1.0)
            _g0 = _bs * _sig_grad - _c0 * _pg0
            _g1 = _bs * eta_p3 * _sig_grad - _c1 * _pg1
            _gamma = 2.0 / (_t + 2)
            _w0 = (1 - _gamma) * _w0
            _w0[int(np.argmax(_g0))] += _gamma
            _w1 = (1 - _gamma) * _w1
            _w1[int(np.argmax(_g1))] += _gamma
        ts_w0[_i] = _w0
        ts_w1[_i] = _w1

    f"Frank-Wolfe Thompson sampling done — {_B} simplex argmaxes, {_T} iters each, {_D} domains."
    return ts_w0, ts_w1


@app.cell
def _(
    PALETTE,
    domains,
    epochs_per_unit_w0,
    epochs_per_unit_w1,
    go,
    make_subplots,
    np,
    pl,
    raw,
    style_fig,
    ts_w0,
    ts_w1,
):
    _comp = [d[:45] for d in domains]
    _w0_mean = ts_w0.mean(axis=0)
    _w1_mean = ts_w1.mean(axis=0)

    _br = raw.filter(pl.col("run_name") == "baseline_proportional").row(0, named=True)
    _bp0 = np.array([_br[f"phase_0_{d}"] for d in domains])
    _bp1 = np.array([_br[f"phase_1_{d}"] for d in domains])

    _order = np.argsort(-_bp0)
    _names = [_comp[i] for i in _order]

    _ep0_base = _bp0 * epochs_per_unit_w0
    _ep1_base = _bp1 * epochs_per_unit_w1
    _ep0_mean = _w0_mean * epochs_per_unit_w0
    _ep1_mean = _w1_mean * epochs_per_unit_w1

    _fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False, horizontal_spacing=0.18,
        subplot_titles=("weight space", "epoch space (full-source passes)"),
    )

    _fig.add_trace(go.Bar(
        x=_bp0[_order], y=_names, orientation="h",
        name="proportional baseline (weight)",
        marker=dict(color="rgba(140,140,140,0.55)", line=dict(width=0)),
    ), row=1, col=1)
    _fig.add_trace(go.Bar(
        x=_w0_mean[_order], y=_names, orientation="h",
        name="phase 0 (Thompson mean)",
        marker=dict(color=PALETTE[0], line=dict(width=0)),
    ), row=1, col=1)
    _fig.add_trace(go.Bar(
        x=_w1_mean[_order], y=_names, orientation="h",
        name="phase 1 (Thompson mean)",
        marker=dict(color=PALETTE[1], line=dict(width=0)),
    ), row=1, col=1)

    _fig.add_trace(go.Bar(
        x=(_ep0_base + _ep1_base)[_order], y=_names, orientation="h",
        name="proportional baseline (total epochs)",
        marker=dict(color="rgba(140,140,140,0.55)", line=dict(width=0)),
    ), row=1, col=2)
    _fig.add_trace(go.Bar(
        x=_ep0_mean[_order], y=_names, orientation="h",
        name="phase 0 epochs (Thompson mean)",
        marker=dict(color=PALETTE[0], line=dict(width=0)),
    ), row=1, col=2)
    _fig.add_trace(go.Bar(
        x=_ep1_mean[_order], y=_names, orientation="h",
        name="phase 1 epochs (Thompson mean)",
        marker=dict(color=PALETTE[1], line=dict(width=0)),
    ), row=1, col=2)

    _fig.update_xaxes(title_text="weight", row=1, col=1)
    _fig.update_xaxes(title_text="epochs over full source", row=1, col=2)
    _fig.update_yaxes(tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)")
    _fig.update_layout(
        height=max(500, 26 * len(_names)), width=1500,
        barmode="group", bargap=0.18,
    )
    _n_zero_p0 = int((_w0_mean < 1e-3).sum())
    _n_zero_p1 = int((_w1_mean < 1e-3).sum())
    style_fig(
        _fig,
        title=(
            f"Thompson-mean optimal mixture (mean over 200 bootstrap simplex argmaxes) vs proportional baseline<br>"
            f"<sub>{_n_zero_p0}/{len(_names)} domains near-zero phase-0 · {_n_zero_p1}/{len(_names)} near-zero phase-1 · sorted by baseline phase-0 weight</sub>"
        ),
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
