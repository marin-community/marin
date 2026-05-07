"""Render the 300M data-mixture pipeline figures for the slidev deck.

Reuses the cached P3 fit (`~/analysis/p3_fit_300m.json`) to skip the slow nested
5×5 CV; re-runs the (fast) bootstrap and Frank-Wolfe Thompson sampling to
produce the recommended-mixture chart, then renders the per-task scaling
validation chart from the hardcoded measured BPBs.

Usage:
    uv run --with plotly --with kaleido --with polars --with scipy --with numpy \\
        scripts/render_data_mix_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from scipy.optimize import nnls as _nnls

ANALYSIS_DIR = Path.home() / "analysis"
OUT_DIR = Path(__file__).resolve().parent.parent / "public" / "charts" / "data-mix"
PALETTE = ["#1877F2", "#F0701A", "#5A24C7", "#E42C97"]
EPS = 1e-4


def nnls_ridge(M, y, alpha):
    p = M.shape[1]
    if alpha > 0.0:
        M_aug = np.vstack([M, np.sqrt(alpha) * np.eye(p)])
        y_aug = np.concatenate([y, np.zeros(p)])
    else:
        M_aug, y_aug = M, y
    coef, _ = _nnls(M_aug, y_aug, maxiter=200 * p)
    return coef


def style_fig(fig, title=None):
    layout = dict(
        template="plotly_white",
        margin=dict(l=60, r=30, t=70, b=100),
        legend=dict(
            font=dict(size=11), orientation="h",
            yanchor="top", y=-0.15, xanchor="center", x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(200,200,200,0.5)", borderwidth=1,
        ),
    )
    if title is not None:
        layout["title"] = dict(text=title, font=dict(size=18), x=0.5, xanchor="center")
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(220,220,220,0.5)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(220,220,220,0.5)")
    return fig


def load_swarm():
    raw = pl.read_csv(
        ANALYSIS_DIR / "raw_metric_matrix_300m.csv", infer_schema_length=500
    ).filter(pl.col("status") == "completed")
    noise_df = pl.read_csv(
        ANALYSIS_DIR / "noise_baseline_run00097_300m.csv", infer_schema_length=200
    )

    MMLU_KEEP = {"lm_eval/mmlu_sl_verb_5shot/bpb"}
    AGG_DROP = {
        "eval/bpb", "eval/macro_bpb",
        "eval/paloma/bpb", "eval/paloma/macro_bpb",
        "eval/uncheatable_eval/bpb", "eval/uncheatable_eval/macro_bpb",
    }
    TASK_DROP = {
        "teacher_forced/gsm8k_5shot_answer_hash/bpb",
        "mcq_smooth/sciq_5shot/bpb",
    }

    def keep(c):
        if not c.endswith("/bpb"):
            return False
        if c in AGG_DROP or c in TASK_DROP:
            return False
        if not c.startswith((
            "eval/uncheatable_eval/", "lm_eval/", "mcq_smooth/", "teacher_forced/",
        )):
            return False
        if c.startswith("lm_eval/mmlu_") and c not in MMLU_KEEP:
            return False
        return True

    candidates = [c for c in raw.columns if keep(c)]
    raw_cols = set(raw.columns)
    task_cols, task_signs = [], []
    for c in candidates:
        base = c.removesuffix("/bpb")
        if base.startswith(("lm_eval/", "mcq_smooth/")):
            alt = base + "/choice_logprob"
            if alt in raw_cols:
                task_cols.append(alt)
                task_signs.append(+1.0)
                continue
        task_cols.append(c)
        task_signs.append(-1.0)
    task_signs = np.array(task_signs)
    domains = sorted(c.removeprefix("phase_0_") for c in raw.columns if c.startswith("phase_0_"))

    meta = pl.read_csv(ANALYSIS_DIR / "two_phase_many_epoch_metadata.csv")
    row_lookup = {row["domain_name"]: row for row in meta.iter_rows(named=True)}
    c0 = np.array([row_lookup[d]["phase_0_epoch_multiplier"] for d in domains], dtype=np.float64)
    c1 = np.array([row_lookup[d]["phase_1_epoch_multiplier"] for d in domains], dtype=np.float64)
    return raw, noise_df, task_cols, task_signs, domains, c0, c1


def nnfa_aggregate(raw, noise_df, task_cols, task_signs, K=4):
    X = raw.select(task_cols).to_numpy().astype(np.float64) * task_signs[None, :]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = (X - mu) / sd
    n, p = Z.shape

    noise_cols = set(noise_df.columns)
    has_noise = np.array([c in noise_cols for c in task_cols])
    present = [c for c in task_cols if c in noise_cols]
    signs_present = task_signs[has_noise]
    nX = noise_df.select(present).to_numpy().astype(np.float64) * signs_present[None, :]
    ns_p = nX.std(axis=0, ddof=1)
    ss_p = X[:, has_noise].std(axis=0, ddof=1)
    noise_share = np.full(len(task_cols), np.nan)
    noise_share[has_noise] = (ns_p / ss_p) ** 2

    psi_anchor = np.where(np.isnan(noise_share), np.nan, np.clip(noise_share, 1e-3, 0.999))
    psi_fixed = ~np.isnan(psi_anchor)
    rng = np.random.default_rng(0)
    Lam = np.abs(rng.normal(scale=0.1, size=(p, K)))
    Psi = np.where(psi_fixed, psi_anchor, 1.0)
    for _ in range(5000):
        Lpi = Lam / Psi[:, None]
        V = np.linalg.inv(np.eye(K) + Lam.T @ Lpi)
        Th = Z @ Lpi @ V
        S = n * V + Th.T @ Th
        ZtT = Z.T @ Th
        Ln = ZtT @ np.linalg.inv(S)
        Ln = np.clip(Ln, 0.0, None)
        Pf = (
            (Z ** 2).mean(axis=0)
            - 2 * (ZtT * Ln).sum(axis=1) / n
            + ((Ln @ S) * Ln).sum(axis=1) / n
        )
        Pf = np.clip(Pf, 1e-6, None)
        Pn = np.where(psi_fixed, psi_anchor, Pf)
        if np.max(np.abs(Ln - Lam)) < 1e-7:
            Lam, Psi = Ln, Pn
            break
        Lam, Psi = Ln, Pn
    order = np.argsort(-(Lam ** 2).sum(axis=0))
    Lam = Lam[:, order]
    Lpi = Lam / Psi[:, None]
    V = np.linalg.inv(np.eye(K) + Lam.T @ Lpi)
    proj = (Lpi @ V).mean(axis=1)
    return Z @ proj


def build_p3_design(w0, w1, c0, c1, a, p_pen, nu):
    sig0 = np.power(np.maximum(w0, EPS), a)
    sig1 = np.power(np.maximum(w1, EPS), a)
    sig = sig0 + nu * sig1
    ne0 = np.maximum(w0 * c0[None, :], EPS)
    ne1 = np.maximum(w1 * c1[None, :], EPS)
    pen0 = np.power(ne0, p_pen).sum(axis=1, keepdims=True)
    pen1 = np.power(ne1, p_pen).sum(axis=1, keepdims=True)
    return np.column_stack([sig, -pen0, -pen1])


def bootstrap_and_thompson(w0, w1, c0, c1, agg, a, p_pen, nu, alpha,
                           n_boot=200, n_iter=300, seed=7):
    D = build_p3_design(w0, w1, c0, c1, a, p_pen, nu)
    mask = D.std(0) > 1e-10
    D_kept = D[:, mask]
    mu, sd = D_kept.mean(0), D_kept.std(0)
    Ds = (D_kept - mu) / sd
    rng = np.random.default_rng(seed)
    boot_coefs = np.zeros((n_boot, Ds.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, len(agg), len(agg))
        boot_coefs[b] = nnls_ridge(Ds[idx], agg[idx] - agg[idx].mean(), alpha)

    n_dom = len(c0)
    coef_natural = boot_coefs / sd[None, :]
    beta_signal = coef_natural[:, :n_dom]
    beta_pen0 = coef_natural[:, n_dom]
    beta_pen1 = coef_natural[:, n_dom + 1]

    eps_fw = 1e-12
    ts_w0 = np.empty((n_boot, n_dom))
    ts_w1 = np.empty((n_boot, n_dom))
    for i in range(n_boot):
        bs, bp0, bp1 = beta_signal[i], beta_pen0[i], beta_pen1[i]
        cw0 = np.full(n_dom, 1.0 / n_dom)
        cw1 = np.full(n_dom, 1.0 / n_dom)
        for t in range(n_iter):
            w0c = np.maximum(cw0, eps_fw)
            w1c = np.maximum(cw1, eps_fw)
            sg0 = a * np.power(w0c, a - 1.0)
            sg1 = a * nu * np.power(w1c, a - 1.0)
            ne0 = np.maximum(c0 * cw0, eps_fw)
            ne1 = np.maximum(c1 * cw1, eps_fw)
            pg0 = bp0 * p_pen * np.power(ne0, p_pen - 1.0)
            pg1 = bp1 * p_pen * np.power(ne1, p_pen - 1.0)
            g0 = bs * sg0 - c0 * pg0
            g1 = bs * sg1 - c1 * pg1
            gamma = 2.0 / (t + 2)
            cw0 = (1 - gamma) * cw0
            cw0[int(np.argmax(g0))] += gamma
            cw1 = (1 - gamma) * cw1
            cw1[int(np.argmax(g1))] += gamma
        ts_w0[i] = cw0
        ts_w1[i] = cw1
    return ts_w0, ts_w1


def render_thompson(raw, domains, c0, c1, ts_w0, ts_w1, out_path, top_n=18):
    w0_mean = ts_w0.mean(axis=0)
    w1_mean = ts_w1.mean(axis=0)
    br = raw.filter(pl.col("run_name") == "baseline_proportional").row(0, named=True)
    bp0 = np.array([br[f"phase_0_{d}"] for d in domains])
    bp1 = np.array([br[f"phase_1_{d}"] for d in domains])

    # Keep the top-N domains by max-weight across (baseline_p0, baseline_p1, ts_p0, ts_p1).
    # Drops domains the baseline ignored AND Thompson zeroed out — they add no information.
    max_weight = np.maximum.reduce([bp0, bp1, w0_mean, w1_mean])
    keep = np.argsort(-max_weight)[:top_n]
    keep = keep[np.argsort(-w0_mean[keep] - w1_mean[keep])]
    names = [domains[i][:45] for i in keep]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=bp0[keep], y=names, orientation="h",
                         name="baseline phase 0",
                         marker=dict(color="rgba(140,140,140,0.85)", line=dict(width=0))))
    fig.add_trace(go.Bar(x=bp1[keep], y=names, orientation="h",
                         name="baseline phase 1",
                         marker=dict(color="rgba(200,200,200,0.85)", line=dict(width=0))))
    fig.add_trace(go.Bar(x=w0_mean[keep], y=names, orientation="h",
                         name="recommended phase 0",
                         marker=dict(color=PALETTE[0], line=dict(width=0))))
    fig.add_trace(go.Bar(x=w1_mean[keep], y=names, orientation="h",
                         name="recommended phase 1",
                         marker=dict(color=PALETTE[1], line=dict(width=0))))
    fig.update_xaxes(title_text="mixture weight")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11),
                     gridcolor="rgba(0,0,0,0)")
    fig.update_layout(barmode="group", bargap=0.22, height=560, width=1400)
    n_zero_p0 = int((w0_mean < 1e-3).sum())
    n_zero_p1 = int((w1_mean < 1e-3).sum())
    style_fig(fig, title=(
        f"Recommended mixture vs proportional baseline — top {top_n} of {len(domains)} sources<br>"
        f"<sub>{n_zero_p0}/{len(domains)} sources zeroed out in phase 0 · "
        f"{n_zero_p1}/{len(domains)} in phase 1</sub>"
    ))
    fig.write_image(out_path, width=1500, height=560, scale=2)


def render_scaling(out_path):
    flops = np.array([2.19e17, 1.70e18, 9.00e18, 2.83e19])
    dim_labels = ["d512", "d768", "d1024", "d1280"]
    tasks = [
        ("MMLU 0-shot bpb",
         np.array([0.844, 0.672, 0.466, 0.441]),
         np.array([0.709, 0.631, 0.442, 0.372])),
        ("MMLU 5-shot bpb",
         np.array([0.549, 0.369, 0.242, 0.217]),
         np.array([0.556, 0.352, 0.201, 0.204])),
        ("GSM8K 5-shot bpb",
         np.array([1.294, 0.964, 0.779, 0.682]),
         np.array([1.322, 1.053, 0.767, 0.632])),
        ("HumanEval 10-shot bpb",
         np.array([0.865, 0.715, 0.574, 0.487]),
         np.array([0.991, 0.763, 0.624, 0.534])),
        ("ARC 5-shot bpb",
         np.array([1.494, 1.299, 1.130, 1.028]),
         np.array([1.490, 1.277, 1.094, 1.005])),
    ]

    def fit_loglog(x, y):
        lx, ly = np.log10(x), np.log10(y)
        m, b = np.polyfit(lx, ly, 1)
        ss_res = ((ly - (m * lx + b)) ** 2).sum()
        ss_tot = ((ly - ly.mean()) ** 2).sum()
        return float(m), float(b), float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    logF = np.log10(flops)
    xgrid = np.linspace(logF.min() - 0.10, logF.max() + 0.20, 80)
    x_pred = 10 ** xgrid

    fig = make_subplots(rows=1, cols=5, shared_yaxes=False, horizontal_spacing=0.04,
                        subplot_titles=[t[0] for t in tasks])
    for col, (label, yp, yo) in enumerate(tasks, start=1):
        mp, bp, rp = fit_loglog(flops, yp)
        mo, bo, ro = fit_loglog(flops, yo)
        yfit_p = 10 ** (mp * xgrid + bp)
        yfit_o = 10 ** (mo * xgrid + bo)
        show = col == 1
        fig.add_trace(go.Scatter(x=x_pred, y=yfit_p, mode="lines",
                                 line=dict(color="rgba(140,140,140,0.55)", width=2, dash="dash"),
                                 name="Baseline fit", legendgroup="prop", showlegend=show),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=flops, y=yp, mode="markers+text",
                                 marker=dict(size=12, color="rgba(110,110,110,0.95)",
                                             line=dict(width=1, color="white")),
                                 text=dim_labels, textposition="top center",
                                 textfont=dict(size=9, color="rgba(80,80,80,0.85)"),
                                 name="Baseline", legendgroup="prop", showlegend=show),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=x_pred, y=yfit_o, mode="lines",
                                 line=dict(color=PALETTE[0], width=2, dash="dash"),
                                 name="Recommended fit", legendgroup="opt", showlegend=show),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=flops, y=yo, mode="markers+text",
                                 marker=dict(size=12, color=PALETTE[0],
                                             line=dict(width=1, color="white")),
                                 text=dim_labels, textposition="bottom center",
                                 textfont=dict(size=9, color=PALETTE[0]),
                                 name="Recommended", legendgroup="opt", showlegend=show),
                      row=1, col=col)
        fig.layout.annotations[col - 1].text = (
            f"{label}<br><sub>prop slope={mp:+.3f} (R²={rp:.2f}) · "
            f"opt slope={mo:+.3f} (R²={ro:.2f})</sub>"
        )
        fig.update_xaxes(type="log", title="training FLOPs", row=1, col=col,
                         showgrid=True, gridcolor="rgba(220,220,220,0.5)")
        fig.update_yaxes(type="log", title="bpb" if col == 1 else "",
                         showgrid=True, gridcolor="rgba(220,220,220,0.5)", row=1, col=col)
    fig.update_layout(height=560, width=2000, margin=dict(t=120, l=60, r=30, b=80))
    style_fig(fig, title="Bits-per-byte across model widths: recommended mixture vs proportional baseline")
    fig.update_layout(title=dict(y=0.98, yanchor="top"))
    for ann in fig.layout.annotations:
        if ann.yref in (None, "paper") and getattr(ann, "y", 0) and ann.y > 0.9:
            ann.y = 0.93
    fig.write_image(out_path, width=2000, height=560, scale=2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Rendering scaling-validation.png (no fit needed)...")
    render_scaling(OUT_DIR / "scaling-validation.png")

    print("Loading swarm CSVs...")
    raw, noise_df, task_cols, task_signs, domains, c0, c1 = load_swarm()
    print(f"  {raw.height} runs · {len(task_cols)} tasks · {len(domains)} domains")

    print("Building NNFA aggregate (k=4)...")
    agg = nnfa_aggregate(raw, noise_df, task_cols, task_signs, K=4)

    w0_raw = raw.select([f"phase_0_{d}" for d in domains]).to_numpy().astype(np.float64)
    w1_raw = raw.select([f"phase_1_{d}" for d in domains]).to_numpy().astype(np.float64)
    w0 = w0_raw / w0_raw.sum(axis=1, keepdims=True)
    w1 = w1_raw / w1_raw.sum(axis=1, keepdims=True)

    fit = json.loads((ANALYSIS_DIR / "p3_fit_300m.json").read_text())["nonlinear_params"]
    a, p_pen, nu, alpha = fit["a"], fit["p_penalty_exponent"], fit["nu_phase1_leverage"], fit["ridge_alpha"]
    print(f"  cached P3 fit: a={a}, p={p_pen}, ν={nu}, α={alpha}")

    print("Bootstrap + Thompson-FW (200 draws × 300 iter)...")
    ts_w0, ts_w1 = bootstrap_and_thompson(w0, w1, c0, c1, agg, a, p_pen, nu, alpha)

    print("Rendering thompson-mixture.png...")
    render_thompson(raw, domains, c0, c1, ts_w0, ts_w1, OUT_DIR / "thompson-mixture.png")

    print(f"Done. Wrote PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
