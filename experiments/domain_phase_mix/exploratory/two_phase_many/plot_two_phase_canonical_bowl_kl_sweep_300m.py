# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""Plot local KL sweeps for two-phase DSP-family candidate optima.

This is a diagnostic companion to
`materialize_two_phase_canonical_bowl_candidates_300m.py`. It does not write
training mixtures; it recomputes the local surrogate optima over a denser KL
grid and plots the tradeoff between predicted objective, distance from
proportional, and simulated epoch exposure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_two_phase_canonical_bowl_candidates_300m as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "reference_outputs" / "two_phase_dsp_canonical_bowl_kl_sweep_20260703"
)
DEFAULT_KL_REGS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)


def kl_label(kl_reg: float) -> str:
    if kl_reg == 0:
        return "0"
    return f"{kl_reg:g}"


def fit_predictors(objective: str, models: set[str]):
    packet, domains, natural, token_counts, target_budget, _folds = candidates.load_objective(objective)
    dsp.LINEAR_REG = candidates.LINEAR_REG
    predict_fns = {}
    if "canonical" in models:
        print(f"  fitting {objective}/canonical", flush=True)
        canonical, _ = candidates.phase_dsp.fit_variant_with_l2(
            packet,
            "canonical",
            candidates.LINEAR_REG,
            maxiter=40,
            coarse_top_k=3,
            basin_hopping_iters=0,
        )
        predict_fns["canonical"] = lambda w: float(dsp.predict(canonical, w[None, :, :])[0])
    if "effective_exposure" in models:
        print(f"  fitting {objective}/effective_exposure", flush=True)
        effexp, _ = candidates.phase_dsp.fit_variant_with_l2(
            packet,
            "effective_exposure",
            candidates.LINEAR_REG,
            maxiter=40,
            coarse_top_k=3,
            basin_hopping_iters=0,
        )
        predict_fns["effective_exposure"] = lambda w: float(dsp.predict(effexp, w[None, :, :])[0])
    if "asymmetric_bowl" in models:
        print(f"  fitting {objective}/asymmetric_bowl", flush=True)
        bowl = candidates.fit_asymmetric_bowl(packet, candidates.LINEAR_REG)
        predict_fns["asymmetric_bowl"] = lambda w: float(
            candidates.abowl_predict(w[None, :, :], packet.c0, packet.c1, bowl)[0]
        )
    return packet, domains, natural, token_counts, target_budget, predict_fns


def weights_to_logits(weights: np.ndarray) -> np.ndarray:
    return np.log(np.clip(weights, 1e-12, 1.0)).reshape(-1)


def optimize_fast(
    predict_fn,
    m: int,
    natural: np.ndarray,
    kl_reg: float,
    starts: list[np.ndarray],
    maxiter: int,
) -> np.ndarray:
    def to_w(logits: np.ndarray) -> np.ndarray:
        out = np.zeros((2, m))
        for ph in range(2):
            zz = logits[ph * m : (ph + 1) * m]
            exp = np.exp(zz - zz.max())
            out[ph] = exp / exp.sum()
        return out

    def obj(logits: np.ndarray) -> float:
        weights = to_w(logits)
        predicted = float(predict_fn(weights))
        if kl_reg <= 0:
            return predicted
        return predicted + kl_reg * float(base.weighted_multiclass_kl(weights, natural, base.PHASE_FRACTIONS))

    best_value = np.inf
    best_weights = None
    for start in starts:
        res = minimize(obj, start, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-8})
        if float(res.fun) < best_value:
            best_value = float(res.fun)
            best_weights = to_w(np.asarray(res.x, float))
    assert best_weights is not None
    return best_weights


def sweep_objective(
    objective: str,
    kl_regs: list[float],
    start_top_k: int,
    maxiter: int,
    models: set[str],
    output_dir: Path,
    write_mixtures: bool,
) -> list[dict[str, object]]:
    packet, domains, natural, token_counts, target_budget, predict_fns = fit_predictors(objective, models)
    starts = candidates.opt_starts(packet, packet.m, k=start_top_k)
    reference = np.stack([natural, natural], axis=0)
    rows: list[dict[str, object]] = []
    for model_name, predict_fn in predict_fns.items():
        warm_starts = list(starts)
        for kl_reg in sorted(kl_regs, reverse=True):
            print(f"    optimizing {objective}/{model_name} KL={kl_reg:g}", flush=True)
            weights = optimize_fast(predict_fn, packet.m, natural, kl_reg, warm_starts, maxiter)
            warm_starts = [weights_to_logits(weights), *starts]
            simulated_epochs = base.simulated_epochs(weights, token_counts, target_budget=target_budget)
            observed_distances = dsp.average_phase_tv_distance(packet.w, weights[None, :, :])
            nearest_idx = int(np.argmin(observed_distances))
            weights_csv = ""
            if write_mixtures:
                mixture_dir = output_dir / objective / model_name / f"kl_{kl_label(kl_reg).replace('.', 'p')}"
                mixture_dir.mkdir(parents=True, exist_ok=True)
                weights_csv = str(mixture_dir / "proposed_mixture_weights.csv")
                candidates.per_component.mixture_frame(
                    domains=domains,
                    natural=natural,
                    weights=weights,
                    token_counts=token_counts,
                    target_budget=target_budget,
                ).to_csv(weights_csv, index=False)
            rows.append(
                {
                    "objective": objective,
                    "model": model_name,
                    "kl_reg": kl_reg,
                    "kl_label": kl_label(kl_reg),
                    "predicted_bpb": float(predict_fn(weights)),
                    "best_observed_bpb": float(np.min(packet.y)),
                    "nearest_observed_bpb": float(packet.y[nearest_idx]),
                    "nearest_observed_tv": float(observed_distances[nearest_idx]),
                    "tv_to_proportional": float(0.5 * np.abs(weights - reference).sum(axis=1).mean()),
                    "max_weight": float(np.max(weights)),
                    "max_simulated_epoch": float(np.max(simulated_epochs)),
                    "q95_simulated_epoch": float(np.quantile(simulated_epochs, 0.95)),
                    "weights_csv": weights_csv,
                }
            )
    return rows


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path, kl_regs: list[float]) -> None:
    subplot_titles = [f"{objective}: {metric}" for objective in ("table9", "uncheatable")]
    fig = make_subplots(rows=1, cols=2, subplot_titles=subplot_titles, shared_yaxes=False)
    palette = px.colors.qualitative.Dark2
    models = list(df["model"].drop_duplicates())
    colors = {model: palette[i % len(palette)] for i, model in enumerate(models)}
    labels = [kl_label(k) for k in kl_regs]
    for col, objective in enumerate(("table9", "uncheatable"), start=1):
        sub = df[df["objective"] == objective].copy()
        for model in models:
            mdf = sub[sub["model"] == model].sort_values("kl_reg")
            fig.add_trace(
                go.Scatter(
                    x=mdf["kl_label"],
                    y=mdf[metric],
                    mode="lines+markers",
                    name=model,
                    legendgroup=model,
                    showlegend=col == 1,
                    line={"color": colors[model]},
                    marker={"color": colors[model], "size": 8},
                    hovertemplate=(
                        "model=%{customdata[0]}<br>"
                        "KL=%{x}<br>"
                        f"{metric}=%{{y:.5g}}<br>"
                        "predicted_bpb=%{customdata[1]:.5g}<br>"
                        "max_epoch=%{customdata[2]:.4g}<br>"
                        "TV=%{customdata[3]:.4g}<extra></extra>"
                    ),
                    customdata=np.column_stack(
                        [
                            mdf["model"],
                            mdf["predicted_bpb"],
                            mdf["max_simulated_epoch"],
                            mdf["tv_to_proportional"],
                        ]
                    ),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(
            title_text="KL regularization",
            categoryorder="array",
            categoryarray=labels,
            row=1,
            col=col,
        )
    fig.update_layout(
        title=f"Two-phase DSP-family local KL sweep: {metric}",
        template="plotly_white",
        width=1250,
        height=520,
        legend_title_text="Surrogate",
    )
    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": output_path.stem,
            "scale": 4,
        }
    }
    output_path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True, config=config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kl-regs", default=",".join(str(k) for k in DEFAULT_KL_REGS))
    parser.add_argument("--models", default="canonical,asymmetric_bowl")
    parser.add_argument("--start-top-k", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--write-mixtures", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kl_regs = [float(k.strip()) for k in args.kl_regs.split(",") if k.strip()]
    models = {m.strip() for m in args.models.split(",") if m.strip()}

    rows = []
    for objective in ("table9", "uncheatable"):
        print(f"==== {objective} ====", flush=True)
        rows.extend(
            sweep_objective(
                objective,
                kl_regs,
                args.start_top_k,
                args.maxiter,
                models,
                args.output_dir,
                args.write_mixtures,
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "kl_sweep_diagnostics.csv", index=False)
    for metric in ("predicted_bpb", "tv_to_proportional", "max_simulated_epoch", "max_weight"):
        plot_metric(df, metric, args.output_dir / f"kl_sweep_{metric}.html", kl_regs)
    print(df.to_string(index=False))
    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
