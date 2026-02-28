# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "scikit-learn", "plotly"]
# ///
"""Interactive 3D visualization of three-phase, two-domain weight space.

Produces interactive HTML files:
  1. scatter_3d.html              — 3D scatter of data points colored by BPB
  2. model_optima_3d.html         — predicted optima per model in 3D
  3. isosurface_3d.html           — 3D isosurface of model predictions (dropdown)
  4. slices_fix_phase{0,1,2}.html — 2D contour slices (small multiples + slider)
  5. optimum_trajectory_3d.html   — how predicted optima drift as training size grows

Usage:
  uv run three_phase_visualization.py
  uv run three_phase_visualization.py --target "lm_eval/arc_challenge/bpb"
  uv run three_phase_visualization.py --quick   # skip CV ranking, show all models
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from general_scaling_models import GENERAL_MODELS, DatasetSpec  # noqa: E402

SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "three_phase_starcoder.csv"
DEFAULT_TARGET = "eval/paloma/dolma_100_programing_languages/bpb"

# Domain configuration (matches three_phase_starcoder_experiment.py)
NEMOTRON_TOKENS = 5_729_908_864_777
STARCODER_TOKENS = 217_000_000_000
TARGET_BUDGET = NEMOTRON_TOKENS
PHASE_FRACS = np.array([0.33, 0.34, 0.33])
DOMAIN_NAMES = ["nemotron_full", "starcoder"]
PHASE_NAMES = ["phase_0", "phase_1", "phase_2"]

# Visualization parameters
GRID_3D_RES = 40
GRID_2D_RES = 50
N_SLIDER_STEPS = 12
N_TOP_MODELS = 8
COLORSCALE = "RdYlGn_r"

MARKER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
]

sys.stdout.reconfigure(line_buffering=True)


# =========================================================================
# Data loading
# =========================================================================
def build_epoch_multipliers():
    """Build (3, 2) epoch multiplier array for three-phase starcoder."""
    domain_tokens = np.array([NEMOTRON_TOKENS, STARCODER_TOKENS])
    return PHASE_FRACS[:, None] * TARGET_BUDGET / domain_tokens[None, :]


def load_spec(target_col):
    """Load CSV and build DatasetSpec for model fitting."""
    df = pd.read_csv(CSV_PATH)
    df = df[df["status"] == "completed"].reset_index(drop=True)

    R = len(df)
    weights = np.zeros((R, 3, 2))
    for ki, ph in enumerate(PHASE_NAMES):
        for di, dom in enumerate(DOMAIN_NAMES):
            weights[:, ki, di] = df[f"{ph}_{dom}"].values

    y = df[target_col].values.astype(float)
    valid = ~np.isnan(y)
    if not valid.all():
        n_drop = int((~valid).sum())
        print(f"Dropping {n_drop} rows with NaN target")
        weights, y, df = weights[valid], y[valid], df[valid].reset_index(drop=True)

    spec = DatasetSpec(
        weights=weights,
        y=y,
        epoch_multipliers=build_epoch_multipliers(),
        domain_names=DOMAIN_NAMES,
        phase_names=PHASE_NAMES,
        small_domains=[1],
        name="three_phase_starcoder",
    )
    return spec, df


# =========================================================================
# Grid construction
# =========================================================================
def build_2d_slice_weights(fixed_phase, fixed_val, g):
    """Build (|g|^2, 3, 2) weight array for a 2D slice with one phase fixed."""
    pa, pb = np.meshgrid(g, g, indexing="ij")
    R = len(g) ** 2
    W = np.zeros((R, 3, 2))
    free = [i for i in range(3) if i != fixed_phase]

    W[:, fixed_phase, 1] = fixed_val
    W[:, fixed_phase, 0] = 1 - fixed_val
    W[:, free[0], 1] = pa.ravel()
    W[:, free[0], 0] = 1 - pa.ravel()
    W[:, free[1], 1] = pb.ravel()
    W[:, free[1], 0] = 1 - pb.ravel()
    return W


def build_3d_grid(resolution):
    """Build (resolution^3, 3, 2) weight array on a regular 3D grid."""
    g = np.linspace(0.005, 0.995, resolution)
    p0, p1, p2 = np.meshgrid(g, g, g, indexing="ij")
    R = resolution**3
    W = np.zeros((R, 3, 2))
    W[:, 0, 1] = p0.ravel()
    W[:, 0, 0] = 1 - p0.ravel()
    W[:, 1, 1] = p1.ravel()
    W[:, 1, 0] = 1 - p1.ravel()
    W[:, 2, 1] = p2.ravel()
    W[:, 2, 0] = 1 - p2.ravel()
    return W, g, p0, p1, p2


def find_3d_optimum(pred_fn, resolution=GRID_3D_RES):
    """Find the predicted optimum in 3D weight space via grid search."""
    W, g, p0, p1, p2 = build_3d_grid(resolution)
    Z = pred_fn(W)
    opt_idx = int(np.argmin(Z))
    return (
        float(p0.ravel()[opt_idx]),
        float(p1.ravel()[opt_idx]),
        float(p2.ravel()[opt_idx]),
        float(Z[opt_idx]),
    )


# =========================================================================
# Model fitting and ranking
# =========================================================================
def fit_and_rank_models(spec, n_top, quick=False, model_names=None):
    """Fit all applicable models, rank by CV R^2, return top N.

    If model_names is provided, fit only those models (by exact name match)
    and skip ranking — return all that succeed.
    """
    fitted = {}
    cv_scores = {}

    if model_names:
        name_set = set(model_names)
        candidates = [m for m in GENERAL_MODELS if m.name in name_set]
    else:
        candidates = GENERAL_MODELS

    print(f"\nFitting models (R={spec.R}, N={spec.N}, M={spec.M})...")
    for model in candidates:
        if not model.applicable(spec):
            print(f"  {model.name:25s}  not applicable (R too small)")
            continue
        try:
            pred_fn, info = model.fit_fn(spec)
            test = pred_fn(spec.weights[:5])
            if np.any(~np.isfinite(test)):
                print(f"  {model.name:25s}  NaN/Inf — skipped")
                continue
            fitted[model.name] = (pred_fn, info)

            if model_names:
                print(f"  {model.name:25s}  OK (n_params={info.get('n_params', '?')})")
            elif not quick:
                from cross_dataset_evaluation import cross_validate

                cv = cross_validate(spec, model, k=5, seed=42)
                r2 = cv.get("R²", float("-inf"))
                cv_scores[model.name] = r2
                print(f"  {model.name:25s}  R²={r2:+.4f}")
            else:
                y_pred = pred_fn(spec.weights)
                ss_res = float(np.sum((spec.y - y_pred) ** 2))
                ss_tot = float(np.sum((spec.y - spec.y.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("-inf")
                cv_scores[model.name] = r2
                print(f"  {model.name:25s}  train R²={r2:+.4f}")
        except Exception as e:
            print(f"  {model.name:25s}  FAILED: {e}")

    if model_names:
        print(f"\nFitted {len(fitted)} models: {list(fitted.keys())}")
        return fitted

    ranked = sorted(cv_scores.items(), key=lambda x: -x[1])[:n_top]
    top_names = [name for name, _ in ranked]
    print(f"\nTop {len(top_names)} models: {top_names}")
    return {n: fitted[n] for n in top_names if n in fitted}


# =========================================================================
# Plot 1: 3D Scatter
# =========================================================================
def plot_3d_scatter(spec, df, out_dir):
    """3D scatter of all data points, colored by BPB."""
    p = [spec.weights[:, k, 1] for k in range(3)]
    best_idx = int(np.argmin(spec.y))

    hover = [
        f"Run {int(df.iloc[i]['run_id'])}<br>"
        f"p0_sc={p[0][i]:.3f}, p1_sc={p[1][i]:.3f}, p2_sc={p[2][i]:.3f}<br>"
        f"BPB={spec.y[i]:.4f}"
        for i in range(spec.R)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=p[0], y=p[1], z=p[2],
            mode="markers",
            marker=dict(
                size=5, color=spec.y, colorscale=COLORSCALE,
                colorbar=dict(title="BPB"), line=dict(width=0.5, color="black"),
            ),
            text=hover, hoverinfo="text", name="Data",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[p[0][best_idx]], y=[p[1][best_idx]], z=[p[2][best_idx]],
            mode="markers",
            marker=dict(
                size=12, color="gold", symbol="diamond",
                line=dict(width=2, color="black"),
            ),
            text=[
                f"BEST: Run {int(df.iloc[best_idx]['run_id'])}<br>"
                f"p0={p[0][best_idx]:.3f}, p1={p[1][best_idx]:.3f}, "
                f"p2={p[2][best_idx]:.3f}<br>BPB={spec.y[best_idx]:.4f}"
            ],
            hoverinfo="text", name="Best observed",
        )
    )

    fig.update_layout(
        title="Three-Phase Weight Space: Observed Data",
        scene=dict(
            xaxis_title="Phase 0 StarCoder weight",
            yaxis_title="Phase 1 StarCoder weight",
            zaxis_title="Phase 2 StarCoder weight",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            zaxis=dict(range=[0, 1]),
        ),
        width=1600, height=1100, margin=dict(l=0, r=0, b=0, t=50),
    )
    out = out_dir / "scatter_3d.html"
    fig.write_html(str(out))
    print(f"  Saved {out}")


# =========================================================================
# Plot 2: Model Optima in 3D
# =========================================================================
def plot_model_optima_3d(spec, fitted, out_dir):
    """3D scatter showing predicted optimum per model + data overlay."""
    p = [spec.weights[:, k, 1] for k in range(3)]
    best_idx = int(np.argmin(spec.y))
    hover = [
        f"BPB={spec.y[i]:.4f}<br>"
        f"w=({p[0][i]:.3f}, {p[1][i]:.3f}, {p[2][i]:.3f})"
        for i in range(spec.R)
    ]

    fig = go.Figure()

    # Data points (semi-transparent) — colorbar on the left to avoid legend overlap
    fig.add_trace(
        go.Scatter3d(
            x=p[0], y=p[1], z=p[2],
            mode="markers",
            marker=dict(
                size=3, color=spec.y, colorscale=COLORSCALE, opacity=0.4,
                colorbar=dict(title="BPB", x=-0.05, len=0.6),
                line=dict(width=0.3, color="black"),
            ),
            hovertext=hover, hoverinfo="text", name="Data",
        )
    )

    # Best observed
    fig.add_trace(
        go.Scatter3d(
            x=[p[0][best_idx]], y=[p[1][best_idx]], z=[p[2][best_idx]],
            mode="markers+text",
            marker=dict(
                size=10, color="gold", symbol="diamond",
                line=dict(width=2, color="black"),
            ),
            text=[f"Best obs (BPB={spec.y[best_idx]:.4f})"],
            textposition="top center", textfont=dict(size=9),
            hoverinfo="text", name="Best observed",
        )
    )

    # Predicted optima per model
    print("  Computing 3D optima...")
    for i, (name, (pred_fn, info)) in enumerate(fitted.items()):
        try:
            opt = find_3d_optimum(pred_fn)
            n_p = info.get("n_params", "?")
            label = f"{name} ({n_p}p)"
            color = MARKER_COLORS[i % len(MARKER_COLORS)]
            fig.add_trace(
                go.Scatter3d(
                    x=[opt[0]], y=[opt[1]], z=[opt[2]],
                    mode="markers+text",
                    marker=dict(
                        size=8, color=color, symbol="cross",
                        line=dict(width=1.5, color="black"),
                    ),
                    text=[f"{label}<br>pred={opt[3]:.4f}"],
                    textposition="top center",
                    textfont=dict(size=8, color=color),
                    hoverinfo="text", name=label,
                )
            )
            print(f"    {label}: opt=({opt[0]:.3f}, {opt[1]:.3f}, {opt[2]:.3f}), BPB={opt[3]:.4f}")
        except Exception as e:
            print(f"    {name}: FAILED ({e})")

    fig.update_layout(
        title="Predicted Optima per Model in 3D Weight Space",
        scene=dict(
            xaxis_title="Phase 0 StarCoder weight",
            yaxis_title="Phase 1 StarCoder weight",
            zaxis_title="Phase 2 StarCoder weight",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            zaxis=dict(range=[0, 1]),
        ),
        width=1600, height=1100, margin=dict(l=0, r=0, b=0, t=50),
    )
    out = out_dir / "model_optima_3d.html"
    fig.write_html(str(out))
    print(f"  Saved {out}")


# =========================================================================
# Plot 3: 3D Isosurface with model dropdown
# =========================================================================
def plot_isosurface_3d(spec, fitted, out_dir):
    """3D isosurface of model predictions with dropdown to switch models."""
    W_grid, g, p0_g, p1_g, p2_g = build_3d_grid(GRID_3D_RES)
    p = [spec.weights[:, k, 1] for k in range(3)]

    iso_min = float(np.min(spec.y)) - 0.01
    iso_max = float(np.percentile(spec.y, 75))

    model_names = list(fitted.keys())
    n_models = len(model_names)

    fig = go.Figure()

    print("  Computing 3D predictions for isosurface...")
    valid_models = []  # list of (name, label) tuples
    for mi, name in enumerate(model_names):
        pred_fn, info = fitted[name]
        n_p = info.get("n_params", "?")
        label = f"{name} ({n_p}p)"
        try:
            Z = pred_fn(W_grid)
            if np.any(~np.isfinite(Z)):
                print(f"    {label}: non-finite predictions, skipping")
                continue
        except Exception as e:
            print(f"    {label}: prediction failed ({e})")
            continue

        visible = (len(valid_models) == 0)  # first valid model visible

        # Isosurface trace
        fig.add_trace(
            go.Isosurface(
                x=p0_g.ravel(), y=p1_g.ravel(), z=p2_g.ravel(),
                value=Z,
                isomin=iso_min, isomax=iso_max,
                surface_count=5,
                colorscale=COLORSCALE,
                opacity=0.3,
                caps=dict(x_show=False, y_show=False, z_show=False),
                visible=visible,
                name=label,
                showscale=visible,
                colorbar=dict(title="Pred BPB"),
            )
        )

        # Optimum marker
        opt_idx = int(np.argmin(Z))
        opt_p0 = float(p0_g.ravel()[opt_idx])
        opt_p1 = float(p1_g.ravel()[opt_idx])
        opt_p2 = float(p2_g.ravel()[opt_idx])
        opt_bpb = float(Z[opt_idx])

        fig.add_trace(
            go.Scatter3d(
                x=[opt_p0], y=[opt_p1], z=[opt_p2],
                mode="markers",
                marker=dict(
                    size=10, color="gold", symbol="diamond",
                    line=dict(width=2, color="black"),
                ),
                text=[f"{label} optimum<br>({opt_p0:.3f}, {opt_p1:.3f}, {opt_p2:.3f})<br>BPB={opt_bpb:.4f}"],
                hoverinfo="text",
                visible=visible,
                showlegend=False,
            )
        )
        valid_models.append((name, label))
        print(f"    {label}: done")

    if not valid_models:
        print("  No models produced valid 3D predictions — skipping isosurface")
        return

    # Data scatter (always visible)
    best_idx = int(np.argmin(spec.y))
    hover = [
        f"BPB={spec.y[i]:.4f}<br>"
        f"w=({p[0][i]:.3f}, {p[1][i]:.3f}, {p[2][i]:.3f})"
        for i in range(spec.R)
    ]
    fig.add_trace(
        go.Scatter3d(
            x=p[0], y=p[1], z=p[2],
            mode="markers",
            marker=dict(
                size=3, color=spec.y, colorscale=COLORSCALE, opacity=0.5,
                line=dict(width=0.3, color="black"), showscale=False,
            ),
            hovertext=hover, hoverinfo="text", name="Data",
        )
    )

    # Best observed (always visible)
    fig.add_trace(
        go.Scatter3d(
            x=[p[0][best_idx]], y=[p[1][best_idx]], z=[p[2][best_idx]],
            mode="markers+text",
            marker=dict(
                size=10, color="gold", symbol="diamond",
                line=dict(width=2, color="black"),
            ),
            text=[f"Best obs (BPB={spec.y[best_idx]:.4f})"],
            textposition="top center", textfont=dict(size=9),
            hoverinfo="text", name="Best observed",
        )
    )

    # Model dropdown
    n_iso_traces = len(valid_models) * 2  # isosurface + optimum per model
    buttons = []
    for mi, (name, label) in enumerate(valid_models):
        vis = [False] * n_iso_traces + [True, True]  # data + best observed always visible
        vis[mi * 2] = True      # isosurface
        vis[mi * 2 + 1] = True  # optimum
        buttons.append(dict(label=label, method="update", args=[{"visible": vis}]))

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons, direction="down", showactive=True,
            x=0.02, xanchor="left", y=0.98, yanchor="top",
        )],
        title="Model Predictions: 3D Isosurface (select model from dropdown)",
        scene=dict(
            xaxis_title="Phase 0 StarCoder",
            yaxis_title="Phase 1 StarCoder",
            zaxis_title="Phase 2 StarCoder",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            zaxis=dict(range=[0, 1]),
        ),
        width=1600, height=1100, margin=dict(l=0, r=0, b=0, t=50),
    )
    out = out_dir / "isosurface_3d.html"
    fig.write_html(str(out))
    print(f"  Saved {out}")


# =========================================================================
# Plot 4: 2D Contour Slices with Slider (small multiples)
# =========================================================================
def plot_slice_panel(spec, fitted, fixed_phase, out_dir):
    """Small-multiples 2D contour (one subplot per model) with slider for the fixed phase."""
    free = [i for i in range(3) if i != fixed_phase]
    free_labels = [f"Phase {i} SC weight" for i in free]

    model_names = list(fitted.keys())
    n_models = len(model_names)
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols

    slider_vals = np.linspace(0.02, 0.98, N_SLIDER_STEPS)
    g = np.linspace(0.005, 0.995, GRID_2D_RES)
    mid_si = N_SLIDER_STEPS // 2

    zmin = float(np.min(spec.y)) - 0.02
    zmax = float(np.percentile(spec.y, 85))

    # Data coordinates in the free dimensions
    dp_free0 = spec.weights[:, free[0], 1]
    dp_free1 = spec.weights[:, free[1], 1]
    dp_fixed = spec.weights[:, fixed_phase, 1]

    subplot_titles = []
    for n in model_names:
        n_p = fitted[n][1].get("n_params", "?")
        subplot_titles.append(f"{n} ({n_p}p)")

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08, vertical_spacing=0.10,
    )

    # Traces are added in this order for each slider step:
    #   n_models contour traces, then n_models scatter traces
    # Total: N_SLIDER_STEPS * n_models * 2
    print(f"  Computing slices (fix phase {fixed_phase})...")
    for si, sv in enumerate(slider_vals):
        visible = (si == mid_si)
        W_grid = build_2d_slice_weights(fixed_phase, sv, g)

        # Contour traces
        for mi, name in enumerate(model_names):
            row, col = mi // ncols + 1, mi % ncols + 1
            pred_fn = fitted[name][0]

            try:
                Z = pred_fn(W_grid).reshape(len(g), len(g))
                # Clip extreme predictions for display
                Z = np.clip(Z, zmin - 0.5, zmax + 0.5)
            except Exception:
                Z = np.full((len(g), len(g)), np.nan)

            fig.add_trace(
                go.Contour(
                    x=g, y=g, z=Z, visible=visible,
                    colorscale=COLORSCALE, zmin=zmin, zmax=zmax,
                    contours=dict(showlabels=True, labelfont=dict(size=7)),
                    showscale=(mi == 0 and si == mid_si),
                    colorbar=dict(title="BPB", x=1.02) if (mi == 0 and si == mid_si) else None,
                    hovertemplate="p_free0=%{x:.3f}<br>p_free1=%{y:.3f}<br>BPB=%{z:.4f}<extra></extra>",
                ),
                row=row, col=col,
            )

        # Scatter traces (data near this slice)
        tol = max(0.05, 0.5 / N_SLIDER_STEPS)
        mask = np.abs(dp_fixed - sv) < tol
        for mi in range(n_models):
            row, col = mi // ncols + 1, mi % ncols + 1
            scatter_x = dp_free0[mask] if mask.any() else np.array([])
            scatter_y = dp_free1[mask] if mask.any() else np.array([])
            scatter_c = spec.y[mask] if mask.any() else np.array([])
            hover_texts = (
                [f"BPB={spec.y[j]:.4f}" for j in np.where(mask)[0]]
                if mask.any() else []
            )
            fig.add_trace(
                go.Scatter(
                    x=scatter_x, y=scatter_y, mode="markers", visible=visible,
                    marker=dict(
                        size=7, color=scatter_c, colorscale=COLORSCALE,
                        cmin=zmin, cmax=zmax, showscale=False,
                        line=dict(width=1, color="black"),
                    ),
                    text=hover_texts, hoverinfo="text", showlegend=False,
                ),
                row=row, col=col,
            )

    # Build slider steps
    traces_per_step = n_models * 2  # contour + scatter per model
    total_traces = N_SLIDER_STEPS * traces_per_step
    steps = []
    for si, sv in enumerate(slider_vals):
        vis = [False] * total_traces
        start = si * traces_per_step
        for j in range(traces_per_step):
            vis[start + j] = True
        steps.append(dict(
            method="update",
            args=[{"visible": vis}],
            label=f"{sv:.2f}",
        ))

    fig.update_layout(
        sliders=[dict(
            active=mid_si,
            currentvalue=dict(prefix=f"Phase {fixed_phase} SC weight = "),
            steps=steps,
            pad=dict(t=50),
        )],
        title=f"Model Predictions: 2D Slices (fixing Phase {fixed_phase} StarCoder weight)",
        height=500 * nrows + 150,
        width=500 * ncols + 70,
        showlegend=False,
    )

    # Axis labels
    for mi in range(n_models):
        row, col = mi // ncols + 1, mi % ncols + 1
        fig.update_xaxes(title_text=free_labels[0] if row == nrows else "", range=[0, 1], row=row, col=col)
        fig.update_yaxes(title_text=free_labels[1] if col == 1 else "", range=[0, 1], row=row, col=col)

    # Hide empty subplots
    for idx in range(n_models, nrows * ncols):
        row, col = idx // ncols + 1, idx % ncols + 1
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    out = out_dir / f"slices_fix_phase{fixed_phase}.html"
    fig.write_html(str(out))
    print(f"  Saved {out}")


# =========================================================================
# Plot 5: 3D Optimum Trajectory over training sizes
# =========================================================================
TRAJ_SEED_BASE = 42
TRAJ_B = 200  # gap between seeds (matches holdout_analysis convention)
TRAJ_GRID_RES = 40


def plot_optimum_trajectory_3d(spec, fitted, out_dir):
    """Show how each model's predicted 3D optimum drifts as training data grows.

    Interactive slider controls the training size. At each step:
      - The cumulative trajectory (optima 1..s) is shown per model.
      - The data points used for fitting at that step are highlighted.
    """
    R = spec.R
    _lo = max(12, R // 10)
    _hi = int(R * 0.87)
    train_sizes = sorted(set(int(round(v)) for v in np.linspace(_lo, _hi, 8)))

    model_by_name = {m.name: m for m in GENERAL_MODELS}

    p = [spec.weights[:, k, 1] for k in range(3)]
    best_idx = int(np.argmin(spec.y))
    hover_texts = [
        f"BPB={spec.y[i]:.4f}<br>"
        f"w=({p[0][i]:.3f}, {p[1][i]:.3f}, {p[2][i]:.3f})"
        for i in range(R)
    ]

    # ------------------------------------------------------------------
    # Phase 1: Pre-compute all optima and training indices
    # ------------------------------------------------------------------
    print("  Computing optimum trajectories...")
    all_optima: dict[str, list[tuple[float, float, float, float]]] = {}
    model_labels: dict[str, str] = {}

    for name, (pred_fn_full, info) in fitted.items():
        n_p = info.get("n_params", "?")
        label = f"{name} ({n_p}p)"
        model_labels[name] = label

        model_obj = model_by_name.get(name)
        if model_obj is None:
            print(f"    {label}: not found in GENERAL_MODELS, skipping")
            continue

        optima: list[tuple[float, float, float, float]] = []
        for si, n_train in enumerate(train_sizes):
            seed = TRAJ_SEED_BASE + si * TRAJ_B
            rng = np.random.RandomState(seed)
            idx = rng.permutation(R)
            train_spec = spec.subset(idx[:n_train])

            if not model_obj.applicable(train_spec):
                optima.append((np.nan, np.nan, np.nan, np.nan))
                continue
            try:
                sub_pred_fn, _ = model_obj.fit_fn(train_spec)
                opt = find_3d_optimum(sub_pred_fn, resolution=TRAJ_GRID_RES)
                optima.append(opt)
                print(
                    f"      {name} n={n_train}: "
                    f"({opt[0]:.3f}, {opt[1]:.3f}, {opt[2]:.3f}) "
                    f"BPB={opt[3]:.4f}"
                )
            except Exception as e:
                print(f"      {name} n={n_train}: fit failed ({e})")
                optima.append((np.nan, np.nan, np.nan, np.nan))

        # Full-data optimum
        try:
            full_opt = find_3d_optimum(pred_fn_full, resolution=TRAJ_GRID_RES)
            optima.append(full_opt)
            print(
                f"      {name} n=full: "
                f"({full_opt[0]:.3f}, {full_opt[1]:.3f}, {full_opt[2]:.3f}) "
                f"BPB={full_opt[3]:.4f}"
            )
        except Exception:
            optima.append((np.nan, np.nan, np.nan, np.nan))

        n_valid = sum(1 for o in optima if not np.isnan(o[0]))
        print(f"    {label}: {n_valid}/{len(optima)} valid optima")
        all_optima[name] = optima

    # Training indices at each step (deterministic seeds)
    train_indices: list[np.ndarray] = []
    for si, n_train in enumerate(train_sizes):
        seed = TRAJ_SEED_BASE + si * TRAJ_B
        rng = np.random.RandomState(seed)
        idx = rng.permutation(R)
        train_indices.append(idx[:n_train])
    train_indices.append(np.arange(R))  # full data

    n_steps = len(train_sizes) + 1  # 8 subsample steps + 1 full
    model_names = [n for n in fitted if n in all_optima]
    n_models = len(model_names)
    LINE_DASHES = ["solid", "dash", "dot", "dashdot"]

    # ------------------------------------------------------------------
    # Phase 2: Build traces
    # ------------------------------------------------------------------
    fig = go.Figure()

    # --- Always-visible background traces (2) ---
    fig.add_trace(
        go.Scatter3d(
            x=p[0], y=p[1], z=p[2],
            mode="markers",
            marker=dict(
                size=3, color=spec.y, colorscale=COLORSCALE, opacity=0.2,
                colorbar=dict(title="BPB", x=-0.05, len=0.5),
                line=dict(width=0.3, color="black"),
            ),
            hovertext=hover_texts, hoverinfo="text", name="Data",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[p[0][best_idx]], y=[p[1][best_idx]], z=[p[2][best_idx]],
            mode="markers+text",
            marker=dict(
                size=10, color="gold", symbol="diamond",
                line=dict(width=2, color="black"),
            ),
            text=[f"Best obs (BPB={spec.y[best_idx]:.4f})"],
            textposition="top center", textfont=dict(size=9),
            hoverinfo="text", name="Best observed",
        )
    )
    n_bg = 2

    # --- Per-step traces ---
    # Order per step: 1 training-highlight, 1 best-train-point,
    #   then per model: line + arrow cones + markers  (3 traces)
    traces_per_step = 2 + n_models * 3

    for s in range(n_steps):
        is_full = (s == n_steps - 1)
        initial_visible = is_full  # start slider at full-data step
        tidx = train_indices[s]
        n_tr = R if is_full else train_sizes[s]

        # Training-data highlight
        train_hover = [hover_texts[i] for i in tidx]
        fig.add_trace(
            go.Scatter3d(
                x=p[0][tidx], y=p[1][tidx], z=p[2][tidx],
                mode="markers",
                marker=dict(
                    size=5, color="rgba(255,165,0,0.7)",
                    line=dict(width=0.5, color="black"),
                ),
                hovertext=train_hover, hoverinfo="text",
                visible=initial_visible,
                name=f"Train pts (n={n_tr})",
                showlegend=True,
                legendgroup="__train__",
            )
        )

        # Best training point (lowest BPB among training subset)
        train_y = spec.y[tidx]
        best_train_local = int(np.argmin(train_y))
        best_train_global = tidx[best_train_local]
        best_train_bpb = spec.y[best_train_global]
        fig.add_trace(
            go.Scatter3d(
                x=[p[0][best_train_global]],
                y=[p[1][best_train_global]],
                z=[p[2][best_train_global]],
                mode="markers+text",
                marker=dict(
                    size=10, color="orange", symbol="diamond",
                    line=dict(width=2, color="black"),
                ),
                text=[f"Best train (BPB={best_train_bpb:.4f})"],
                textposition="top center",
                textfont=dict(size=9),
                hoverinfo="text",
                visible=initial_visible,
                name="Best in training set",
                showlegend=True,
                legendgroup="__best_train__",
            )
        )

        # Per model: cumulative line + arrows + markers up to step s
        for mi, name in enumerate(model_names):
            label = model_labels[name]
            color = MARKER_COLORS[mi % len(MARKER_COLORS)]
            dash = LINE_DASHES[mi % len(LINE_DASHES)]
            optima = all_optima[name]
            cum = optima[: s + 1]

            # Build cumulative arrays
            lx, ly, lz = [], [], []
            # Valid points for markers and arrows
            valid_pts: list[tuple[int, tuple[float, ...]]] = []
            for oi, opt in enumerate(cum):
                if np.isnan(opt[0]):
                    lx.append(None)
                    ly.append(None)
                    lz.append(None)
                else:
                    lx.append(opt[0])
                    ly.append(opt[1])
                    lz.append(opt[2])
                    valid_pts.append((oi, opt))

            # Markers: newest point (last valid) gets a star, others circle
            mx, my, mz = [], [], []
            mt, mh, msz, msym = [], [], [], []
            for vi, (oi, opt) in enumerate(valid_pts):
                is_newest = (vi == len(valid_pts) - 1)
                oi_full = (oi == len(optima) - 1)
                pt_lbl = "F" if oi_full else str(oi + 1)
                oi_n = "full" if oi_full else str(train_sizes[oi])
                mx.append(opt[0])
                my.append(opt[1])
                mz.append(opt[2])
                mt.append(pt_lbl)
                mh.append(
                    f"{label}<br>n_train={oi_n}<br>"
                    f"opt=({opt[0]:.3f}, {opt[1]:.3f}, "
                    f"{opt[2]:.3f})<br>"
                    f"pred BPB={opt[3]:.4f}"
                )
                if is_newest:
                    msz.append(12)
                    msym.append("diamond")
                else:
                    msz.append(6)
                    msym.append("circle")

            # Arrow cones between consecutive valid points
            cx, cy, cz, cu, cv, cw = [], [], [], [], [], []
            for vi in range(1, len(valid_pts)):
                _, prev = valid_pts[vi - 1]
                _, curr = valid_pts[vi]
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                dz = curr[2] - prev[2]
                # Place cone at 80% along the segment (near destination)
                cx.append(prev[0] + 0.8 * dx)
                cy.append(prev[1] + 0.8 * dy)
                cz.append(prev[2] + 0.8 * dz)
                cu.append(dx)
                cv.append(dy)
                cw.append(dz)

            # Line trace
            fig.add_trace(
                go.Scatter3d(
                    x=lx, y=ly, z=lz,
                    mode="lines",
                    line=dict(color=color, width=4, dash=dash),
                    hoverinfo="skip", showlegend=False,
                    legendgroup=name,
                    visible=initial_visible,
                )
            )
            # Arrow cones
            if cx:
                fig.add_trace(
                    go.Cone(
                        x=cx, y=cy, z=cz, u=cu, v=cv, w=cw,
                        sizemode="absolute", sizeref=0.03,
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        hoverinfo="skip", showlegend=False,
                        legendgroup=name,
                        visible=initial_visible,
                    )
                )
            else:
                # Placeholder empty trace to keep trace count consistent
                fig.add_trace(
                    go.Scatter3d(
                        x=[], y=[], z=[],
                        mode="markers",
                        hoverinfo="skip", showlegend=False,
                        legendgroup=name,
                        visible=initial_visible,
                    )
                )
            # Markers
            fig.add_trace(
                go.Scatter3d(
                    x=mx, y=my, z=mz,
                    mode="markers+text",
                    marker=dict(
                        size=msz, color=color, symbol=msym,
                        line=dict(width=1.5, color="black"),
                    ),
                    text=mt,
                    textposition="top center",
                    textfont=dict(size=8, color=color),
                    hovertext=mh, hoverinfo="text",
                    showlegend=True,
                    legendgroup=name,
                    name=label,
                    visible=initial_visible,
                )
            )

    # ------------------------------------------------------------------
    # Phase 3: Slider
    # ------------------------------------------------------------------
    total_step_traces = n_steps * traces_per_step
    slider_steps = []
    for s in range(n_steps):
        is_full = (s == n_steps - 1)
        vis = [True] * n_bg + [False] * total_step_traces
        start = n_bg + s * traces_per_step
        for j in range(traces_per_step):
            vis[start + j] = True
        lbl = f"full (n={R})" if is_full else f"n={train_sizes[s]}"
        slider_steps.append(dict(
            method="update", args=[{"visible": vis}], label=lbl,
        ))

    fig.update_layout(
        sliders=[dict(
            active=n_steps - 1,
            currentvalue=dict(prefix="Training size: "),
            steps=slider_steps,
            pad=dict(t=50),
        )],
        title="Predicted Optimum Trajectory over Training Size",
        scene=dict(
            xaxis_title="Phase 0 StarCoder weight",
            yaxis_title="Phase 1 StarCoder weight",
            zaxis_title="Phase 2 StarCoder weight",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            zaxis=dict(range=[0, 1]),
        ),
        width=1600, height=1100, margin=dict(l=0, r=0, b=0, t=50),
    )
    out = out_dir / "optimum_trajectory_3d.html"
    fig.write_html(str(out))
    print(f"  Saved {out}")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="3D visualization of three-phase weight space")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target metric column")
    parser.add_argument("--n_top", type=int, default=N_TOP_MODELS, help="Number of top models to show")
    parser.add_argument("--quick", action="store_true", help="Skip CV, rank by train R²")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Exact model names to fit (skip ranking). E.g. --models CES 'CEQ-SUM soft' 'NCEQ(k)'")
    args = parser.parse_args()

    out_dir = SCRIPT_DIR / "three_phase_plots"
    out_dir.mkdir(exist_ok=True)

    spec, df = load_spec(args.target)
    print(f"Loaded {spec.R} runs, target: {args.target}")
    print(f"y range: [{spec.y.min():.4f}, {spec.y.max():.4f}], median={np.median(spec.y):.4f}")

    fitted = fit_and_rank_models(spec, args.n_top, quick=args.quick, model_names=args.models)

    print("\n--- Generating plots ---")
    print("Plot 1: 3D scatter...")
    plot_3d_scatter(spec, df, out_dir)

    print("Plot 2: Model optima in 3D...")
    plot_model_optima_3d(spec, fitted, out_dir)

    print("Plot 3: 3D isosurface...")
    plot_isosurface_3d(spec, fitted, out_dir)

    for fp in range(3):
        print(f"Plot 4: 2D slices (fix phase {fp})...")
        plot_slice_panel(spec, fitted, fp, out_dir)

    print("Plot 5: Optimum trajectory in 3D...")
    plot_optimum_trajectory_3d(spec, fitted, out_dir)

    print(f"\nAll outputs saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
