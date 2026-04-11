# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["lightgbm", "pandas", "numpy", "plotly", "kaleido", "scikit-learn"]
# ///
"""Regenerate the 2D LightGBM heatmap using all available 2-phase data.

Uses two_phase_starcoder_combined.csv (116 runs) instead of the original 66-run subset.

Usage:
    uv run experiments/domain_phase_mix/exploratory/lgbm_2d_heatmap.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import KFold

FEATURE_COLS = ["phase_0_starcoder", "phase_1_starcoder"]
TARGET = "eval/paloma/dolma_100_programing_languages/bpb"

IMG_DIR = Path(
    "/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/"
    "My Drive/Research/Marin/presentation-20260302/img/"
)


def main():
    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir / "two_phase_starcoder_combined.csv")
    df = df[df["status"] == "completed"].copy()
    print(f"Loaded {len(df)} completed runs")

    X = df[FEATURE_COLS].values
    y = df[TARGET].values

    # Fit 5-fold ensemble
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    for train_idx, val_idx in kfold.split(X):
        gbm = lgb.LGBMRegressor(
            boosting_type="gbdt",
            objective="regression",
            num_iterations=1000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=1,
            verbosity=-1,
            seed=42,
        )
        gbm.fit(
            X[train_idx],
            y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        models.append(gbm)

    # Predict on a fine grid
    resolution = 500
    g0 = np.linspace(0, 1, resolution)
    g1 = np.linspace(0, 1, resolution)
    G0, G1 = np.meshgrid(g0, g1)
    grid = np.column_stack([G0.ravel(), G1.ravel()])
    preds = np.mean([m.predict(grid) for m in models], axis=0)
    pred_grid = preds.reshape(resolution, resolution)

    # Find optimum
    opt_idx = int(np.argmin(preds))
    opt_p0 = grid[opt_idx, 0]
    opt_p1 = grid[opt_idx, 1]
    opt_val = preds[opt_idx]
    print(f"Predicted optimum: p0_sc={opt_p0:.3f}, p1_sc={opt_p1:.3f}, BPB={opt_val:.4f}")

    # Best observed
    best_idx = int(np.argmin(y))
    print(f"Best observed: p0_sc={X[best_idx, 0]:.3f}, p1_sc={X[best_idx, 1]:.3f}, BPB={y[best_idx]:.4f}")

    p0 = df["phase_0_starcoder"].values
    p1 = df["phase_1_starcoder"].values
    run_ids = df["run_id"].values

    # Match holdout_analysis.py color range
    vmin = float(np.min(y)) - 0.05
    vmax = float(np.percentile(y, 90))

    fig = go.Figure()

    # Heatmap
    fig.add_trace(
        go.Heatmap(
            z=pred_grid,
            x=g0,
            y=g1,
            colorscale="RdYlGn_r",
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(
                title=dict(
                    text="eval/<br>paloma/<br>dolma_100_programing_languages/<br>bpb",
                    font=dict(size=10),
                ),
            ),
            hovertemplate="p0_sc=%{x:.3f}<br>p1_sc=%{y:.3f}<br>pred=%{z:.4f}<extra></extra>",
        )
    )

    # Training runs
    fig.add_trace(
        go.Scatter(
            x=p0,
            y=p1,
            mode="markers",
            marker=dict(
                size=8,
                color=y,
                colorscale="RdYlGn_r",
                cmin=vmin,
                cmax=vmax,
                line=dict(width=1.5, color="white"),
                showscale=False,
            ),
            text=[
                f"run_id={int(rid)}<br>p0_sc={x:.3f}<br>p1_sc={y_:.3f}<br>actual={v:.4f}"
                for rid, x, y_, v in zip(run_ids, p0, p1, y)
            ],
            hoverinfo="text",
            name="Training runs",
        )
    )

    # Predicted optimum
    fig.add_trace(
        go.Scatter(
            x=[opt_p0],
            y=[opt_p1],
            mode="markers",
            marker=dict(size=14, symbol="star", color="red", line=dict(width=1, color="darkred")),
            name=f"Predicted opt: ({opt_p0:.3f}, {opt_p1:.3f}) = {opt_val:.4f}",
            hoverinfo="name",
        )
    )

    # Best observed
    fig.add_trace(
        go.Scatter(
            x=[X[best_idx, 0]],
            y=[X[best_idx, 1]],
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(width=2.5, color="red")),
            name=f"Best observed: {y[best_idx]:.4f}",
            hoverinfo="name",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"{TARGET} (predicted)",
            font=dict(size=16),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(title="Phase 0 StarCoder weight", range=[0, 1], constrain="domain"),
        yaxis=dict(title="Phase 1 StarCoder weight", range=[0, 1], scaleanchor="x", scaleratio=1),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        width=800,
        height=700,
        margin=dict(l=60, r=20, t=50, b=60),
    )

    out_dir = script_dir / "two_phase_starcoder_plots"
    out_dir.mkdir(exist_ok=True)

    html_path = out_dir / "heatmap_combined_dolma_100_programing_languages_bpb.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"Saved {html_path}")

    png_path = IMG_DIR / "heatmap_eval_paloma_dolma_100_programing_languages_bpb.png"
    fig.write_image(str(png_path), scale=2, width=800, height=700)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
