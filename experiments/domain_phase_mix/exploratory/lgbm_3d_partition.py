# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["lightgbm", "pandas", "numpy", "plotly", "kaleido", "scikit-learn"]
# ///
"""3D visualization of LightGBM's axis-aligned partitioning on 3-phase StarCoder data.

Fits a LightGBM model then renders the actual partition boundaries as flat
axis-aligned quads colored by predicted value on each side.

Usage:
    uv run experiments/domain_phase_mix/exploratory/lgbm_3d_partition.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import KFold

FEATURE_COLS = ["phase_0_starcoder", "phase_1_starcoder", "phase_2_starcoder"]
TARGET = "eval/paloma/dolma_100_programing_languages/bpb"

IMG_DIR = Path(
    "/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/"
    "My Drive/Research/Marin/presentation-20260302/img/"
)


def fit_lgbm_ensemble(X, y, n_splits=5):
    """Fit a 5-fold LightGBM ensemble."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
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
    return models


def extract_boundary_faces(pred_vol, edges, threshold=0.005):
    """Find faces between adjacent voxels where prediction changes.

    Returns lists of quad vertices and the average predicted value at each face.
    """
    nx, ny, nz = pred_vol.shape
    faces_verts = []  # list of (4, 3) arrays — quad corners
    faces_color = []  # average pred value at each face

    # Check x-boundaries (faces perpendicular to x axis)
    for i in range(nx - 1):
        for j in range(ny):
            for k in range(nz):
                diff = abs(pred_vol[i, j, k] - pred_vol[i + 1, j, k])
                if diff > threshold:
                    x = edges[i + 1]
                    y0, y1 = edges[j], edges[j + 1] if j + 1 < len(edges) else 1.0
                    z0, z1 = edges[k], edges[k + 1] if k + 1 < len(edges) else 1.0
                    faces_verts.append(np.array([[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]]))
                    faces_color.append((pred_vol[i, j, k] + pred_vol[i + 1, j, k]) / 2)

    # Check y-boundaries
    for i in range(nx):
        for j in range(ny - 1):
            for k in range(nz):
                diff = abs(pred_vol[i, j, k] - pred_vol[i, j + 1, k])
                if diff > threshold:
                    y = edges[j + 1]
                    x0, x1 = edges[i], edges[i + 1] if i + 1 < len(edges) else 1.0
                    z0, z1 = edges[k], edges[k + 1] if k + 1 < len(edges) else 1.0
                    faces_verts.append(np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]]))
                    faces_color.append((pred_vol[i, j, k] + pred_vol[i, j + 1, k]) / 2)

    # Check z-boundaries
    for i in range(nx):
        for j in range(ny):
            for k in range(nz - 1):
                diff = abs(pred_vol[i, j, k] - pred_vol[i, j, k + 1])
                if diff > threshold:
                    z = edges[k + 1]
                    x0, x1 = edges[i], edges[i + 1] if i + 1 < len(edges) else 1.0
                    y0, y1 = edges[j], edges[j + 1] if j + 1 < len(edges) else 1.0
                    faces_verts.append(np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]))
                    faces_color.append((pred_vol[i, j, k] + pred_vol[i, j, k + 1]) / 2)

    # Also add the 6 outer faces of the unit cube so it's enclosed
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x0, x1 = edges[i], edges[i + 1] if i + 1 < len(edges) else 1.0
                y0, y1 = edges[j], edges[j + 1] if j + 1 < len(edges) else 1.0
                z0, z1 = edges[k], edges[k + 1] if k + 1 < len(edges) else 1.0
                val = pred_vol[i, j, k]
                if i == 0:
                    faces_verts.append(np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]))
                    faces_color.append(val)
                if i == nx - 1:
                    faces_verts.append(np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]))
                    faces_color.append(val)
                if j == 0:
                    faces_verts.append(np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]))
                    faces_color.append(val)
                if j == ny - 1:
                    faces_verts.append(np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]))
                    faces_color.append(val)
                if k == 0:
                    faces_verts.append(np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]))
                    faces_color.append(val)
                if k == nz - 1:
                    faces_verts.append(np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]))
                    faces_color.append(val)

    return faces_verts, faces_color


def main():
    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir / "three_phase_starcoder.csv")
    df = df[df["status"] == "completed"].copy()
    print(f"Loaded {len(df)} completed runs")

    X = df[FEATURE_COLS].values
    y = df[TARGET].values

    models = fit_lgbm_ensemble(X, y)

    # Predict on a grid
    resolution = 25
    edges = np.linspace(0, 1, resolution + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    C0, C1, C2 = np.meshgrid(centers, centers, centers, indexing="ij")
    grid = np.column_stack([C0.ravel(), C1.ravel(), C2.ravel()])
    preds = np.mean([m.predict(grid) for m in models], axis=0)
    pred_vol = preds.reshape(resolution, resolution, resolution)

    print(f"Pred range: [{preds.min():.4f}, {preds.max():.4f}]")

    # Match holdout_analysis.py color range
    vmin = float(np.min(y)) - 0.05
    vmax = float(np.percentile(y, 90))

    # Extract boundary faces where prediction changes
    faces_verts, faces_color = extract_boundary_faces(pred_vol, edges, threshold=0.003)
    print(f"Extracted {len(faces_verts)} boundary faces")

    # Build a single Mesh3d from all quads (2 triangles per quad)
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    all_intensity = []
    offset = 0

    for quad, val in zip(faces_verts, faces_color):
        for v in quad:
            all_x.append(v[0])
            all_y.append(v[1])
            all_z.append(v[2])
            all_intensity.append(val)
        # Two triangles: 0-1-2 and 0-2-3
        all_i.extend([offset, offset])
        all_j.extend([offset + 1, offset + 2])
        all_k.extend([offset + 2, offset + 3])
        offset += 4

    fig = go.Figure()

    fig.add_trace(
        go.Mesh3d(
            x=all_x,
            y=all_y,
            z=all_z,
            i=all_i,
            j=all_j,
            k=all_k,
            intensity=all_intensity,
            intensitymode="vertex",
            colorscale="RdYlGn_r",
            cmin=vmin,
            cmax=vmax,
            flatshading=True,
            lighting=dict(ambient=0.9, diffuse=0.1, specular=0.0),
            lightposition=dict(x=0, y=0, z=1000),
            colorbar=dict(title=dict(text="Code BPB", font=dict(size=12)), len=0.7),
            opacity=0.25,
            hoverinfo="skip",
            name="LightGBM partition boundaries",
        )
    )

    # Overlay training runs
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=y,
                colorscale="RdYlGn_r",
                cmin=vmin,
                cmax=vmax,
                line=dict(width=0.8, color="black"),
                showscale=False,
            ),
            text=[f"p0_sc={x[0]:.2f}, p1_sc={x[1]:.2f}, p2_sc={x[2]:.2f}<br>BPB={v:.4f}" for x, v in zip(X, y)],
            hoverinfo="text",
            name="Training runs",
        )
    )

    fig.update_layout(
        title=dict(
            text="LightGBM Predicted Landscape (3-Phase StarCoder → Code BPB)",
            font=dict(size=14),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(title="Phase 0 StarCoder", range=[0, 1]),
            yaxis=dict(title="Phase 1 StarCoder", range=[0, 1]),
            zaxis=dict(title="Phase 2 StarCoder", range=[0, 1]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        width=900,
        height=750,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    out_dir = script_dir / "three_phase_starcoder_plots"
    out_dir.mkdir(exist_ok=True)

    html_path = out_dir / "lgbm_3d_cuboids.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"Saved {html_path}")

    png_path = IMG_DIR / "lgbm_3d_cuboids.png"
    fig.write_image(str(png_path), scale=2, width=900, height=750)
    print(f"Saved {png_path}")

    # --- Isosurface visualization ---
    iso_res = 40
    iso_edges = np.linspace(0, 1, iso_res)
    IG0, IG1, IG2 = np.meshgrid(iso_edges, iso_edges, iso_edges, indexing="ij")
    iso_grid = np.column_stack([IG0.ravel(), IG1.ravel(), IG2.ravel()])
    iso_preds = np.mean([m.predict(iso_grid) for m in models], axis=0)

    n_levels = 8
    iso_values = np.linspace(
        float(np.percentile(iso_preds, 5)),
        float(np.percentile(iso_preds, 95)),
        n_levels,
    )

    fig_iso = go.Figure()
    fig_iso.add_trace(
        go.Isosurface(
            x=IG0.ravel(),
            y=IG1.ravel(),
            z=IG2.ravel(),
            value=iso_preds,
            isomin=float(iso_values[0]),
            isomax=float(iso_values[-1]),
            surface_count=n_levels,
            colorscale="RdYlGn_r",
            cmin=vmin,
            cmax=vmax,
            caps=dict(x_show=False, y_show=False, z_show=False),
            opacity=0.3,
            colorbar=dict(title=dict(text="Code BPB", font=dict(size=12)), len=0.7),
        )
    )

    fig_iso.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=y,
                colorscale="RdYlGn_r",
                cmin=vmin,
                cmax=vmax,
                line=dict(width=0.8, color="black"),
                showscale=False,
            ),
            text=[f"p0_sc={x[0]:.2f}, p1_sc={x[1]:.2f}, p2_sc={x[2]:.2f}<br>BPB={v:.4f}" for x, v in zip(X, y)],
            hoverinfo="text",
            name="Training runs",
        )
    )

    fig_iso.update_layout(
        title=dict(
            text="LightGBM Predicted Landscape — Isosurfaces (3-Phase StarCoder → Code BPB)",
            font=dict(size=14),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis=dict(title="Phase 0 StarCoder", range=[0, 1]),
            yaxis=dict(title="Phase 1 StarCoder", range=[0, 1]),
            zaxis=dict(title="Phase 2 StarCoder", range=[0, 1]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        width=900,
        height=750,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    iso_html = out_dir / "lgbm_3d_isosurface.html"
    fig_iso.write_html(str(iso_html), include_plotlyjs="cdn")
    print(f"Saved {iso_html}")

    iso_png = IMG_DIR / "lgbm_3d_partition.png"
    fig_iso.write_image(str(iso_png), scale=2, width=900, height=750)
    print(f"Saved {iso_png}")


if __name__ == "__main__":
    main()
