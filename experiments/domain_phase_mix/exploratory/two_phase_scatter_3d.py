"""Interactive 3D scatter plot of all 2-phase StarCoder data.

Loads all two_phase_starcoder*.csv files, deduplicates by weight config
(keeping the first occurrence), and plots:
  x = phase_0_starcoder weight
  y = phase_1_starcoder weight
  z = eval/paloma/dolma_100_programing_languages/bpb

Includes a smooth TPS-interpolated 3D surface passing through the data
and marks the global minimum.

Usage:
  uv run experiments/domain_phase_mix/exploratory/two_phase_scatter_3d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay

SCRIPT_DIR = Path(__file__).resolve().parent
TARGET = "eval/paloma/dolma_100_programing_languages/bpb"


def load_and_merge() -> pd.DataFrame:
    """Load all two_phase_starcoder CSVs and deduplicate by weight config."""
    csv_files = sorted(SCRIPT_DIR.glob("two_phase_starcoder*.csv"))
    if not csv_files:
        print("No two_phase_starcoder*.csv files found", file=sys.stderr)
        sys.exit(1)

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["_source"] = f.name
        if "status" in df.columns:
            df = df[df["status"] == "completed"]
        frames.append(df)
        print(f"  {f.name}: {len(df)} completed rows")

    merged = pd.concat(frames, ignore_index=True)

    # Deduplicate by (phase_0_starcoder, phase_1_starcoder), keep first
    merged["_key"] = (
        merged["phase_0_starcoder"].round(6).astype(str)
        + "|"
        + merged["phase_1_starcoder"].round(6).astype(str)
    )
    merged = merged.drop_duplicates(subset="_key", keep="first")

    # Filter to valid target
    merged = merged[merged[TARGET].notna()].reset_index(drop=True)
    print(f"  After dedup + filter: {len(merged)} points")
    return merged


def build_surface(x: np.ndarray, y: np.ndarray, z: np.ndarray, grid_n: int = 100):
    """Build a smooth RBF-interpolated surface, masked to the convex hull of data."""
    points = np.column_stack([x, y])
    rbf = RBFInterpolator(points, z, kernel="thin_plate_spline", smoothing=1e-2)

    gx = np.linspace(float(x.min()), float(x.max()), grid_n)
    gy = np.linspace(float(y.min()), float(y.max()), grid_n)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    GZ = rbf(grid_pts).reshape(GX.shape)

    # Mask grid points outside the convex hull of observations to avoid
    # wild extrapolation artifacts at the boundary.
    hull = Delaunay(points)
    outside = hull.find_simplex(grid_pts) < 0
    GZ_flat = GZ.ravel()
    GZ_flat[outside] = np.nan
    GZ = GZ_flat.reshape(GX.shape)

    return GX, GY, GZ


def main():
    print("Loading CSVs...")
    df = load_and_merge()

    x = df["phase_0_starcoder"].values.astype(float)
    y = df["phase_1_starcoder"].values.astype(float)
    z = df[TARGET].values.astype(float)

    print("Building interpolated surface...")
    GX, GY, GZ = build_surface(x, y, z)

    # Find global minimum (observed)
    min_idx = int(np.argmin(z))
    min_x, min_y, min_z = float(x[min_idx]), float(y[min_idx]), float(z[min_idx])
    print(f"Global minimum: p0_sc={min_x:.4f}, p1_sc={min_y:.4f}, BPB={min_z:.4f}")

    fig = go.Figure()

    # Smooth interpolated 3D surface (z = interpolated BPB)
    # fig.add_trace(
    #     go.Surface(
    #         x=GX[0],
    #         y=GY[:, 0],
    #         z=GZ,
    #         colorscale="Viridis",
    #         opacity=0.75,
    #         showscale=True,
    #         colorbar=dict(title="BPB<br>(interp)", x=1.02, len=0.5, y=0.3),
    #         name="TPS interpolation",
    #         hoverinfo="skip",
    #     )
    # )

    # Scatter points
    hover_text = [
        f"p0_sc={xi:.4f}<br>p1_sc={yi:.4f}<br>BPB={zi:.4f}<br>src={src}"
        for xi, yi, zi, src in zip(x, y, z, df["_source"])
    ]
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=3.5,
                color=z,
                colorscale="Viridis",
                cmin=float(np.nanmin(GZ)),
                cmax=float(np.nanmax(GZ)),
                showscale=False,
                line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
            ),
            text=hover_text,
            hoverinfo="text",
            name="Runs",
        )
    )

    # Mark the global minimum
    fig.add_trace(
        go.Scatter3d(
            x=[min_x],
            y=[min_y],
            z=[min_z],
            mode="markers+text",
            marker=dict(
                size=10,
                color="red",
                symbol="diamond",
                line=dict(width=2, color="white"),
            ),
            text=[f"MIN: ({min_x:.3f}, {min_y:.3f})<br>BPB={min_z:.4f}"],
            textposition="top center",
            textfont=dict(size=11, color="red"),
            hovertext=f"GLOBAL MIN<br>p0_sc={min_x:.4f}<br>p1_sc={min_y:.4f}<br>BPB={min_z:.4f}",
            hoverinfo="text",
            name="Global minimum",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"2-Phase StarCoder: {TARGET}<br>"
                f"<sub>{len(df)} runs | min BPB={min_z:.4f} "
                f"at (p0={min_x:.3f}, p1={min_y:.3f})</sub>"
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis_title="phase_0 StarCoder weight",
            yaxis_title="phase_1 StarCoder weight",
            zaxis_title="BPB (lower = better)",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        width=1000,
        height=750,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    out_path = SCRIPT_DIR / "two_phase_plots" / "scatter_3d_all.html"
    out_path.parent.mkdir(exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
