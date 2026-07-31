# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["kaleido==0.2.1", "numpy", "pandas", "plotly", "scikit-learn", "scipy"]
# ///
"""Look at the fitted surface against the measured one, from several fixed angles.

Scalar error summaries say a model is worse without saying where. This renders the fitted response as
a continuous sheet and the measurements as bare points on top, from orthographic cameras, so a
systematic miss shows up as the points sitting consistently above or below the sheet in one region.

Two deliberate choices. The measured data is never interpolated: a triangulated measurement surface
hides exactly the disagreements worth seeing, because the eye reads two smooth sheets as agreeing.
Only the model is drawn as a surface, and it is evaluated on a dense grid rather than at the sampled
coordinates, so the sheet is the model's actual response and not a smoothing of its predictions.

The projection is orthographic. Under perspective, depth changes apparent height, and a sheet that
looks like it passes through a cloud of points from one angle can be far from them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

DEFAULT_OUTPUT_DIR = wsd80.REFERENCE_OUTPUTS / "wsd80_model_vs_surface_20260728"
GRID = 121
CV_SPLITS = 5
FOLD_SEED = 100
# Above this the surface is dominated by the two corner blowups and nothing near the optimum is
# legible. Points beyond it are still drawn, flattened onto the ceiling and marked.
CEILING = 1.30
# Orthographic cameras. The two axis-aligned views are the ones that show a systematic miss in one
# phase, and the diagonal view looks along the tied policies so the fibers are seen edge on.
VIEWS = (
    ("overview", {"x": 1.05, "y": 1.05, "z": 0.72}),
    ("along_phase_0", {"x": 1.45, "y": 0.0, "z": 0.22}),
    ("along_phase_1", {"x": 0.0, "y": 1.45, "z": 0.22}),
    ("down_the_diagonal", {"x": 1.1, "y": -1.1, "z": 0.30}),
    ("from_above", {"x": 0.0, "y": 0.0, "z": 1.5}),
)


def grid_weights(phase_0: np.ndarray, phase_1: np.ndarray) -> np.ndarray:
    return np.stack([np.column_stack([1.0 - phase_0, phase_0]), np.column_stack([1.0 - phase_1, phase_1])], axis=1)


def blocked_folds(weights: np.ndarray, n_splits: int, seed: int) -> tuple:
    coordinates = np.column_stack([weights[:, 0, :], weights[:, 1, :]])
    blocks = KMeans(n_clusters=n_splits, n_init=10, random_state=seed).fit_predict(coordinates)
    rows = np.arange(len(blocks))
    return tuple((~np.isin(rows, rows[blocks == b]), np.isin(rows, rows[blocks == b])) for b in np.unique(blocks))


def fitted_sheet(model, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, resolution)
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    predicted = model.predict(grid_weights(grid_0.ravel(), grid_1.ravel())).reshape(grid_0.shape)
    return axis, axis, predicted


def build_figure(panel, model, title: str) -> go.Figure:
    axis_0, axis_1, sheet = fitted_sheet(model, GRID)
    measured = panel.y
    predicted_at_points = model.predict(panel.weights)
    residual = measured - predicted_at_points
    above = measured > CEILING

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=axis_1,
            y=axis_0,
            z=np.clip(sheet, None, CEILING),
            colorscale="Blues",
            opacity=0.72,
            showscale=False,
            name="fitted response",
            hovertemplate="model %{z:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=panel.phase_1[:, 1],
            y=panel.phase_0[:, 1],
            z=np.clip(measured, None, CEILING),
            mode="markers",
            marker={
                "size": np.where(above, 3.0, 4.2),
                "color": residual,
                "colorscale": "RdBu",
                "cmid": 0.0,
                "colorbar": {"title": "measured<br>minus model", "len": 0.5, "thickness": 11, "x": 0.94},
                "line": {"width": 0.4, "color": "rgba(40,40,40,0.55)"},
                "symbol": np.where(above, "diamond", "circle"),
            },
            name="measured",
            hovertemplate=(
                "p0 %{y:.3f}, p1 %{x:.3f}<br>measured %{customdata[0]:.4f}"
                "<br>model %{customdata[1]:.4f}<br>residual %{marker.color:+.4f}<extra></extra>"
            ),
            customdata=np.column_stack([measured, predicted_at_points]),
        )
    )
    figure.update_layout(
        template="simple_white",
        title={"text": title},
        scene={
            "xaxis": {"title": "phase 1 StarCoder", "range": [0, 1]},
            "yaxis": {"title": "phase 0 StarCoder", "range": [0, 1]},
            "zaxis": {"title": "code BPB", "range": [0.92, CEILING]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.72},
            "domain": {"x": [0.0, 0.88], "y": [0.0, 1.0]},
        },
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        showlegend=False,
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = wsd80.load_surface()
    sigma = wsd80.training_seed_sigma(wsd80.load_fiber_replicates())
    geometry = retained.Geometry(c0=panel.c0, c1=panel.c1, phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION)
    model = retained.fit(panel.weights, panel.y, geometry, folds=blocked_folds(panel.weights, CV_SPLITS, FOLD_SEED))

    residual = panel.y - model.predict(panel.weights)
    print(f"full-panel fit over {len(panel.y)} coordinates, training-seed sigma {sigma:.6f}")
    print(f"  rmse {np.sqrt(np.mean(residual**2)) / sigma:.2f} sigma")
    print(f"  median |residual| {np.median(np.abs(residual)) / sigma:.2f} sigma")
    print(f"  worst over-prediction {residual.min() / sigma:+.1f} sigma")
    print(f"  worst under-prediction {residual.max() / sigma:+.1f} sigma")
    print(f"  shape {model.shape}")

    figure = build_figure(panel, model, f"Fitted response against measured 80/20 WSD surface ({len(panel.y)} points)")
    figure.write_html(args.output_dir / "model_vs_surface.html", include_plotlyjs="cdn")
    for name, eye in VIEWS:
        figure.update_layout(scene_camera={"eye": eye, "projection": {"type": "orthographic"}})
        figure.write_image(args.output_dir / f"model_vs_surface_{name}.png", width=1000, height=780, scale=2)
    print(f"wrote {len(VIEWS)} views to {args.output_dir}")


if __name__ == "__main__":
    main()
