# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chinchilla-style scaling fit on the *first* (step-0) math eval loss.

At 0% of midtraining, the math eval loss is the base model's loss before any
midtraining data is seen, so it is identical across every mix/lr within a scale
(one clean value per compute scale). That makes it a clean pretraining scaling
law: base-model math loss vs pretraining compute C.

We fit the same Chinchilla floor+power form used for endpoint scaling in
``delphi_small_final_loss_scaling.py`` -- ``L(C) = E + A * (C/1e18)^(-alpha)``
via ``scipy.optimize.curve_fit`` -- training on the 3e18->3e20 ladder and
holding out 1e21/1e22 to check extrapolation. A log-linear reference fit is
shown alongside.

Run:
    uv run python scripts/analysis/delphi_first_math_loss_scaling.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

OUT_DIR = Path("midtrain_analysis_outputs/small_final_loss_scaling")
TRAJECTORY_POINTS = OUT_DIR / "trajectory_points.csv"
HTML_OUT = OUT_DIR / "first_math_loss_chinchilla.html"

METRIC_LABEL = "math_val_loss"
FLOP_NORM = 1e18  # normalize compute the same way the endpoint fit does

# Training ladder vs held-out scales (mirrors the rest of the report).
TRAIN_SCALES = [3e18, 9e18, 2e19, 3e19, 9e19, 2e20, 3e20]
HELDOUT_SCALES = [1e21, 1e22]

# Mixtures for the best-LR endpoint overlays. Name p{paloma%}m{math%}, so p33m67 is the
# most math-heavy recipe. Each gets its own color and an independent floor+power fit.
MIX_COLORS = {
    "p33m67": "#2ca02c",  # 67% math
    "p50m50": "#ff7f0e",  # 50% math
    "p67m33": "#d62728",  # 33% math
}
BASE_COLOR = "#1f77b4"


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def floor_power_model(x: np.ndarray, floor: float, amplitude: float, alpha: float) -> np.ndarray:
    """Chinchilla form: irreducible floor + power law in (normalized) compute."""
    return floor + amplitude * np.power(x, -alpha)


def fit_floor_power(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y_min = float(np.min(y))
    y_span = float(np.max(y) - np.min(y))
    floor0 = max(1e-8, y_min - 0.2 * y_span)
    params, _ = curve_fit(
        floor_power_model,
        x,
        y,
        p0=(floor0, max(y_span, 1e-6), 0.1),
        bounds=([0.0, 0.0, 0.0], [y_min * 0.999, max(float(np.max(y)) * 100, 1.0), 5.0]),
        maxfev=20_000,
    )
    pred = floor_power_model(x, *params)
    return {
        "floor": float(params[0]),
        "amplitude": float(params[1]),
        "exponent": float(params[2]),
        "r2": r2_score(y, pred),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
    }


def load_first_math_loss() -> pd.DataFrame:
    df = pd.read_csv(TRAJECTORY_POINTS)
    first = df[(df["metric_label"] == METRIC_LABEL) & (df["step"] == 0)]
    base = (
        first.groupby("scale")["value"].mean().reset_index().rename(columns={"value": "math_loss"}).sort_values("scale")
    )
    base["split"] = np.where(base["scale"].isin(TRAIN_SCALES), "train (3e18->3e20)", "held-out")
    return base.reset_index(drop=True)


def load_best_endpoint_math_loss(mix: str) -> pd.DataFrame:
    """Best (lowest) final math val loss across the LR sweep at each scale, for one mix."""
    df = pd.read_csv(TRAJECTORY_POINTS)
    m = df[(df["metric_label"] == METRIC_LABEL) & (df["mix"] == mix)]
    per_run = m.groupby(["scale", "lr", "run_id"])["final_value"].first().reset_index()
    best_idx = per_run.groupby("scale")["final_value"].idxmin()
    best = per_run.loc[best_idx].rename(columns={"final_value": "math_loss", "lr": "best_lr"}).sort_values("scale")
    best["split"] = np.where(best["scale"].isin(TRAIN_SCALES), "train (3e18->3e20)", "held-out")
    return best.reset_index(drop=True)


def fit_log_linear(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Pure power law L = A·x^(-alpha) (a straight line in log-log)."""
    slope, intercept = np.polyfit(np.log(x), np.log(y), deg=1)
    pred = np.exp(intercept + slope * np.log(x))
    return {
        "floor": float("nan"),
        "amplitude": float(math.exp(intercept)),
        "exponent": float(-slope),
        "r2": r2_score(y, pred),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
    }


# Fit forms the dropdown switches between. Each entry: (key, button label, equation, fitter).
FIT_KINDS = [
    ("floor_power", "Chinchilla floor+power", "L(C)=E+A*(C/1e18)<sup>-alpha</sup>", fit_floor_power),
    ("log_linear", "Log-linear (power law)", "L(C)=A*(C/1e18)<sup>-alpha</sup>", fit_log_linear),
]
FIT_LABEL = {key: label for key, label, _, _ in FIT_KINDS}
FIT_EQUATION = {key: eq for key, _, eq, _ in FIT_KINDS}


def eval_model(kind: str, x: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    power = fit["amplitude"] * np.power(x, -fit["exponent"])
    return power + fit["floor"] if kind == "floor_power" else power


def fit_on_train(points: pd.DataFrame) -> dict[str, dict[str, float]]:
    train = points[points["split"].str.startswith("train")]
    x = train["scale"].to_numpy(dtype=float) / FLOP_NORM
    y = train["math_loss"].to_numpy(dtype=float)
    return {key: fitter(x, y) for key, _, _, fitter in FIT_KINDS}


@dataclass(frozen=True)
class Series:
    label: str
    color: str
    points: pd.DataFrame
    fits: dict[str, dict[str, float]]  # keyed by FIT_KINDS key
    has_lr: bool  # endpoint series carry a best_lr column; the base series does not


def fmt_scale(scale: float) -> str:
    exp = round(math.log10(scale))
    mant = scale / 10**exp
    return f"{mant:.0f}e{exp}"


def _add_curve(fig: go.Figure, s: Series) -> None:
    """Add one fitted line per fit kind (only the active kind starts visible) plus
    train/held-out observation markers. Only lines carry legend entries; marker
    shapes are explained once by separate legend keys. The dropdown toggles which
    fit-kind line is visible."""
    train = s.points[s.points["split"].str.startswith("train")]
    held = s.points[s.points["split"] == "held-out"].sort_values("scale")
    x_grid = np.logspace(math.log10(s.points["scale"].min()), math.log10(s.points["scale"].max()), 200)

    for i, (key, _, _, _) in enumerate(FIT_KINDS):
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=eval_model(key, x_grid / FLOP_NORM, s.fits[key]),
                mode="lines",
                name=s.label,
                line=dict(color=s.color, width=3),
                visible=(i == 0),
            )
        )

    def lr_suffix(frame: pd.DataFrame) -> list[str]:
        if not s.has_lr:
            return [""] * len(frame)
        return [f"<br>picked LR {int(lr) / 100:.2f}x" for lr in frame["best_lr"]]

    fig.add_trace(
        go.Scatter(
            x=train["scale"],
            y=train["math_loss"],
            mode="markers",
            showlegend=False,
            marker=dict(color=s.color, size=12, symbol="diamond", line=dict(color="white", width=1)),
            text=[
                f"{fmt_scale(sc)}: {v:.4f}{lr}"
                for sc, v, lr in zip(train["scale"], train["math_loss"], lr_suffix(train), strict=True)
            ],
            hovertemplate=f"{s.label}<br>%{{text}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=held["scale"],
            y=held["math_loss"],
            mode="markers",
            showlegend=False,
            marker=dict(color=s.color, size=15, symbol="star", line=dict(color="white", width=1)),
            text=[
                f"{fmt_scale(sc)}: observed {v:.4f}{lr}"
                for sc, v, lr in zip(held["scale"], held["math_loss"], lr_suffix(held), strict=True)
            ],
            hovertemplate=f"{s.label}<br>%{{text}}<extra></extra>",
        )
    )


def _visibility(series: list[Series], kind: str) -> list[bool]:
    """Trace-visibility array selecting the active fit kind; markers/keys stay on."""
    active = next(i for i, (key, *_) in enumerate(FIT_KINDS) if key == kind)
    vis: list[bool] = []
    for _ in series:
        vis.extend(i == active for i in range(len(FIT_KINDS)))
        vis.extend([True, True])  # train + held markers
    vis.extend([True, True])  # marker-shape legend keys
    return vis


def _fit_summary_line(s: Series, kind: str) -> str:
    fit = s.fits[kind]
    held = s.points[s.points["split"] == "held-out"].sort_values("scale")
    pred = eval_model(kind, held["scale"].to_numpy() / FLOP_NORM, fit)
    err = [
        f"{fmt_scale(sc)} {100 * (p - o) / o:+.1f}%"
        for sc, o, p in zip(held["scale"], held["math_loss"], pred, strict=True)
    ]
    floor = f"E={fit['floor']:.3f} " if kind == "floor_power" else ""
    return (
        f"<span style='color:{s.color}'>{s.label}: {floor}A={fit['amplitude']:.3f} "
        f"alpha={fit['exponent']:.3f} R²={fit['r2']:.4f} | held-out {' '.join(err)}</span>"
    )


def _fit_box(series: list[Series], kind: str) -> dict:
    return dict(
        x=0.99,
        y=0.99,
        xref="paper",
        yref="paper",
        align="left",
        showarrow=False,
        text=f"<b>{FIT_LABEL[kind]} fits — {FIT_EQUATION[kind]} (train 3e18→3e20)</b><br>"
        + "<br>".join(_fit_summary_line(s, kind) for s in series),
        font=dict(size=12),
        bordercolor="#888",
        borderwidth=1,
        borderpad=6,
        bgcolor="rgba(255,255,255,0.9)",
    )


def _picked_lr_box(endpoints: list[Series]) -> dict:
    """Monospace table of the LR picked at each scale below 1e22, per mix."""
    scales = sorted({sc for s in endpoints for sc in s.points["scale"] if sc < 1e22})
    lookup = {s.label: dict(zip(s.points["scale"], s.points["best_lr"], strict=True)) for s in endpoints}
    header = "scale  " + " ".join(f"{s.label:>7}" for s in endpoints)
    rows = [header]
    for sc in scales:
        cells = " ".join(f"{int(lookup[s.label][sc]) / 100:>7.2f}" for s in endpoints)
        rows.append(f"{fmt_scale(sc):<6} {cells}")
    return dict(
        x=0.01,
        y=0.02,
        xref="paper",
        yref="paper",
        align="left",
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        text="<b>Best LRx picked per scale (models below 1e22)</b><br>" + "<br>".join(rows),
        font=dict(size=11, family="Courier New, monospace"),
        bordercolor="#888",
        borderwidth=1,
        borderpad=6,
        bgcolor="rgba(255,255,255,0.9)",
    )


def _annotations(series: list[Series], endpoints: list[Series], kind: str) -> list[dict]:
    return [_fit_box(series, kind), _picked_lr_box(endpoints)]


def build_figure(base: Series, endpoints: list[Series]) -> go.Figure:
    series = [base, *endpoints]
    fig = go.Figure()
    for s in series:
        _add_curve(fig, s)

    # Legend keys explaining the two marker shapes (shared across all series).
    for symbol, name in (("diamond", "observed (train, fit 3e18→3e20)"), ("star", "observed (held-out 1e21/1e22)")):
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers", name=name, marker=dict(color="#555", size=12, symbol=symbol))
        )

    default_kind = FIT_KINDS[0][0]
    fig.update_layout(
        title=dict(
            text="Delphi math eval loss vs pretraining compute — base (step-0) vs best-LR endpoint by mix",
            x=0.5,
            xanchor="center",
            y=0.97,
        ),
        xaxis=dict(title="pretraining compute C (FLOPs)", type="log"),
        yaxis=dict(title="math eval loss (nemotron_cc_math 4plus)"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        margin=dict(t=95, r=40, b=130, l=80),
        autosize=True,
        height=760,
        annotations=_annotations(series, endpoints, default_kind),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.12,
                yanchor="top",
                pad=dict(l=4, r=4, t=2, b=2),
                buttons=[
                    dict(
                        label=label,
                        method="update",
                        args=[
                            {"visible": _visibility(series, key)},
                            {"annotations": _annotations(series, endpoints, key)},
                        ],
                    )
                    for key, label, _, _ in FIT_KINDS
                ],
            )
        ],
    )
    return fig


def _report(name: str, points: pd.DataFrame, fits: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {name} ===")
    print(points.to_string(index=False))
    held = points[points["split"] == "held-out"]
    for kind, label, _, _ in FIT_KINDS:
        fit = fits[kind]
        floor = f"{fit['floor']:.4f} + " if kind == "floor_power" else ""
        print(
            f"{label} (train 3e18->3e20):  L(C) = {floor}{fit['amplitude']:.4f} * (C/1e18)^(-{fit['exponent']:.4f})"
            f"   R²={fit['r2']:.4f}  RMSE={fit['rmse']:.4f}"
        )
        for _, row in held.iterrows():
            pred = float(eval_model(kind, np.array([row["scale"] / FLOP_NORM]), fit)[0])
            pct = 100 * (pred - row["math_loss"]) / row["math_loss"]
            print(
                f"    held-out {row['scale']:.0e}: observed {row['math_loss']:.4f}  predicted {pred:.4f}  "
                f"rel_err {pct:+.2f}%"
            )


def main() -> None:
    base_points = load_first_math_loss()
    base = Series("base (step-0)", BASE_COLOR, base_points, fit_on_train(base_points), has_lr=False)
    _report("Base-model step-0 math loss per scale", base.points, base.fits)

    endpoints: list[Series] = []
    for mix, color in MIX_COLORS.items():
        pts = load_best_endpoint_math_loss(mix)
        fits = fit_on_train(pts)
        endpoints.append(Series(f"best-LR endpoint ({mix})", color, pts, fits, has_lr=True))
        _report(f"Best-LR endpoint math loss per scale (mix {mix})", pts, fits)

    fig = build_figure(base, endpoints)
    fig.write_html(HTML_OUT, include_plotlyjs="cdn", full_html=True, default_width="100%", config={"responsive": True})
    print(f"\nWrote {HTML_OUT}")


if __name__ == "__main__":
    main()
