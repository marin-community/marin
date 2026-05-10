# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: B018,B905,F841

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
#     "tabulate",
# ]
# ///
"""Marimo notebook for phase-shift-like mixture-scale interaction diagnostics."""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, make_subplots, mo, np, pd


@app.cell
def _(Path):
    REPO_ROOT = Path(__file__).resolve().parents[4]
    PAPER_ROOT = Path(__file__).resolve().parent
    IMG_DIR = PAPER_ROOT / "img"
    TWO_PHASE_ROOT = REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many"
    ANALYSIS_DATASET = TWO_PHASE_ROOT / "analysis_dataset/nd_scale_runs.csv"
    BASELINE_POINTS = PAPER_ROOT / "img/baseline_scaling_trajectories_points.csv"

    METRIC = "eval/uncheatable_eval/bpb"
    SCALE_NAMES = {
        "130m_2p6b": "20M/2.6B",
        "60m_1p2b": "60M/1.2B",
        "300m_6b": "100M/6B",
        "520m_10p4b": "340M/10.4B",
        "1_2b_24b": "900M/24B",
    }
    SCALE_ORDER = ["130m_2p6b", "60m_1p2b", "300m_6b", "520m_10p4b", "1_2b_24b"]

    TEXT_COLOR = "#232B32"
    MUTED_COLOR = "#6C6F7D"
    GRID_COLOR = "#E6E2DA"
    AXIS_COLOR = "#A8A29A"
    ACCENT_BLUE = "#4C78A8"
    ACCENT_RED = "#E24731"
    ACCENT_GREEN = "#59A14F"
    ACCENT_BROWN = "#8F6B38"
    COLOR_SCALE = "RdYlGn_r"
    return (
        ACCENT_BLUE,
        ACCENT_RED,
        ANALYSIS_DATASET,
        AXIS_COLOR,
        BASELINE_POINTS,
        COLOR_SCALE,
        GRID_COLOR,
        METRIC,
        SCALE_NAMES,
        SCALE_ORDER,
        TEXT_COLOR,
    )


@app.cell
def _(AXIS_COLOR, GRID_COLOR, TEXT_COLOR, go, np, pd):
    def dedupe_mixture_scale(frame: pd.DataFrame) -> pd.DataFrame:
        """Keep one target-ready BPB value per mixture, scale, and multiplier."""
        return frame.sort_values("bpb").drop_duplicates(["mixture_id", "scale", "target_budget_multiplier"])

    def format_scale_ticks(scale_keys: list[str], scale_names: dict[str, str]) -> list[str]:
        return [scale_names.get(scale_key, scale_key) for scale_key in scale_keys]

    def apply_layout(fig: go.Figure, *, title: str, x_title: str, y_title: str, height: int = 720) -> go.Figure:
        fig.update_layout(
            template="plotly_white",
            height=height,
            title={
                "text": title,
                "x": 0.02,
                "xanchor": "left",
                "font": {"size": 26, "family": "Times New Roman, Times, serif", "color": TEXT_COLOR},
            },
            font={"family": "Times New Roman, Times, serif", "size": 17, "color": TEXT_COLOR},
            xaxis_title=x_title,
            yaxis_title=y_title,
            margin={"l": 82, "r": 54, "t": 110, "b": 82},
            legend={
                "font": {"size": 15},
                "bgcolor": "rgba(255,255,255,0.86)",
                "bordercolor": "rgba(0,0,0,0)",
            },
            hoverlabel={"bgcolor": "white", "bordercolor": AXIS_COLOR, "font": {"size": 13}},
        )
        fig.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor=AXIS_COLOR,
            ticks="outside",
            tickcolor=AXIS_COLOR,
            gridcolor=GRID_COLOR,
            zeroline=False,
        )
        fig.update_yaxes(
            showline=True,
            linewidth=1,
            linecolor=AXIS_COLOR,
            ticks="outside",
            tickcolor=AXIS_COLOR,
            gridcolor=GRID_COLOR,
            zeroline=False,
        )
        return fig

    def corr_summary(frame: pd.DataFrame, left: str, right: str) -> tuple[float, float]:
        return (
            float(frame[[left, right]].corr(method="spearman").iloc[0, 1]),
            float(frame[[left, right]].corr(method="pearson").iloc[0, 1]),
        )

    def finite_polyfit(x_values: pd.Series, y_values: pd.Series) -> tuple[float, float]:
        x = x_values.to_numpy(dtype=float)
        y = y_values.to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        return tuple(float(v) for v in np.polyfit(x[mask], y[mask], deg=1))

    return apply_layout, dedupe_mixture_scale, finite_polyfit


@app.cell
def _(mo):
    mo.md(
        r"""
    # Phase-shift-like mixture-scale interaction

    This notebook visualizes the empirical evidence that the data mixture and scaling trajectory are
    not separable in the current two-phase domain-mix experiments. The claim is deliberately modest:
    the data support **smooth phase-shift-like mixture-scale interaction**, not a hard critical
    epoch. We see mixture rankings and scaling speeds change with scale, but our measurements are
    mostly endpoint validation BPB rather than dense training-time curves.

    The main diagnostic target is `eval/uncheatable_eval/bpb`, because it is smooth, high-SNR at
    100M/6B, and present across the corrected N/D scaling panels.
    """
    )
    return


@app.cell
def _(ANALYSIS_DATASET, BASELINE_POINTS, METRIC, pd):
    nd_raw = pd.read_csv(ANALYSIS_DATASET, low_memory=False)
    nd_ready = nd_raw.loc[nd_raw[METRIC].notna()].copy()
    nd_ready["bpb"] = nd_ready[METRIC].astype(float)
    nd_ready["target_budget_multiplier"] = nd_ready["target_budget_multiplier"].astype(float)

    baseline_scaling = pd.read_csv(BASELINE_POINTS)
    baseline_scaling = baseline_scaling.sort_values(["method", "x_order"]).copy()
    return baseline_scaling, nd_ready


@app.cell
def _(METRIC, SCALE_NAMES, mo, nd_ready):
    _scale_counts = nd_ready["scale"].map(SCALE_NAMES).value_counts().to_dict()
    _multiplier_counts = nd_ready.groupby(["scale", "target_budget_multiplier"]).size().reset_index(name="rows")
    _multiplier_counts["scale"] = _multiplier_counts["scale"].map(SCALE_NAMES)
    _multiplier_table = (
        _multiplier_counts.pivot(index="scale", columns="target_budget_multiplier", values="rows").fillna(0).astype(int)
    )
    mo.md(
        f"""
    ## Data slice

    - Source rows: `{len(nd_ready):,}` target-ready endpoint rows from
      `analysis_dataset/nd_scale_runs.csv`.
    - Target metric: `{METRIC}`.
    - Corrected scale coverage: `{_scale_counts}`.
    - Budget multipliers are available at 0.5x/1x/2x for the representative panels; the broad
      60M and 100M swarms are primarily 1x.

    Multiplier coverage:

    {_multiplier_table.to_markdown()}
    """
    )
    return


@app.cell
def _(dedupe_mixture_scale, nd_ready, np):
    one_x = nd_ready.loc[np.isclose(nd_ready["target_budget_multiplier"], 1.0)].copy()
    one_x_dedup = dedupe_mixture_scale(one_x)

    transfer_pivot = one_x_dedup.pivot(index="mixture_id", columns="scale", values="bpb")
    common_60_100 = transfer_pivot.dropna(subset=["60m_1p2b", "300m_6b"]).copy()
    common_60_100["drop_60_to_100"] = common_60_100["60m_1p2b"] - common_60_100["300m_6b"]
    common_60_100["rank_60"] = common_60_100["60m_1p2b"].rank(method="average")
    common_60_100["rank_100"] = common_60_100["300m_6b"].rank(method="average")
    common_60_100["rank_shift"] = common_60_100["rank_60"] - common_60_100["rank_100"]
    common_60_100 = common_60_100.reset_index()

    transfer_stats = {
        "n_common": len(common_60_100),
        "spearman": float(common_60_100[["60m_1p2b", "300m_6b"]].corr(method="spearman").iloc[0, 1]),
        "pearson": float(common_60_100[["60m_1p2b", "300m_6b"]].corr(method="pearson").iloc[0, 1]),
        "drop_mean": float(common_60_100["drop_60_to_100"].mean()),
        "drop_std": float(common_60_100["drop_60_to_100"].std()),
        "drop_min": float(common_60_100["drop_60_to_100"].min()),
        "drop_max": float(common_60_100["drop_60_to_100"].max()),
        "rank_abs_median": float(common_60_100["rank_shift"].abs().median()),
        "rank_abs_p90": float(common_60_100["rank_shift"].abs().quantile(0.9)),
        "rank_abs_max": float(common_60_100["rank_shift"].abs().max()),
    }
    return common_60_100, one_x_dedup, transfer_stats


@app.cell
def _(mo, transfer_stats):
    mo.md(
        f"""
    ## Main evidence: broad-swarm transfer is not rank-invariant

    If mixture and scale were approximately separable, the 60M/1.2B and 100M/6B rankings would be
    nearly identical. They are related, but not invariant:

    - common mixtures: `{transfer_stats["n_common"]}`
    - BPB Spearman: `{transfer_stats["spearman"]:.3f}`
    - BPB Pearson: `{transfer_stats["pearson"]:.3f}`
    - 60M→100M BPB drop: mean `{transfer_stats["drop_mean"]:.4f}`, std
      `{transfer_stats["drop_std"]:.4f}`, range `{transfer_stats["drop_min"]:.4f}` to
      `{transfer_stats["drop_max"]:.4f}`
    - absolute rank shift: median `{transfer_stats["rank_abs_median"]:.0f}`, p90
      `{transfer_stats["rank_abs_p90"]:.0f}`, max `{transfer_stats["rank_abs_max"]:.0f}`

    The rank-shift numbers are too large to treat the 60M mixture ranking as a scale-invariant
    ordering. The interpretation is a mixture-dependent scaling speed: some mixtures improve faster
    as N and D increase.
    """
    )
    return


@app.cell
def _(
    ACCENT_BLUE,
    COLOR_SCALE,
    TEXT_COLOR,
    apply_layout,
    common_60_100,
    finite_polyfit,
    go,
    np,
    transfer_stats,
):
    _slope, _intercept = finite_polyfit(common_60_100["60m_1p2b"], common_60_100["300m_6b"])
    _x_line = np.linspace(common_60_100["60m_1p2b"].min(), common_60_100["60m_1p2b"].max(), 200)

    fig_transfer_scatter = go.Figure()
    fig_transfer_scatter.add_trace(
        go.Scatter(
            x=common_60_100["60m_1p2b"],
            y=common_60_100["300m_6b"],
            mode="markers",
            marker={
                "size": 8,
                "color": common_60_100["drop_60_to_100"],
                "colorscale": COLOR_SCALE,
                "showscale": True,
                "colorbar": {"title": "BPB drop<br>60M→100M"},
                "line": {"width": 0.35, "color": "white"},
            },
            text=common_60_100["mixture_id"],
            customdata=common_60_100[["rank_60", "rank_100", "rank_shift", "drop_60_to_100"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "60M/1.2B BPB=%{x:.4f}<br>"
                "100M/6B BPB=%{y:.4f}<br>"
                "60M rank=%{customdata[0]:.0f}<br>"
                "100M rank=%{customdata[1]:.0f}<br>"
                "rank shift=%{customdata[2]:+.0f}<br>"
                "BPB drop=%{customdata[3]:.4f}<extra></extra>"
            ),
            name="242 common mixtures",
        )
    )
    fig_transfer_scatter.add_trace(
        go.Scatter(
            x=_x_line,
            y=_slope * _x_line + _intercept,
            mode="lines",
            line={"color": ACCENT_BLUE, "width": 2, "dash": "dash"},
            name="linear fit",
            hoverinfo="skip",
        )
    )
    fig_transfer_scatter.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="rgba(168,162,154,0.55)",
        borderwidth=1,
        font={"color": TEXT_COLOR, "size": 15},
        text=(
            f"Spearman={transfer_stats['spearman']:.3f}<br>"
            f"median |rank shift|={transfer_stats['rank_abs_median']:.0f}<br>"
            f"max |rank shift|={transfer_stats['rank_abs_max']:.0f}"
        ),
    )
    apply_layout(
        fig_transfer_scatter,
        title="Broad 60M→100M transfer: related but not rank-invariant",
        x_title="60M/1.2B BPB (lower is better)",
        y_title="100M/6B BPB (lower is better)",
    )
    fig_transfer_scatter.update_layout(legend={"orientation": "h", "x": 0.02, "y": 1.05})
    fig_transfer_scatter
    return


@app.cell
def _(mo):
    mo.md(
        """
    The scatter above is the most direct endpoint test. The cloud is positively correlated, so small
    runs are informative. But the color variation and rank shifts show that different mixtures move
    down the scaling curve at different speeds.
    """
    )
    return


@app.cell
def _(ACCENT_BLUE, COLOR_SCALE, apply_layout, common_60_100, go):
    _max_rank = max(common_60_100["rank_60"].max(), common_60_100["rank_100"].max())
    fig_rank_transfer = go.Figure()
    fig_rank_transfer.add_trace(
        go.Scatter(
            x=common_60_100["rank_60"],
            y=common_60_100["rank_100"],
            mode="markers",
            marker={
                "size": 8,
                "color": common_60_100["rank_shift"],
                "colorscale": COLOR_SCALE,
                "cmid": 0,
                "showscale": True,
                "colorbar": {"title": "rank shift<br>60M - 100M"},
                "line": {"width": 0.35, "color": "white"},
            },
            text=common_60_100["mixture_id"],
            customdata=common_60_100[["60m_1p2b", "300m_6b", "drop_60_to_100"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "60M rank=%{x:.0f}<br>"
                "100M rank=%{y:.0f}<br>"
                "60M BPB=%{customdata[0]:.4f}<br>"
                "100M BPB=%{customdata[1]:.4f}<br>"
                "BPB drop=%{customdata[2]:.4f}<extra></extra>"
            ),
            name="mixture",
        )
    )
    fig_rank_transfer.add_trace(
        go.Scatter(
            x=[1, _max_rank],
            y=[1, _max_rank],
            mode="lines",
            line={"color": ACCENT_BLUE, "width": 2, "dash": "dash"},
            name="rank invariant",
            hoverinfo="skip",
        )
    )
    apply_layout(
        fig_rank_transfer,
        title="Rank-transfer view: many mixtures cross as scale changes",
        x_title="rank at 60M/1.2B (lower is better)",
        y_title="rank at 100M/6B (lower is better)",
    )
    fig_rank_transfer.update_yaxes(autorange="reversed")
    fig_rank_transfer.update_xaxes(autorange="reversed")
    fig_rank_transfer.update_layout(legend={"orientation": "h", "x": 0.02, "y": 1.05})
    fig_rank_transfer
    return


@app.cell
def _(mo):
    mo.md(
        """
    A strict rank-invariance story would put almost every point on the diagonal. The actual pattern is
    broad: many mid-ranked 60M mixtures become strong at 100M, and some apparently good 60M mixtures
    fall substantially. This is exactly the visible signature expected from scale-dependent domain
    marginal returns.
    """
    )
    return


@app.cell
def _(common_60_100, nd_ready, one_x_dedup, pd):
    phase0_cols = [col for col in nd_ready.columns if col.startswith("phase_0_")]
    phase_domains = [col[len("phase_0_") :] for col in phase0_cols]
    phase1_cols = [f"phase_1_{domain}" for domain in phase_domains]
    weight_source = one_x_dedup.loc[one_x_dedup["scale"].eq("60m_1p2b")].set_index("mixture_id")
    weight_joined = common_60_100.set_index("mixture_id").join(weight_source[phase0_cols + phase1_cols], how="left")

    domain_records = []
    for domain in phase_domains:
        phase0_col = f"phase_0_{domain}"
        phase1_col = f"phase_1_{domain}"
        if phase0_col not in weight_joined.columns or phase1_col not in weight_joined.columns:
            continue
        exposure = 0.8 * weight_joined[phase0_col].astype(float) + 0.2 * weight_joined[phase1_col].astype(float)
        if exposure.std() <= 1e-12:
            continue
        domain_records.append(
            {
                "domain": domain,
                "spearman_drop": exposure.corr(weight_joined["drop_60_to_100"], method="spearman"),
                "spearman_rank_shift": exposure.corr(weight_joined["rank_shift"], method="spearman"),
                "mean_exposure": exposure.mean(),
                "std_exposure": exposure.std(),
            }
        )

    domain_correlations = pd.DataFrame(domain_records).sort_values("spearman_drop", ascending=False)
    top_domain_correlations = pd.concat(
        [
            domain_correlations.head(10).assign(direction="larger 60M→100M drop"),
            domain_correlations.tail(10).assign(direction="smaller 60M→100M drop"),
        ],
        ignore_index=True,
    )
    return domain_correlations, top_domain_correlations


@app.cell
def _(ACCENT_BLUE, ACCENT_RED, apply_layout, go, np, top_domain_correlations):
    _bar_frame = top_domain_correlations.copy()
    _bar_frame["domain_label"] = _bar_frame["domain"].str.replace("dolma3_cc/", "cc/", regex=False)
    _bar_frame = _bar_frame.sort_values("spearman_drop")
    _colors = np.where(_bar_frame["spearman_drop"] >= 0, ACCENT_RED, ACCENT_BLUE)

    fig_domain_corr = go.Figure()
    fig_domain_corr.add_trace(
        go.Bar(
            x=_bar_frame["spearman_drop"],
            y=_bar_frame["domain_label"],
            orientation="h",
            marker={"color": _colors},
            customdata=_bar_frame[["spearman_rank_shift", "mean_exposure", "std_exposure"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "corr(exposure, BPB drop)=%{x:.3f}<br>"
                "corr(exposure, rank shift)=%{customdata[0]:.3f}<br>"
                "mean exposure=%{customdata[1]:.3f}<br>"
                "std exposure=%{customdata[2]:.3f}<extra></extra>"
            ),
            name="domain exposure correlation",
        )
    )
    fig_domain_corr.add_vline(x=0, line={"color": "rgba(0,0,0,0.35)", "dash": "dash"})
    apply_layout(
        fig_domain_corr,
        title="Which mixture dimensions move faster as scale increases?",
        x_title="Spearman corr(exposure, 60M→100M BPB drop)",
        y_title="domain",
        height=800,
    )
    fig_domain_corr.update_layout(showlegend=False, margin={"l": 270, "r": 70, "t": 110, "b": 80})
    fig_domain_corr
    return


@app.cell
def _(domain_correlations, mo):
    _fast = domain_correlations.sort_values("spearman_drop", ascending=False).head(5)[
        ["domain", "spearman_drop", "spearman_rank_shift"]
    ]
    _slow = domain_correlations.sort_values("spearman_drop").head(5)[["domain", "spearman_drop", "spearman_rank_shift"]]
    mo.md(
        f"""
    The domain-correlation plot is not a causal proof, but it rules out a purely scalar scaling
    explanation. Exposure to some domains is associated with a faster 60M→100M drop; exposure to
    others is associated with a slower drop.

    Top positive correlations:

    {_fast.to_markdown(index=False, floatfmt=".3f")}

    Top negative correlations:

    {_slow.to_markdown(index=False, floatfmt=".3f")}
    """
    )
    return


@app.cell
def _(SCALE_NAMES, SCALE_ORDER, baseline_scaling, pd):
    baseline_display = baseline_scaling.copy()
    baseline_display["scale_key"] = pd.Categorical(baseline_display["scale"], categories=SCALE_ORDER, ordered=True)
    baseline_display["scale_display"] = baseline_display["scale"].map(SCALE_NAMES)
    baseline_display = baseline_display.sort_values(["method", "scale_key"])

    segment_records = []
    for _method, _group in baseline_display.groupby("method", sort=False):
        _ordered = _group.sort_values("scale_key")
        _values = _ordered["metric_value"].to_numpy(dtype=float)
        _scales = _ordered["scale_display"].tolist()
        for _left_idx, _right_idx in zip(range(len(_values) - 1), range(1, len(_values))):
            segment_records.append(
                {
                    "method": _method,
                    "segment": f"{_scales[_left_idx]}→{_scales[_right_idx]}",
                    "drop": _values[_left_idx] - _values[_right_idx],
                }
            )
    baseline_segments = pd.DataFrame(segment_records)
    return baseline_display, baseline_segments


@app.cell
def _(apply_layout, baseline_display, go):
    _method_colors = {
        "GRP no-L2": "#232B32",
        "Proportional": "#8F6B38",
        "Olmix": "#4C78A8",
        "Uniform": "#E24731",
        "UniMax": "#59A14F",
    }
    fig_baseline_scaling = go.Figure()
    for _method, _group in baseline_display.groupby("method", sort=False):
        _ordered = _group.sort_values("scale_key")
        fig_baseline_scaling.add_trace(
            go.Scatter(
                x=_ordered["scale_display"],
                y=_ordered["metric_value"],
                mode="lines+markers",
                line={"width": 3, "color": _method_colors.get(_method, "#777")},
                marker={"size": 10, "line": {"width": 1, "color": "white"}},
                text=_ordered["run_name"],
                customdata=_ordered[["non_embedding_params", "realized_train_tokens"]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "scale=%{x}<br>"
                    "BPB=%{y:.4f}<br>"
                    "N=%{customdata[0]:,.0f}<br>"
                    "D=%{customdata[1]:,.0f}<extra></extra>"
                ),
                name=_method,
            )
        )
    apply_layout(
        fig_baseline_scaling,
        title="Baseline scaling curves are not merely vertical translations",
        x_title="corrected scale label with realized D",
        y_title="uncheatable BPB (lower is better)",
    )
    fig_baseline_scaling.update_layout(legend={"orientation": "h", "x": 0.02, "y": 1.08})
    fig_baseline_scaling
    return


@app.cell
def _(baseline_segments, mo):
    _segment_table = baseline_segments.pivot(index="method", columns="segment", values="drop")
    mo.md(
        f"""
    ## Baseline curves: useful, but not the strongest evidence

    The five deployed/baseline curves all improve with scale, and their segment drops differ. This is
    consistent with mixture-scale interaction, but it is a weaker diagnostic than the 242-row swarm
    transfer because there are only five methods and the N/D path is uneven.

    Segment drops:

    {_segment_table.to_markdown(floatfmt=".4f")}
    """
    )
    return


@app.cell
def _(dedupe_mixture_scale, nd_ready, np, pd):
    multiplier_records = []
    for scale_key in ["130m_2p6b", "300m_6b", "520m_10p4b"]:
        _subset = nd_ready.loc[
            nd_ready["scale"].eq(scale_key) & nd_ready["study_panel"].fillna("").eq("representative12")
        ].copy()
        _subset = dedupe_mixture_scale(_subset)
        _pivot = _subset.pivot(index="mixture_id", columns="target_budget_multiplier", values="bpb")
        _pivot = _pivot.dropna(subset=[0.5, 1.0, 2.0]).copy()
        if _pivot.empty:
            continue
        _pivot["scale"] = scale_key
        _pivot["drop_0.5→1.0"] = _pivot[0.5] - _pivot[1.0]
        _pivot["drop_1.0→2.0"] = _pivot[1.0] - _pivot[2.0]
        _pivot["second_drop_ratio"] = _pivot["drop_1.0→2.0"] / _pivot["drop_0.5→1.0"].replace(0, np.nan)
        multiplier_records.append(_pivot.reset_index())
    multiplier_panel = pd.concat(multiplier_records, ignore_index=True)

    multiplier_long = multiplier_panel.melt(
        id_vars=["mixture_id", "scale"],
        value_vars=["drop_0.5→1.0", "drop_1.0→2.0"],
        var_name="segment",
        value_name="bpb_drop",
    )
    return multiplier_long, multiplier_panel


@app.cell
def _(
    SCALE_NAMES,
    apply_layout,
    go,
    make_subplots,
    multiplier_long,
    multiplier_panel,
):
    _drop_plot = multiplier_long.copy()
    _drop_plot["scale_display"] = _drop_plot["scale"].map(SCALE_NAMES)
    _ratio_plot = multiplier_panel.copy()
    _ratio_plot["scale_display"] = _ratio_plot["scale"].map(SCALE_NAMES)

    fig_multiplier = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Budget-multiplier BPB drops", "Second-drop / first-drop ratio"),
        horizontal_spacing=0.14,
    )
    for _segment, _color in [("drop_0.5→1.0", "#4C78A8"), ("drop_1.0→2.0", "#E24731")]:
        _subset = _drop_plot.loc[_drop_plot["segment"].eq(_segment)]
        fig_multiplier.add_trace(
            go.Box(
                x=_subset["scale_display"],
                y=_subset["bpb_drop"],
                name=_segment,
                marker={"color": _color},
                boxpoints="all",
                jitter=0.42,
                pointpos=0,
                hovertemplate="scale=%{x}<br>drop=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig_multiplier.add_trace(
        go.Box(
            x=_ratio_plot["scale_display"],
            y=_ratio_plot["second_drop_ratio"],
            name="drop ratio",
            marker={"color": "#59A14F"},
            boxpoints="all",
            jitter=0.42,
            pointpos=0,
            hovertemplate="scale=%{x}<br>ratio=%{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig_multiplier.add_hline(y=1.0, row=1, col=2, line={"color": "rgba(0,0,0,0.28)", "dash": "dash"})
    apply_layout(
        fig_multiplier,
        title="Representative-12 budget continuation looks smooth, not cliff-like",
        x_title="corrected scale",
        y_title="BPB drop / ratio",
        height=700,
    )
    fig_multiplier.update_layout(boxmode="group", legend={"orientation": "h", "x": 0.02, "y": 1.05})
    fig_multiplier.update_yaxes(title_text="BPB drop", row=1, col=1)
    fig_multiplier.update_yaxes(title_text="drop ratio", row=1, col=2)
    fig_multiplier
    return


@app.cell
def _(mo, multiplier_panel):
    _summary = (
        multiplier_panel.groupby("scale")[["drop_0.5→1.0", "drop_1.0→2.0", "second_drop_ratio"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    mo.md(
        f"""
    ## What this does and does not prove

    The multiplier panel is a useful negative control. If we had a sharp budget-threshold effect, the
    0.5x→1x and 1x→2x drops might show extreme mixture-specific kinks. Instead, the second drop is
    consistently smaller than the first and the ratio is usually well below one. This is smooth
    diminishing returns, not a critical boundary.

    {_summary.to_markdown()}
    """
    )
    return


@app.cell
def _(mo, transfer_stats):
    mo.md(
        f"""
    # Bottom line

    The evidence supports **phase-shift-like mixture-scale interaction** in the practical sense used
    by scale-aware data-mixing papers:

    1. Mixture rankings transfer imperfectly from 60M/1.2B to 100M/6B: Spearman
       `{transfer_stats["spearman"]:.3f}`, median absolute rank shift
       `{transfer_stats["rank_abs_median"]:.0f}`, max `{transfer_stats["rank_abs_max"]:.0f}`.
    2. Mixtures have measurably different scaling speeds: 60M→100M BPB drop ranges from
       `{transfer_stats["drop_min"]:.4f}` to `{transfer_stats["drop_max"]:.4f}`.
    3. Domain exposures correlate with those scaling-speed differences, which points to genuine
       mixture-scale interaction rather than a scalar scale offset.
    4. The representative-12 budget-multiplier panel is smooth; it does **not** by itself show a hard
       critical threshold.

    Operationally, this argues for fitting a joint law `L(w, N, D, μ)` whose mixture features can
    affect the scale terms. It does not justify claiming a training-time critical epoch without
    denser checkpoint curves and seed-level distributions.
    """
    )
    return


if __name__ == "__main__":
    app.run()
