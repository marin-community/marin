import logging
import os
import re
from dataclasses import dataclass
from typing import Sequence

import fsspec
import jax.numpy as jnp
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jaxopt import ScipyMinimize

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from marin.execution.executor import ExecutorStep, InputName, output_path_of
from marin.scaling_laws.eval_metrics_reader import (
    EvalMetricsAnalysisConfig,
    create_analysis_step,
    extract_run_name_from_path,
    read_metrics_dataframe,
)


logger = logging.getLogger(__name__)


def build_wandb_run_overrides(wandb_sources: list[tuple[str, str]]) -> dict[str, str]:
    """
    Builds a mapping from clean run names to full WandB displayNames.
    This is used to find WandB runs for backfill, even when checkpoint paths
    use the new clean names but WandB displayNames have legacy hash suffixes.
    """
    if not WANDB_AVAILABLE:
        logger.warning("wandb not available, cannot build run overrides")
        return {}

    api = wandb.Api()
    overrides = {}  # clean_name -> full_displayName

    for entity_project, fragment in wandb_sources:
        if "/" not in entity_project:
            raise ValueError(f"Bad ENTITY/PROJECT: {entity_project}")

        regex = rf"isoflop.*({fragment}).*"
        filters = {"displayName": {"$regex": regex}, "state": "finished"}
        try:
            runs = api.runs(entity_project.strip(), filters=filters)
            for run in runs:
                display_name = run.displayName
                # The key for the override map is the "clean" name, without hash
                clean_name = re.sub(r"-[0-9a-fA-F]{6}$", "", display_name)
                # The value is the full name, which is used as the run ID for backfill
                overrides[clean_name] = display_name
        except Exception as e:
            logger.warning(f"Failed to query WandB for {entity_project}: {e}")

    logger.info(f"Built {len(overrides)} WandB run overrides")
    return overrides


# ---------------- Theme ----------------
pio.templates.default = "plotly_white"

# ---------------- Constants ----------------
PALETTE = [
    "#1877F2",
    "#F0701A",
    "#5A24C7",
    "#E42C97",
    "#00487C",
    "#0EAC96",
    "#AB76FF",
    "#B50550",
    "#0099E6",
    "#22085F",
    "#783301",
]
MARKERS = [
    "circle",
    "square",
    "cross",
    "x",
    "triangle-up",
    "triangle-down",
    "triangle-left",
    "triangle-right",
    "pentagon",
    "hexagon",
    "hexagon2",
    "star",
    "star-triangle-up",
    "star-triangle-down",
    "star-square",
    "star-diamond",
    "hourglass",
    "bowtie",
]
DASHES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
DEFAULT_METRIC_KEY = "eval/paloma/c4_en/bpb"
SEQ_LEN = 4096

_MIN_MARKER = dict(symbol="diamond", size=10, color="#000000")
_SCALE_MARKER = dict(symbol="circle", size=9, color=PALETTE[0])
_SCALE_LINE = dict(dash="dot", width=2, color=PALETTE[0])

REQUIRED_TAGS = {"steps", "B", "FLOPs", "d", "L"}
CANON_LABELS = ["nemo", "comma", "dclm"]  # canonical dataset names we detect in displayName


# ---------------- Helpers ----------------
def _tags_to_dict(tags):
    return {k: v for k, v in (t.split("=", 1) for t in tags if "=" in t)}


def _parse_isoflop_run_name(run_name: str) -> dict | None:
    """Parse metadata from isoflop run name.

    Expected format: isoflop-{budget}-d{hidden}-L{layers}-B{batch}-{experiment_name}
    Optionally with a trailing -<hash> which is ignored.
    E.g., 'isoflop-1e+19-d2048-L16-B1024-nemo-wider-depth-adapt'
    or 'isoflop-1e+19-d2048-L16-B1024-nemo-wider-depth-adapt-a1b2c3'

    Returns dict with: flops, d, L, B, experiment_name or None if parsing fails.
    """
    # Strip optional -<hash> suffix
    run_name = re.sub(r"-[0-9a-fA-F]{6}$", "", run_name)

    pattern = r"isoflop-([0-9.e+]+)-d(\d+)-L(\d+)-B(\d+)-(.+)"
    match = re.match(pattern, run_name)
    if not match:
        return None

    flops_str, d, L, B, exp_name = match.groups()
    return {
        "flops": float(flops_str),
        "d": int(d),
        "L": int(L),
        "B": int(B),
        "experiment_name": exp_name,
    }


def df_from_sources(source_runs: list[tuple[list, str]], metric_key: str = DEFAULT_METRIC_KEY) -> pd.DataFrame:
    """Build a dataframe from [(runs, fragment), ...] and compute a 'label' per row."""
    records = []
    for runs, fragment in source_runs:
        for run in runs:
            summary = run.summary
            tags = _tags_to_dict(run.tags)
            if not REQUIRED_TAGS.issubset(tags):
                continue

            steps = float(tags["steps"])
            batch = float(tags["B"])
            flops = float(tags["FLOPs"])
            if flops < 1e18:
                continue

            tokens = steps * batch * SEQ_LEN
            loss = summary.get(metric_key)
            if loss is None:
                continue

            params = summary.get("parameter_count")
            name = run.displayName

            records.append(dict(tokens=tokens, loss=loss, flops=flops, params=params, name=name, label=fragment))
    return pd.DataFrame.from_records(records)


def _robust_quad_logx(x: jnp.ndarray, y: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    L = jnp.log10(x)

    def huber(residual):
        abs_r = jnp.abs(residual)
        quad = 0.5 * residual**2
        linear = delta * (abs_r - 0.5 * delta)
        return jnp.where(abs_r <= delta, quad, linear)

    def objective(params):
        a, b, c = params
        pred = a * L**2 + b * L + c
        residuals = y - pred
        return jnp.sum(huber(residuals))

    opt = ScipyMinimize(fun=objective, method="BFGS", value_and_grad=False)
    init = jnp.array(jnp.polyfit(L, y, 2)) if len(L) >= 3 else jnp.array([0.0, *jnp.polyfit(L, y, 1)])
    return opt.run(init_params=init).params


def iso_plot_with_minima_df(df: pd.DataFrame):
    """
    Expects df columns: tokens, loss, flops, params, name, label.
    ISO plot:
      - points: color by compute bucket (FLOPs), marker shape by dataset label
      - dashed parabolas: per-(label, FLOPs) robust quadratic fits (restored)
      - minima per (label, FLOPs): black diamonds
    SCALING plot:
      - one N* ~ A*C^alpha fit line per dataset (distinct color/dash)
      - dataset minima as points in matching color
    """
    if df is None or df.empty:
        return go.Figure(), go.Figure()

    present = list(dict.fromkeys(df["label"].tolist()))
    datasets = [lab for lab in CANON_LABELS if lab in present] + [lab for lab in present if lab not in CANON_LABELS]

    # Visual maps
    buckets = sorted(df.flops.unique())
    bucket_color = {C: PALETTE[i % len(PALETTE)] for i, C in enumerate(buckets)}  # ISO: color = compute bucket
    ds_marker = {lab: MARKERS[i % len(MARKERS)] for i, lab in enumerate(datasets)}  # ISO: shape = dataset
    DS_COLORS = PALETTE
    DASHES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

    fig_iso = go.Figure()
    minima = []  # (label, C, N_star, loss)

    # ---- ISO: scatter, per-(label,C) parabola (RESTORED), and minima
    for lab in datasets:
        for C in buckets:
            sub = df[(df.flops == C) & (df.label == lab)].sort_values("tokens")
            if sub.empty:
                continue

            # scatter
            fig_iso.add_trace(
                go.Scatter(
                    x=sub.tokens,
                    y=sub.loss,
                    mode="markers",
                    marker=dict(symbol=ds_marker[lab], color=bucket_color[C], size=8),
                    name=f"{lab}, {C:.2e} FLOPs",
                    legendgroup=f"{lab}, {C:.2e}",
                    hovertemplate=(
                        "C=%{text:.2e} FLOPs<br>tokens=%{x:.3e}<br>"
                        "loss=%{y:.4f}<br>params=%{customdata:.3e}<extra></extra>"
                    ),
                    text=[C] * len(sub),
                    customdata=sub.params.values,
                )
            )

            # robust quadratic fit in log10(tokens)
            a, b, c = _robust_quad_logx(jnp.array(sub.tokens.values), jnp.array(sub.loss.values))
            if a == 0:
                continue

            # draw the parabola for this (lab, C)
            Ls = jnp.linspace(jnp.log10(sub.tokens.min()), jnp.log10(sub.tokens.max()), 200)
            fig_iso.add_trace(
                go.Scatter(
                    x=10**Ls,
                    y=a * Ls**2 + b * Ls + c,
                    mode="lines",
                    line=dict(color=bucket_color[C], dash="dash", width=2),
                    showlegend=False,  # avoid legend clutter
                    legendgroup=f"{lab}, {C:.2e}",
                )
            )

            # compute and draw minimum
            L_opt = -b / (2 * a)
            N_star = float(10**L_opt)
            loss_opt = float(a * L_opt**2 + b * L_opt + c)
            params_opt = sub.iloc[(sub.tokens - N_star).abs().argmin()].params
            minima.append((lab, float(C), N_star, loss_opt))

            fig_iso.add_trace(
                go.Scatter(
                    x=[N_star],
                    y=[loss_opt],
                    mode="markers",
                    marker=_MIN_MARKER,
                    showlegend=False,
                    legendgroup=f"{lab}, {C:.2e}",
                    hovertemplate=(
                        "<b>Compute-optimal</b><br>"
                        "C=%{text:.2e} FLOPs<br>tokens=%{x:.3e}<br>"
                        "loss=%{y:.4f}<br>params=%{customdata:.3e}<extra></extra>"
                    ),
                    text=[C],
                    customdata=[params_opt],
                )
            )

    fig_iso.update_layout(
        template="plotly_white",
        xaxis_type="log",
        xaxis_title="Tokens (log scale)",
        yaxis_title="Bits Per Byte Validation",
        title="Marin IsoFLOP Suite",
        width=1000,
        height=600,
    )

    # ---- SCALING: separate line per dataset
    if not minima:
        return fig_iso, go.Figure()

    fig_scale = go.Figure()
    by_lab = {}
    for lab, C, N_star, _ in minima:
        by_lab.setdefault(lab, []).append((C, N_star))

    for i, lab in enumerate(datasets):
        pts = by_lab.get(lab, [])
        if not pts:
            continue
        pts = sorted(pts)
        Cs, Ns = zip(*pts, strict=False)
        Cs = jnp.array(Cs)
        Ns = jnp.array(Ns)

        color = DS_COLORS[i % len(DS_COLORS)]
        dash = DASHES[i % len(DASHES)]

        # plot minima points
        fig_scale.add_trace(
            go.Scatter(
                x=list(map(float, Cs)),
                y=list(map(float, Ns)),
                mode="markers",
                marker=dict(symbol=_SCALE_MARKER["symbol"], size=_SCALE_MARKER["size"], color=color),
                name=f"{lab} minima",
                legendgroup=lab,
            )
        )

        if len(Cs) >= 2:
            alpha, logA = jnp.polyfit(jnp.log10(Cs), jnp.log10(Ns), 1)
            A = 10**logA
            Cmin, Cmax = float(Cs.min()), float(Cs.max())
            C_fit = jnp.logspace(jnp.log10(Cmin) - 0.1, jnp.log10(Cmax) + 0.1, 400)
            N_fit = A * (C_fit**alpha)

            fig_scale.add_trace(
                go.Scatter(
                    x=list(map(float, C_fit)),
                    y=list(map(float, N_fit)),
                    mode="lines",
                    line=dict(color=color, dash=dash, width=_SCALE_LINE["width"]),
                    name=f"{lab} fit",
                    legendgroup=lab,
                )
            )

    fig_scale.update_layout(
        template="plotly_white",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="Compute budget C (FLOPs, log)",
        yaxis_title="Optimal tokens N* (log)",
        title="Scaling fits per dataset",
    )

    return fig_iso, fig_scale


# ---------------- Executor Integration ----------------
@dataclass(frozen=True)
class IsoFlopAnalysisConfig(EvalMetricsAnalysisConfig):
    """Config for isoflop analysis - extends base eval metrics analysis.

    Inherits training_runs, output_path, and backfill settings from base.
    Adds isoflop-specific parameters.
    """

    metric_key: str = DEFAULT_METRIC_KEY
    """Which metric to use for loss (default: eval/paloma/c4_en/bpb)."""

    label_map: dict[str, str] | None = None
    """Optional mapping from experiment_name -> display label."""

    upload_to_wandb: bool = True
    """Whether to upload plots to WandB."""

    wandb_entity: str = "marin-community"
    wandb_project: str = "marin-analysis"
    wandb_run_name: str = "isoflop-analysis"


def _transform_metrics_for_isoflop(
    df: pd.DataFrame,
    metric_key: str,
    label_map: dict[str, str] | None,
) -> pd.DataFrame:
    """Transform raw metrics DataFrame into isoflop plotting format.

    Takes the generic metrics DataFrame from read_metrics_dataframe() and
    transforms it into the format expected by iso_plot_with_minima_df():
    columns: tokens, loss, flops, params, name, label
    """
    if df.empty:
        return pd.DataFrame(columns=["tokens", "loss", "flops", "params", "name", "label"])

    # Get final metrics for each run (max step)
    final_metrics = df.loc[df.groupby("run_path")["step"].idxmax()].copy()

    records = []
    for _, row in final_metrics.iterrows():
        run_path = row["run_path"]
        run_name = extract_run_name_from_path(run_path)

        # Parse metadata from run name
        meta = _parse_isoflop_run_name(run_name)
        if meta is None:
            print(f"Warning: Could not parse metadata from run name: {run_name}")
            continue

        flops = meta["flops"]
        if flops < 1e18:
            continue

        # Calculate tokens = steps * batch * seq_len
        steps = row["step"]
        batch = meta["B"]
        tokens = steps * batch * SEQ_LEN

        # Get loss from the metric column
        loss = row.get(metric_key)
        if loss is None or pd.isna(loss):
            print(f"Warning: Missing metric {metric_key} for run {run_name}")
            continue

        params = row.get("parameter_count")
        if params is None or pd.isna(params):
            params = None

        # Determine label
        exp_name = meta["experiment_name"]
        if label_map and exp_name in label_map:
            label = label_map[exp_name]
        else:
            label = exp_name
            for canon in CANON_LABELS:
                if canon in exp_name.lower():
                    label = canon
                    break

        records.append(
            dict(
                tokens=tokens,
                loss=loss,
                flops=flops,
                params=params,
                name=run_name,
                label=label,
            )
        )

    return pd.DataFrame.from_records(records)


def run_isoflop_analysis(config: IsoFlopAnalysisConfig) -> None:
    """Run isoflop analysis from training runs.

    This is a subtype of eval metrics analysis that:
    1. Reads metrics using the base read_metrics_dataframe()
    2. Transforms them for isoflop plotting
    3. Generates and saves isoflop/scaling plots
    """
    # Use inherited metrics reading from base
    raw_df = read_metrics_dataframe(config)

    if raw_df.empty:
        print("Warning: No eval metrics found")
        return

    # Transform to isoflop format
    df = _transform_metrics_for_isoflop(raw_df, config.metric_key, config.label_map)

    if df.empty:
        print("Warning: No valid isoflop data after transformation")
        return

    print(f"Transformed {len(df)} runs for isoflop analysis")
    fig_iso, fig_scaling = iso_plot_with_minima_df(df)

    # Save plots locally
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)

    iso_path = os.path.join(config.output_path, "isoflop_plot.html")
    scaling_path = os.path.join(config.output_path, "scaling_plot.html")

    with fs.open(iso_path, "w") as f:
        f.write(fig_iso.to_html())
    print(f"Wrote isoflop plot to {iso_path}")

    with fs.open(scaling_path, "w") as f:
        f.write(fig_scaling.to_html())
    print(f"Wrote scaling plot to {scaling_path}")

    # Optionally upload to WandB
    if config.upload_to_wandb and WANDB_AVAILABLE:
        wandb.login()
        run = wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            job_type="isoflop-analysis",
            name=config.wandb_run_name,
            resume="allow",
        )
        wandb.log(
            {
                "isoFLOP_plot": wandb.Plotly(fig_iso),
                "scaling_plot": wandb.Plotly(fig_scaling),
            }
        )
        run.finish()
        print("Uploaded plots to WandB")


def create_isoflop_analysis_step(
    name: str,
    training_runs: Sequence[ExecutorStep | InputName],
    metric_key: str = DEFAULT_METRIC_KEY,
    label_map: dict[str, str] | None = None,
    upload_to_wandb: bool = True,
    description: str | None = None,
) -> ExecutorStep:
    """Create an ExecutorStep for isoflop analysis.

    This uses the base create_analysis_step() with IsoFlopAnalysisConfig.

    Args:
        name: Name for this executor step
        training_runs: Training run ExecutorSteps (creates blocking dependencies)
        metric_key: Which metric to use for loss
        label_map: Optional mapping from experiment_name -> display label
        upload_to_wandb: Whether to upload plots to WandB
        description: Optional description

    Returns:
        ExecutorStep configured to run isoflop analysis
    """
    return create_analysis_step(
        name=name,
        training_runs=training_runs,
        analysis_fn=run_isoflop_analysis,
        config_class=IsoFlopAnalysisConfig,
        description=description or f"IsoFLOP analysis for {len(training_runs)} runs",
        metric_key=metric_key,
        label_map=label_map,
        upload_to_wandb=upload_to_wandb,
    )


# ---------------- Main (Legacy WandB-based) ----------------
def main(sources: list[tuple[str, str]]):
    """
    sources: list of (ENTITY/PROJECT, REGEX_FRAGMENT) with single fragments (no '|').
    We query with r'isoflop.*(<fragment>)' and infer dataset labels from displayName,
    falling back to the fragment so nothing gets dropped.
    """
    RUN_ID = "marin-scaling-suite-isoflop"
    wandb.login()
    run = wandb.init(
        entity="marin-community",
        project="marin-analysis",
        job_type="isoflop-analysis",
        id=RUN_ID,
        resume="allow",
        name="isoflop-analysis",
    )

    api = wandb.Api()
    source_runs = []
    for entity_project, fragment in sources:
        if "/" not in entity_project:
            raise ValueError(f"Bad ENTITY/PROJECT: {entity_project}")
        if not fragment:
            raise ValueError("Empty regex fragment")

        regex = rf"isoflop.*({fragment}).*"
        filters = {"displayName": {"$regex": regex}, "state": "finished"}
        runs = api.runs(entity_project.strip(), filters=filters)
        source_runs.append((runs, fragment.strip()))

    df = df_from_sources(source_runs)
    fig_iso, fig_scaling = iso_plot_with_minima_df(df)

    wandb.log(
        {
            "isoFLOP_plot": wandb.Plotly(fig_iso),
            "scaling_plot": wandb.Plotly(fig_scaling),
        }
    )
    run.finish()


if __name__ == "__main__":
    SOURCES = [
        ("marin-community/marin", "nemo-wider-depth-adapt"),
        ("marin-community/marin", "comma"),
        ("stanford-mercury/marin", "dclm-default"),
    ]
    main(SOURCES)
