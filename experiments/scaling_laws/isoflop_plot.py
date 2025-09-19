import wandb
import plotly.io as pio
import jax.numpy as jnp
from jaxopt import ScipyMinimize
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

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
# Allow multiple metrics; each becomes a panel in the plots
DEFAULT_METRIC_KEYS = [
    "eval/paloma/c4_en/bpb",
    "lm_eval/mmlu_sl_verb_5_shot/choice_logprob",
    "lm_eval/mmlu_sl_verb_5_shot/acc_norm",
    "eval/paloma/bpb",
]
# DEFAULT_METRIC_KEY = "lm_eval/mmlu_sl_verb_5_shot/choice_logprob"
SEQ_LEN = 4096

_MIN_MARKER = dict(symbol="diamond", size=10, color="#000000")
_SCALE_MARKER = dict(symbol="circle", size=9, color=PALETTE[0])
_SCALE_LINE = dict(dash="dot", width=2, color=PALETTE[0])

REQUIRED_TAGS = {"steps", "B", "FLOPs", "d", "L"}
CANON_LABELS = ["nemo", "comma", "dclm"]  # canonical dataset names we detect in displayName


# ---------------- Helpers ----------------
def _tags_to_dict(tags):
    return {k: v for k, v in (t.split("=", 1) for t in tags if "=" in t)}


def _parse_flops_from_name(name: str) -> float | None:
    """Parse FLOPs value from run name like 'isoflop-1e+19-...'.

    Returns float on success, None otherwise.
    """
    m = re.search(r"isoflop-([0-9.+\-eE]+)", name)
    if not m:
        return None
    try:
        # Remove trailing dash if present
        flops_str = m.group(1).rstrip("-")
        return float(flops_str)
    except ValueError:
        return None


def df_from_sources(source_runs: list[tuple[list, str]], metric_key: str = DEFAULT_METRIC_KEYS[0]) -> pd.DataFrame:
    """Build a dataframe from [(runs, fragment), ...] and compute a 'label' per row."""
    records = []
    for runs, fragment in source_runs:
        for run in runs:
            tags = _tags_to_dict(run.tags)
            # Allow fallback when FLOPs tag is missing: require other tags only
            required_no_flops = REQUIRED_TAGS - {"FLOPs"}
            if not required_no_flops.issubset(tags):
                continue

            steps = float(tags["steps"])
            batch = float(tags["B"])
            name = run.displayName
            flops_tag = tags.get("FLOPs")
            # breakpoint()
            if flops_tag is not None:
                flops = float(flops_tag)
            else:
                flops_parsed = _parse_flops_from_name(name)
                if flops_parsed is None:
                    continue
                flops = float(flops_parsed)
            if flops < 1e18:
                continue

            tokens = steps * batch * SEQ_LEN
            loss = run.summaryMetrics.get(metric_key)
            # print(loss)
            if loss is None:
                continue

            if "choice_logprob" in metric_key:
                loss = -loss

            params = run.summaryMetrics.get("parameter_count")

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
        width=1400,
        height=1400,
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
        width=1400,
        height=800,
    )

    return fig_iso, fig_scale


# ---------------- Multi-metric helpers ----------------
def multi_metric_subplots_from_sources(
    source_runs: list[tuple[list, str]], metric_keys: list[str] = DEFAULT_METRIC_KEYS
):
    """
    Build two multi-panel figures (ISO and SCALING), one column per metric key.
    """
    if not metric_keys:
        metric_keys = DEFAULT_METRIC_KEYS

    num_cols = len(metric_keys)
    fig_iso_all = make_subplots(rows=1, cols=num_cols, subplot_titles=metric_keys)
    fig_scaling_all = make_subplots(rows=1, cols=num_cols, subplot_titles=metric_keys)

    for i, mkey in enumerate(metric_keys, start=1):
        df_m = df_from_sources(source_runs, metric_key=mkey)
        fig_iso_m, fig_scaling_m = iso_plot_with_minima_df(df_m)

        # Add traces to ISO panel i
        for tr in fig_iso_m.data:
            fig_iso_all.add_trace(tr, row=1, col=i)
        # Log-scale axes and titles per panel
        fig_iso_all.update_xaxes(type="log", title_text="Tokens (log)", row=1, col=i)
        fig_iso_all.update_yaxes(
            title_text=("Loss" if "choice_logprob" not in mkey else "-Choice LogProb"), row=1, col=i
        )

        # Add traces to SCALING panel i
        for tr in fig_scaling_m.data:
            fig_scaling_all.add_trace(tr, row=1, col=i)
        fig_scaling_all.update_xaxes(type="log", title_text="Compute C (FLOPs, log)", row=1, col=i)
        fig_scaling_all.update_yaxes(type="log", title_text="Optimal tokens N* (log)", row=1, col=i)

    fig_iso_all.update_layout(
        template="plotly_white", title="Marin IsoFLOP Suite (multi-metric)", width=1400 * num_cols, height=1400
    )
    fig_scaling_all.update_layout(
        template="plotly_white", title="Scaling fits per dataset (multi-metric)", width=1400 * num_cols, height=800
    )

    return fig_iso_all, fig_scaling_all


# ---------------- Main ----------------
def main(sources: list[tuple[str, str]]):
    """
    sources: list of (ENTITY/PROJECT, REGEX_FRAGMENT) with single fragments (no '|').
    We query with r'isoflop.*(<fragment>)' and infer dataset labels from displayName,
    falling back to the fragment so nothing gets dropped.
    """
    wandb.login()
    run = wandb.init(
        entity="marin-community",
        project="marin-analysis",
        job_type="isoflop-analysis",
        resume="never",
        name="isoflop-analysis",
    )

    api = wandb.Api()
    source_runs = []
    for entity_project, fragment in sources:
        if "/" not in entity_project:
            raise ValueError(f"Bad ENTITY/PROJECT: {entity_project}")
        if not fragment:
            raise ValueError("Empty regex fragment")

        # NOTE(chris): bit hacky, we don't wanna include the midtrain runs directly after
        # e.g. nemo-wider-depth-adapt should not include runs from nemo-wider-depth-adapt-mt-flan-80-20
        # regex = rf"isoflop.*({fragment})(?!-mt).*lmeval.*"
        # regex = rf"isoflop.*({fragment})(?!-mt).*"
        regex = rf"isoflop.*({fragment}).*"
        filters = {"displayName": {"$regex": regex}, "state": "finished"}
        runs = api.runs(entity_project.strip(), filters=filters)
        # breakpoint()
        fragment = fragment.strip()
        source_runs.append((runs, fragment))

        # Per-source plots
        # per_df = df_from_sources([(runs, fragment)])
        # fig_iso_src, fig_scaling_src = iso_plot_with_minima_df(per_df)
        # wandb.log(
        #     {
        #         f"isoFLOP_plot/{fragment}": wandb.Plotly(fig_iso_src),
        #         f"scaling_plot/{fragment}": wandb.Plotly(fig_scaling_src),
        #     }
        # )

    # Aggregated overlay plots across all sources, with one panel per metric key
    fig_iso_all, fig_scaling_all = multi_metric_subplots_from_sources(source_runs, DEFAULT_METRIC_KEYS)
    wandb.log(
        {
            "isoFLOP_plot/all": wandb.Plotly(fig_iso_all),
            "scaling_plot/all": wandb.Plotly(fig_scaling_all),
        }
    )
    run.finish()


if __name__ == "__main__":
    SOURCES = [
        # ("stanford-mercury/marin", "nemo-wider-depth-adapt"),
        ("marin-community/marin", "nemo-wider-depth-adapt"),
        # ("marin-community/marin", "comma"),
        # ("stanford-mercury/marin", "dclm-default"),
        # ("stanford-mercury/marin", "nemo-wider-depth-adapt-decay"),
        # ("stanford-mercury/marin", "nemo-wider-depth-adapt-mt-flan-80-20"),
        # ("stanford-mercury/marin", "nemo-wider-depth-adapt-mt-flan-70-30"),
        # ("stanford-mercury/marin", "dolmino-dclm-sweep"),
        # ("stanford-mercury/marin", "dclm-shard-1-1"),
        # ("stanford-mercury/marin", "dclm-wrap-qa-1024-v2"),
        # ("stanford-mercury/marin", "dclm-wrap-med-1024"),
        # ("stanford-mercury/marin", "dclm-nemo-qa-1024"),
        # ("stanford-mercury/marin", "dclm-mcq-1024"),
        # ("stanford-mercury/marin", "dclm-qa-1b-1024"),
    ]
    main(SOURCES)
