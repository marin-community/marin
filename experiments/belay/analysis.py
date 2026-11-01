# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The analysis report of one Belay trial: ``experiments/belay/analysis/<X####>/<trial>.md`` plus its PNGs, on the
registry branch, written by ``run-finished`` in the same commit as the results.

Every report has the same sections, plots and styling:

1. a header with the registry id, title, track and verdict, plus the commit, approver and idea link;
2. the final eval table, control against variant;
3. train loss against step, raw and smoothed;
4. Δ train loss (variant - control) against step;
5. step time and tokens/s;
6. optional sections, only for metrics the runs logged (MoE routing, gradient norm).

W&B is read anonymously through its GraphQL API (``marin_moe`` is public), so no key is needed.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import requests
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from experiments.belay.config import CONFIG

GRAPHQL = "https://api.wandb.ai/graphql"
ANALYSIS_DIR = "experiments/belay/analysis"
LOSS = "train/loss"
STEP_TIME = "throughput/duration"
TOKENS = "throughput/tokens_per_second"
OPTIONAL = {
    "train/router/capacity_overflow_rate_mean": "MoE routing: capacity drops (mean over layers)",
    "train/router/bias_max": "MoE routing: router bias max",
    "grad/norm/total": "Gradient norm",
}
FIGSIZE = (7.0, 3.6)
CONTROL_COLOR, VARIANT_COLOR = "#5f6b64", "#c2410c"
SMOOTH = 10  # points of W&B's sampled history (about 10 steps apart for a 3,000-step run)
SKIP_STEPS = 5  # compile and warm-up steps, left out of the step-time medians
_QUERY = """query($e:String!,$p:String!,$r:String!,$s:[JSONString!]!){project(name:$p,entityName:$e){
run(name:$r){historyKeys sampledHistory(specs:$s)}}}"""


@dataclass(frozen=True)
class Series:
    """One run's logged history: ``{key: [(step, value), ...]}``."""

    run: str
    history: dict[str, list[tuple[int, float]]]


@dataclass(frozen=True)
class Header:
    registry_id: str
    trial: str
    title: str
    track: str
    verdict: str
    commit: str
    approver: str | None
    idea_url: str | None
    rule: str


def fetch_series(run: str) -> Series:
    """The run's loss, throughput and optional metrics from W&B, read anonymously."""
    entity, project = CONFIG.wandb_project.split("/")
    variables = {"e": entity, "p": project, "r": run, "s": []}
    keys_reply = requests.post(GRAPHQL, json={"query": _QUERY, "variables": variables}, timeout=60)
    available = set(keys_reply.json()["data"]["project"]["run"]["historyKeys"]["keys"])
    keys = [k for k in (LOSS, STEP_TIME, TOKENS, *OPTIONAL) if k in available]
    spec = json.dumps({"keys": ["_step", *keys], "samples": 100000})
    reply = requests.post(GRAPHQL, json={"query": _QUERY, "variables": {**variables, "s": [spec]}}, timeout=120)
    rows = reply.json()["data"]["project"]["run"]["sampledHistory"][0]
    return series_from_rows(run, rows)


def series_from_rows(run: str, rows: list[dict]) -> Series:
    history: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        for key, value in row.items():
            if key != "_step" and isinstance(value, int | float):
                history.setdefault(key, []).append((int(row["_step"]), float(value)))
    return Series(run, {k: sorted(v) for k, v in history.items()})


def smoothed(values: list[float], window: int = SMOOTH) -> list[float]:
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def _figure(title: str) -> tuple[Figure, Axes]:
    fig = Figure(figsize=FIGSIZE, dpi=110)
    ax = fig.subplots()
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.3)
    return fig, ax


def plot_loss(control: Series, variant: Series, path: Path) -> None:
    fig, ax = _figure("Train loss")
    for series, color, label in ((control, CONTROL_COLOR, "control"), (variant, VARIANT_COLOR, "variant")):
        steps, values = zip(*series.history[LOSS], strict=True)
        ax.plot(steps, values, color=color, alpha=0.2, linewidth=0.6)
        ax.plot(steps, smoothed(list(values)), color=color, linewidth=1.4, label=f"{label} ({SMOOTH}-point mean)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_ylim(top=min(6.0, ax.get_ylim()[1]))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def plot_delta(control: Series, variant: Series, path: Path) -> None:
    by_step = dict(control.history[LOSS])
    pairs = [(s, v - by_step[s]) for s, v in variant.history[LOSS] if s in by_step]
    steps, deltas = zip(*pairs, strict=True)
    fig, ax = _figure("Δ train loss (variant - control)")
    ax.plot(steps, deltas, color=VARIANT_COLOR, alpha=0.2, linewidth=0.6)
    ax.plot(steps, smoothed(list(deltas)), color=VARIANT_COLOR, linewidth=1.4, label=f"{SMOOTH}-point mean")
    ax.axhline(0, color=CONTROL_COLOR, linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("Δ loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def _median(series: Series, key: str) -> float | None:
    values = [v for s, v in series.history.get(key, []) if s >= SKIP_STEPS]
    return median(values) if values else None


def _signed(delta: float | None) -> str:
    return "" if delta is None else f"{delta:+.4f}"


def _fmt(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_report(
    out_dir: Path,
    header: Header,
    control: Series,
    variant: Series,
    final: dict[str, tuple[float, float | None, float | None]],
) -> Path:
    """Write ``<trial>.md`` and its PNGs into ``out_dir``; ``final`` maps a run name to (loss, Δ, BPB)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_png, delta_png = f"{header.trial}_loss.png", f"{header.trial}_delta.png"
    plot_loss(control, variant, out_dir / loss_png)
    plot_delta(control, variant, out_dir / delta_png)
    idea = f"[#9514 comment]({header.idea_url})" if header.idea_url else "maintainer proposal"
    approver = f" · approved by @{header.approver}" if header.approver else ""
    eval_rows = "\n".join(
        f"| {name} | {loss:.4f} | {_signed(delta)} | {_fmt(bpb, 4)} |" for name, (loss, delta, bpb) in final.items()
    )
    speed_rows = "\n".join(
        f"| {s.run} | {_fmt(_median(s, STEP_TIME), 4)} | {_fmt(_median(s, TOKENS), 0)} |" for s in (control, variant)
    )
    sections = [
        f"# {header.registry_id} · {header.title} · {header.track} · {header.verdict}",
        f"Trial {header.trial} · commit `{header.commit[:10]}` · {idea}{approver}",
        f"## Final eval\n\n| Run | Paloma macro loss | Δ vs control | Paloma macro BPB |\n|---|---|---|---|\n"
        f"{eval_rows}\n\n{header.rule}",
        f"## Train loss\n\n![train loss]({loss_png})",
        f"## Δ train loss\n\n![delta loss]({delta_png})",
        f"## Speed (median after step {SKIP_STEPS})\n\n| Run | s/step | tokens/s |\n|---|---|---|\n{speed_rows}",
    ]
    for key, label in OPTIONAL.items():
        if key in control.history and key in variant.history:
            values = " | ".join(_fmt(_median(s, key), 6) for s in (control, variant))
            sections.append(f"## {label}\n\n| Control | Variant |\n|---|---|\n| {values} |")
    path = out_dir / f"{header.trial}.md"
    path.write_text("\n\n".join(sections) + "\n")
    return path


def report_dir(registry_checkout: Path, registry_id: str) -> Path:
    return registry_checkout / ANALYSIS_DIR / registry_id


def report_url(registry_id: str, trial: str) -> str:
    return (
        f"https://github.com/{CONFIG.repository}/blob/{CONFIG.registry_branch}/{ANALYSIS_DIR}/{registry_id}/{trial}.md"
    )
