# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""X0224: stable rank, row-norm ratio and norm gains of every layer's matrices over a d512 recipe run.

Every saved checkpoint step is measured; each statistic is compared with its value at initialization:

- stable rank ``||W||_F^2 / sigma_max^2``, against a Gaussian matrix of the same shape, ``mn / (sqrt(m) + sqrt(n))^2``;
- row-norm ratio, the largest over the median L2 norm of the matrix's rows (the stored first axis) and of its columns,
  whichever is larger; a Gaussian matrix stays close to 1;
- norm gains (RMSNorm ``weight`` vectors), initialized to 1.

Stacked block arrays carry the layer on their first axis; expert arrays carry the expert on their second, and are
summarized per layer by the worst expert.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

STACKED = "/stacked/"
NON_MATRIX_PARTS = ("sconv", "rel_pos/proj")
MIN_MATRIX_DIM = 8

# A matrix is flagged when its stable rank falls below this fraction of the Gaussian reference, or below an absolute
# floor (rank collapse onto a few directions).
STABLE_RANK_FRACTION_FLOOR = 0.05
STABLE_RANK_FLOOR = 2.0
ROW_NORM_RATIO_CEILING = 10.0
# A gain vector is flagged when its mean |gain| leaves this band, or one gain is this many times the median.
GAIN_BAND = (0.2, 5.0)
GAIN_SPREAD_CEILING = 20.0

DEPTH_FIGURE = "stable_rank.png"
TRAINING_FIGURE = "over_training.png"
LOSS_KEY = "train/loss"


@dataclass(frozen=True)
class MatrixStats:
    name: str
    layer: int | None
    shape: tuple[int, ...]
    stable_rank: float
    reference: float
    row_norm_ratio: float

    @property
    def fraction(self) -> float:
        return self.stable_rank / self.reference

    @property
    def flags(self) -> list[str]:
        out = []
        if self.fraction < STABLE_RANK_FRACTION_FLOOR or self.stable_rank < STABLE_RANK_FLOOR:
            out.append("stable-rank collapse")
        if self.row_norm_ratio > ROW_NORM_RATIO_CEILING:
            out.append("row-norm outlier")
        return out


@dataclass(frozen=True)
class GainStats:
    name: str
    layer: int | None
    size: int
    mean_abs: float
    min: float
    max: float
    spread: float

    @property
    def flags(self) -> list[str]:
        out = []
        if not GAIN_BAND[0] <= self.mean_abs <= GAIN_BAND[1]:
            out.append("gain scale drift")
        if self.spread > GAIN_SPREAD_CEILING:
            out.append("gain outlier")
        return out


def short_name(path: str) -> str:
    return path.removeprefix("params/").replace(STACKED, "/")


def gaussian_stable_rank(m: int, n: int) -> float:
    return m * n / (np.sqrt(m) + np.sqrt(n)) ** 2


def norm_ratio(norms: np.ndarray) -> float:
    median = float(np.median(norms))
    return float(norms.max() / median) if median > 0 else float("inf")


def matrix_stats(name: str, layer: int | None, w: np.ndarray) -> MatrixStats | None:
    """Statistics of a stack of same-shape matrices ``(..., m, n)``; the worst one when there are several."""
    w = w.astype(np.float64)
    m, n = w.shape[-2:]
    w = w.reshape(-1, m, n)
    singular = np.linalg.svd(w, compute_uv=False)
    live = singular[:, 0] > 0
    if not live.any():
        return None
    singular = singular[live]
    w = w[live]
    stable = (singular**2).sum(axis=1) / singular[:, 0] ** 2
    ratios = [max(norm_ratio(np.linalg.norm(x, axis=1)), norm_ratio(np.linalg.norm(x, axis=0))) for x in w]
    return MatrixStats(name, layer, (m, n), float(stable.min()), gaussian_stable_rank(m, n), float(max(ratios)))


def gain_stats(name: str, layer: int | None, g: np.ndarray) -> GainStats:
    g = g.astype(np.float64).ravel()
    magnitude = np.abs(g)
    median = float(np.median(magnitude))
    spread = float(magnitude.max() / median) if median > 0 else float("inf")
    return GainStats(name, layer, g.size, float(magnitude.mean()), float(g.min()), float(g.max()), spread)


def is_gain(path: str) -> bool:
    return path.endswith("/weight") and not any(part in path for part in NON_MATRIX_PARTS)


def is_matrix(path: str, per_layer: np.ndarray) -> bool:
    if any(part in path for part in NON_MATRIX_PARTS) or per_layer.ndim < 2:
        return False
    return min(per_layer.shape[-2:]) >= MIN_MATRIX_DIM


def layer_slices(path: str, array: np.ndarray) -> list[tuple[int | None, np.ndarray]]:
    if STACKED in path:
        return [(i, array[i]) for i in range(array.shape[0])]
    return [(None, array)]


def collect(params: dict[str, np.ndarray]) -> tuple[list[MatrixStats], list[GainStats]]:
    matrices: list[MatrixStats] = []
    gains: list[GainStats] = []
    for path, array in sorted(params.items()):
        name = short_name(path)
        for layer, part in layer_slices(path, array):
            if is_gain(path):
                gains.append(gain_stats(name, layer, part))
            elif is_matrix(path, part):
                stats = matrix_stats(name, layer, part)
                if stats is not None:
                    matrices.append(stats)
    return matrices, gains


def layer_label(layer: int | None) -> str:
    return "-" if layer is None else str(layer)


def matrix_table(matrices: list[MatrixStats]) -> list[str]:
    rows = [
        "| Matrix | Layer | Shape | Stable rank | Gaussian ref | Fraction | Row-norm ratio | Flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in matrices:
        rows.append(
            f"| `{s.name}` | {layer_label(s.layer)} | {s.shape[0]}x{s.shape[1]} | {s.stable_rank:.2f} | "
            f"{s.reference:.1f} | {s.fraction:.3f} | {s.row_norm_ratio:.2f} | {', '.join(s.flags) or '-'} |"
        )
    return rows


def gain_table(gains: list[GainStats]) -> list[str]:
    rows = [
        "| Gain | Layer | Size | Mean abs | Min | Max | Max/median abs | Flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for g in gains:
        rows.append(
            f"| `{g.name}` | {layer_label(g.layer)} | {g.size} | {g.mean_abs:.3f} | {g.min:.3f} | {g.max:.3f} | "
            f"{g.spread:.2f} | {', '.join(g.flags) or '-'} |"
        )
    return rows


def depth_trends(matrices: list[MatrixStats]) -> list[str]:
    """Per stacked matrix kind: the stable-rank fraction at the first and last layer, and the layer of its minimum."""
    rows = ["| Matrix | First layer | Last layer | Lowest (layer) |", "|---|---|---|---|"]
    by_name: dict[str, list[MatrixStats]] = {}
    for s in matrices:
        if s.layer is not None:
            by_name.setdefault(s.name, []).append(s)
    for name, stack in sorted(by_name.items()):
        lowest = min(stack, key=lambda s: s.fraction)
        rows.append(
            f"| `{name}` | {stack[0].fraction:.3f} | {stack[-1].fraction:.3f} | "
            f"{lowest.fraction:.3f} ({lowest.layer}) |"
        )
    return rows


@dataclass(frozen=True)
class StepStats:
    step: int
    matrices: list[MatrixStats]
    gains: list[GainStats]

    @property
    def flagged_matrices(self) -> list[MatrixStats]:
        return [s for s in self.matrices if s.flags]

    @property
    def flagged_gains(self) -> list[GainStats]:
        return [g for g in self.gains if g.flags]


def by_layer_stack(matrices: list[MatrixStats]) -> dict[str, list[MatrixStats]]:
    by_name: dict[str, list[MatrixStats]] = {}
    for s in matrices:
        if s.layer is not None:
            by_name.setdefault(s.name, []).append(s)
    return dict(sorted(by_name.items()))


def plot_depth(matrices: list[MatrixStats], step: int, out: Path) -> None:
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    for name, stack in by_layer_stack(matrices).items():
        ax.plot([s.layer for s in stack], [s.fraction for s in stack], marker="o", markersize=3, label=name)
    ax.axhline(STABLE_RANK_FRACTION_FLOOR, color="black", linestyle="--", linewidth=1, label="collapse threshold")
    ax.set_yscale("log")
    ax.set_xlabel("layer (within its stack)")
    ax.set_ylabel("stable rank / Gaussian reference")
    ax.set_title(f"Stable rank relative to initialization, per layer, at step {step}")
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def plot_training(steps: list[StepStats], out: Path) -> None:
    """Per matrix kind, its lowest stable-rank fraction and largest row-norm ratio over layers, at every step."""
    names = sorted({s.name for stats in steps for s in stats.matrices})
    fig = Figure(figsize=(14, 6))
    rank_ax, ratio_ax = fig.subplots(1, 2)
    for name in names:
        xs, fractions, ratios = [], [], []
        for stats in steps:
            kind = [s for s in stats.matrices if s.name == name]
            if kind:
                xs.append(stats.step)
                fractions.append(min(s.fraction for s in kind))
                ratios.append(max(s.row_norm_ratio for s in kind))
        rank_ax.plot(xs, fractions, marker="o", markersize=3, label=name)
        ratio_ax.plot(xs, ratios, marker="o", markersize=3, label=name)
    rank_ax.axhline(STABLE_RANK_FRACTION_FLOOR, color="black", linestyle="--", linewidth=1)
    ratio_ax.axhline(ROW_NORM_RATIO_CEILING, color="black", linestyle="--", linewidth=1)
    rank_ax.set_yscale("log")
    ratio_ax.set_yscale("log")
    rank_ax.set_ylabel("lowest stable rank / Gaussian reference over layers")
    ratio_ax.set_ylabel("largest row-norm ratio over layers")
    for ax in (rank_ax, ratio_ax):
        ax.set_xlabel("step")
    rank_ax.legend(fontsize=6, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def logged_loss(history: list[dict]) -> dict[int, float]:
    return {int(row["_step"]): float(row[LOSS_KEY]) for row in history if row.get(LOSS_KEY) is not None}


def nearest_loss(losses: dict[int, float], step: int) -> str:
    if not losses:
        return "-"
    nearest = min(losses, key=lambda s: abs(s - step))
    return f"{losses[nearest]:.3f}"


def training_table(steps: list[StepStats], losses: dict[int, float]) -> list[str]:
    rows = [
        "| Step | Loss | Flagged matrices | Flagged gains | Lowest stable-rank fraction | Largest row-norm ratio "
        "| Largest mean abs gain |",
        "|---|---|---|---|---|---|---|",
    ]
    for stats in steps:
        lowest = min(stats.matrices, key=lambda s: s.fraction)
        widest = max(stats.matrices, key=lambda s: s.row_norm_ratio)
        largest = max(stats.gains, key=lambda g: g.mean_abs)
        rows.append(
            f"| {stats.step} | {nearest_loss(losses, stats.step)} | {len(stats.flagged_matrices)} | "
            f"{len(stats.flagged_gains)} | {lowest.fraction:.3f} (`{lowest.name}` {layer_label(lowest.layer)}) | "
            f"{widest.row_norm_ratio:.2f} (`{widest.name}` {layer_label(widest.layer)}) | "
            f"{largest.mean_abs:.3f} (`{largest.name}` {layer_label(largest.layer)}) |"
        )
    return rows


def first_flagged(steps: list[StepStats]) -> list[str]:
    """Every matrix or gain flagged at some step: the first step it was flagged, and its flags then."""
    first: dict[tuple[str, int | None], tuple[int, list[str]]] = {}
    for stats in steps:
        for item in [*stats.flagged_matrices, *stats.flagged_gains]:
            first.setdefault((item.name, item.layer), (stats.step, item.flags))
    if not first:
        return ["No matrix or gain vector flagged at any saved step."]
    rows = ["| Array | Layer | First flagged at step | Flags then |", "|---|---|---|---|"]
    for (name, layer), (step, flags) in sorted(first.items(), key=lambda item: (item[1][0], item[0][0])):
        rows.append(f"| `{name}` | {layer_label(layer)} | {step} | {', '.join(flags)} |")
    return rows


def analyze(checkpoints: Mapping[int, dict[str, np.ndarray]], history: list[dict], out_dir: Path) -> None:
    steps = [StepStats(step, *collect(checkpoints[step])) for step in checkpoints]
    final = steps[-1]
    plot_depth(final.matrices, final.step, out_dir / DEPTH_FIGURE)
    plot_training(steps, out_dir / TRAINING_FIGURE)
    lowest = min(final.matrices, key=lambda s: s.fraction)
    widest = max(final.matrices, key=lambda s: s.row_norm_ratio)
    largest_gain = max(final.gains, key=lambda g: g.mean_abs)
    lines = [
        "# X0224: training-dynamics statistics of the d512 recipe",
        "",
        f"Statistics of every saved checkpoint ({len(steps)} steps, {steps[0].step} to {final.step}), each against its",
        "value at initialization: stable rank against a Gaussian matrix of the same shape, row-norm ratio against 1,",
        "and norm gains against 1.",
        "",
        "## Answer",
        "",
        f"- {len(final.matrices)} matrices and {len(final.gains)} gain vectors measured at each step.",
        f"- At step {final.step}: {len(final.flagged_matrices)} matrices flagged (stable rank below "
        f"{STABLE_RANK_FRACTION_FLOOR:g} of the Gaussian reference or below {STABLE_RANK_FLOOR:g}, or row-norm ratio "
        f"above {ROW_NORM_RATIO_CEILING:g}); {len(final.flagged_gains)} gain vectors flagged (mean abs gain outside "
        f"{GAIN_BAND[0]:g} to {GAIN_BAND[1]:g}, or one gain above {GAIN_SPREAD_CEILING:g} times the median).",
        f"- Lowest stable-rank fraction at step {final.step}: `{lowest.name}` layer {layer_label(lowest.layer)}, "
        f"{lowest.fraction:.3f} ({lowest.stable_rank:.2f} of {lowest.reference:.1f}).",
        f"- Largest row-norm ratio at step {final.step}: `{widest.name}` layer {layer_label(widest.layer)}, "
        f"{widest.row_norm_ratio:.2f}.",
        f"- Largest mean abs gain at step {final.step}: `{largest_gain.name}` layer "
        f"{layer_label(largest_gain.layer)}, {largest_gain.mean_abs:.3f}.",
        "",
        "## Over training",
        "",
        f"![lowest stable rank and largest row-norm ratio per matrix kind over training]({TRAINING_FIGURE})",
        "",
        *training_table(steps, logged_loss(history)),
        "",
        "### First step each array was flagged",
        "",
        *first_flagged(steps),
        "",
        f"## Stable rank across depth at step {final.step}",
        "",
        f"![stable rank]({DEPTH_FIGURE})",
        "",
        *depth_trends(final.matrices),
        "",
        f"## Flagged at step {final.step}",
        "",
        *(matrix_table(final.flagged_matrices) if final.flagged_matrices else ["No matrix flagged."]),
        "",
        *(gain_table(final.flagged_gains) if final.flagged_gains else ["No gain vector flagged."]),
        "",
        f"## All matrices at step {final.step}",
        "",
        "Expert arrays show the worst expert of each layer.",
        "",
        *matrix_table(final.matrices),
        "",
        f"## All gains at step {final.step}",
        "",
        *gain_table(final.gains),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))
