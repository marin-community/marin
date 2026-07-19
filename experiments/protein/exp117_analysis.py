"""exp117 sweep analysis — pairwise final-loss difference matrices.

Pulls *completed* W&B runs for an exp117 adaptive sweep and renders, per resource
rung, one square matrix for each search axis (batch_size, learning_rate,
weight_decay). A cell (row=value_i, col=value_j) of an axis matrix holds

    mean over matched holdings of [ final_loss(value_j) - final_loss(value_i) ]

where a "matched holding" is a pair of completed runs that are IDENTICAL on every
other axis (the two axes not being compared, plus epochs) and differ only in the
compared axis. Blue = column config is better (lower loss); red = column worse.

Two figures are produced, sharing one color scale so cells are directly comparable:

  * pairwise_loss_diff_marginal.png — every cell backed by >=1 matched holding is
    filled with the mean difference across those holdings; the holding count `n` is
    annotated. This is the marginal effect of the axis, averaged over the rest of
    the grid. Denser, but a cell can mix several otherwise-identical pairs.

  * pairwise_loss_diff_joint.png — only cells backed by exactly one matched holding
    are filled: a single pair of configs sharing one joint setting of every other
    axis and differing only in the paired hyperparameter values. No averaging. Much
    sparser, but every cell is an unambiguous head-to-head comparison.

Matrices are expected to be sparse (the isolated figure especially). Currently
restricted to the 8-epoch rung (`ALLOWED_EPOCHS`), the only rung complete enough to
be informative — the figure still lays out rows as rungs so later rungs drop in
without a rewrite.

Run identity and completion are read from structured W&B **tags**, never parsed from
the run display name. Each run tags `epochs`, `lr`, `wd`, `global_batch`, `steps`
(expected total), and `sweep_subversion` (the "sNN" selector).

Guard on the "should not be possible" case: a single full trial identity
(epochs, lr, wd, batch) with >1 completed W&B run raises immediately — a completed
trial is unique, so duplicates mean a data problem (e.g. two regions both finished
the same point).

Run:
    cd <repo> && set -a; source ~/marin.env; set +a
    uv run --with 'wandb,matplotlib,numpy' python -m experiments.protein.exp117_analysis
    # optional: SWEEP_VERSION=s02 (default)

Output: experiments/protein/exp117_results/<version>/*.png
(the exp117_results/ tree is git-ignored via experiments/protein/.gitignore).
"""

from __future__ import annotations

import datetime as _dt
import itertools
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

# --------------------------------------------------------------------------------------
# Sweep specification. One entry per sweep subversion so the script serves s02, s03, ...
# Identity is read from W&B tags; `subversion` is the `sweep_subversion` tag value.
# --------------------------------------------------------------------------------------

OBJECTIVE_KEY = "eval/tokenized/contacts-v1-val/loss"
COMPLETE_FRACTION = 0.999  # _step / tags['steps'] at/above which a run counts as finished
ALLOWED_EPOCHS = {8}  # rungs to render; only 8ep is complete enough for now

# The three search axes, in the column order requested (BS, LR, WD). Each maps to the
# W&B tag key that carries that axis's value.
AXES = ("batch_size", "learning_rate", "weight_decay")
AXIS_TAG = {"batch_size": "global_batch", "learning_rate": "lr", "weight_decay": "wd"}
AXIS_LABEL = {
    "batch_size": "batch size",
    "learning_rate": "learning rate",
    "weight_decay": "weight decay",
}


@dataclass(frozen=True)
class HoldingMode:
    """How a cell aggregates the matched holdings behind it."""

    key: str
    filename: str
    blurb: str  # one-line description for the figure subtitle
    min_holdings: int  # fill a cell only when its holding count is >= this
    max_holdings: int  # ...and <= this (use a large sentinel for "no upper bound")
    annotate_count: bool  # annotate the holding count `n` in each cell


MODES = (
    HoldingMode(
        key="marginal",
        filename="pairwise_loss_diff_marginal.png",
        blurb="marginal mean over all otherwise-identical pairs (n = holding count)",
        min_holdings=1,
        max_holdings=10**9,
        annotate_count=True,
    ),
    HoldingMode(
        key="joint",
        filename="pairwise_loss_diff_joint.png",
        blurb="joint — only cells with a single otherwise-identical pair (no averaging)",
        min_holdings=1,
        max_holdings=1,
        annotate_count=False,
    ),
)


@dataclass(frozen=True)
class SweepSpec:
    version: str  # e.g. "s02"
    entity: str
    project: str
    membership_tag: str  # tag scoping the experiment (e.g. "exp117")
    subversion: str  # `sweep_subversion` tag value selecting this subversion


SWEEPS = {
    "s02": SweepSpec(
        version="s02",
        entity=os.environ.get("WANDB_ENTITY", "eric-czech"),
        # WANDB_PROJECT env is stale for these runs; they live in <entity>/marin.
        project="marin",
        membership_tag="exp117",
        subversion="2",
    ),
}

RESULTS_DIR = Path(__file__).resolve().parent / "exp117_results"


# --------------------------------------------------------------------------------------
# Run identity — read from structured W&B tags, never parsed from the run name.
# Tag values (e.g. "0.00031623", "0.2") are clean decimal strings; they double as stable
# categorical keys and parse to float only for axis ordering / labels.
# --------------------------------------------------------------------------------------


def tag_map(tags) -> dict[str, str]:
    """Split `key=value` W&B tags into a dict; bare tags are dropped."""
    return dict(tag.split("=", 1) for tag in tags if "=" in tag)


@dataclass(frozen=True)
class Identity:
    epochs: int
    lr: str  # tag value verbatim, categorical key
    wd: str
    batch_size: int

    def axis_key(self, axis: str) -> str:
        """Categorical key for the compared axis."""
        return {
            "learning_rate": self.lr,
            "weight_decay": self.wd,
            "batch_size": str(self.batch_size),
        }[axis]

    def holding_key(self, axis: str) -> tuple:
        """Everything held EQUAL while `axis` varies (the other two axes + epochs)."""
        parts = {
            "learning_rate": ("wd", self.wd, "bs", self.batch_size),
            "weight_decay": ("lr", self.lr, "bs", self.batch_size),
            "batch_size": ("lr", self.lr, "wd", self.wd),
        }[axis]
        return (self.epochs,) + parts


# --------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CompletedRun:
    name: str
    identity: Identity
    objective: float


def fetch_completed_runs(spec: SweepSpec) -> list[CompletedRun]:
    """Return completed runs (final step reached, objective present) for the sweep.

    Membership, identity, and expected step count all come from W&B tags. Raises if a
    single trial identity resolves to more than one completed run — a completed trial
    is unique, so this should not happen.
    """
    import wandb

    api = wandb.Api()
    runs = api.runs(
        f"{spec.entity}/{spec.project}",
        filters={"tags": spec.membership_tag},
        per_page=200,
    )

    by_identity: dict[Identity, list[CompletedRun]] = defaultdict(list)
    for run in runs:
        tags = tag_map(run.tags)
        if tags.get("sweep_subversion") != spec.subversion:
            continue
        epochs = int(tags["epochs"])
        if epochs not in ALLOWED_EPOCHS:
            continue
        summary = run.summary or {}
        step = summary.get("_step")
        objective = summary.get(OBJECTIVE_KEY)
        if not isinstance(step, (int, float)) or not isinstance(objective, (int, float)):
            continue
        if step / int(tags["steps"]) < COMPLETE_FRACTION:  # not at final step -> not done
            continue
        ident = Identity(
            epochs=epochs,
            lr=tags["lr"],
            wd=tags["wd"],
            batch_size=int(tags["global_batch"]),
        )
        by_identity[ident].append(
            CompletedRun(name=run.name, identity=ident, objective=float(objective))
        )

    completed: list[CompletedRun] = []
    for ident, group in by_identity.items():
        if len(group) > 1:
            names = ", ".join(f"{r.name} (loss={r.objective:.4f})" for r in group)
            raise ValueError(
                "Multiple completed W&B runs for one trial identity "
                f"{ident} — should be unique. Runs: {names}"
            )
        completed.append(group[0])
    return completed


# --------------------------------------------------------------------------------------
# Matrix construction
# --------------------------------------------------------------------------------------


@dataclass
class AxisMatrix:
    axis: str
    epochs: int
    values: list[str]  # categorical axis keys, ordered by numeric value
    labels: list[str]  # display labels
    diff: np.ndarray  # [n, n] mean( loss(col) - loss(row) ), NaN where unmatched
    holdings: np.ndarray  # [n, n] int count of matched holdings per cell


def _order_axis_values(axis: str, keys: set[str]) -> tuple[list[str], list[str]]:
    """Order categorical keys by numeric value; produce display labels."""
    if axis == "batch_size":
        ordered = sorted(keys, key=int)
        return ordered, list(ordered)
    ordered = sorted(keys, key=float)
    if axis == "learning_rate":
        return ordered, [f"{float(k):.2e}" for k in ordered]
    return ordered, [f"{float(k):g}" for k in ordered]  # weight_decay


def build_axis_matrix(axis: str, epochs: int, runs: list[CompletedRun]) -> AxisMatrix:
    rung_runs = [r for r in runs if r.identity.epochs == epochs]

    # Group completed losses by (holding, axis-value); each is unique by the fetch guard.
    losses: dict[tuple, dict[str, float]] = defaultdict(dict)
    axis_keys: set[str] = set()
    for r in rung_runs:
        key = r.identity.axis_key(axis)
        axis_keys.add(key)
        losses[r.identity.holding_key(axis)][key] = r.objective

    values, labels = _order_axis_values(axis, axis_keys)
    index = {v: i for i, v in enumerate(values)}
    n = len(values)

    # Accumulate differences per cell across every holding that has both endpoints.
    sums = np.zeros((n, n))
    counts = np.zeros((n, n), dtype=int)
    for val_to_loss in losses.values():
        present = [v for v in values if v in val_to_loss]
        for a, b in itertools.permutations(present, 2):  # ordered (row=a, col=b)
            i, j = index[a], index[b]
            sums[i, j] += val_to_loss[b] - val_to_loss[a]
            counts[i, j] += 1

    diff = np.divide(sums, counts, out=np.full((n, n), np.nan), where=counts > 0)
    np.fill_diagonal(diff, np.nan)  # self-difference is trivially 0; keep it neutral
    return AxisMatrix(axis, epochs, values, labels, diff, counts)


def masked_diff(m: AxisMatrix, mode: HoldingMode) -> np.ndarray:
    """Diff array with cells outside the mode's holding-count window blanked to NaN."""
    keep = (m.holdings >= mode.min_holdings) & (m.holdings <= mode.max_holdings)
    return np.where(keep, m.diff, np.nan)


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _draw_matrix(ax, m: AxisMatrix, display: np.ndarray, mode: HoldingMode, norm, cmap):
    n = len(m.values)
    ax.imshow(display, cmap=cmap, norm=norm, aspect="equal")
    ax.set_xticks(range(n), m.labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), m.labels, fontsize=8)
    ax.set_xlabel(f"{AXIS_LABEL[m.axis]}  (column)", fontsize=9)
    ax.set_ylabel(f"{AXIS_LABEL[m.axis]}  (row)", fontsize=9)
    ax.set_title(f"{AXIS_LABEL[m.axis]}  ·  {m.epochs}ep", fontsize=11, pad=8)
    # Minor grid between cells.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    vmax = norm.vmax
    for i in range(n):
        for j in range(n):
            v = display[i, j]
            if np.isnan(v):
                if i != j:  # unmatched (not the diagonal): mark faintly
                    ax.text(j, i, "·", ha="center", va="center",
                            color="#c8c8c8", fontsize=10)
                continue
            txt = f"{v:+.3f}"
            if mode.annotate_count and m.holdings[i, j] > 1:
                txt += f"\nn={m.holdings[i, j]}"
            color = "white" if abs(v) > 0.55 * vmax else "#1a1a1a"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7.5)


def render_mode(spec, matrices, rungs, mode: HoldingMode, norm, n_runs, out_path):
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f4f4f4")

    nrows, ncols = len(rungs), len(AXES)
    # Extra bottom band (in inches) reserved for the shared colorbar, kept constant
    # regardless of row count so it never collides with the bottom row's tick labels.
    cbar_band = 1.4
    fig_h = 4.8 * nrows + cbar_band
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, fig_h), squeeze=False)
    filled = {axis: 0 for axis in AXES}
    for ri, ep in enumerate(rungs):
        for ci, axis in enumerate(AXES):
            m = matrices[ep][axis]
            display = masked_diff(m, mode)
            filled[axis] += int(np.isfinite(display).sum())
            _draw_matrix(axes[ri][ci], m, display, mode, norm, cmap)

    bottom = cbar_band / fig_h
    fig.subplots_adjust(top=1 - 0.9 / fig_h, bottom=bottom + 0.02,
                        left=0.06, right=0.98, wspace=0.42, hspace=0.45)
    cax = fig.add_axes([0.30, 0.40 * bottom, 0.40, 0.14 * bottom])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(
        "Δ final val loss  =  loss(column value) − loss(row value)   "
        "[ blue: column better (lower) · red: column worse ]",
        fontsize=9,
    )

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(
        f"exp117 {spec.version} · pairwise final-loss differences · {mode.blurb}\n"
        f"{n_runs} completed runs · objective: {OBJECTIVE_KEY} · {stamp}",
        fontsize=12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return filled


def main() -> None:
    spec = SWEEPS[os.environ.get("SWEEP_VERSION", "s02")]
    runs = fetch_completed_runs(spec)
    rungs = sorted({r.identity.epochs for r in runs} & ALLOWED_EPOCHS)
    if not rungs:
        raise SystemExit("No completed runs in the allowed rungs — nothing to plot.")

    matrices = {
        ep: {axis: build_axis_matrix(axis, ep, runs) for axis in AXES} for ep in rungs
    }

    # One shared color scale (from every filled cell) so the two figures are comparable.
    finite = [
        m.diff[np.isfinite(m.diff)]
        for per_axis in matrices.values()
        for m in per_axis.values()
    ]
    all_vals = np.concatenate([f for f in finite if f.size]) if finite else np.array([0.0])
    vmax = float(np.nanmax(np.abs(all_vals))) or 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)

    print(f"Completed runs: {len(runs)} · rungs: {rungs} · color scale ±{vmax:.3f}")
    for mode in MODES:
        out_path = RESULTS_DIR / spec.version / mode.filename
        filled = render_mode(spec, matrices, rungs, mode, norm, len(runs), out_path)
        print(f"  [{mode.key}] wrote {out_path.name} · filled cells {filled}")


if __name__ == "__main__":
    main()
