# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Overlay the 4plus-only and union decon val-loss sweeps on one figure.

Same loss-vs-tau sweep (p33m67 x lr0.33, one curve per compute budget) drawn for
two decon-source families on shared axes:

- **4plus-only** (decontaminated against the actually-trained source) — solid.
- **union(3, 4plus, 4plus_mind)** (the over-decontaminated version) — dotted.

Same viridis color per compute budget; the ★ "no-filter" anchor (original
12,500-window val) is identical for both, drawn once. Reads the small per-run
eval_results.json from both GCS roots.

    uv run --with matplotlib --with gcsfs python scripts/analysis/plot_decon_sweep_union_vs_4plus.py
"""

import argparse
import json
import logging
import re
from pathlib import Path

import fsspec
import matplotlib
from marin.utils import fsspec_glob

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets"
# (label, eval_root, linestyle)
FAMILIES = [
    ("4plus-only", f"{ROOT}/evals_4plus", "-"),
    ("union(3+4plus+4plus_mind)", f"{ROOT}/evals_sweep9", ":"),
]
CUTOFFS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
DECON_TAGS = [f"j{round(c * 100):03d}" for c in CUTOFFS]
ANCHOR_KEY = "eval/nemotron_cc_math_v1/4plus/loss"
SLICE = "p33m67_lr0.33"
RUNS = [
    ("3e18", "delphi-3e18-p33m67-k0p20-lr33-a003"),
    ("9e18", "delphi-9e18-p33m67-k0p20-lr33-a002"),
    ("2e19", "delphi-2e19-p33m67-k0p20-lr33-a002"),
    ("3e19", "delphi-3e19-p33m67-k0p20-lr33-a002"),
    ("9e19", "delphi-9e19-p33m67-k0p20-lr33-a002"),
    ("2e20", "delphi-2e20-p33m67-k0p20-lr33-a001"),
    ("3e20", "delphi-3e20-p33m67-k0p20-lr33-a001"),
    ("1e21", "delphi-1e21-p33m67-9p25b-lr0.33-58ebcb"),
    ("1e22", "delphi-1e22-p33m67-32p07b-lr0.33-e9132105"),
]


def latest_eval_results(run: str, eval_root: str) -> dict:
    hits = fsspec_glob(f"{eval_root}/{run}/step-*/metrics.jsonl/eval_results.json")
    if not hits:
        raise FileNotFoundError(f"no eval_results.json under {eval_root}/{run}/")
    by_step = {int(m.group(1)): p for p in hits if (m := re.search(r"step-(\d+)/", p))}
    with fsspec.open(by_step[max(by_step)]) as f:
        return json.load(f)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    cmap = plt.get_cmap("viridis")
    anchor_x = 0.95

    for i, (_scale, run) in enumerate(RUNS):
        color = cmap(i / (len(RUNS) - 1))
        anchor_drawn = False
        for _, eval_root, style in FAMILIES:
            res = latest_eval_results(run, eval_root)
            ys = [res[f"eval/decon_{tag}/loss"] for tag in DECON_TAGS]
            ax.plot(CUTOFFS, ys, style, color=color, linewidth=1.6, marker="o", markersize=3.5)
            if not anchor_drawn:
                ax.plot([anchor_x], [res[ANCHOR_KEY]], marker="*", color=color, markersize=11, linestyle="none")
                anchor_drawn = True

    color_handles = [
        plt.Line2D([], [], color=cmap(i / (len(RUNS) - 1)), marker="o", label=s) for i, (s, _) in enumerate(RUNS)
    ]
    style_handles = [
        plt.Line2D([], [], color="0.3", linestyle="-", label="4plus-only (solid)"),
        plt.Line2D([], [], color="0.3", linestyle=":", label="union 3+4plus+4plus_mind (dotted)"),
        plt.Line2D([], [], color="0.3", marker="*", linestyle="none", label="★ original 'no-filter' val"),
    ]
    legend1 = ax.legend(handles=color_handles, title="compute", fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(legend1)
    ax.legend(handles=style_handles, fontsize=8, loc="lower left", bbox_to_anchor=(1.01, 0.0))

    ax.axvline(0.925, color="0.6", linestyle=":", linewidth=1)
    ax.set_xlabel("Jaccard decontamination cutoff τ  (drop docs with max train J ≥ τ; right = more aggressive)")
    ax.set_ylabel("math val loss (nats/token)")
    ax.set_title(
        f"Val loss vs decontamination cutoff — {SLICE}\n"
        "solid = 4plus-only decon · dotted = union(3+4plus+4plus_mind) · ★ = original no-filter val"
    )
    ax.set_xticks([*CUTOFFS, anchor_x])
    ax.set_xticklabels([f"{c:.2f}" for c in CUTOFFS] + ["none"])
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / f"decon_val_loss_vs_cutoff_{SLICE}_4plus_vs_union.png"
    fig.savefig(out, dpi=150)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
