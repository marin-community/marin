# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot math val loss vs Jaccard decontamination cutoff, one curve per budget.

For a fixed (mix, lr) slice of the Delphi K=0.20 ladder, reads each scale's
9-cutoff eval (written by ``eval_decon_val_sets.py``) and draws:

  x = Jaccard decontamination cutoff tau (drop val docs whose max train
      Jaccard is >= tau; lower tau = more aggressive = smaller, cleaner set)
  y = math val loss on the tau-decontaminated paranoid val set
  one line per compute budget (3e18 ... 1e22)

The nine paranoid points per budget share the same "fully contained in val
windows" filter and differ ONLY in tau, so each line is a controlled sweep.
The original 12,500-window val ("no filter") anchor is drawn as a separate
marker: it is a different (long-doc-inclusive, window-split) distribution, so
the anchor->j090 step mixes a short-doc distribution shift with contamination
and is not directly on the paranoid curve.

Reads the small per-run eval_results.json files from GCS (a few KB each) and
writes a PNG + a CSV of every loss value locally.

    .venv/bin/python scripts/analysis/plot_decon_val_loss_vs_cutoff.py
"""

import argparse
import csv
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

EVAL_OUT_ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_sweep9"
CUTOFFS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
DECON_TAGS = [f"j{round(c * 100):03d}" for c in CUTOFFS]
ANCHOR_KEY = "eval/nemotron_cc_math_v1/4plus/loss"

# p33m67 (67% math, strongest contamination signal) x lr0.33 ladder. The eval
# is lr-invariant (established), and the decon caches are mix/lr independent;
# this is the slice whose 3-cutoff table is already published, so the 9-point
# curve extends a known result.
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
    """Read the newest-step eval_results.json for a run."""
    candidates = fsspec_glob(f"{eval_root}/{run}/step-*/metrics.jsonl/eval_results.json")
    if not candidates:
        raise FileNotFoundError(f"no eval_results.json under {eval_root}/{run}/")
    by_step = {int(m.group(1)): p for p in candidates if (m := re.search(r"step-(\d+)/", p))}
    with fsspec.open(by_step[max(by_step)]) as f:
        return json.load(f)


def collect_losses(eval_root: str) -> list[dict]:
    """Per-scale anchor + 9 decon losses, in budget order."""
    rows = []
    for scale, run in RUNS:
        results = latest_eval_results(run, eval_root)
        row = {"scale": scale, "run": run, "anchor": results[ANCHOR_KEY]}
        for tag in DECON_TAGS:
            row[tag] = results[f"eval/decon_{tag}/loss"]
        rows.append(row)
        logger.info("%s: anchor %.4f, j050 %.4f .. j090 %.4f", scale, row["anchor"], row["j050"], row["j090"])
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = ["scale", "run", "anchor", *DECON_TAGS]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows: list[dict], path: Path, slice_label: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")
    anchor_x = 0.95  # park the no-filter anchor just right of the tau sweep

    for i, row in enumerate(rows):
        color = cmap(i / (len(rows) - 1))
        ys = [row[tag] for tag in DECON_TAGS]
        ax.plot(CUTOFFS, ys, "-o", color=color, label=row["scale"], markersize=4)
        ax.plot([anchor_x], [row["anchor"]], marker="*", color=color, markersize=11, linestyle="none")

    ax.axvline(0.925, color="0.6", linestyle=":", linewidth=1)
    ax.set_xlabel("Jaccard decontamination cutoff τ  (drop docs with max train J ≥ τ; right = more aggressive)")
    ax.set_ylabel("math val loss (nats/token)")
    ax.set_title(
        f"Val loss vs decontamination cutoff — {slice_label}\n"
        "(★ = original 'no-filter' val; its short-doc distribution differs from the paranoid sets)"
    )
    ax.set_xticks([*CUTOFFS, anchor_x])
    ax.set_xticklabels([f"{c:.2f}" for c in CUTOFFS] + ["none"])
    ax.invert_xaxis()  # no-filter on the left, aggressive (0.50) on the right
    ax.grid(True, alpha=0.3)
    ax.legend(title="compute", ncol=1, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logger.info("wrote %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="plots", help="Local output directory for the PNG + CSV.")
    parser.add_argument("--eval-root", default=EVAL_OUT_ROOT, help="GCS root of the per-run eval_results.json.")
    parser.add_argument("--slice", dest="slice_label", default=SLICE, help="Slice label for filename + title.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    rows = collect_losses(args.eval_root)
    write_csv(rows, out_dir / f"decon_val_loss_vs_cutoff_{args.slice_label}.csv")
    make_plot(rows, out_dir / f"decon_val_loss_vs_cutoff_{args.slice_label}.png", args.slice_label)


if __name__ == "__main__":
    main()
