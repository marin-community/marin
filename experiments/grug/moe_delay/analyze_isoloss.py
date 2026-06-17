# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-overhead (iso-loss) analysis for the delayed-gradient / PP cohort.

The headline PP-viability question is not "how big is the loss gap at a fixed
step budget" but "how many *extra tokens* does a delayed (pipeline-parallel) run
need to reach the same loss as synchronous training" -- and whether the gap is a
recoverable token tax or a permanent quality floor.

For each run in a W&B group this pulls the train-loss-vs-step history, takes one
run as the synchronous reference, and reports:

  * plateau    -- robust final loss (trailing raw mean; no boundary smoothing)
  * gap        -- plateau minus the sync run's plateau
  * slope/1k   -- loss change per 1000 steps over the tail (a still-negative
                  slope means the run is still descending == token tax, not a
                  converged quality floor)
  * overhead   -- token-overhead multiplier: the step at which this run reaches
                  its own plateau loss, divided by the step at which the sync run
                  first reached that same loss (tokens = steps * batch for a
                  fixed batch). This is the "extra tokens to match" number that
                  feeds the throughput break-even in pp_throughput_model.py.

Smoothing is edge-safe (a centered rolling mean with ``min_periods``) so the tail
is not dragged toward zero -- a plain ``np.convolve(mode="same")`` pads with
zeros and corrupts exactly the final-loss region this analysis depends on.

    .venv/bin/python -m experiments.grug.moe_delay.analyze_isoloss \
        --group delay-pp-isoloss --sync delay-muon-d512-tau0-none-s0-st6000
"""

import argparse

import numpy as np
import pandas as pd
import wandb

# Tail window (in steps) over which the plateau loss and end-slope are measured.
TAIL_STEPS = 1500
PLATEAU_STEPS = 150
# A run is a quality floor (vs a still-converging token tax) if it has stopped
# descending (|slope| below this, per 1k steps) yet sits above the sync plateau.
FLOOR_SLOPE = 0.01
FLOOR_GAP = 0.02


def _curve(run) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, edge-safe-smoothed loss) for a run, sorted by step."""
    hist = run.history(keys=["train/loss", "_step"], samples=20000)
    if hist is None or "train/loss" not in getattr(hist, "columns", []):
        return np.array([]), np.array([])
    df = hist.dropna(subset=["train/loss"]).sort_values("_step")
    steps = df["_step"].to_numpy(dtype=float)
    raw = df["train/loss"].to_numpy(dtype=float)
    # Centered rolling mean with min_periods keeps the boundaries honest (unlike a
    # zero-padded convolution, which pulls the final-loss region toward zero).
    loss = pd.Series(raw).rolling(41, center=True, min_periods=5).mean().to_numpy()
    return steps, loss


def _plateau(steps: np.ndarray, loss: np.ndarray) -> float:
    """Robust final loss: mean over the last PLATEAU_STEPS steps."""
    return float(np.nanmean(loss[steps >= steps[-1] - PLATEAU_STEPS]))


def _slope_per_1k(steps: np.ndarray, loss: np.ndarray) -> float:
    """Loss change per 1000 steps from a linear fit over the tail window."""
    mask = steps >= steps[-1] - TAIL_STEPS
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(steps[mask], loss[mask], 1)
    return float(slope) * 1000.0


def _first_step_below(steps: np.ndarray, loss: np.ndarray, target: float) -> float | None:
    """First step at which the (decreasing) loss reaches target."""
    below = np.where(loss <= target)[0]
    return float(steps[below[0]]) if len(below) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="delay-pp-isoloss")
    ap.add_argument("--project", default="marin-community/marin_moe")
    ap.add_argument("--sync", required=True, help="display_name of the synchronous reference run")
    args = ap.parse_args()

    runs = list(wandb.Api().runs(args.project, filters={"group": args.group}))
    curves = {r.name: _curve(r) for r in runs}
    curves = {name: (s, lo) for name, (s, lo) in curves.items() if len(s)}
    if args.sync not in curves:
        raise SystemExit(f"sync run {args.sync!r} not found in group {args.group!r}; have {sorted(curves)}")

    sync_steps, sync_loss = curves[args.sync]
    sync_plateau = _plateau(sync_steps, sync_loss)

    print(f"group={args.group}  sync={args.sync}  sync_plateau={sync_plateau:.4f}  steps={int(sync_steps[-1])}\n")
    print(f"{'run':<46}{'plateau':>9}{'gap':>9}{'slope/1k':>10}{'overhead':>10}  tag")
    for name in sorted(curves):
        steps, loss = curves[name]
        plateau = _plateau(steps, loss)
        gap = plateau - sync_plateau
        slope = _slope_per_1k(steps, loss)
        # Token-overhead at this run's own plateau loss: it reaches that loss at
        # ~its final step; when did the sync run first reach it?
        s_sync = _first_step_below(sync_steps, sync_loss, plateau)
        s_self = _first_step_below(steps, loss, plateau)
        overhead = (s_self / s_sync) if (s_sync and s_self) else float("inf")
        if name == args.sync:
            tag = "sync-ref"
        elif gap > FLOOR_GAP and slope > -FLOOR_SLOPE:
            tag = "QUALITY-FLOOR (flat, above sync)"
        else:
            tag = "token-tax (still descending)"
        print(f"{name:<46}{plateau:>9.4f}{gap:>+9.4f}{slope:>+10.4f}{overhead:>9.2f}x  {tag}")


if __name__ == "__main__":
    main()
