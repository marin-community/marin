# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-overhead (iso-loss) analysis for the delayed-gradient / PP cohort.

The headline PP-viability question is not "how big is the loss gap at a fixed
step budget" but "how many *extra tokens* does a delayed (pipeline-parallel) run
need to reach the same loss as synchronous training" -- and whether the gap is a
recoverable token tax or a permanent quality floor.

This pulls the train-loss-vs-step history for every run in a W&B group, takes one
run as the synchronous reference, and for a grid of target losses reports, per
run, the step at which it first reaches that loss and the ratio to the sync run
(the token-overhead multiplier, since tokens = steps * batch_size for a fixed
batch). A run whose *final* loss never reaches the sync run's final loss is
flagged as a quality floor rather than a token tax.

    .venv/bin/python -m experiments.grug.moe_delay.analyze_isoloss \
        --group delay-pp-isoloss --sync delay-muon-d512-tau0-none-s0-st6000
"""

import argparse

import numpy as np
import wandb


def _loss_curve(run) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, smoothed_loss) for a run, sorted by step."""
    hist = run.history(keys=["train/loss", "_step"], samples=20000)
    if hist is None or "train/loss" not in getattr(hist, "columns", []):
        return np.array([]), np.array([])
    df = hist.dropna(subset=["train/loss"]).sort_values("_step")
    steps = df["_step"].to_numpy(dtype=float)
    loss = df["train/loss"].to_numpy(dtype=float)
    # Light smoothing so the first-crossing step is robust to per-step noise.
    if len(loss) >= 25:
        k = 25
        loss = np.convolve(loss, np.ones(k) / k, mode="same")
    return steps, loss


def _first_step_below(steps: np.ndarray, loss: np.ndarray, target: float) -> float | None:
    """First step at which the (monotone-ish decreasing) loss drops to target."""
    below = np.where(loss <= target)[0]
    return float(steps[below[0]]) if len(below) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="delay-pp-isoloss")
    ap.add_argument("--project", default="marin-community/marin_moe")
    ap.add_argument("--sync", required=True, help="display_name of the synchronous reference run")
    ap.add_argument("--num-targets", type=int, default=8)
    args = ap.parse_args()

    runs = list(wandb.Api().runs(args.project, filters={"group": args.group}))
    curves = {r.name: _loss_curve(r) for r in runs}
    curves = {name: (s, lo) for name, (s, lo) in curves.items() if len(s)}
    if args.sync not in curves:
        raise SystemExit(f"sync run {args.sync!r} not found in group {args.group!r}; have {sorted(curves)}")

    sync_steps, sync_loss = curves[args.sync]
    sync_final = float(sync_loss[-1])
    # Target band: from a loss both sync and the worst run comfortably pass, down
    # toward (but not below) the sync run's final loss.
    worst_final = max(float(lo[-1]) for _, lo in curves.values())
    lo_target, hi_target = sync_final + 0.05, worst_final
    targets = np.linspace(hi_target, lo_target, args.num_targets)

    print(f"group={args.group}  sync={args.sync}  sync_final_loss={sync_final:.4f}")
    print(f"target band: {hi_target:.3f} -> {lo_target:.3f}\n")

    for name in sorted(curves):
        steps, loss = curves[name]
        final = float(loss[-1])
        floor = final > sync_final + 0.02  # never reaches sync quality within budget
        ratios = []
        for t in targets:
            s_run = _first_step_below(steps, loss, t)
            s_sync = _first_step_below(sync_steps, sync_loss, t)
            if s_run and s_sync and s_sync > 0:
                ratios.append(s_run / s_sync)
        overhead = float(np.median(ratios)) if ratios else float("nan")
        tag = "QUALITY-FLOOR" if floor else "token-tax"
        print(f"{name:<46} final={final:.4f}  median_token_overhead={overhead:.2f}x  [{tag}]")


if __name__ == "__main__":
    main()
