# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score an autoresearch EP64 rung run: throughput metric gated on drops and finite loss.

Extends experiments/grug/moe_hero_ep/measure.py with a machine-readable contract:
diagnostics go to stderr, and stdout's LAST line is exactly one number — tokens/s over
the scored window — so the autoresearch loop can parse it. Exits 1 (no number) when the
drop gate or the finite-loss gate fails, which the loop records as a crash and reverts.
"""

import math
import statistics
import sys

import click
import wandb

WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "marin_moe"
GB200_BF16_DENSE_PEAK = 2.5e15  # dense per-GPU peak; see measure.py for provenance


def _window(rows: dict[int, dict], start: int, end: int, key: str) -> list[float]:
    return [rows[s][key] for s in range(start, end + 1) if s in rows and rows[s].get(key) is not None]


@click.command()
@click.option("--run-id", required=True)
@click.option("--start-step", type=int, required=True, help="Throughput window start (inclusive).")
@click.option("--end-step", type=int, required=True, help="Throughput window end (inclusive).")
@click.option("--drop-start", type=int, required=True, help="Drop-gate window start (inclusive).")
@click.option("--drop-end", type=int, required=True, help="Drop-gate window end (inclusive).")
@click.option("--drop-budget", type=float, required=True, help="Max mean moe/drop_fraction in the gate window.")
@click.option("--gpus", type=int, required=True)
@click.option("--peak-flops-per-gpu", type=float, default=GB200_BF16_DENSE_PEAK, show_default=True)
def main(
    run_id: str,
    start_step: int,
    end_step: int,
    drop_start: int,
    drop_end: int,
    drop_budget: float,
    gpus: int,
    peak_flops_per_gpu: float,
) -> None:
    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    config = run.config
    batch_size = config["trainer"]["trainer"]["train_batch_size"]
    seq_len = config["model"]["max_seq_len"]
    tokens_per_step = batch_size * seq_len

    keys = ["throughput/duration", "train/loss", "moe/drop_fraction"]
    rows: dict[int, dict] = {}
    lo, hi = min(start_step, drop_start), max(end_step, drop_end)
    for row in run.scan_history(keys=["_step", *keys], page_size=1000):
        step = row.get("_step")
        if step is None or not (lo <= step <= hi):
            continue
        rows.setdefault(step, {}).update({k: v for k, v in row.items() if v is not None})

    durations = _window(rows, start_step, end_step, "throughput/duration")
    expected = end_step - start_step + 1
    if len(durations) < expected:
        raise click.ClickException(f"throughput window incomplete: {len(durations)}/{expected} durations")

    losses = _window(rows, start_step, drop_end, "train/loss")
    bad = [x for x in losses if not math.isfinite(x)]
    if bad or not losses:
        raise click.ClickException(f"finite-loss gate FAILED: {len(bad)} non-finite of {len(losses)} scored losses")

    drops = _window(rows, drop_start, drop_end, "moe/drop_fraction")
    if not drops:
        raise click.ClickException("drop gate FAILED: moe/drop_fraction not logged in gate window")
    mean_drop, worst_drop = statistics.mean(drops), max(drops)

    mean_duration = statistics.mean(durations)
    tokens_per_second = tokens_per_step / mean_duration
    flops_per_example = run.summary.get("throughput/flops_per_example_analytic")
    mfu = (
        100.0 * flops_per_example * batch_size / mean_duration / (gpus * peak_flops_per_gpu)
        if flops_per_example
        else float("nan")
    )

    print(
        f"run={run_id} window={start_step}-{end_step} "
        f"step_mean={mean_duration:.4f}s step_median={statistics.median(durations):.4f}s "
        f"mfu={mfu:.3f}% drop_mean[{drop_start}-{drop_end}]={mean_drop:.4%} drop_worst={worst_drop:.4%} "
        f"budget={drop_budget:.2%}",
        file=sys.stderr,
    )
    if mean_drop > drop_budget:
        raise click.ClickException(f"drop gate FAILED: mean {mean_drop:.4%} > budget {drop_budget:.2%}")

    print(f"{tokens_per_second:.1f}")


if __name__ == "__main__":
    main()
