# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score one arm of the ragged EP64 tuning loop from its W&B history.

The primary metric is the run MEDIAN of ``throughput/mfu`` over a fixed step window (sparse
environmental stall steps dominate means; see DESIGN "Metric and guards"); the mean and a stall
count are emitted alongside. The window is fixed for the whole loop: comparing arms scored over
different windows compares warmup fractions, not transports.

Also emits the two guard quantities that come from the same run -- the worst drop fraction and the
loss at the window's last step -- so a faster arm that quietly dropped tokens or changed the model
cannot be recorded as an improvement.

usage: uv run --with wandb python score.py <run-id> [--lo 5] [--hi 19]
"""

import argparse
import json
import math
import statistics
import sys

import wandb

MFU_KEY = "throughput/mfu"
LOSS_KEY = "train/loss"
DROP_KEY = "moe/drop_fraction"
TOKENS_KEY = "throughput/tokens_per_second"
PEAK_KEY = "memory/peak_gib"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--lo", type=int, default=5, help="first scored step")
    parser.add_argument("--hi", type=int, default=19, help="last scored step")
    parser.add_argument("--project", default="marin-community/marin_moe")
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Read lo/hi as offsets from the run's first logged step. A restored run resumes at "
        "the checkpoint's step, so an absolute window would miss it entirely.",
    )
    parser.add_argument(
        "--expect-min-first-step",
        type=int,
        default=30000,
        help="Fail unless the run's first logged step is at least this. A restore that silently "
        "fell back to scratch logs from step 1 and would otherwise score as a valid window.",
    )
    args = parser.parse_args()

    run = wandb.Api().run(f"{args.project}/{args.run_id}")
    rows = list(run.scan_history())
    steps = [int(r["_step"]) for r in rows if r.get("_step") is not None]
    by_step: dict[int, dict] = {}
    for row in rows:
        step = row.get("_step")
        if step is None:
            continue
        by_step.setdefault(int(step), {}).update({k: v for k, v in row.items() if v is not None})

    first_step = min(by_step) if by_step else 0
    lo, hi = (first_step + args.lo, first_step + args.hi) if args.relative else (args.lo, args.hi)

    def series(key: str) -> list[float]:
        return [float(by_step[s][key]) for s in sorted(by_step) if lo <= s <= hi and key in by_step[s]]

    mfu = series(MFU_KEY)
    drops = series(DROP_KEY)
    peaks = series(PEAK_KEY)
    losses = [(s, by_step[s][LOSS_KEY]) for s in sorted(by_step) if lo <= s <= hi and LOSS_KEY in by_step[s]]
    tokens = series(TOKENS_KEY)

    result = {
        "run_id": args.run_id,
        "state": run.state,
        "max_step": max(steps) if steps else -1,
        "window": [lo, hi],
        "first_step": first_step,
        "scored_points": len(mfu),
        # A window with holes in it is not the window the loop agreed to compare, so say so loudly
        # rather than averaging whatever happens to be there.
        "window_complete": len(mfu) == hi - lo + 1,
        # The PRIMARY decision statistic (DESIGN, calibrated at iteration 0): sparse environmental
        # stall steps swing the mean by >2x the keep bar while medians hold; stall count guards
        # the median's blind spot (a treatment that CAUSES stalls must not hide behind it).
        "mfu_median": round(statistics.median(mfu), 4) if mfu else None,
        "stall_steps": (
            sum(1 for v in mfu if v < statistics.median(mfu) * 0.97) if mfu else None
        ),
        "mfu_mean": round(statistics.fmean(mfu), 4) if mfu else None,
        "mfu_stdev": round(statistics.stdev(mfu), 4) if len(mfu) > 1 else None,
        "tokens_per_second_mean": round(statistics.fmean(tokens), 1) if tokens else None,
        # Drops are an outcome here, not only a guard: the whole point of the ragged transport is
        # moving tokens the pooled one clips, so a throughput win that costs drops is not a win.
        # Engagement/confound telemetry: HloRematerialization flips on allocator-limit crossings
        # (issue #8054's +9.08% artifact); a memory-moving arm whose peak did not move did not
        # engage, and an MFU delta with a big peak move may be a remat-boundary crossing instead.
        "peak_gib_max": round(max(peaks), 2) if peaks else None,
        "drop_fraction_max": max(drops) if drops else None,
        "drop_fraction_mean": round(statistics.fmean(drops), 6) if drops else None,
        # Guard-series completeness: a killed run can leave the last rows of less-frequent keys
        # unflushed, so a "last loss" alone can silently come from an earlier step. Emit the full
        # series; the loop compares max |delta| against the paired control (data is deterministic
        # across arms at a fixed restore step, so the series pair pointwise).
        "drop_points": len(drops),
        "loss_points": len(losses),
        "loss_series": [[s, round(float(v), 6)] for s, v in losses],
        "loss_last": float(losses[-1][1]) if losses else None,
        "loss_last_step": losses[-1][0] if losses else None,
    }

    # One enforced verdict instead of a pile of fields a tired reader can skim past. Anything
    # that makes the window not the window the loop agreed to compare -- a scratch fallback, a
    # hole in any guard series, a NaN -- fails the arm here, not downstream.
    expected_points = hi - lo + 1
    numeric = [x for x in (mfu + drops + [v for _, v in losses] + peaks) if x is not None]
    problems = []
    if first_step < args.expect_min_first_step:
        problems.append(f"first_step {first_step} < {args.expect_min_first_step}: restore fell back to scratch")
    if len(mfu) != expected_points:
        problems.append(f"mfu series has {len(mfu)}/{expected_points} points")
    if len(drops) != expected_points:
        problems.append(f"drop series has {len(drops)}/{expected_points} points")
    if len(losses) != expected_points:
        problems.append(f"loss series has {len(losses)}/{expected_points} points")
    if losses and losses[-1][0] != hi:
        problems.append(f"last loss is from step {losses[-1][0]}, not window end {hi}")
    if any(not math.isfinite(float(x)) for x in numeric):
        problems.append("non-finite value in a scored series")
    result["valid"] = not problems
    result["problems"] = problems
    print(json.dumps(result, indent=2))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
