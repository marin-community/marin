# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score one arm of the ragged EP64 tuning loop from its W&B history.

The metric is mean ``throughput/mfu`` over a fixed step window. The window is fixed for the whole
loop: comparing arms scored over different windows compares warmup fractions, not transports.

Also emits the two guard quantities that come from the same run -- the worst drop fraction and the
loss at the window's last step -- so a faster arm that quietly dropped tokens or changed the model
cannot be recorded as an improvement.

usage: uv run --with wandb python score.py <run-id> [--lo 5] [--hi 19]
"""

import argparse
import json
import statistics

import wandb

MFU_KEY = "throughput/mfu"
LOSS_KEY = "train/loss"
DROP_KEY = "moe/drop_fraction"
TOKENS_KEY = "throughput/tokens_per_second"


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
        "mfu_mean": round(statistics.fmean(mfu), 4) if mfu else None,
        "mfu_stdev": round(statistics.stdev(mfu), 4) if len(mfu) > 1 else None,
        "tokens_per_second_mean": round(statistics.fmean(tokens), 1) if tokens else None,
        # Drops are an outcome here, not only a guard: the whole point of the ragged transport is
        # moving tokens the pooled one clips, so a throughput win that costs drops is not a win.
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
