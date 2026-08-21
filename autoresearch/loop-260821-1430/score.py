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

    def series(key: str) -> list[float]:
        return [float(by_step[s][key]) for s in sorted(by_step) if args.lo <= s <= args.hi and key in by_step[s]]

    mfu = series(MFU_KEY)
    drops = series(DROP_KEY)
    losses = [(s, by_step[s][LOSS_KEY]) for s in sorted(by_step) if args.lo <= s <= args.hi and LOSS_KEY in by_step[s]]
    tokens = series(TOKENS_KEY)

    result = {
        "run_id": args.run_id,
        "state": run.state,
        "max_step": max(steps) if steps else -1,
        "window": [args.lo, args.hi],
        "scored_points": len(mfu),
        # A window with holes in it is not the window the loop agreed to compare, so say so loudly
        # rather than averaging whatever happens to be there.
        "window_complete": len(mfu) == args.hi - args.lo + 1,
        "mfu_mean": round(statistics.fmean(mfu), 4) if mfu else None,
        "mfu_stdev": round(statistics.stdev(mfu), 4) if len(mfu) > 1 else None,
        "tokens_per_second_mean": round(statistics.fmean(tokens), 1) if tokens else None,
        "drop_fraction_max": max(drops) if drops else None,
        "loss_last": float(losses[-1][1]) if losses else None,
        "loss_last_step": losses[-1][0] if losses else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
